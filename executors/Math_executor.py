import os
import re
from typing import Any, Optional
from .executor_types import ExecuteResult, Executor
from openai import OpenAI  # if not already imported

# Judge model is served through the same relay endpoint as MultihopQA.
# The API key is read from the OPENAI_API_KEY environment variable.
JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "https://www.openai-labs.com/v1")


def _normalize_math_answer(ans: str) -> str:
    """
    Lowercase, collapse whitespace, strip wrapping quotes/brackets and trailing
    punctuation, so punctuation/space-only differences don't trigger the GPT
    judge (mirrors qa/eval_hotpot_simple.normalize_qa_answer).
    """
    ans = ans.strip().lower()
    ans = re.sub(r"\s+", " ", ans)
    ans = ans.strip('"\'`()[]{}*')
    ans = ans.rstrip(".,;:!?")
    return ans.strip()


def _find_boxed(text: str) -> Optional[str]:
    """Return the contents of the last well-formed ``\\boxed{...}`` in *text*.

    Brace-aware, so nested LaTeX such as ``\\boxed{\\frac{1}{36}}`` is captured
    whole. A box left open by a truncated response is ignored, and an earlier
    complete box is preferred over it.
    """
    marker = r"\boxed{"
    last = None
    pos = 0
    while True:
        idx = text.find(marker, pos)
        if idx == -1:
            return last
        start = idx + len(marker)
        depth = 1
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    val = text[start:i].strip()
                    if val:
                        last = val
                    pos = i + 1
                    break
        else:
            return last  # ran off the end with this box still open


def _unwrap_boxed(val: str) -> str:
    """Strip a ``\\boxed{...}`` that wraps *val* entirely (``$``/``$$`` allowed).

    Models very often answer with ``**Answer:** $\\boxed{\\frac{1}{36}}$``; the
    bare value compares exactly far more often than the wrapped one does, which
    saves a GPT-judge round trip and its occasional false "No".
    """
    s = val.strip().strip("$").strip()
    if not s.startswith(r"\boxed{"):
        return val
    start = len(r"\boxed{")
    depth = 1
    for i in range(start, len(s)):
        if s[i] == "{":
            depth += 1
        elif s[i] == "}":
            depth -= 1
            if depth == 0:
                inner = s[start:i].strip()
                # Only unwrap a box spanning the entire value.
                return inner if inner and i == len(s) - 1 else val
    return val


def _clean_value(val: str) -> str:
    """Normalize a captured answer value for comparison.

    Strips outer punctuation and ``$``/``$$`` math delimiters and unwraps a
    ``\\boxed{...}`` that spans the whole value. The dataset's gold answers are
    stored bare (``\\frac{18}{5}``, never ``$\\frac{18}{5}$``), so removing the
    wrapping delimiters lets the exact-match fast path fire instead of always
    paying for a GPT-judge round trip.
    """
    val = val.strip().strip('.,;:!? ').strip()
    val = val.strip("$").strip()
    return _unwrap_boxed(val)


def _extract_final_answer(text: str) -> Optional[str]:
    """Pull the final answer from a model response.

    The math system prompt asks the model to finish with a line of the form
    ``**`Answer:`** <final answer>``. Match that (and a few common variants)
    and return just the answer value. When no such line is present — e.g. the
    response was cut off, or the model answered with ``\\boxed{...}`` only —
    fall back to the last ``\\boxed{...}`` and finally to None, in which case
    callers use the raw response as-is.
    """
    if not text:
        return None
    # The last "Answer: <x>" line, matching bold/backtick variants. The first
    # pattern is the format the system prompt actually asks for
    # ("**`Answer:`** <value>"); the looser ones below catch the variants.
    patterns = [
        r'\*\*\s*`\s*Answer\s*:\s*`\s*\*\*\s*(.+?)(?:\n|$)',
        r'\*\*\s*`?\s*Answer\s*`?\s*\*\*\s*:\s*(.+?)(?:\n|$)',
        r'\*\*\s*`?\s*Answer\s*`?\s*:\s*\*\*\s*(.+?)(?:\n|$)',
        r'\*\*Answer\*\*\s*:\s*(.+?)(?:\n|$)',
        r'Answer\s*:\s*(.+?)(?:\n|$)',
        r'\*\*answer\*\*\s*:\s*(.+?)(?:\n|$)',
        r'answer\s*:\s*(.+?)(?:\n|$)',
    ]
    for pattern in patterns:
        matches = list(re.finditer(pattern, text, re.IGNORECASE))
        if matches:
            val = _clean_value(matches[-1].group(1))
            if val:
                return val
    boxed = _find_boxed(text)
    return _clean_value(boxed) if boxed else None


class MathExecutor(Executor):
    def execute(self, func: str, tests: list, timeout: int = 5) -> ExecuteResult:
        # MultihopQA does not use execute in this implementation
        raise NotImplementedError("MultiHopQAExecutor.execute is not used for evaluation.")

    def evaluate(self, answer: str, golden_truth: str, timeout: int = 5) -> bool:
        """
        Evaluate whether the predicted answer matches the gold answer.

        The model response may be verbose reasoning ending in an
        ``**`Answer:`** <value>`` line; extract that final value before
        comparing so a correct answer isn't marked wrong just because it was
        wrapped in reasoning text.

        Args:
            answer (str): The predicted answer string (possibly verbose).
            golden_truth (str): The ground-truth answer string.
            timeout (int): Timeout (unused).

        Returns:
            bool: True if answers are considered equivalent (exact match,
            punctuation/whitespace-normalized match, or semantically).
        """

        # A verbose model response may carry reasoning before the final answer;
        # prefer the value on the last "Answer:" line when present.
        ans = _extract_final_answer(answer) or answer

        # First check exact match
        if ans.strip().lower() == golden_truth.strip().lower():
            return True

        # Second check normalized match (case/punctuation/whitespace-insensitive)
        if _normalize_math_answer(ans) == _normalize_math_answer(golden_truth):
            return True

        prompt = f"""
        Determine if the following two answers are equivalent in the context of a mathematical problem.

        Answer 1: {ans}
        Answer 2: {golden_truth}

        Respond only with "Yes" or "No".
        """

        try:
            client = OpenAI(
                base_url=JUDGE_BASE_URL,
                api_key=os.getenv("OPENAI_API_KEY"),
                timeout=60.0,
            )  # 60 second timeout to prevent hanging
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt.strip()}],
                temperature=0.00001,
                max_tokens=8,
            )
            result = response.choices[0].message.content.strip().lower()
            print("executor eval:", result)
            return result.startswith("yes")
        except Exception:
            print("executor eval: exception")
            return False
