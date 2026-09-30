#!/usr/bin/env python3
"""
Generate pitfalls & flawed implementations for dataset/code/train/code.json
using Gemini via the yunwu.ai OpenAI-compatible API.

References:
- code/gen_code_pitfalls_qwen35.py             — prompts, output schema, resume logic
- qa/gen_multihop_decomposition_data.py        — API client / retry patterns

Output: {output_dir}/gemini.jsonl (append-only resume log) and gemini.json,
with the same schema as the qwen files: the original item plus 'pitfalls',
'flawed_impl', and 'raw_text' filled with Gemini-generated content.

Quality controls (each item is retried up to 3 attempts):
- truncated output (finish_reason == 'length')   -> double max_tokens (cap 8192)
- empty pitfalls / flawed implementations        -> regenerate
- flawed_impl that does not compile as Python    -> regenerate
- damaged `->` arrows are repaired with the same logic as gen_code_pitfalls_qwen35.py

Resume: completed items (keyed by sample id) are skipped on re-run, so the
script can be interrupted and restarted safely. Failed items are logged to
{output_dir}/gemini_errors.jsonl and retried automatically on the next run.

Example:
  python code/gen_code_pitfalls_gemini.py \
    --input_json dataset/code/train/apps_original.json \
    --output_dir dataset/code/gemini_new \
    --workers 8
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import jsonlines
from openai import OpenAI, APIStatusError, APIConnectionError, RateLimitError, APITimeoutError
from tenacity import retry, stop_after_attempt, wait_random_exponential, retry_if_exception_type
from tqdm import tqdm

# ──────────────────── API configuration ────────────────────
API_BASE = os.getenv("OPENAI_API_BASE", "https://yunwu.ai/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "<REDACTED_HARDCODED_KEY>")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gemini-3.5-flash")

MODEL = DEFAULT_MODEL
_client = None

def get_client() -> OpenAI:
    global _client
    if _client is None:
        api_key = API_KEY or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Set OPENAI_API_KEY before running this script.")
        _client = OpenAI(base_url=API_BASE, api_key=api_key)
    return _client


# ──────────────────── Prompts (same as gen_code_pitfalls_qwen35.py) ────────────────────

SYSTEM_PROMPT = (
    "You are an AI assistant for Python coding. Given a function signature and docstring, "
    "use your knowledge to propose potential pitfalls for the implementation, "
    "and list the possible pitfalls, and generate up to 6 flawed implementations "
    "specific to the function signature that cover as many pitfalls as possible. "
    "Use <Pitfalls> and <Flawed Implementations> before pitfalls and implementations."
)

FEW_SHOT = """Example:
[Function Signature]:
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    \"\"\"Check if any two numbers in the list are closer than the threshold.\"\"\"\n
<Pitfalls>:
1. **Empty or Single-Element Lists** must return `False`, not `True`.
2. **Duplicate Values** must be compared (difference 0), so never drop duplicates.
3. Always use **absolute difference** (`abs(a - b)`), not raw subtraction.
4. Use the correct **strictness** (`< threshold`, not `<=`).
5. Ensure you don't **exit too early**—check all distinct pairs.

[Flawed Implementations]:

```python
def has_close_elements_v1(numbers: List[float], threshold: float) -> bool:
    # BUG: returns True for empty or single-element lists
    if len(numbers) < 2:
        return True
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False

def has_close_elements_v2(numbers: List[float], threshold: float) -> bool:
    # BUG: removes duplicates, so identical values never compared
    numbers = sorted(set(numbers))
    for i in range(len(numbers)-1):
        if abs(numbers[i+1] - numbers[i]) < threshold:
            return True
    return False

def has_close_elements_v3(numbers: List[float], threshold: float) -> bool:
    # BUG: uses raw subtraction instead of abs()
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if (numbers[i] - numbers[j]) < threshold:
                return True
    return False

def has_close_elements_v4(numbers: List[float], threshold: float) -> bool:
    # BUG: uses <= instead of <, misclassifies exactly-threshold pairs
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) <= threshold:
                return True
    return False

def has_close_elements_v5(numbers: List[float], threshold: float) -> bool:
    # BUG: breaks out of outer loop too soon
    for i in range(len(numbers)-1):
        for j in range(i+1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
            break   # <-- this break prevents checking all j for each i
    return False
```"""


# ──────────────────── Text helpers (kept in sync with gen_code_pitfalls_qwen35.py) ────────────────────

def _normalize_func_sign_for_prompt(item: dict) -> str:
    func_sign = str(item.get("func_sign", "") or item.get("prompt", ""))
    docstring = str(item.get("docstring", ""))
    func_sign = re.sub(r"^```python\s*\n?", "", func_sign)
    func_sign = re.sub(r"\n?```\s*$", "", func_sign)
    func_sign = func_sign.strip()
    has_docstring_in_sign = '"""' in func_sign or "'''" in func_sign
    if docstring and not has_docstring_in_sign:
        if not func_sign.endswith("\n"):
            func_sign += "\n"
        func_sign += f'    """{docstring}"""\n'
    return func_sign


def split_raw_text(raw_text: str):
    pitfalls = ""
    flawed_impl = ""
    split_markers = [
        "<Flawed Implementations>",
        "[Flawed Implementations]:",
        "[Flawed Implementations]",
    ]
    for marker in split_markers:
        idx = raw_text.find(marker)
        if idx != -1:
            pitfalls = raw_text[:idx].strip()
            pitfalls = pitfalls.replace("<Pitfalls>:", "<Pitfalls>:").strip()
            flawed_impl = raw_text[idx + len(marker):].strip()
            if flawed_impl and not flawed_impl.startswith(":"):
                flawed_impl = ":\n\n" + flawed_impl
            return pitfalls, flawed_impl
    pitfalls = raw_text
    return pitfalls, flawed_impl


_DEF_HEAD_RE = re.compile(r"^(\s*(?:async\s+)?def\s+\w+\s*)\(")


def _find_close_paren(line: str, open_idx: int) -> Optional[int]:
    """Return the index of the ')' matching open_idx, or None if unbalanced.

    Skips quoted strings so defaults like `def f(x: str = ")")` don't confuse
    the depth count.
    """
    n = len(line)
    depth = 0
    i = open_idx
    while i < n:
        ch = line[i]
        if ch in "'\"":
            quote = ch
            i += 1
            while i < n and line[i] != quote:
                i += 1
            if i >= n:
                return None
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return None


def _match_def_dash(line: str):
    """Return (close_idx, type_start) if the line starts with `def name (` whose
    argument list is directly followed by ` - ` (a damaged arrow); else None.

    Intact `->` arrows return None.
    """
    match = _DEF_HEAD_RE.match(line)
    if not match:
        return None
    close = _find_close_paren(line, match.end() - 1)
    if close is None:
        return None
    n = len(line)
    j = close + 1
    while j < n and line[j] in " \t":
        j += 1
    if j >= n or line[j] != "-":
        return None
    k = j + 1
    while k < n and line[k] in " \t":
        k += 1
    if k < n and line[k] == ">":
        return None  # intact '->'
    return close, k


def _parse_annotation(text: str, start: int) -> Optional[int]:
    """Parse one annotation expression at `start`; return the index just past it.

    Supports plain names (`int`), dotted names (`np.ndarray`), quoted
    annotations (`'Cons'`), parenthesized tuples (`(bool, bool)`), and bracket
    generics with arbitrary nesting (`Optional[List[int]]`).
    """
    n = len(text)
    i = start
    while i < n and text[i] in " \t":
        i += 1
    if i >= n:
        return None
    if text[i] in "'\"":
        quote = text[i]
        i += 1
        while i < n and text[i] != quote:
            i += 1
        if i >= n:
            return None
        return i + 1
    if text[i] == "(":
        depth = 0
        while i < n:
            ch = text[i]
            if ch in "'\"":
                quote = ch
                i += 1
                while i < n and text[i] != quote:
                    i += 1
                if i >= n:
                    return None
            elif ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        if depth != 0:
            return None
    else:
        if not (text[i].isalpha() or text[i] == "_"):
            return None
        while i < n and (text[i].isalnum() or text[i] == "_"):
            i += 1
        while i + 1 < n and text[i] == "." and (text[i + 1].isalpha() or text[i + 1] == "_"):
            i += 1
            while i < n and (text[i].isalnum() or text[i] == "_"):
                i += 1
    while True:
        j = i
        while j < n and text[j] in " \t":
            j += 1
        if j >= n or text[j] != "[":
            break
        depth = 0
        i = j
        while i < n:
            ch = text[i]
            if ch == "[":
                depth += 1
            elif ch == "]":
                depth -= 1
                if depth == 0:
                    i += 1
                    break
            i += 1
        if depth != 0:
            return None
    return i


def _parse_ret_type_end(text: str, start: int) -> Optional[int]:
    """Parse a return annotation starting at `start`; return the index of its ':'.

    Handles plain types (`int`), quoted annotations (`'Cons'`), nested generics
    (`Optional[List[int]]`), and unions (`dict[str, int] | None`, `int or bool`).
    Returns None unless a well-formed `<annotation> :` tail follows, in which
    case the line is left untouched.
    """
    end = _parse_annotation(text, start)
    if end is None:
        return None
    n = len(text)
    while True:
        j = end
        while j < n and text[j] in " \t":
            j += 1
        if j < n and text[j] == "|":
            nxt = _parse_annotation(text, j + 1)
        elif (
            j + 1 < n
            and text[j:j + 2] == "or"
            and (j + 2 >= n or not (text[j + 2].isalnum() or text[j + 2] == "_"))
        ):
            nxt = _parse_annotation(text, j + 2)
        else:
            nxt = None
        if nxt is None:
            break
        end = nxt
    j = end
    while j < n and text[j] in " \t":
        j += 1
    return j if j < n and text[j] == ":" else None


def _fix_damaged_arrows(text: str) -> str:
    """Repair `def f(...) - Type:` -> `def f(...) -> Type:` in model output.

    Defensive backport of the repair from gen_code_pitfalls_qwen35.py: the
    fine-tuned 2B model sometimes dropped the '>' of '->'. Only a line whose
    `def` argument list is directly followed by ` - <well-formed annotation>:`
    is rewritten; intact `->` arrows, ambiguous lines (e.g. `f(x) - y`), and
    truncated generations are left alone.
    """
    if not text:
        return text
    fixed_lines = []
    for line in text.split("\n"):
        hit = _match_def_dash(line)
        if hit is not None:
            close, type_start = hit
            colon = _parse_ret_type_end(line, type_start)
            if colon is not None:
                line = line[:close + 1] + " -> " + line[type_start:colon] + line[colon:]
        fixed_lines.append(line)
    return "\n".join(fixed_lines)


def get_sample_id(item: dict, prompt_key: str) -> str:
    if "unique_id" in item:
        return str(item["unique_id"])
    if "task_id" in item:
        return str(item["task_id"])
    if "id" in item:
        return str(item["id"])
    func_sign = str(item.get("func_sign", "") or item.get("prompt", ""))
    entry_point = str(item.get("entry_point", ""))
    question = str(item.get("question", ""))
    raw = f"{func_sign}|||{question}|||{entry_point}"
    if raw.strip("|||"):
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    raise KeyError(f"Cannot generate sample ID: missing keys in {list(item.keys())}")


def load_completed_ids(jsonl_path: str, prompt_key: str) -> set:
    completed = set()
    if not os.path.exists(jsonl_path):
        return completed
    try:
        with jsonlines.open(jsonl_path, mode="r") as reader:
            for item in reader:
                try:
                    completed.add(get_sample_id(item, prompt_key))
                except (KeyError, jsonlines.InvalidLineError):
                    continue
    except Exception as e:
        print(f"Warning: error reading {jsonl_path}: {e}. Treating as empty.")
        return set()
    return completed


def filter_pending(data: list, completed: set, prompt_key: str) -> list:
    pending = []
    for item in data:
        try:
            if get_sample_id(item, prompt_key) not in completed:
                pending.append(item)
        except KeyError:
            pending.append(item)
    return pending


def jsonl_to_json(jsonl_path: str, json_path: str) -> None:
    items = []
    with jsonlines.open(jsonl_path, mode="r") as reader:
        for item in reader:
            items.append(item)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)
    print(f"Converted {len(items)} records: {jsonl_path} -> {json_path}")


# ──────────────────── LLM call + validation ────────────────────

TRANSIENT_ERRORS = (
    RateLimitError,
    APIStatusError,
    APIConnectionError,
    APITimeoutError,
)

MAX_MAX_TOKENS = 8192


@retry(
    retry=retry_if_exception_type(TRANSIENT_ERRORS),
    wait=wait_random_exponential(min=1, max=60),
    stop=stop_after_attempt(6),
)
def call_api(client: OpenAI, messages: list, temperature: float, max_tokens: int):
    return client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        timeout=120,
    )


def _clean_code_for_compile(flawed_impl: str) -> str:
    """Strip the ':\\n\\n' prefix and markdown fences before compile()."""
    code = flawed_impl.lstrip(":\n ")
    code = re.sub(r"^```[a-zA-Z]*\s*\n?", "", code)
    code = re.sub(r"\n?```\s*$", "", code)
    return code


def validate_output(pitfalls: str, flawed_impl: str) -> Optional[str]:
    """Return None if the output looks usable, else a reason string."""
    if not pitfalls.strip():
        return "empty pitfalls"
    if not flawed_impl.strip():
        return "empty flawed implementations"
    if "def " not in flawed_impl:
        return "no function definitions in flawed implementations"
    try:
        compile(_clean_code_for_compile(flawed_impl), "<flawed_impl>", "exec")
    except SyntaxError as e:
        return f"syntax error: {e}"
    return None


def generate_item(client: OpenAI, item: dict, args: argparse.Namespace) -> dict:
    """Generate pitfalls + flawed implementations for one item.

    Retries up to 3 times on truncation / empty / malformed / non-compiling
    output; raises RuntimeError with the last reason if all attempts fail.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": f"{FEW_SHOT}\n\n[Function Signature]:\n{_normalize_func_sign_for_prompt(item).strip()}",
        },
    ]
    max_tok = args.max_tokens
    last_reason = "unknown"
    for _ in range(3):
        resp = call_api(client, messages, args.temperature, max_tok)
        content = resp.choices[0].message.content
        finish = resp.choices[0].finish_reason
        if finish == "length":
            if max_tok < MAX_MAX_TOKENS:
                max_tok = min(max_tok * 2, MAX_MAX_TOKENS)
                last_reason = f"truncated, retrying with max_tokens={max_tok}"
                continue
            last_reason = "truncated at max_tokens limit"
            break
        if not content or not content.strip():
            last_reason = "empty completion"
            continue
        pitfalls, flawed_impl = split_raw_text(content.strip())
        flawed_impl = _fix_damaged_arrows(flawed_impl)
        reason = validate_output(pitfalls, flawed_impl)
        if reason:
            last_reason = reason
            continue
        out_item = dict(item)
        out_item["pitfalls"] = pitfalls
        out_item["flawed_impl"] = flawed_impl
        if "raw_texts" in item:
            out_item["raw_texts"] = content
        else:
            out_item["raw_text"] = content
        return out_item
    raise RuntimeError(f"failed after 3 attempts: {last_reason}")


# ──────────────────── Main ────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Generate code pitfalls with Gemini via yunwu.ai")
    parser.add_argument("--input_json", default="dataset/code/train/code.json")
    parser.add_argument("--output_dir", default="dataset/code/gemini_new")
    parser.add_argument("--model", default=None,
                        help=f"Model name (default: OPENAI_MODEL env or {DEFAULT_MODEL})")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_tokens", type=int, default=2048,
                        help="Doubled on truncation, capped at 8192 (default: 2048)")
    parser.add_argument("--workers", type=int, default=8, help="Concurrent API workers")
    parser.add_argument("--prompt_key", default="func_sign")
    parser.add_argument("--test", action="store_true", help="Test mode: only run on 4 samples")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    global MODEL
    MODEL = args.model or DEFAULT_MODEL

    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)
    if args.limit:
        data = data[: args.limit]
    if args.test:
        data = data[:4]
        print(f"Test mode — using first {len(data)} samples")

    os.makedirs(args.output_dir, exist_ok=True)
    jsonl_path = os.path.join(args.output_dir, "gemini.jsonl")
    json_path = os.path.join(args.output_dir, "gemini.json")
    errors_path = os.path.join(args.output_dir, "gemini_errors.jsonl")

    completed_ids = load_completed_ids(jsonl_path, args.prompt_key)
    pending = filter_pending(data, completed_ids, args.prompt_key)
    print(f"API: {API_BASE} | model={MODEL} | temperature={args.temperature} | "
          f"max_tokens={args.max_tokens} | workers={args.workers}")
    print(f"Resume: {len(completed_ids)}/{len(data)} done, {len(pending)} pending")

    if not pending:
        jsonl_to_json(jsonl_path, json_path)
        return

    client = get_client()
    write_lock = threading.Lock()
    done = 0
    errors = 0
    start = time.time()

    with jsonlines.open(jsonl_path, mode="a") as writer, tqdm(total=len(pending), desc="gemini") as pbar:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(generate_item, client, item, args): item for item in pending}
            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    out_item = fut.result()
                    with write_lock:
                        writer.write(out_item)
                        writer._fp.flush()
                        done += 1
                        if done % 50 == 0:
                            os.fsync(writer._fp.fileno())
                except Exception as e:
                    errors += 1
                    with write_lock:
                        with open(errors_path, "a", encoding="utf-8") as ef:
                            ef.write(json.dumps({
                                "sample_id": get_sample_id(item, args.prompt_key),
                                "func_sign": str(item.get("func_sign", ""))[:200],
                                "error": str(e),
                            }, ensure_ascii=False) + "\n")
                pbar.update(1)
                pbar.set_postfix({"done": done, "err": errors})

    jsonl_to_json(jsonl_path, json_path)
    elapsed = time.time() - start
    print(f"Finished: {done} new items, {errors} errors in {elapsed / 60:.1f} min.")
    if errors:
        print(f"Errors logged to {errors_path} — rerun the script to retry them automatically.")


if __name__ == "__main__":
    main()
