import argparse
import json
import os
import random
import re
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from openai import OpenAI

from generators.model_HF import (
    HFModelBase,
    Message,
    _force_sdpa_backend,
    _warmup_generate,
)

ACTOR_LABEL = "llama3_1_8b"  # label used in result filenames / cost table

# yunwu.ai/v1 relay endpoint for running the actor as an API call instead of
# loading it locally (opt-in via --use_api_actor). Same key convention as
# code/eval_pitfalls_v2.py.
YUNWU_BASE_URL = "https://yunwu.ai/v1"
DEFAULT_YUNWU_API_KEY = os.getenv(
    "YUNWU_API_KEY",
    "<REDACTED_HARDCODED_KEY>",
)

# Decomposition prompt, identical to the math SFT/DPO training prompt
# (math/LoRA_Qwen35_Math_4090.py SYSTEM_PROMPT + "PROBLEM:" marker, no few-shot).
MATH_DECOMP_SYSTEM_PROMPT = (
    "You are a mathematics education assistant. Given a math problem, identify "
    "and clearly explain common mistakes and pitfalls students may encounter."
)


# ───────────────────────────  API actor model  ──────────────────────────

class YunwuActor:
    """
    Actor backed by an OpenAI-compatible chat endpoint (default: yunwu.ai/v1),
    used when --use_api_actor is passed. Implements the same surface the dot
    loop expects from a local HF model: `is_chat`, `name`, and
    `generate_chat(messages, max_tokens, temperature, num_comps) -> str`.
    """

    def __init__(self, model_name: str, base_url: str, api_key: str):
        self.name = model_name
        self.is_chat = True
        self._client = OpenAI(base_url=base_url, api_key=api_key, timeout=180.0)

    def generate_chat(self, messages, max_tokens: int = 2048,
                      temperature: float = 0.2, num_comps: int = 1):
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        last_err = None
        for attempt in range(1, 4):
            try:
                resp = self._client.chat.completions.create(
                    model=self.name,
                    messages=msgs,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=0.95,
                    n=num_comps,
                )
                contents = [c.message.content or "" for c in resp.choices]
                return contents[0] if num_comps == 1 else contents
            except Exception as e:
                last_err = e
                print(f"  ⚠ API actor error (attempt {attempt}/3): {type(e).__name__}: {e}")
                if attempt < 3:
                    time.sleep(2 ** attempt)
        print(f"  ⛔ API actor failed after retries: {last_err}")
        return "" if num_comps == 1 else [""] * num_comps


# ───────────────────────────  local HF model  ───────────────────────────

class LocalHFModel(HFModelBase):
    """
    Chat model loaded from a LOCAL directory (e.g. models/Meta-Llama-3.1-8B-Instruct,
    models/Qwen3.5-2B-math-merged2). Inherits the robust generate_chat() of
    HFModelBase (left padding, OOM guard, logits clamp, gpt_usage accounting).
    """

    def __init__(self, model_path: str, label: str, device: str = "cuda:0"):
        try:
            _force_sdpa_backend()
        except Exception:
            pass

        tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token

        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        # Qwen3.5 is a multimodal model — load it with the same class the
        # project's own pitfall-generation script uses (math/gen_math_pitfalls_qwen35.py),
        # then fall back to AutoModelForCausalLM for the Llama actor.
        model = None
        try:
            from transformers import AutoModelForMultimodalLM
            model = AutoModelForMultimodalLM.from_pretrained(
                model_path,
                torch_dtype=dtype,
                device_map={"": device},
            )
        except Exception:
            model = None

        if model is None:
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map={"": device},
                    attn_implementation="sdpa",
                )
            except TypeError:
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=dtype,
                    device_map={"": device},
                )

        model.eval()

        super().__init__(label, model, tokenizer)
        _warmup_generate(self.model, self.tokenizer, self.model.device)

    def prepare_prompt(self, messages) -> str:
        chat = [{"role": m.role, "content": m.content} for m in messages]
        try:
            # Qwen3.5 chat template takes enable_thinking; Llama's ignores it
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except Exception:
            return self.tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=True
            )

    def extract_output(self, output: str, prompt: str) -> str:
        # Some chat templates (e.g. Llama-3.1) produce a duplicated BOS — strip one.
        if output.startswith("<|begin_of_text|><|begin_of_text|>"):
            output = output[len("<|begin_of_text|>"):]

        if output.startswith(prompt):
            output = output[len(prompt):]
        else:
            # Fallback: take everything after the last assistant header.
            for marker in (
                "<|start_header_id|>assistant<|end_header_id|>",
                "<|im_start|>assistant\n",
            ):
                if marker in output:
                    output = output.split(marker)[-1]
                    break

        # Trim any remaining special tokens (after assistant-section extraction).
        for tok in (
            "<|eot_id|>", "<|im_end|>", "<|endoftext|>", "</s>",
            "<|start_header_id|>",
        ):
            if tok in output:
                output = output.split(tok)[0]
                break
        return output.strip()


def load_local_model(model_path: str, label: str, device: str = "cuda:0") -> LocalHFModel:
    if not os.path.isdir(model_path):
        raise FileNotFoundError(
            f"Model directory not found: {model_path}\n"
            f"Please upload it to the server first."
        )
    return LocalHFModel(model_path, label, device)


# ────────────────────────  mistake-insights decomposer  ─────────────────

class MathDecomposer:
    """
    Wraps a local Qwen3.5-2B (base / SFT-merged / DPO-merged) and produces the
    math "mistake insights"/pitfalls decomposition, using the exact prompt format
    the SFT/DPO models were trained on (system prompt + `PROBLEM:` marker,
    no few-shot demonstrations).
    """

    def __init__(self, model: LocalHFModel):
        self.model = model
        self.label = model.name

    def generate(self, problem: str, temperature: float = 0.1) -> str:
        messages = [
            Message(role="system", content=MATH_DECOMP_SYSTEM_PROMPT),
            Message(role="user", content=f"PROBLEM:\n{problem.strip()}"),
        ]
        for attempt, temp in enumerate((temperature, 0.3)):
            out = self.model.generate_chat(
                messages=messages, max_tokens=900, temperature=temp
            )
            if out and out.strip():
                return out.strip()
            print(f"  ⚠ empty decomposition (attempt {attempt + 1}), retrying...")
        return ""


# ────────────────────────  dataset helpers  ─────────────────────────────

def load_dataset(path: str) -> list:
    """Load a JSON array or JSONL file into a list of dicts."""
    with open(path, encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f)
        try:
            obj = json.load(f)
            return [obj]
        except json.JSONDecodeError:
            f.seek(0)
            return [json.loads(line) for line in f if line.strip()]


def extract_golden_answer(solution: str) -> str:
    """
    The dataset `solution` field is a full reference solution ending in
    `\boxed{<answer>}`. Return the content of the LAST `\boxed{...}` (with
    balanced-brace handling so `\boxed{\frac{1}{36}}` works). Falls back to the
    whole string when no box is present.
    """
    s = (solution or "").strip()
    idx = s.rfind("\\boxed{")
    if idx == -1:
        return s
    i = idx + len("\\boxed{")
    depth, j = 1, i
    while j < len(s) and depth > 0:
        if s[j] == "{":
            depth += 1
        elif s[j] == "}":
            depth -= 1
        j += 1
    return s[i:j - 1].strip()


# ────────────────────────  answer extraction  ───────────────────────────

def _clean_extracted(ans: str) -> str:
    """Strip markdown/LaTeX artifacts around an extracted answer."""
    ans = ans.strip()
    ans = re.sub(r'^\*{1,2}\s*', '', ans)
    ans = re.sub(r'^`+\*{0,2}\s*', '', ans)
    ans = re.sub(r'\s*\*{0,2}`+$', '', ans)
    ans = re.sub(r'\s*\*{1,2}$', '', ans)
    ans = re.sub(r'^\\\(\s*', '', ans)
    ans = re.sub(r'\s*\\\)$', '', ans)
    ans = re.sub(r'^\$\s*', '', ans)
    ans = re.sub(r'\s*\$$', '', ans)
    return ans.strip()


def extract_final_answer(text: str) -> str:
    """Reduce the raw actor output to the final answer string."""
    patterns = [
        r'\*\*\s*`?\s*Answer\s*`?\s*\*\*\s*:\s*(.+?)(?:\n|$)',
        r'\*\*\s*`?\s*Answer\s*`?\s*:\s*\*\*\s*(.+?)(?:\n|$)',
        r'\*\*Answer\*\*\s*:\s*(.+?)(?:\n|$)',
        r'Answer\s*:\s*(.+?)(?:\n|$)',
        r'\*\*answer\*\*\s*:\s*(.+?)(?:\n|$)',
        r'answer\s*:\s*(.+?)(?:\n|$)',
        r'\\boxed\{([^}]+)\}',
        r'(?:^|\n)\s*(?:Therefore|Thus|So|Hence|The\s+final\s+answer\s+is)[,:]\s*(.+?)(?:\n|$)',
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            return _clean_extracted(matches[-1])

    # ── No explicit answer marker: the model may still have *computed* the
    # answer inline (or been cut off by max_tokens).  Extract the most likely
    # conclusion from the tail instead of blindly returning the last line —
    # which is often an intermediate step or a half-finished expression.
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    if not lines:
        return text.strip()
    tail = "\n".join(lines[-2:])  # conclusions live in the last 1-2 lines

    def _tidy(s: str) -> str:
        """Strip leading/trailing markdown/LaTeX delimiters and punctuation."""
        s = _clean_extracted(s)
        s = re.sub(r'[\s.,;:!]+$', '', s)
        return s.rstrip('$').strip()

    # a) the value after the last "=" sign — handles chained equations, e.g.
    #    "... = 1344/2500 = 336/625." → "336/625"
    eq = re.findall(r'=\s*([^=\n]+?)(?:[\n$]|$)', tail)
    if eq:
        cand = _tidy(eq[-1])
        if cand and len(cand) <= 200:
            return cand

    # b) the last inline-math span $...$; keep its trailing "= X" if present
    spans = re.findall(r'\$([^$]+?)\$', tail)
    if spans:
        span = spans[-1].strip()
        eq2 = re.findall(r'=\s*([^=\n]+?)$', span)
        cand = _tidy(eq2[-1] if eq2 else span)
        if cand:
            return cand

    # c) a concluding verb phrase: "... is / equals / charge / walk ... <value>."
    concl = re.findall(
        r'\b(?:is|equals?|equal\s+to|are|charge[sd]?|costs?|walk[sd]?|get[sd]?)\s+'
        r'([^\n,;]+?)(?:\.|\n|$)',
        tail, re.IGNORECASE,
    )
    if concl:
        cand = _tidy(concl[-1])
        if cand and len(cand) <= 200:
            return cand

    # d) previous behavior: last non-blank, non-fence line
    for line in reversed(lines):
        if line not in (']', '[', '$$', '$', ')', '(', '```'):
            return _clean_extracted(line)
    return _clean_extracted(lines[-1])


def normalize_answer(ans: str) -> str:
    """Normalize a math answer for robust string comparison."""
    ans = ans.strip()
    ans = re.sub(r'^\$|\$$|^\\\(|\\\)$', '', ans)
    ans = re.sub(r'\\frac\{([^}]*)\}\{([^}]*)\}', r'\1/\2', ans)
    ans = re.sub(r'\\text\{([^}]*)\}', r'\1', ans)
    ans = re.sub(
        r'\\(?:mathrm|mathbf|mathit|mathsf|mathtt|textrm|textsf|texttt|bm|cal|scr|frak|bb)\{([^}]*)\}',
        r'\1', ans,
    )
    ans = re.sub(r'\s*([+\-])\s*', r'\1', ans)
    ans = re.sub(r'\s+', ' ', ans)
    ans = ans.rstrip('.,;:')
    return ans.strip()


def _strip_units(s: str) -> str:
    """Drop trailing English unit words before comparison."""
    s = re.sub(
        r'\s+(feet|foot|inches|inch|miles|mile|meters|meter|dollars?|cents?|hours?|minutes?|seconds?|per\s+hour|mph|km/h)\s*$',
        '', s, flags=re.IGNORECASE,
    )
    return s.strip()


# ────────────────────────  answer judge  ────────────────────────────────

def judge_answer(raw_output: str, golden: str, exe) -> tuple:
    """
    4-level judge, cheapest first:
      1. case-insensitive exact match on the extracted answer
      2. normalized match (LaTeX/whitespace/operator-insensitive)
      3. unit-stripped match
      4. MathExecutor.evaluate → GPT-4o-mini semantic equivalence
    Returns (is_solved, judge_info).
    """
    pred = extract_final_answer(raw_output)
    if not pred or not pred.strip():
        return False, "empty answer"

    if pred.strip().lower() == golden.strip().lower():
        return True, "exact match"

    norm_p = normalize_answer(pred)
    norm_g = normalize_answer(golden)
    if norm_p == norm_g:
        return True, "normalized match"

    if _strip_units(norm_p) == _strip_units(norm_g):
        return True, "unit-stripped match"

    return bool(exe.evaluate(pred, golden, timeout=5)), "executor eval"


# ────────────────────────  resume / log helpers  ────────────────────────

def load_processed_problems(results_path: str) -> dict:
    """Return {problem_text: is_solved} for every result already written.

    Keyed on the dataset's ``problem`` field (not line number), so resume is
    robust to reordering, to a changed --num_samples, and to switching the
    actor backend (local model vs. API) which both write the same file.
    """
    processed = {}
    if os.path.exists(results_path):
        with open(results_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                key = rec.get("problem", "")
                if key:
                    processed[key] = bool(rec.get("is_solved"))
    return processed


def log_detail(detail_path: str, text: str) -> None:
    """Append one chunk to the human-readable log and flush immediately."""
    with open(detail_path, "a", encoding="utf-8") as f:
        f.write(text)
        f.flush()


# ────────────────────────────  dot loop  ─────────────────────────────────

class DotSolver:
    """
    DoT first-pass loop for one run: init attempt + reflection iterations, with
    mistake-insights decompositions generated ON THE FLY by a local Qwen model
    (or no decomposition at all for the direct config). Faithful to the first
    pass of math/dot_Math_parametric_with_bank.run_dot.
    """

    def __init__(self, gen, exe, actor, decomposer, init_attempts, max_iters):
        self.gen = gen
        self.exe = exe
        self.actor = actor
        self.decomposer = decomposer  # None in direct config
        self.init_attempts = init_attempts
        self.max_iters = max_iters

    def _gen_insight(self, problem: str, temperature: float) -> str:
        if self.decomposer is None:
            return ""
        return self.decomposer.generate(problem, temperature=temperature)

    def _gen_answer(
        self,
        problem: str,
        strategy: str,
        temperature: float = 0.1,
        insight: str = "",
        prev_answers: str = None,
        feedback: str = None,
        self_reflection: str = None,
    ) -> str:
        """Call gen.func_impl, retrying a few times on None (e.g. OOM)."""
        for _ in range(3):
            out = self.gen.func_impl(
                problem,
                self.actor,
                strategy,
                temperature=temperature,
                mistake_insights=insight or None,
                prev_answers=prev_answers,
                feedback=feedback,
                self_reflection=self_reflection,
            )
            if out:
                return out
        return ""  # all attempts returned None

    def _gen_reflections(self, problem, cur_answer, feedback,
                         diverse_reflections, insight) -> list:
        if self.decomposer is not None:
            text = self.gen.self_reflection_diverse_parametric(
                problem, cur_answer, feedback, self.actor,
                diverse_reflections, insight,
            )
        else:
            text = self.gen.self_reflection_diverse(
                problem, cur_answer, feedback, self.actor,
                diverse_reflections,
            )
        if not text:
            return []
        # math generator separates reflections by '\n' (see dot_Math_parametric_with_bank)
        return [r.strip() for r in text.split("\n") if len(r.strip()) > 7]

    def solve(self, problem: str, golden: str) -> dict:
        """
        Run the full dot budget for one problem.
        Returns dict with keys: solution, is_solved, judge, num_attempts,
        attempts (list of per-attempt dicts), insights/answers/feedbacks/reflections.
        """
        insights, answers, feedbacks = [], [], []
        attempts = []          # per-attempt detail for the log
        judge_info = "error"

        # ── 1) init phase: a single `simple` attempt (init_attempts=1) ──
        for a in range(self.init_attempts):
            temp = 0.1 if a == 0 else 1.0
            insight = self._gen_insight(problem, temp)
            answer = self._gen_answer(problem, "simple",
                                      temperature=0.1, insight=insight)
            is_solved, judge_info = judge_answer(answer, golden, self.exe)

            insights.append(insight)
            answers.append(answer)
            feedbacks.append("Correct answer" if is_solved else "Incorrect answer")
            attempts.append({
                "phase": "init",
                "insight_temp": temp if self.decomposer is not None else None,
                "answer": answer,
                "reflection": None,
                "judge": judge_info,
                "is_solved": is_solved,
            })
            if is_solved:
                return {
                    "solution": answer, "is_solved": True, "judge": judge_info,
                    "num_attempts": a + 1, "attempts": attempts,
                    "insights": insights, "answers": answers, "feedbacks": feedbacks,
                    "reflections": [],
                }

        # ── 2) reflection phase ──
        # `direct` injects no decomposition, so the dot loop degenerates to a
        # single `simple` attempt: answer once and stop (no reflection rounds).
        if self.decomposer is None:
            return {
                "solution": answers[-1] if answers else "",
                "is_solved": False, "judge": judge_info,
                "num_attempts": len(attempts), "attempts": attempts,
                "insights": insights, "answers": answers, "feedbacks": feedbacks,
                "reflections": [],
            }

        cur_answer = answers[-1] if answers else ""
        cur_feedback = "Incorrect answer"
        diverse_reflections = []
        solved = False

        for it in range(max(0, self.max_iters - 1)):
            insight = self._gen_insight(problem, 1.0)
            reflections = self._gen_reflections(
                problem, cur_answer, cur_feedback, diverse_reflections, insight,
            )
            # Snapshot the base answer ONCE per iteration (the original run_dot
            # deep-copies cur_func_impl before the inner loop): every reflection
            # in this iteration shares the same `prev_answers`.
            base_answer = cur_answer
            temp_solutions = []
            reflections_scores = []

            for reflection in reflections[:2]:
                new_answer = self._gen_answer(
                    problem, "reflexion", temperature=0.1,
                    insight=insight, prev_answers=base_answer,
                    feedback=cur_feedback, self_reflection=reflection,
                )
                is_solved, judge_info = judge_answer(new_answer, golden, self.exe)

                insights.append(insight)
                answers.append(new_answer)
                feedbacks.append("Correct answer" if is_solved else "Incorrect answer")
                attempts.append({
                    "phase": f"refl-it{it + 1}",
                    "insight_temp": 1.0 if self.decomposer is not None else None,
                    "answer": new_answer,
                    "reflection": reflection,
                    "judge": judge_info,
                    "is_solved": is_solved,
                })
                temp_solutions.append(new_answer)
                reflections_scores.append((1.0 if is_solved else 0.0) + 1e-8)

                # The last iteration (`it == max_iters - 2`) tries only one
                # reflection, matching the original `cur_iter == max_iters - 1`
                # early break that bounds the total number of attempts.
                if is_solved or it == self.max_iters - 2:
                    if is_solved:
                        solved = True
                    break

            diverse_reflections += reflections
            if solved:
                break

            # Weighted-sample the base answer for the NEXT iteration, exactly like
            # the original run_dot (cur_feedback stays "Incorrect answer": we only
            # reach here when none of this iteration's reflections solved it).
            if temp_solutions:
                sampled_idx = random.choices(
                    range(len(temp_solutions)), weights=reflections_scores, k=1
                )[0]
                cur_answer = temp_solutions[sampled_idx]

        return {
            "solution": answers[-1] if answers else "",
            "is_solved": solved, "judge": judge_info,
            "num_attempts": len(attempts), "attempts": attempts,
            "insights": insights, "answers": answers, "feedbacks": feedbacks,
            "reflections": diverse_reflections,
        }


# ────────────────────────────  main loop  ───────────────────────────────

def run_eval(args, decomp_config=None) -> None:
    from executors.factory import executor_factory
    from generators.factory import generator_factory
    from utils import write_jsonl

    log_dir = os.path.join(args.output_dir, args.run_name)
    os.makedirs(log_dir, exist_ok=True)
    dataset_name = os.path.splitext(os.path.basename(args.dataset_path))[0]

    # ── actor selection (local HF dir by default; yunwu.ai relay with --use_api_actor) ──
    # The result filename uses the SAME label for both, so a run can be resumed
    # interchangeably regardless of which actor backend produced the earlier lines.
    if args.use_api_actor:
        actor_desc = f"{args.api_actor_model} @ {args.api_base_url}"
        print(f"Using API actor: {actor_desc}")
        actor = YunwuActor(args.api_actor_model, args.api_base_url, args.api_key)
    else:
        actor_desc = f"{args.actor_model} (device={args.device})"
        print(f"Loading actor model: {actor_desc}")
        actor = load_local_model(args.actor_model, ACTOR_LABEL, args.device)

    log_path = os.path.join(
        log_dir,
        f"{dataset_name}_dot_{args.max_iters}_{ACTOR_LABEL}_pass_at_k_1_math.jsonl",
    )
    detail_path = os.path.join(log_dir, "eval_detail.log")

    dataset = load_dataset(args.dataset_path)
    print(f"Loaded {len(dataset)} examples from {args.dataset_path}")
    if args.num_samples and args.num_samples > 0:
        dataset = dataset[: args.num_samples]
        print(f"Limited to first {len(dataset)} samples")

    decomposer = None
    if decomp_config:
        print(f"Loading decomposition model: {decomp_config['model_path']}")
        decomp_model = load_local_model(
            decomp_config["model_path"], decomp_config["label"], args.device
        )
        decomposer = MathDecomposer(decomp_model)

    gen = generator_factory("math")
    exe = executor_factory("math")
    solver = DotSolver(gen, exe, actor, decomposer,
                       args.init_attempts, args.max_iters)

    # ── resume (keyed on the `problem` field, not line number) ──
    processed = load_processed_problems(log_path)
    num_done = len(processed)
    num_right = sum(1 for v in processed.values() if v)
    if num_done:
        print(f"Resuming: {num_done} already processed ({num_right} right), will skip them")

    header = (
        f"{'=' * 80}\n"
        f"RUN START  run={args.run_name}  dataset={args.dataset_path}  "
        f"actor={actor_desc}  "
        f"decomp={decomp_config['model_path'] if decomp_config else 'NONE'}  "
        f"strategy=dot  init_attempts={args.init_attempts}  "
        f"max_iters={args.max_iters}  device={args.device}\n"
        f"resume: {num_done} done / {num_right} right\n"
        f"{'=' * 80}\n\n"
    )
    log_detail(detail_path, header)

    n = len(dataset)
    for i, item in enumerate(dataset):
        idx = i + 1
        problem = item.get("problem", "")
        solution = item.get("solution", "")
        if not problem or not solution:
            print(f"\n[{idx}/{n}] SKIP (no problem/solution): {problem[:60]}...")
            continue
        if problem in processed:
            print(f"\n[{idx}/{n}] SKIP (already processed): {problem[:60]}...")
            continue

        golden = extract_golden_answer(solution)
        print(f"\n[{idx}/{n}] {problem[:80]}...")
        start_time = time.time()

        try:
            result = solver.solve(problem, golden)
        except Exception as e:
            result = {
                "solution": f"(error) {type(e).__name__}: {e}",
                "is_solved": False, "judge": "error", "num_attempts": 0,
                "attempts": [], "insights": [], "answers": [], "feedbacks": [],
                "reflections": [],
            }
            print(f"  ⚠ Error processing item {idx}: {e}")

        elapsed = time.time() - start_time
        if result["is_solved"]:
            num_right += 1

        # Keep the reference `solution` untouched; store the model answer separately.
        item["golden_answer"] = golden
        item["predicted"] = result["solution"]
        item["is_solved"] = result["is_solved"]
        item["judge"] = result["judge"]
        item["runtime"] = round(elapsed, 2)
        item["strategy"] = "dot"
        item["num_attempts"] = result["num_attempts"]
        item["insights"] = result["insights"]
        item["answers"] = result["answers"]
        item["feedbacks"] = result["feedbacks"]
        item["reflections"] = result["reflections"]
        write_jsonl(log_path, [item], append=True)

        num_done += 1
        acc = num_right / num_done if num_done else 0.0

        # ── detail log: problem / GT first, then every attempt ──
        entry = []
        entry.append("=" * 80)
        entry.append(
            f"[{idx}/{n}]  {'✓ RIGHT' if result['is_solved'] else '✗ WRONG'}  "
            f"attempts: {result['num_attempts']}"
        )
        entry.append("=" * 80)
        entry.append(f"--- Problem ---\n{problem}\n")
        entry.append(f"--- Ground Truth (boxed) ---\n{golden}\n")
        for att_i, att in enumerate(result["attempts"], 1):
            if att["insight_temp"] is not None:
                label = f"{att['phase']}, insight temp={att['insight_temp']}"
            else:
                label = att["phase"]
            entry.append(
                f"--- Attempt {att_i} ({label}) ---\n"
                f"Mistake insights:\n{result['insights'][att_i - 1] or '(none)'}\n"
                f"Model Output:\n{att['answer'] or '(empty)'}\n"
                f"Judge: {att['judge']}\n"
            )
        entry.append(f"--- Runtime ---\n{elapsed:.1f}s\n")
        entry.append(f"--- Accuracy ---\n{num_right}/{num_done} = {acc:.4f}\n\n")
        log_detail(detail_path, "\n".join(entry))

        print(f"  → {'RIGHT' if result['is_solved'] else 'WRONG'} "
              f"({result['judge']}, {result['num_attempts']} attempts) | "
              f"running acc: {acc:.4f} ({num_right}/{num_done})")

    acc = num_right / num_done if num_done else 0.0
    summary = (
        f"\n{'=' * 80}\n"
        f"RUN FINISHED  run={args.run_name}\n"
        f"correct: {num_right} / {num_done}\n"
        f"accuracy: {acc:.4f}\n"
        f"results: {log_path}\n"
        f"{'=' * 80}\n"
    )
    log_detail(detail_path, summary)
    print(f"\nDone! {args.run_name}: right={num_right}, total={num_done}, acc={acc:.4f}")
    print(f"Results  → {log_path}")
    print(f"Detail   → {detail_path}")


# ─────────────────────────────  argparse  ───────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--run_name", type=str, required=True,
        help="subdirectory under --output_dir, e.g. direct / decomp_base / decomp_sft / decomp_dpo",
    )
    p.add_argument(
        "--dataset_path", type=str,
        default="dataset/math/test/test_150_per_type.jsonl",
    )
    p.add_argument(
        "--actor_model", type=str, default="models/Meta-Llama-3.1-8B-Instruct"
    )
    p.add_argument(
        "--use_api_actor", action="store_true",
        help="run the actor via an OpenAI-compatible API (yunwu.ai/v1 relay) "
             "instead of loading a local HF model",
    )
    p.add_argument(
        "--api_actor_model", type=str, default="llama-3.1-8b",
        help="API model name for --use_api_actor (e.g. llama-3.1-8b on yunwu.ai)",
    )
    p.add_argument(
        "--api_base_url", type=str, default=YUNWU_BASE_URL,
        help="API base URL for --use_api_actor",
    )
    p.add_argument(
        "--api_key", type=str, default=DEFAULT_YUNWU_API_KEY,
        help="API key for --use_api_actor (or set YUNWU_API_KEY env var)",
    )
    p.add_argument(
        "--decomp_model", type=str, default=None,
        help="local dir of the decomposition model (Qwen3.5-2B variants); "
             "if omitted, the dot loop runs WITHOUT mistake-insight injection",
    )
    p.add_argument("--output_dir", type=str, default="results/math_dot")
    p.add_argument("--device", type=str, default="cuda:0")
    p.add_argument("--num_samples", type=int, default=None, help="limit to first N samples")
    p.add_argument(
        "--init_attempts", type=int, default=1,
        help="initial `simple` attempts per question (unified to 1)",
    )
    p.add_argument(
        "--max_iters", type=int, default=8,
        help="max total attempts per question incl. the initial one "
             "(dot strategy; reflection rounds = max_iters - 1)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    decomp_config = None
    if args.decomp_model:
        decomp_config = {
            "model_path": args.decomp_model,
            "label": os.path.basename(args.decomp_model.rstrip("/")),
        }

    run_eval(args, decomp_config=decomp_config)


if __name__ == "__main__":
    main()
