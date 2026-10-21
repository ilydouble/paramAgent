#!/usr/bin/env python3
"""Audit the existing two-stage ParamMem runners on a frozen small test set.

This adapter preserves the domain runners and their prompt/search logic. It
replaces external API clients with a logged local actor, task-conditioned local
SFT/DPO memory, CPU retrieval embeddings, and explicit deterministic evaluators.
It is an oracle-feedback diagnostic, not a router-training or blind benchmark.
"""
from __future__ import annotations

import argparse
import ast
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import re
import subprocess
import sys
import time
import urllib.request


ROOT = Path(__file__).resolve().parents[1]
RUNNERS = {"code": "code/dot_parametric_with_bank.py",
           "math": "math/dot_Math_parametric_with_bank.py",
           "qa": "qa/dot_QA_parametric_with_bank.py"}
BASES = {"code": "Qwen3.5-2B-code-merged2", "math": "Qwen3.5-2B-math-merged2",
         "qa": "Qwen3.5-2B-qa-merged"}
ADAPTERS = {"code": "lora-qwen3.5-2b-code2-dpo-20261003",
            "math": "lora-qwen3.5-2b-math-dpo3", "qa": "lora-qwen3.5-2b-qa-dpo3"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def file_digest(path):
    with Path(path).open("rb") as handle:
        value = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
        return value.hexdigest()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def read_frozen_rows(path, expected_tasks=10):
    rows = read_rows(path)
    if expected_tasks <= 0 or len(rows) != expected_tasks or len({row["task_id"] for row in rows}) != expected_tasks:
        raise ValueError(f"Expected {expected_tasks} distinct frozen test rows")
    return rows


def resolve_dpo_adapter(args):
    explicit = getattr(args, "dpo_adapter_path", None)
    expected = getattr(args, "expected_dpo_sha256", None)
    if bool(explicit) != bool(expected):
        raise ValueError("Explicit DPO path and expected SHA256 must be supplied together")
    if args.method != "parammem-dpo":
        if explicit:
            raise ValueError("Explicit DPO weights require method parammem-dpo")
        return None, None
    if explicit:
        adapter = Path(explicit).resolve(strict=True)
    else:
        adapter = args.model_root / "lora-qwen3.5-2b-dpo" / ADAPTERS[args.domain]
        registry = json.loads((ROOT / "docs/models/model-registry.json").read_text())
        expected = next(model["adapter_sha256"] for model in registry["models"]
                        if model["path"] == "lora-qwen3.5-2b-dpo/" + ADAPTERS[args.domain])
    actual = file_digest(adapter / "adapter_model.safetensors")
    if actual != expected:
        raise ValueError("DPO weights differ from expected checkpoint SHA256")
    metadata = json.loads((adapter / "adapter_config.json").read_text())
    declared_base = Path(metadata["base_model_name_or_path"])
    if not declared_base.is_absolute():
        declared_base = args.model_root / declared_base
    if declared_base.resolve() != (args.model_root / "models" / BASES[args.domain]).resolve():
        raise ValueError("DPO adapter/SFT base mismatch")
    return adapter, actual


def stage_results(directory):
    """The legacy results.jsonl stops at Stage 1; Stage 2 has its own file."""
    directory = Path(directory)
    first = read_rows(directory / "first_stage_log.jsonl")
    second = directory / "second_stage_log.jsonl"
    final = read_rows(second) if second.exists() else first
    initial_ids = {r["task_id"] for r in first}
    if len(first) != len(initial_ids) or len(final) != len(first) or {r["task_id"] for r in final} != initial_ids:
        raise RuntimeError("Stage result IDs are incomplete or duplicated")
    by_id = {r["task_id"]: r for r in final}
    for row in first:
        if not row["is_solved"] and not by_id[row["task_id"]].get("stage2"):
            raise RuntimeError("Stage 1 failure was not completed in Stage 2")
        if row["is_solved"] and not by_id[row["task_id"]]["is_solved"]:
            raise RuntimeError("Stage 2 lost a Stage 1 success")
    return first, final


def literal_assignment(path, name):
    for node in ast.parse(Path(path).read_text()).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError(f"Missing literal {name}: {path}")


def memory_messages(domain, question):
    stem = {"code": "Code", "math": "Math", "qa": "QA"}[domain]
    path = ROOT / domain / f"LoRA_Qwen35_{stem}_4090.py"
    system = literal_assignment(path, "SYSTEM_PROMPT")
    if domain == "qa":
        examples = literal_assignment(path, "PRE_INSIGHT_FEWSHOT")
        user = f"Here are some examples:{examples}\n\n[Question]: {question}"
    else:
        marker = "FUNC_SIGNATURE" if domain == "code" else "PROBLEM"
        user = f"{marker}:\n{question}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def normalize_qa(value):
    import string
    value = value.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def qa_score(answer, gold):
    pred, ref = normalize_qa(answer), normalize_qa(gold)
    a, b = pred.split(), ref.split()
    common = sum((collections.Counter(a) & collections.Counter(b)).values())
    f1 = 2 * common / (len(a) + len(b)) if a and b else float(a == b)
    if pred in {"yes", "no", "noanswer"} or ref in {"yes", "no", "noanswer"}:
        f1 = float(pred == ref)
    return {"success": pred == ref, "em": int(pred == ref), "f1": f1}


def clean_math(value):
    value = re.sub(r"\\(?:text|mathrm)\{[^{}]*\}", "", value)
    value = value.replace(r"^\circ", "").replace(r"^{\circ}", "")
    value = value.replace(r"\left", "").replace(r"\right", "")
    value = value.strip().strip("$ \t\n.,")
    value = re.sub(r"\s+(?:degrees?|cm|centimeters?|square centimeters?)\.?$", "", value)
    return value


def math_score(answer, gold, finish_reason=None):
    # Preserve the domain's brace-aware final-answer extraction. Only its remote
    # judge is replaced; unparseable expressions remain explicitly unresolved.
    from executors.Math_executor import _extract_final_answer
    from sympy import Symbol, pi, simplify
    from sympy.parsing.latex import parse_latex
    extracted = _extract_final_answer(answer)
    pred = clean_math(extracted or answer)
    ref = clean_math(gold)
    if pred == ref:
        return {"success": True, "prediction": pred, "status": "exact"}
    left = right = None
    try:
        right = parse_latex(ref, strict=True).subs(Symbol("pi"), pi)
        if right.free_symbols:
            raise ValueError("Scalar diagnostic received free symbols")
        left = parse_latex(pred, strict=True)
        # SymPy's LaTeX parser represents \pi as a free Symbol, not its
        # mathematical constant. Resolve only this explicitly named constant.
        left = left.subs(Symbol("pi"), pi)
        right = right.subs(Symbol("pi"), pi)
        if left.free_symbols or right.free_symbols:
            raise ValueError("Scalar diagnostic received free symbols")
        delta = simplify(left - right)
        correct = delta == 0 or delta.is_zero is True
        if not correct and delta.is_zero is not False:
            raise ValueError("Equivalence remains undecided")
        return {"success": bool(correct), "prediction": pred, "status": "symbolic"}
    except Exception as exc:
        if finish_reason == "length" and not extracted and left is None and right is not None and not right.free_symbols:
            return {"success": False, "prediction": pred, "status": "budget_exhausted_without_final",
                    "finish_reason": finish_reason, "error": f"{type(exc).__name__}: {exc}"}
        return {"success": None, "prediction": pred, "status": "unresolved",
                "error": f"{type(exc).__name__}: {exc}"}


def code_check(code, test, timeout=8):
    import tempfile
    # A subprocess can actually be stopped on timeout; the legacy thread-based
    # executor cannot terminate a generated infinite loop.
    preamble = "import resource, sys\nresource.setrlimit(resource.RLIMIT_CPU,(5,5))\n"
    preamble += "if sys.platform == 'linux': resource.setrlimit(resource.RLIMIT_AS,(536870912,536870912))\nfrom typing import *\n"
    with tempfile.TemporaryDirectory(prefix="parammem-check-") as directory:
        path = Path(directory) / "candidate.py"
        path.write_text(preamble + code + "\n" + test + "\n")
        try:
            result = subprocess.run([sys.executable, "-I", str(path)], cwd=directory,
                                    capture_output=True, text=True, timeout=timeout)
            return result.returncode == 0, result.stderr[-1500:]
        except subprocess.TimeoutExpired:
            return False, "timeout"


class Journal:
    def __init__(self, directory):
        self.directory = directory
        self.cache = {}
        self.context = {}
        self.occurrences = collections.Counter()
        path = directory / "calls.jsonl"
        if path.exists():
            for value in read_rows(path):
                if value.get("type") in {"actor", "memory"}:
                    self.cache[value["key"]] = value

    def sample_request(self, kind, request):
        # Repeated high-temperature draws are distinct samples, even when the
        # text is identical. Fresh replay reproduces the same occurrence seeds.
        key = (kind, digest(request))
        occurrence = self.occurrences[key]
        self.occurrences[key] += 1
        sampled = {**request, "seed": request["seed"] + occurrence}
        self.write({"type": "sample_use", "kind": kind, "key": digest(sampled),
                    "seed": sampled["seed"], "cached": digest(sampled) in self.cache})
        return sampled

    def actor_finish_reason(self, answer):
        reasons = {choice.get("finish_reason") for call in self.cache.values()
                   if call.get("type") == "actor" for choice in call["response"]["choices"]
                   if choice["message"].get("content") == answer}
        return reasons.pop() if len(reasons) == 1 else None

    def write(self, value):
        with (self.directory / "calls.jsonl").open("a") as handle:
            handle.write(json.dumps({**self.context, **value}, ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class LocalActor:
    is_chat = True
    name = "Qwen3.5-4B"

    def __init__(self, args, journal):
        self.args, self.journal = args, journal
        self.token_counts = {}

    def completion_budget(self, messages, cap):
        if cap <= 0:
            raise ValueError("Actor output budget must be positive")
        messages = [{"role": message.role, "content": message.content}
                    if not isinstance(message, dict) else message for message in messages]
        payload = {"model": self.name, "messages": messages, "add_generation_prompt": True,
                   "chat_template_kwargs": {"enable_thinking": False}}
        key = digest(payload)
        if key not in self.token_counts:
            request = urllib.request.Request(self.args.actor_url.rstrip("/").removesuffix("/v1") + "/tokenize",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(request, timeout=30) as response:
                metadata = json.load(response)
            if type(metadata.get("count")) is not int or metadata["count"] < 0:
                raise ValueError("Actor tokenizer returned an invalid prompt count")
            if metadata.get("max_model_len") != self.args.actor_context_tokens:
                raise ValueError("Actor server/context configuration mismatch")
            self.token_counts[key] = metadata["count"]
        self.last_prompt_tokens = self.token_counts[key]
        available = self.args.actor_context_tokens - self.last_prompt_tokens - self.args.context_reserve_tokens
        if available <= 0:
            raise ValueError("Actor prompt exhausts the configured context window")
        return min(cap, available)

    def generate_chat(self, messages, max_tokens=1024, temperature=0.2, num_comps=1):
        messages = [{"role": m.role, "content": m.content} for m in messages]
        requested_tokens = max_tokens
        max_tokens = self.completion_budget(messages, max_tokens)
        self.journal.write({"type": "actor_budget", "requested_max_tokens": requested_tokens,
                            "max_tokens": max_tokens, "context_tokens": self.args.actor_context_tokens,
                            "prompt_tokens": self.last_prompt_tokens,
                            "reserve_tokens": self.args.context_reserve_tokens})
        payload = {"model": self.name, "messages": messages, "max_tokens": max_tokens,
                   "temperature": temperature, "top_p": 0.9, "n": num_comps,
                   "seed": self.args.seed, "chat_template_kwargs": {"enable_thinking": False}}
        payload = self.journal.sample_request("actor", payload)
        key = digest(payload)
        cached = self.journal.cache.get(key)
        if cached is None:
            request = urllib.request.Request(self.args.actor_url.rstrip("/") + "/chat/completions",
                data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
            started = time.monotonic()
            with urllib.request.urlopen(request, timeout=300) as response:
                body = json.load(response)
            cached = {"type": "actor", "key": key, "request": payload, "response": body,
                      "seconds": time.monotonic() - started}
            self.journal.cache[key] = cached
            self.journal.write(cached)
            print("ACTOR", self.journal.context, body.get("usage"), flush=True)
        choices = sorted(cached["response"]["choices"], key=lambda c: c["index"])
        outputs = [c["message"].get("content") or "" for c in choices]
        if any(not output.strip() for output in outputs):
            raise RuntimeError("Empty actor output; not a scored model failure")
        return outputs[0] if num_comps == 1 else outputs


class LocalMemory:
    def __init__(self, args, journal, rows):
        self.args, self.journal = args, journal
        self.ids = {r.get("prompt", r.get("problem", r.get("question"))): r["task_id"] for r in rows}
        self.model = self.tokenizer = None

    def load(self):
        if self.model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        base = self.args.model_root / "models" / BASES[self.args.domain]
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        self.model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant,
                                                       device_map={"": 0}, dtype=torch.bfloat16)
        if self.args.method == "parammem-dpo":
            adapter, _ = resolve_dpo_adapter(self.args)
            self.model = PeftModel.from_pretrained(self.model, adapter)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(base)

    def generate(self, question, temperature=0.1):
        self.journal.context = {"task_id": self.ids.get(question), "domain": self.args.domain,
                                "method": self.args.method}
        if self.args.method == "qwen":
            return ""
        messages = memory_messages(self.args.domain, question)
        request = {"messages": messages, "method": self.args.method,
                   "temperature": temperature, "max_tokens": 3072, "seed": self.args.seed}
        request = self.journal.sample_request("memory", request)
        key = digest(request)
        if key in self.journal.cache:
            return self.journal.cache[key]["output"]
        self.load()
        import torch
        from gain_router.local_preference import generation_stop_ids, generation_finish
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        if inputs["input_ids"].shape[1] + 3072 > 16384:
            raise ValueError("Memory prompt exceeds context; no silent truncation")
        stop_ids = generation_stop_ids(self.tokenizer, self.model.generation_config)
        options = {"max_new_tokens": 3072, "do_sample": temperature > 0,
                   "eos_token_id": stop_ids, "pad_token_id": self.tokenizer.pad_token_id or stop_ids[0]}
        if temperature > 0:
            options.update(temperature=temperature, top_p=0.9)
        started = time.monotonic()
        # Memory sampling must not reset the runner's Python/NumPy search RNGs,
        # or make search behavior depend on whether a response was cached.
        with torch.random.fork_rng(devices=[0]), torch.inference_mode():
            torch.manual_seed(request["seed"])
            tokens = self.model.generate(**inputs, **options)[0, inputs["input_ids"].shape[1]:]
        output = self.tokenizer.decode(tokens, skip_special_tokens=True)
        if not output.strip():
            raise RuntimeError("Empty memory output")
        value = {"type": "memory", "key": key, "request": request, "output": output,
                 "completion_tokens": len(tokens), "prompt_tokens": inputs["input_ids"].shape[1],
                 "finish_reason": generation_finish(tokens.tolist(), stop_ids, 3072),
                 "seconds": time.monotonic() - started}
        self.journal.cache[key] = value
        self.journal.write(value)
        return output


class CpuEmbeddings:
    def __init__(self, path):
        self.path, self.cache = path, {}
        self.model = self.tokenizer = None

    def __call__(self, texts, verbose=False):
        if len(texts) != 1:
            raise ValueError("Legacy retrieval expects exactly one text")
        text = str(texts[0])
        if text not in self.cache:
            import torch
            from transformers import AutoModel, AutoTokenizer
            if self.model is None:
                torch.set_num_threads(2)
                self.tokenizer = AutoTokenizer.from_pretrained(self.path, local_files_only=True)
                self.model = AutoModel.from_pretrained(self.path, local_files_only=True).eval().to("cpu")
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1)
                vector = (hidden * mask).sum(1) / mask.sum(1)
                self.cache[text] = torch.nn.functional.normalize(vector, p=2, dim=1)[0].numpy()
        return self.cache[text]


def make_executor(domain, journal):
    from executors.executor_types import ExecuteResult

    class DiagnosticExecutor:
        def execute(self, code, tests, timeout=8):
            states = [code_check(code, test, timeout)[0] for test in tests]
            passed = "\n".join(t for t, ok in zip(tests, states) if ok)
            failed = "\n".join(t for t, ok in zip(tests, states) if not ok)
            return ExecuteResult(all(states), f"Tests passed:\n{passed}\n\nTests failed:\n{failed}", tuple(states))

        def evaluate(self, *values, timeout=8):
            if domain == "code":
                name, code, test = values
                success, error = code_check(code, test + f"\ncheck({name})", timeout)
                result = {"success": success, "error": error}
            elif domain == "math":
                result = math_score(*values, finish_reason=journal.actor_finish_reason(values[0]))
            else:
                result = qa_score(*values)
            journal.write({"type": "evaluation", "inputs": list(values), "result": result})
            if result["success"] is None:
                raise ValueError("Unresolved math scoring; inspect saved evaluation instead of counting as failure")
            return result["success"]

    return DiagnosticExecutor()


def configure_generation_budget(args):
    if (args.actor_context_tokens <= 0 or args.math_answer_tokens <= 0 or args.context_reserve_tokens < 0
            or args.math_answer_tokens + args.context_reserve_tokens >= args.actor_context_tokens):
        raise ValueError("Invalid actor context/output/reserve budget")
    os.environ["VLLM_MAX_MODEL_LEN"] = str(args.actor_context_tokens)
    os.environ["MATH_MAX_ANSWER_TOKENS"] = str(args.math_answer_tokens)


def run(args):
    configure_generation_budget(args)
    adapter, adapter_sha256 = resolve_dpo_adapter(args)
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.jsonl").exists():
        raise ValueError("Use a fresh output directory; seed only calls.jsonl for a full deterministic replay")
    rows = read_frozen_rows(args.dataset, args.expected_tasks)
    identity = {"protocol": "legacy_parammem_triad_diagnostic_v1", "domain": args.domain,
        "method": args.method, "dataset_sha256": file_digest(args.dataset),
        "runner_sha256": file_digest(ROOT / RUNNERS[args.domain]), "adapter_sha256": None,
        "harness_sha256": file_digest(__file__), "seed": args.seed, "iterations": args.iterations,
        "expected_tasks": args.expected_tasks,
        "sampling": "seed plus occurrence of identical request; fresh full replay only",
        "feedback": "oracle diagnostic; HumanEval full tests / numeric Math equality / QA EM",
        "embedding": str(args.embedding_path), "embedding_max_tokens": 512,
        "actor_url": args.actor_url, "actor": "Qwen3.5-4B",
        "actor_context_tokens": args.actor_context_tokens, "math_answer_tokens": args.math_answer_tokens,
        "context_reserve_tokens": args.context_reserve_tokens, "actor_token_counting": "vllm_tokenize"}
    if args.method != "qwen":
        base = args.model_root / "models" / BASES[args.domain]
        identity["base_weights_sha256"] = file_digest(base / "model.safetensors")
        if args.method == "parammem-dpo":
            identity["adapter_sha256"] = adapter_sha256
            identity["adapter_path"] = str(adapter.resolve())
    meta = args.output / "identity.json"
    if meta.exists() and json.loads(meta.read_text()) != identity:
        raise ValueError("Existing run identity differs; use a new output directory")
    meta.write_text(json.dumps(identity, indent=2) + "\n")
    journal = Journal(args.output)
    random.seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)
    os.environ.setdefault("ANTHROPIC_API_KEY", "unused-local-diagnostic")
    os.environ["AWS_EC2_METADATA_DISABLED"] = "true"
    import generators
    from generators import generator_utils
    generator_utils.MAX_CONTEXT_TOKENS = args.actor_context_tokens
    generator_utils.MAX_ANSWER_TOKENS = args.math_answer_tokens
    generator_utils.RESERVED_TOKENS = args.context_reserve_tokens
    import memory_utils
    actor = LocalActor(args, journal)
    generators.model_factory = lambda name: actor
    memory_utils.get_openai_embedding = CpuEmbeddings(args.embedding_path)
    spec = importlib.util.spec_from_file_location("pilot_legacy_runner", ROOT / RUNNERS[args.domain])
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.executor_factory = lambda *a, **kw: make_executor(args.domain, journal)
    memory = LocalMemory(args, journal, rows)
    kwargs = dict(dataset=rows, model_name="vllm:Qwen3.5-4B", language="py",
                  max_iters=args.iterations, pass_at_k=1, log_path=str(args.output / "results.jsonl"), verbose=True)
    if args.domain == "code":
        kwargs.update(use_mistakes=True, pitfall_agent=memory, mistake_json_file=[{} for _ in rows], inner_iter=1)
    else:
        kwargs.update(use_parsing=True, insight_json_file=[{} for _ in rows],
                      **{"Math_agent" if args.domain == "math" else "QA_agent": memory})
    runner.run_dot(**kwargs)
    if any(c.get("type") == "evaluation" and c["result"]["success"] is None
           for c in read_rows(args.output / "calls.jsonl")):
        raise RuntimeError("Unresolved evaluations remain; run cannot be reported as complete")
    first, final = stage_results(args.output)
    if len(final) != args.expected_tasks or {r["task_id"] for r in final} != {r["task_id"] for r in rows}:
        raise RuntimeError("Legacy runner produced incomplete or mismatched results")
    if any("is_solved" not in r for r in final):
        raise RuntimeError("Legacy result lacks is_solved")
    summary = {"domain": args.domain, "method": args.method, "n": args.expected_tasks,
               "stage1_successes": sum(bool(r["is_solved"]) for r in first),
               "successes": sum(bool(r["is_solved"]) for r in final),
               "calls": len(journal.cache), "identity": digest(identity)}
    (args.output / "completed.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("PILOT_COMPLETED", json.dumps(summary), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=RUNNERS, required=True)
    parser.add_argument("--method", choices=["qwen", "parammem", "parammem-dpo"], required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--expected-tasks", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--dpo-adapter-path", type=Path)
    parser.add_argument("--expected-dpo-sha256")
    parser.add_argument("--embedding-path", type=Path, required=True)
    parser.add_argument("--actor-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--actor-context-tokens", type=int, default=16384)
    parser.add_argument("--math-answer-tokens", type=int, default=8192)
    parser.add_argument("--context-reserve-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    from urllib.parse import urlparse
    if urlparse(args.actor_url).hostname not in {"localhost", "127.0.0.1"}:
        parser.error("This diagnostic only sends tasks to the local actor")
    sys.path.insert(0, str(ROOT))
    run(args)


if __name__ == "__main__":
    main()
