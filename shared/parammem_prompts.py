"""Author ParamMem text templates, adapted only to native Qwen chat wrapping."""
import ast
import hashlib
from functools import lru_cache
from pathlib import Path


AUTHOR_COMMIT = "8b36cddda863ce38cc762976d9ffc55c3540af07"
ROOT = Path(__file__).resolve().parents[1]
AUTHOR_SOURCE_SHA256 = {
    "code/LoRA_Llama3_Code_multigpu.py": "3e6b5c9ccbbeaff3e8b36aa143dd0cb065b574ca4ad03a9eaf3d7456e4e5fdec",
    "code/LoRA_Llama3_Code_multigpu_inference.py": "637800f480271fc03eb64590fc070de4b247929733bb51c01d776b8f96d0ffba",
    "math/LoRA_Llama3_Math_multigpu.py": "b24bcbab0b5dd85601fc2682d50c7967b45c8b00fba4444802c04bb377f3109c",
    "math/LoRA_Llama3_Math_multigpu_inference.py": "12215dd5d7dda35a439782774266f0a4cbc24d92f8ba46d5f7e07f83c41278b7",
    "qa/LoRA_Llama3_QA_multigpu.py": "cd3383899a76024f4703d8ee18a3ed02ddaf45f53f5d92078f7c2022fb5183ae",
    "qa/LoRA_Llama3_QA_multigpu_inference.py": "a996cbc0ef0354e58e2c5e4734efb33064f7b40adb10fef627597cb7c5253807",
}


@lru_cache(maxsize=None)
def author_constants(domain, stage):
    if domain not in {"code", "math", "qa"} or stage not in {"train", "inference"}:
        raise ValueError("Unknown ParamMem domain or prompt stage")
    label = {"code": "Code", "math": "Math", "qa": "QA"}[domain]
    suffix = "_inference" if stage == "inference" else ""
    relative = f"{domain}/LoRA_Llama3_{label}_multigpu{suffix}.py"
    path = ROOT / relative
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != AUTHOR_SOURCE_SHA256[relative]:
        raise ValueError(f"Author prompt source changed: {path}")
    names = {"SYS_PROMPT", "PRE_INSIGHT_FEWSHOT"} if domain == "qa" else {"SYSTEM_PROMPT"}
    values = {}
    for node in ast.parse(source.decode("utf-8")).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = ast.literal_eval(node.value)
    if set(values) != names or not all(isinstance(value, str) for value in values.values()):
        raise ValueError(f"Missing author prompt literals in {path}")
    return values["SYS_PROMPT" if domain == "qa" else "SYSTEM_PROMPT"], values.get("PRE_INSIGHT_FEWSHOT")


def memory_messages(domain, question, stage="inference"):
    system, examples = author_constants(domain, stage)
    if domain == "qa":
        user = f"Here are some examples:{examples}\n\n[Question]: {question.strip()}"
    else:
        marker = "FUNC_SIGNATURE" if domain == "code" else "PROBLEM"
        user = f"{marker}:\n{question.strip()}"
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def tokenize_sft(processor, tokenizer, domain, question, answer, max_seq_len, dynamic_padding=False,
                 preserve_eos=False):
    if max_seq_len <= 0 or tokenizer.pad_token_id is None:
        raise ValueError("SFT requires a positive sequence length and a configured padding token")
    messages = memory_messages(domain, question, stage="train")
    prompt = processor.apply_chat_template(messages, tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(answer.strip() + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_seq_len]
    labels = (input_ids.copy() if preserve_eos else
              [token_id if token_id != tokenizer.pad_token_id else -100 for token_id in input_ids])
    result = {"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels}
    if dynamic_padding:
        result["length"] = len(input_ids)
    else:
        padding = max_seq_len - len(input_ids)
        result["input_ids"] += [tokenizer.pad_token_id] * padding
        result["attention_mask"] += [0] * padding
        result["labels"] += [-100] * padding
    return result
