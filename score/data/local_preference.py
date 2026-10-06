"""Sequential local QLoRA inference; no supervision or verifier access."""
from pathlib import Path
import gc
import json
import time

from .actor import prompt_hash
from .common import file_sha256


def generation_stop_ids(tokenizer, generation_config):
    # Chat EOS can differ from the base model's end-of-text EOS.
    configured = generation_config.eos_token_id
    candidates = [tokenizer.eos_token_id] + (configured if isinstance(configured, list) else [configured])
    ids = list(dict.fromkeys(v for v in candidates if type(v) is int and v >= 0))
    if not ids:
        raise ValueError("Local preference requires a valid EOS token")
    return ids


def generation_finish(token_ids, stop_ids, max_tokens):
    return "stop" if token_ids and token_ids[-1] in stop_ids else ("length" if len(token_ids) >= max_tokens else "stop")


class LocalPreference:
    def __init__(self):
        self.identity = None
        self.model = None
        self.tokenizer = None

    def load(self, config):
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        if not config.weights_path or not config.adapter_path:
            raise ValueError("Local preference requires absolute weights_path and adapter_path")
        base, adapter = Path(config.weights_path), Path(config.adapter_path)
        if not base.is_absolute() or not adapter.is_absolute():
            raise ValueError("Local model paths must be absolute")
        identity = (str(base), str(adapter), config.revision)
        if self.identity == identity:
            return
        adapter_config = json.loads((adapter / "adapter_config.json").read_text())
        if Path(adapter_config["base_model_name_or_path"]).name != base.name:
            raise ValueError("Adapter base model does not match configured SFT")
        digest = file_sha256(adapter / "adapter_model.safetensors")
        if config.revision != "sha256:" + digest:
            raise ValueError("Local adapter SHA256 differs from declared revision")
        self.model = None
        self.tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
        model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant,
            device_map={"": 0}, dtype=torch.bfloat16)
        self.model = PeftModel.from_pretrained(model, adapter).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(base)
        self.identity = identity

    def __call__(self, config, system, user, seed):
        import torch
        from transformers import set_seed

        self.load(config)
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False,
            add_generation_prompt=True, enable_thinking=False)
        inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
        count = inputs["input_ids"].shape[1]
        if count + config.max_tokens > 16384:
            raise ValueError("Local preference prompt exceeds declared 16384-token total budget")
        set_seed(seed)
        started = time.monotonic()
        stop_ids = generation_stop_ids(self.tokenizer, self.model.generation_config)
        options = {"eos_token_id": stop_ids, "max_new_tokens": config.max_tokens, "do_sample": config.temperature > 0,
                   "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else stop_ids[0]}
        if config.temperature > 0:
            options.update(temperature=config.temperature, top_p=config.top_p)
        with torch.inference_mode():
            generated = self.model.generate(**inputs, **options)
        ids = generated[0, count:]
        token_ids = ids.tolist()
        finish = generation_finish(token_ids, stop_ids, config.max_tokens)
        return {"status": "ok", "output": self.tokenizer.decode(ids, skip_special_tokens=True),
            "generated_token_ids": token_ids, "stop_token_ids": stop_ids,
            "finish_reason": finish, "quality_flags": ["truncated"] if finish == "length" else [],
            "usage": {"prompt_tokens": count, "completion_tokens": len(ids), "total_tokens": count + len(ids)},
            "latency_seconds": time.monotonic() - started, "confidence": {"available": False},
            "prompt_sha256": prompt_hash(messages), "attempts": 1, "logprobs_fallback": False}
