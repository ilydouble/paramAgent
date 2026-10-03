"""Central, strict pilot configuration; no paths hidden in policy code."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from .schema import DOMAINS, nonempty, strict_keys


@dataclass(frozen=True)
class ModelConfig:
    endpoint: str
    model_id: str
    revision: str
    temperature: float
    max_tokens: int
    weights_path: str | None = None
    adapter_path: str | None = None
    top_p: float = 0.9
    enable_thinking: bool | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelConfig":
        keys = set(cls.__dataclass_fields__)
        strict_keys(value, keys, {"endpoint", "model_id", "revision", "temperature", "max_tokens"}, "ModelConfig")
        model = cls(**value)
        for key in ("endpoint", "model_id", "revision"):
            nonempty(getattr(model, key), key)
        url = urlsplit(model.endpoint)
        if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
            raise ValueError("Endpoint must be an HTTP(S) URL without credentials/query/fragment")
        if type(model.max_tokens) is not int or model.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        for key in ("temperature", "top_p"):
            number = getattr(model, key)
            if isinstance(number, bool) or not isinstance(number, (int, float)) or not math.isfinite(number):
                raise ValueError(f"Invalid {key}")
        if model.temperature < 0 or not 0 < model.top_p <= 1:
            raise ValueError("Invalid sampling range")
        if model.enable_thinking is not None and type(model.enable_thinking) is not bool:
            raise ValueError("enable_thinking must be bool or null")
        for key in ("weights_path", "adapter_path"):
            if getattr(model, key) is not None:
                nonempty(getattr(model, key), key)
        return model


@dataclass(frozen=True)
class LabelConfig:
    version: str
    verifier_version: str
    lambda_tokens: float
    lambda_latency: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "LabelConfig":
        keys = set(cls.__dataclass_fields__)
        strict_keys(value, keys, keys, "LabelConfig")
        result = cls(**value)
        if result.version != "binary_success_net_gain_v1" or result.verifier_version != "legacy_deterministic_v1":
            raise ValueError("Unsupported label/verifier version; no silent rule changes")
        for key in ("lambda_tokens", "lambda_latency"):
            number = getattr(result, key)
            if isinstance(number, bool) or not isinstance(number, (float, int)) or not math.isfinite(number) or number < 0:
                raise ValueError(f"{key} must be finite and nonnegative")
        return result


@dataclass(frozen=True)
class ExperimentConfig:
    version: str
    purpose: str
    tasks_path: str
    split_manifest: str
    split: str
    output_dir: str
    seed: int
    max_per_domain: int
    actor: ModelConfig
    preference_models: dict[str, ModelConfig]
    policy_version: str
    labels: LabelConfig
    request_timeout: float = 180.0
    retries: int = 2

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ExperimentConfig":
        keys = set(cls.__dataclass_fields__)
        strict_keys(value, keys, keys - {"request_timeout", "retries"}, "ExperimentConfig")
        if value["version"] != "router_experiment_v1" or value["purpose"] != "pilot":
            raise ValueError("Only versioned pilot configuration is implemented; formal protocol is not certified")
        if value["split"] not in {"train", "val"}:
            raise ValueError("Training-data collection only permits train/val, not test")
        if value["policy_version"] != "preference_repair_single_v1":
            raise ValueError("Unsupported policy version")
        for key in ("tasks_path", "split_manifest", "output_dir"):
            nonempty(value[key], key)
        if type(value["seed"]) is not int or value["seed"] < 0:
            raise ValueError("seed must be nonnegative integer")
        if type(value["max_per_domain"]) is not int or value["max_per_domain"] <= 0:
            raise ValueError("max_per_domain must be positive integer")
        preferences = value["preference_models"]
        if not isinstance(preferences, dict) or not preferences or set(preferences) - DOMAINS:
            raise ValueError("Preference models must be indexed by code/math/qa")
        data = dict(value)
        data["actor"] = ModelConfig.from_dict(value["actor"])
        data["preference_models"] = {domain: ModelConfig.from_dict(model) for domain, model in preferences.items()}
        data["labels"] = LabelConfig.from_dict(value["labels"])
        result = cls(**data)
        if type(result.retries) is not int or result.retries < 0:
            raise ValueError("retries must be nonnegative integer")
        if isinstance(result.request_timeout, bool) or not isinstance(result.request_timeout, (int, float)) or not math.isfinite(result.request_timeout) or result.request_timeout <= 0:
            raise ValueError("request_timeout must be finite and positive")
        return result

    def collection_config(self) -> dict[str, Any]:
        # Relabeling must NOT change the immutable collection identity.
        result = asdict(self)
        result.pop("labels")
        result.pop("output_dir")
        return result


def load_config(path: Path) -> ExperimentConfig:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError("YAML configuration requires requirements-router.txt (PyYAML)") from exc
    class UniqueLoader(yaml.SafeLoader):
        pass

    def unique_mapping(loader, node, deep=False):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in result:
                raise ValueError(f"Duplicate YAML key: {key}")
            result[key] = loader.construct_object(value_node, deep=deep)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
    value = yaml.load(path.read_text(encoding="utf-8"), Loader=UniqueLoader)
    return ExperimentConfig.from_dict(value)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def validate_model_paths(root: Path, config: ExperimentConfig) -> None:
    for model in [config.actor, *config.preference_models.values()]:
        for value in (model.weights_path, model.adapter_path):
            if value is not None and not resolve_path(root, value).is_dir():
                raise FileNotFoundError(f"Missing model dependency: {resolve_path(root, value)}")

    from split_protocol import load_manifest
    from training_splits import validate_training_artifact
    manifest = load_manifest(resolve_path(root, config.split_manifest))
    for domain, model in config.preference_models.items():
        if model.weights_path is None or model.adapter_path is None:
            raise ValueError("Preference collection requires SFT base and DPO adapter paths for split verification")
        base = validate_training_artifact(resolve_path(root, model.weights_path), manifest, stage="sft", domain=domain)
        adapter = validate_training_artifact(resolve_path(root, model.adapter_path), manifest, stage="dpo", domain=domain)
        if adapter.get("base_artifacts") != base["artifacts"]:
            raise ValueError("DPO provenance refers to a different SFT base artifact")
