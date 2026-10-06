"""Strict boundaries between online state, raw calls, and offline supervision."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any

SCHEMA_VERSION = "router_call_v1"
DOMAINS = {"code", "math", "qa"}
CONFIDENCE_KEYS = {"available", "token_count", "mean_logprob", "min_logprob", "p10_logprob"}


def strict_keys(value: dict[str, Any], allowed: set[str], required: set[str], name: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    extra, missing = set(value) - allowed, required - set(value)
    if extra or missing:
        raise ValueError(f"{name}: unknown={sorted(extra)}, missing={sorted(missing)}")


def nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


@dataclass(frozen=True)
class TaskSample:
    sample_id: str
    group_id: str
    domain: str
    problem: str
    split: str
    split_seed: int
    context: str = ""

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TaskSample":
        keys = set(cls.__dataclass_fields__)
        strict_keys(value, keys, keys - {"context"}, "TaskSample")
        for key in ("sample_id", "group_id", "problem"):
            nonempty(value[key], key)
        if value["domain"] not in DOMAINS or value["split"] not in {"train", "val", "test"}:
            raise ValueError("Unknown domain or split")
        if type(value["split_seed"]) is not int or not isinstance(value.get("context", ""), str):
            raise ValueError("Invalid split_seed or context")
        if value["domain"] == "qa" and not value.get("context", "").strip():
            raise ValueError("QA requires inference-visible context")
        return cls(**value)


@dataclass(frozen=True)
class ModelCall:
    role: str
    model_id: str
    revision: str
    result: dict[str, Any]

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelCall":
        strict_keys(value, {"role", "model_id", "revision", "result"},
                    {"role", "model_id", "revision", "result"}, "ModelCall")
        if value["role"] not in {"actor", "preference"}:
            raise ValueError("Unknown model role")
        nonempty(value["model_id"], "model_id")
        nonempty(value["revision"], "revision")
        if not isinstance(value["result"], dict) or not isinstance(value["result"].get("output"), str):
            raise ValueError("ModelCall requires a textual output")
        return cls(**value)

    @property
    def output(self) -> str:
        return self.result["output"]

    @property
    def quality_flags(self) -> list[str]:
        flags = list(self.result.get("quality_flags") or [])
        if self.result.get("finish_reason") == "length":
            flags.append("truncated")
        if not self.output.strip():
            flags.append("empty_output")
        return sorted(set(flags))


@dataclass(frozen=True)
class DecisionState:
    task: TaskSample
    initial: ModelCall

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "DecisionState":
        strict_keys(value, {"task", "initial"}, {"task", "initial"}, "DecisionState")
        initial = ModelCall.from_dict(value["initial"])
        if initial.role != "actor":
            raise ValueError("Initial execution must use actor")
        return cls(TaskSample.from_dict(value["task"]), initial)

    def router_features(self) -> dict[str, Any]:
        """Build a whitelist, NEVER a dump of the full call/raw response."""
        confidence = self.initial.result.get("confidence") or {}
        safe_confidence = {}
        for key in CONFIDENCE_KEYS:
            value = confidence.get(key)
            if isinstance(value, bool) or (isinstance(value, (float, int)) and math.isfinite(value)):
                safe_confidence[key] = value
        reasoning = self.initial.result.get("reasoning_content")
        return {
            "task_id": self.task.domain,
            "problem": self.task.problem,
            "context": self.task.context,
            "initial_answer": self.initial.output,
            "initial_reasoning": reasoning if isinstance(reasoning, str) else "",
            "confidence": safe_confidence,
            "online_feedback": {
                "output_nonempty": bool(self.initial.output.strip()),
                "truncated": "truncated" in self.initial.quality_flags,
            },
        }


@dataclass(frozen=True)
class StrategyTrace:
    policy_version: str
    calls: tuple[ModelCall, ...]
    final_output: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "StrategyTrace":
        strict_keys(value, {"policy_version", "calls", "final_output"},
                    {"policy_version", "calls", "final_output"}, "StrategyTrace")
        nonempty(value["policy_version"], "policy_version")
        if not isinstance(value["calls"], list) or not isinstance(value["final_output"], str):
            raise ValueError("Invalid StrategyTrace")
        return cls(value["policy_version"], tuple(ModelCall.from_dict(c) for c in value["calls"]), value["final_output"])


@dataclass(frozen=True)
class CollectionRecord:
    schema_version: str
    purpose: str
    run_id: str
    attempt_id: str
    state: DecisionState
    no_call: StrategyTrace
    call: StrategyTrace

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CollectionRecord":
        keys = set(cls.__dataclass_fields__)
        strict_keys(value, keys, keys, "CollectionRecord")
        if value["schema_version"] != SCHEMA_VERSION or value["purpose"] != "pilot":
            raise ValueError("Unsupported collection schema/purpose")
        nonempty(value["run_id"], "run_id")
        nonempty(value["attempt_id"], "attempt_id")
        state = DecisionState.from_dict(value["state"])
        no_call, call = StrategyTrace.from_dict(value["no_call"]), StrategyTrace.from_dict(value["call"])
        if no_call.policy_version != "retain_initial_v1" or no_call.calls or no_call.final_output != state.initial.output:
            raise ValueError("No-call baseline must retain exactly the shared initial result")
        if call.policy_version != "preference_repair_single_v1" or len(call.calls) != 2:
            raise ValueError("Unsupported call strategy/budget")
        if [c.role for c in call.calls] != ["preference", "actor"] or call.final_output != call.calls[-1].output:
            raise ValueError("Call strategy must use preference then actor; no oracle candidate selection")
        return cls(value["schema_version"], value["purpose"], value["run_id"], value["attempt_id"], state, no_call, call)


@dataclass(frozen=True)
class OfflineSupervision:
    sample_id: str
    group_id: str
    domain: str
    gold: str | None = None
    tests: dict[str, Any] | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "OfflineSupervision":
        keys = set(cls.__dataclass_fields__)
        strict_keys(value, keys, {"sample_id", "group_id", "domain"}, "OfflineSupervision")
        for key in ("sample_id", "group_id"):
            nonempty(value[key], key)
        if value["domain"] not in DOMAINS:
            raise ValueError("Unknown supervision domain")
        if value["domain"] == "code":
            tests = value.get("tests")
            if not isinstance(tests, dict) or not isinstance(tests.get("inputs"), list) or not isinstance(tests.get("outputs"), list):
                raise ValueError("Code supervision requires tests")
            if not tests["inputs"] or len(tests["inputs"]) != len(tests["outputs"]):
                raise ValueError("Code tests must have equal nonempty input/output lengths")
            nonempty(tests.get("fn_name"), "fn_name")
        elif not isinstance(value.get("gold"), str) or not value["gold"].strip():
            raise ValueError("Math/QA supervision requires gold")
        return cls(**value)


def serialize(value: Any) -> dict[str, Any]:
    return asdict(value)
