"""OpenAI-compatible actor client and generation confidence extraction."""
from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
import urllib.error
import urllib.request
from typing import Any, Callable


class TraceWriteError(OSError):
    """A journal failure is not a model/API failure and must never be retried."""

def prompt_hash(messages: list[dict[str, str]]) -> str:
    packed = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(packed.encode("utf-8")).hexdigest()


def summarize_logprobs(choice: dict[str, Any]) -> dict[str, Any]:
    content = ((choice.get("logprobs") or {}).get("content") or [])
    values = [float(item["logprob"]) for item in content if isinstance(item, dict) and item.get("logprob") is not None]
    if not values:
        return {"available": False}
    ordered = sorted(values)
    p10_index = max(0, math.ceil(len(ordered) * 0.10) - 1)
    return {
        "available": True,
        "token_count": len(values),
        "mean_logprob": statistics.fmean(values),
        "min_logprob": min(values),
        "p10_logprob": ordered[p10_index],
    }


def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_actor(
    *,
    base_url: str,
    model: str,
    system: str,
    user: str,
    temperature: float,
    max_tokens: int,
    seed: int,
    timeout: float,
    retries: int,
    trace: Callable[[str, dict[str, Any]], None] | None = None,
    top_p: float = 0.9,
    enable_thinking: bool | None = None,
) -> dict[str, Any]:
    def emit(event: str, data: dict[str, Any]) -> None:
        if trace:
            try:
                trace(event, data)
            except Exception as exc:
                raise TraceWriteError(f"Cannot persist {event}: {exc}") from exc

    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
        "seed": seed,
        "logprobs": True,
        "top_logprobs": 5,
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    last_error: Exception | None = None
    logprobs_fallback = False
    started = time.monotonic()
    for attempt in range(retries + 1):
        # Journal writes deliberately sit outside request error handling: a
        # disk failure must stop the run, not cause another model request.
        emit("request.started", {"request_attempt": attempt + 1, "payload": dict(payload)})
        try:
            try:
                response = post_json(f"{base_url.rstrip('/')}/chat/completions", payload, timeout)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")
                if payload.get("logprobs") and exc.code == 400:
                    logprobs_fallback = True
                    emit("request.logprobs_fallback", {"request_attempt": attempt + 1,
                                                       "http_status": exc.code, "body": body})
                    payload.pop("logprobs", None)
                    payload.pop("top_logprobs", None)
                    response = post_json(f"{base_url.rstrip('/')}/chat/completions", payload, timeout)
                else:
                    raise RuntimeError(f"HTTP {exc.code}: {body[:1000]}") from exc
            emit("response.received", {"request_attempt": attempt + 1, "raw_response": response})
            choice = response["choices"][0]
            output = (choice.get("message") or {}).get("content") or ""
            result = {
                "status": "ok",
                "output": output,
                "reasoning_content": (choice.get("message") or {}).get("reasoning_content"),
                "raw_response": response,
                "quality_flags": [flag for flag, condition in (
                    ("truncated", choice.get("finish_reason") == "length"),
                    ("empty_output", not output.strip()),
                ) if condition],
                "finish_reason": choice.get("finish_reason"),
                "usage": response.get("usage") or {},
                "confidence": summarize_logprobs(choice),
                "latency_seconds": time.monotonic() - started,
                "seed": seed,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "prompt_hash": prompt_hash(messages),
                "messages": messages,
                "attempts": attempt + 1,
                "logprobs_fallback": logprobs_fallback,
            }
        except TraceWriteError:
            raise
        except Exception as exc:  # infrastructure error; retry, never label as model failure
            last_error = exc
            emit("request.failed", {"request_attempt": attempt + 1,
                                    "error_type": type(exc).__name__, "error": str(exc)})
            if attempt < retries:
                time.sleep(min(2**attempt, 8))
            continue
        emit("request.completed", {"request_attempt": attempt + 1, "call": result})
        return result
    raise RuntimeError(f"actor request failed after {retries + 1} attempts: {last_error}")
