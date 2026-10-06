"""Compact paired-run records; generation and scoring fields remain unchanged."""
import hashlib
import json

STORAGE_FORMAT = "paired_compact_v1"


def compact_result(result):
    # Checkpoints already carry the exact system/user request. Token-level raw
    # responses and duplicate messages are not inputs to scoring or routing.
    return {k: v for k, v in result.items() if k not in {"raw_response", "messages"}}


def text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def result_summary(result):
    fields = {k: v for k, v in compact_result(result).items()
              if k not in {"output", "reasoning_content", "generated_token_ids"}}
    fields.update(output_sha256=text_digest(result.get("output", "")),
                  output_chars=len(result.get("output", "")))
    return fields


def compact_event_data(data):
    result = dict(data)
    if "payload" in result:
        payload = dict(result["payload"])
        messages = payload.pop("messages", [])
        payload["prompt_hash"] = text_digest(json.dumps(messages, ensure_ascii=False, sort_keys=True))
        result["payload"] = payload
    if "raw_response" in result:
        raw = result.pop("raw_response")
        result["response_summary"] = {
            "id": raw.get("id"), "model": raw.get("model"), "usage": raw.get("usage"),
            "finish_reasons": [c.get("finish_reason") for c in raw.get("choices", [])],
        }
    for key in ("call", "result"):
        if key in result:
            result[key] = result_summary(result[key])
    if "request" in result:
        request = result.pop("request")
        result["request_summary"] = {k: v for k, v in request.items() if k not in {"system", "user"}}
        result["request_summary"]["request_sha256"] = text_digest(
            json.dumps(request, ensure_ascii=False, sort_keys=True))
    return result
