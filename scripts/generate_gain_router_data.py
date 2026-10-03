#!/usr/bin/env python3
"""Compatibility entry point for the explicitly opt-in historical G/D pilot.

Use --help. This is NOT the final call/no-call router data generator.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Preserve existing imports used by pilot notebooks and tests.
from gain_router.common import (  # noqa: F401
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    PROTOCOL_VERSION,
    SCHEMA_VERSION,
    stable_id,
    normalize_space,
    split_for_group,
    ranked_sample,
    read_jsonl,
    load_json_array,
    append_jsonl,
    completed_ids,
    summarize_completed,
    file_sha256,
)
from gain_router.datasets import (  # noqa: F401
    format_context,
    find_qa_context_files,
    load_math_rows,
    load_code_rows,
    load_qa_rows,
)
from gain_router.verifiers import (  # noqa: F401
    extract_boxed,
    extract_final_answer,
    normalize_qa,
    qa_metrics,
    normalize_math,
    simple_numeric_math,
    math_metrics,
    extract_code,
    _limit_child_resources,
    code_metrics,
    verify,
)
from gain_router.actor import (  # noqa: F401
    prompt_hash,
    summarize_logprobs,
    post_json,
    call_actor,
)
from gain_router.legacy import (  # noqa: F401
    sample_seed,
    base_task_prompt,
    repair_prompt,
    derive_case,
    trajectory_payload,
    generate_one,
    build_parser,
    main,
    run_samples,
)

if __name__ == "__main__":
    raise SystemExit(main())
