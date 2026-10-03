import importlib.util
import json
from pathlib import Path
import tempfile


SCRIPT = Path(__file__).parents[1] / "scripts" / "generate_gain_router_data.py"
SPEC = importlib.util.spec_from_file_location("generate_gain_router_data", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_case_derivation_preserves_all_router_outcomes():
    assert MODULE.derive_case(True, None, None) == ("initial_success", None)
    assert MODULE.derive_case(False, False, True) == ("strict_positive_gain", 1)
    assert MODULE.derive_case(False, True, True) == ("redundant_intervention", 0)
    assert MODULE.derive_case(False, True, False) == ("negative_transfer", 0)
    assert MODULE.derive_case(False, False, False) == ("shared_failure", 0)


def test_qa_metrics_use_normalized_exact_and_retain_f1():
    exact = MODULE.qa_metrics("FINAL_ANSWER: The United Kingdom.", "United Kingdom")
    assert exact["success"] is True
    assert exact["f1"] == 1.0

    partial = MODULE.qa_metrics("FINAL_ANSWER: Berkeley California", "Berkeley")
    assert partial["success"] is False
    assert 0.0 < partial["f1"] < 1.0


def test_math_extraction_and_numeric_equivalence():
    assert MODULE.extract_final_answer("work\nFINAL_ANSWER: 0.5") == "0.5"
    metrics = MODULE.math_metrics("FINAL_ANSWER: 0.5", r"\frac{1}{2}")
    assert metrics["success"] is True
    assert metrics["equivalence"] == "numeric"


def test_hash_split_is_stable_and_disjoint():
    first = MODULE.split_for_group("qa:abc", 42)
    assert first == MODULE.split_for_group("qa:abc", 42)
    assert first in {"train", "validation", "test"}


def test_code_verifier_runs_function_tests_in_subprocess():
    tests = {"fn_name": "add", "inputs": [[1, 2], [3, 4]], "outputs": [3, 7]}
    passed = MODULE.code_metrics("```python\ndef add(a, b):\n    return a + b\n```", tests)
    assert passed["success"] is True
    assert passed["reward"] == 1.0

    failed = MODULE.code_metrics("def add(a, b):\n    return a - b", tests)
    assert failed["success"] is False
    assert failed["reward"] == 0.0


def test_code_loader_accepts_backed_up_input_slash_output_key():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "code.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "question": "Add two integers.",
                        "func_sign": "def add(a, b):\n    pass",
                        "input/output": {"fn_name": "add", "inputs": [[1, 2]], "outputs": [3]},
                        "pitfalls": "Check the sum.",
                    }
                ]
            )
        )
        rows = []
        for split in ("train", "validation", "test"):
            split_rows, _ = MODULE.load_code_rows(path, 42, split)
            rows.extend(split_rows)
        assert len(rows) == 1
        assert rows[0]["tests"]["fn_name"] == "add"


def test_code_loader_joins_apps_metadata_directory_by_exact_question():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        path = root / "dataset" / "code" / "train" / "apps_original.json"
        path.parent.mkdir(parents=True)
        path.write_text(
            json.dumps(
                [
                    {
                        "question": "Multiply two integers.",
                        "func_sign": "def multiply(a, b):\n    pass",
                        "pitfalls": "Check signs.",
                    }
                ]
            )
        )
        meta = root / "dataset" / "code" / "meta" / "train" / "17"
        meta.mkdir(parents=True)
        (meta / "question.txt").write_text("Multiply two integers.\n")
        (meta / "input_output.json").write_text(
            json.dumps({"fn_name": "multiply", "inputs": [[2, 3]], "outputs": [6]})
        )
        rows = []
        for split in ("train", "validation", "test"):
            split_rows, stats = MODULE.load_code_rows(path, 42, split)
            rows.extend(split_rows)
        assert len(rows) == 1
        assert rows[0]["tests"]["fn_name"] == "multiply"
        assert rows[0]["source_meta"]["problem_id"] == "17"
        assert stats["meta_questions_indexed"] == 1


def test_qa_loader_exact_matches_across_multiple_context_sources():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train = root / "train.jsonl"
        context_a = root / "2w.jsonl"
        context_b = root / "hotpot.jsonl"
        train.write_text(
            json.dumps({"question": "Question A?", "answer": "A", "decomposition": "steps"}) + "\n"
            + json.dumps({"question": "Question B?", "answer": "B", "decomposition": "steps"}) + "\n"
        )
        context_a.write_text(
            json.dumps({"id": "a", "question": "Question A?", "context": {"title": ["A"], "sentences": [["fact"]]}}) + "\n"
        )
        context_b.write_text(
            json.dumps({"id": "b", "question": "Question B?", "context": {"title": ["B"], "sentences": [["fact"]]}}) + "\n"
        )
        rows = []
        for split in ("train", "validation", "test"):
            split_rows, stats = MODULE.load_qa_rows(train, [context_a, context_b], 42, split)
            rows.extend(split_rows)
        assert len(rows) == 2
        assert stats["matched_context_questions"] == 2


def test_qa_loader_formats_musique_paragraph_context():
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        train = root / "train.jsonl"
        context = root / "mus_train.jsonl"
        train.write_text(json.dumps({"question": "Who wrote it?", "answer": "Ada"}) + "\n")
        context.write_text(
            json.dumps(
                {
                    "id": "mus-1",
                    "question": "Who wrote it?",
                    "paragraphs": [{"idx": 0, "title": "Biography", "paragraph_text": "Ada wrote it."}],
                }
            )
            + "\n"
        )
        rows = []
        for split in ("train", "validation", "test"):
            split_rows, _ = MODULE.load_qa_rows(train, [context], 42, split)
            rows.extend(split_rows)
        assert len(rows) == 1
        assert "[Biography] Ada wrote it." in rows[0]["context"]


def test_completed_summary_is_cumulative_by_label_domain_and_case():
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "trajectories.jsonl"
        records = [
            {"domain": "code", "outcome": {"case_type": "shared_failure", "label": 0}},
            {"domain": "math", "outcome": {"case_type": "initial_success", "label": None}},
            {"domain": "qa", "outcome": {"case_type": "strict_positive_gain", "label": 1}},
        ]
        path.write_text("".join(json.dumps(row) + "\n" for row in records))
        summary = MODULE.summarize_completed(path)
        assert summary["completed_total"] == 3
        assert summary["case_counts"]["shared_failure"] == 1
        assert summary["label_counts"] == {"0": 1, "null": 1, "1": 1}
        assert summary["domain_case_counts"]["qa"]["strict_positive_gain"] == 1
