import copy

import pytest

from nl2sql.common import digest, file_hash, write_json, write_jsonl
from nl2sql.freeze import verify_conditions
from nl2sql.provenance import verify_evaluation_archive


def test_freeze_checks_every_condition_not_only_combined(tmp_path):
    prepared, conditions = tmp_path / "prepared", tmp_path / "conditions"
    accepted_file = tmp_path / "accepted.jsonl"
    cfg = {"paths": {"prepared": str(prepared), "conditions": str(conditions), "validated": str(accepted_file)},
           "data": {"augmentation_budget": 1, "split_seed": 42}}
    original = {"id": "o", "gold_sql": "SELECT 1", "transformation_type": "original"}
    checks = {key: True for key in ("natural", "unambiguous", "sql_matches_question", "constraints_checked", "null_semantics_checked")}
    annotations = []
    for identifier, kind in (("p", "paraphrase"), ("c", "counterfactual")):
        row = {"id": identifier, "transformation_type": kind, "gold_sql": "SELECT 1", "validation_status": "accepted",
               "semantic_review_status": "reviewed", "review": {**checks, "decision": "accept", "reviewer": "test",
                   "reviewed_at": "2026-11-10", "notes": "Synthetic unit-test record"}}
        row["validation_fingerprint"] = digest(row)
        annotations.append(row)
    write_jsonl(prepared / "train.jsonl", [original])
    write_json(prepared / "manifest.json", {"test": "fixture"})
    write_jsonl(accepted_file, annotations)
    para, counter = annotations
    variants = {"A0": [original], "A1": [original, para], "A2": [original, counter],
                "A3": [original, para, counter], "A3_matched": [original, counter]}
    for name, rows in variants.items():
        write_jsonl(conditions / f"{name}.jsonl", rows)
    manifest = {"budget": 1, "seed": 42, "prepared_manifest_sha256": file_hash(prepared / "manifest.json"),
                "accepted_sha256": file_hash(accepted_file), "files": {name: {"count": len(rows), "sha256": file_hash(conditions / f"{name}.jsonl")}
                    for name, rows in variants.items()}}
    write_json(conditions / "manifest.json", manifest)
    verify_conditions(cfg)
    altered = copy.deepcopy(variants["A2"])
    altered[0]["gold_sql"] = "SELECT 2"
    write_jsonl(conditions / "A2.jsonl", altered)
    with pytest.raises(ValueError, match="A2"):
        verify_conditions(cfg)


def test_scoring_archive_cannot_be_silently_edited(tmp_path):
    write_json(tmp_path / "metrics.json", {"condition": "A0"})
    write_jsonl(tmp_path / "items.jsonl", [{"id": "o"}])
    write_jsonl(tmp_path / "pairs.jsonl", [])
    names = ("metrics.json", "items.jsonl", "pairs.jsonl")
    write_json(tmp_path / "completion.json", {"files": {name: file_hash(tmp_path / name) for name in names}})
    assert verify_evaluation_archive(tmp_path)["condition"] == "A0"
    write_jsonl(tmp_path / "items.jsonl", [{"id": "changed"}])
    with pytest.raises(ValueError, match="changed or is incomplete"):
        verify_evaluation_archive(tmp_path)
