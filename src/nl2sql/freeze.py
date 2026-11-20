"""Bind final evaluations to frozen data, fixtures, conditions and configuration."""
from __future__ import annotations

from pathlib import Path

from .common import (digest, file_hash, fingerprint_tree, planned_conditions, read_json,
                     read_jsonl, require_gate, require_unique, validate_families,
                     verify_record_fingerprint, write_json)
from .data import near_pairs, verify_prepared


def final_review_path(cfg: dict) -> str:
    return cfg["gates"].get("final_evaluation_review", "docs/final_evaluation_review.json")


def verify_conditions(cfg: dict) -> dict:
    """Verify every condition, not just the file used by the current run."""
    root = Path(cfg["paths"]["conditions"])
    manifest = read_json(root / "manifest.json")
    if set(manifest["files"]) != {"A0", "A1", "A2", "A3", "A3_matched"}:
        raise ValueError("Condition manifest must contain A0–A3 and A3_matched")
    prepared = Path(cfg["paths"]["prepared"])
    if manifest["prepared_manifest_sha256"] != file_hash(prepared / "manifest.json"):
        raise ValueError("Conditions belong to a different prepared dataset")
    if manifest["budget"] != cfg["data"]["augmentation_budget"] or manifest["seed"] != cfg["data"]["split_seed"]:
        raise ValueError("Augmentation budget/selection seed differs from the frozen conditions")
    if manifest["accepted_sha256"] != file_hash(cfg["paths"]["validated"]):
        raise ValueError("Accepted augmentation changed since conditions were built")
    originals = read_jsonl(prepared / "train.jsonl")
    accepted = read_jsonl(cfg["paths"]["validated"])
    for row in accepted:
        verify_record_fingerprint(row)
    source_rows, approved_rows = require_unique(originals), require_unique(accepted)
    budget = manifest["budget"]
    expected_counts = {"A0": (0, 0), "A1": (budget, 0), "A2": (0, budget),
                       "A3": (budget, budget), "A3_matched": (budget // 2, budget - budget // 2)}
    for condition, entry in manifest["files"].items():
        path = root / f"{condition}.jsonl"
        if file_hash(path) != entry["sha256"]:
            raise ValueError(f"Frozen condition modified: {condition}")
        rows = read_jsonl(path)
        actual = require_unique(rows)
        if len(rows) != entry["count"] or len(rows) != len(originals) + sum(expected_counts[condition]):
            raise ValueError(f"Incorrect condition size: {condition}")
        for identifier, row in source_rows.items():
            if actual.get(identifier) != row:
                raise ValueError(f"Original missing or changed in {condition}: {identifier}")
        additions = [r for r in rows if r["id"] not in source_rows]
        for row in additions:
            if row != approved_rows.get(row["id"]):
                raise ValueError(f"Unapproved augmentation in {condition}: {row['id']}")
        counts = tuple(sum(r["transformation_type"] == kind for r in additions) for kind in ("paraphrase", "counterfactual"))
        if counts != expected_counts[condition]:
            raise ValueError(f"Condition type budget differs: {condition}")
    return manifest


def verify_dataset_manifest(path: str | Path) -> dict:
    """Require a producer manifest, rather than trusting validation status strings."""
    path = Path(path)
    local_manifest = path.parent / "manifest.json"
    if local_manifest.is_file():
        manifest = read_json(local_manifest)
        if "test_sha256" in manifest:
            if file_hash(path) != manifest["test_sha256"]:
                raise ValueError(f"Diagnostic dataset changed after review/freeze: {path}")
            for db, expected in manifest["databases"].items():
                if file_hash(db) != expected:
                    raise ValueError(f"Diagnostic fixture changed: {db}")
            for row in read_jsonl(path):
                verify_record_fingerprint(row)
            return manifest
    manifest = read_json(str(path) + ".manifest.json")
    if manifest.get("output_sha256") != file_hash(path):
        raise ValueError(f"External benchmark changed since import: {path}")
    for db, expected in manifest["databases"].items():
        if file_hash(db) != expected:
            raise ValueError(f"External benchmark fixture changed: {db}")
    return manifest


def freeze_experiment(cfg: dict, datasets: list[str]) -> None:
    verify_prepared(cfg)
    verify_conditions(cfg)
    for name, path in cfg["gates"].items():
        require_gate(path, name)
    review_path = final_review_path(cfg)
    require_gate(review_path, "Final evaluation review")
    review = read_json(review_path)
    checks = ("all_hyperparameters_selected_on_internal_validation", "diagnostic_annotations_reviewed_before_final_predictions")
    if not all(review.get(key) is True for key in checks):
        raise ValueError("Complete both final evaluation review checks before freezing")
    if sorted(review.get("planned_seeds", [])) != sorted(cfg["training"]["seeds"]):
        raise ValueError("Final review seeds differ from configured seeds")
    if sorted(review.get("planned_conditions", [])) != sorted(planned_conditions(cfg)):
        raise ValueError("Final review conditions differ from training.conditions")
    baseline_review = read_json(cfg["gates"]["baseline_review"])
    if baseline_review.get("baseline_is_suitable") is not True or not baseline_review.get("pilot_run"):
        raise ValueError("Baseline review must identify a suitable completed pilot run")
    pilot_path = Path(baseline_review["pilot_run"])
    if pilot_path.is_dir():
        pilot_path = pilot_path / "run.json"
    pilot = read_json(pilot_path)
    if (pilot.get("status") != "complete" or pilot.get("pilot") is not True or pilot.get("condition") != "A0"
            or pilot["config"]["model"] != cfg["model"]
            or pilot["pretrained_model"]["sha256"] != fingerprint_tree(cfg["model"]["path"])["sha256"]
            or pilot["validation_sha256"] != file_hash(Path(cfg["paths"]["prepared"]) / "validation.jsonl")):
        raise ValueError("Baseline review refers to a different or incomplete pilot/model/validation set")
    quality_review = read_json(cfg["gates"]["augmentation_pilot_review"])
    if type(quality_review.get("pilot_examples_reviewed")) is not int or quality_review["pilot_examples_reviewed"] < 1:
        raise ValueError("Augmentation pilot review must record a positive reviewed example count")
    conditions = Path(cfg["paths"]["conditions"])
    dest = conditions / "final_freeze.json"
    if dest.exists():
        raise FileExistsError("Final experiment is already frozen; use new artifact roots for a new protocol")
    supplied = [str(Path(p).resolve()) for p in datasets]
    prepared_test = (Path(cfg["paths"]["prepared"]) / "test.jsonl").resolve()
    controlled_test = (Path(cfg["paths"]["diagnostics"]) / "test.jsonl").resolve()
    if len(supplied) != len(set(supplied)) or not {str(prepared_test), str(controlled_test)} <= set(supplied):
        raise ValueError("Register Spider test and controlled diagnostics exactly once, plus any external tests")
    training = read_jsonl(conditions / "A3.jsonl")
    heldout = read_jsonl(Path(cfg["paths"]["prepared"]) / "validation.jsonl")
    evaluation, all_rows = {}, {}
    for path in supplied:
        rows = read_jsonl(path)
        validate_families(rows)
        if not rows or any(r.get("split") != "test" for r in rows):
            raise ValueError(f"Final dataset must contain only test examples: {path}")
        for field in ("family_id", "source_id", "db_id"):
            if {r[field] for r in rows} & {r[field] for r in training + heldout}:
                raise ValueError(f"Final {field} overlaps training/validation: {path}")
        if next(near_pairs(rows, training + heldout, cfg["data"]["near_duplicate_threshold"]), None) is not None:
            raise ValueError(f"Final question near-duplicates training/validation: {path}")
        producer = None
        if Path(path) != prepared_test:
            producer = verify_dataset_manifest(path)
            for row in rows:
                if row.get("semantic_review_status") == "benchmark_gold":
                    if not row.get("provenance", {}).get("benchmark_version") or not row.get("provenance", {}).get("license"):
                        raise ValueError("External benchmark records need version/license provenance")
                else:
                    verify_record_fingerprint(row)
        databases = {p for row in rows for p in row["database_paths"]}
        evaluation[path] = {"sha256": file_hash(path), "count": len(rows), "producer_manifest": producer,
                            "databases": {p: file_hash(p) for p in sorted(databases)}}
        all_rows[path] = rows
    controlled = all_rows[str(controlled_test)]
    for path, rows in all_rows.items():
        if path == str(controlled_test):
            continue
        for field in ("family_id", "source_id", "db_id"):
            if {r[field] for r in controlled} & {r[field] for r in rows}:
                raise ValueError(f"Controlled diagnostics overlap a benchmark's {field}")
        if next(near_pairs(controlled, rows, cfg["data"]["near_duplicate_threshold"]), None) is not None:
            raise ValueError("Controlled diagnostic questions overlap final benchmark questions")
    evidence = {*cfg["gates"].values(), review_path, str(pilot_path), "docs/research_protocol.md", "docs/related_work.csv"}
    write_json(dest, {"config_sha256": digest(cfg), "datasets": evaluation,
        "conditions": planned_conditions(cfg), "pretrained_model": fingerprint_tree(cfg["model"]["path"]),
        "conditions_manifest_sha256": file_hash(conditions / "manifest.json"),
        "prepared_manifest_sha256": file_hash(Path(cfg["paths"]["prepared"]) / "manifest.json"),
        "gates": {p: file_hash(p) for p in sorted(evidence)},
        "source": fingerprint_tree(Path(__file__).parent, suffixes=(".py",)),
        "policy": "No hyperparameter/prompt/augmentation selection using final results"})


def verify_final(cfg: dict, dataset: str | None = None) -> dict:
    conditions = Path(cfg["paths"]["conditions"])
    freeze = read_json(conditions / "final_freeze.json")
    if freeze["config_sha256"] != digest(cfg):
        raise ValueError("Configuration changed after final freeze")
    if freeze["conditions_manifest_sha256"] != file_hash(conditions / "manifest.json"):
        raise ValueError("Conditions manifest changed after freeze")
    if freeze["prepared_manifest_sha256"] != file_hash(Path(cfg["paths"]["prepared"]) / "manifest.json"):
        raise ValueError("Prepared manifest changed after freeze")
    verify_conditions(cfg)
    for path, expected in freeze["gates"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Frozen protocol/evidence modified: {path}")
    if fingerprint_tree(Path(__file__).parent, suffixes=(".py",))["sha256"] != freeze["source"]["sha256"]:
        raise ValueError("Pipeline source changed after final freeze")
    selected = freeze["datasets"] if dataset is None else {str(Path(dataset).resolve()): freeze["datasets"].get(str(Path(dataset).resolve()))}
    for path, entry in selected.items():
        if entry is None or entry["sha256"] != file_hash(path):
            raise ValueError(f"Dataset unregistered or changed since freeze: {path}")
        for db, expected in entry["databases"].items():
            if file_hash(db) != expected:
                raise ValueError(f"Frozen fixture modified: {db}")
    return freeze
