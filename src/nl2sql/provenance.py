"""Shared prediction/archive verification for local and official scoring."""
from __future__ import annotations

from pathlib import Path

from .common import digest, file_hash, read_json, read_jsonl, require_unique, validate_families


def verify_predictions(dataset: str, predictions: str, cfg: dict | None = None):
    rows, predicted = read_jsonl(dataset), read_jsonl(predictions)
    if not rows:
        raise ValueError("Cannot evaluate an empty dataset")
    validate_families(rows)
    by_id, pred_by_id = require_unique(rows), require_unique(predicted)
    if set(by_id) != set(pred_by_id):
        raise ValueError("Prediction IDs must exactly match all evaluation IDs")
    manifest = read_json(predictions + ".manifest.json")
    if manifest["dataset_sha256"] != file_hash(dataset) or manifest["prediction_sha256"] != file_hash(predictions):
        raise ValueError("Prediction provenance does not match dataset/prediction content")
    database_paths = {p for row in rows for p in row["database_paths"]}
    if set(manifest["databases"]) != database_paths:
        raise ValueError("Prediction manifest does not cover exactly the evaluation fixtures")
    for path, expected in manifest["databases"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Database changed since prediction: {path}")
    if cfg is not None and digest(manifest["config"]) != digest(cfg):
        raise ValueError("Evaluation configuration differs from prediction configuration")
    for row in predicted:
        if row.get("condition") != manifest["condition"] or row.get("seed") != manifest["seed"]:
            raise ValueError("Mixed or incorrect condition/seed in predictions")
        if not isinstance(row.get("predicted_sql"), str):
            raise ValueError(f"predicted_sql must be a string: {row['id']}")
    if manifest.get("final"):
        if cfg is None:
            raise ValueError("Final evaluation requires the frozen configuration")
        from .freeze import verify_final
        frozen = verify_final(cfg, dataset)
        run_path = manifest["run_metadata_path"]
        if file_hash(run_path) != manifest["run_metadata_sha256"]:
            raise ValueError("Run metadata changed since prediction")
        run = read_json(run_path)
        if run.get("status") != "complete" or run.get("pilot") or run["condition"] != manifest["condition"] or run["seed"] != manifest["seed"]:
            raise ValueError("Final predictions must come from a completed matching final run")
        if (digest(run["config"]) != digest(cfg) or run["checkpoint"]["sha256"] != manifest["checkpoint"]["sha256"]
                or run["pretrained_model"]["sha256"] != frozen["pretrained_model"]["sha256"]):
            raise ValueError("Final prediction run/checkpoint provenance differs from the frozen experiment")
    local_manifest = Path(dataset).parent / "manifest.json"
    if local_manifest.is_file():
        local = read_json(local_manifest)
        if "test_sha256" in local and local["test_sha256"] != file_hash(dataset):
            raise ValueError("Controlled diagnostic changed after freeze")
    return rows, pred_by_id, manifest


def verify_evaluation_archive(path: str | Path) -> dict:
    root = Path(path)
    completion = read_json(root / "completion.json")
    for name, expected in completion["files"].items():
        if file_hash(root / name) != expected:
            raise ValueError(f"Evaluation archive changed or is incomplete: {root / name}")
    if not {"items.jsonl", "pairs.jsonl", "metrics.json"} <= set(completion["files"]):
        raise ValueError("Evaluation completion manifest is incomplete")
    return read_json(root / "metrics.json")
