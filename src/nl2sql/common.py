"""File formats, offline configuration, identities, and reproducibility."""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

import yaml


def offline() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["WANDB_DISABLED"] = "true"


def read_json(path: str | Path) -> Any:
    with Path(path).open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def read_jsonl(path: str | Path) -> list[dict]:
    records = []
    with Path(path).open(encoding="utf-8-sig") as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(f"{path}:{number}: invalid JSON: {error}") from error
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{number}: expected a JSON object")
                records.append(record)
    return records


def _atomic_text(path: str | Path, content: str) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(destination)


def write_json(path: str | Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def write_jsonl(path: str | Path, records: Iterable[dict]) -> None:
    _atomic_text(path, "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records))


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    result = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def fingerprint_tree(path: str | Path, suffixes: tuple[str, ...] | None = None) -> dict:
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f"Expected local directory: {root}")
    files = {p.relative_to(root).as_posix(): file_hash(p) for p in sorted(root.rglob("*"))
             if p.is_file() and (suffixes is None or p.suffix in suffixes)}
    if not files:
        raise ValueError(f"Empty local directory: {root}")
    return {"path": str(root.resolve()), "sha256": digest(files), "files": files}


def config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8-sig") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict) or cfg.get("version") != 1:
        raise ValueError("Expected configuration version: 1")
    for section in ("data", "model", "training", "paths", "gates"):
        if not isinstance(cfg.get(section), dict):
            raise ValueError(f"Configuration section {section} must be an object")
    if not 0 < cfg["data"]["validation_fraction"] < 1:
        raise ValueError("validation_fraction must be between zero and one")
    if cfg["training"]["precision"] not in {"fp32", "fp16", "bf16"}:
        raise ValueError("precision must be fp32, fp16, or bf16")
    settings = cfg["training"]
    for key in ("max_steps", "eval_steps", "save_total_limit", "per_device_train_batch_size", "per_device_eval_batch_size", "gradient_accumulation_steps"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"training.{key} must be a positive integer")
    if (not isinstance(settings["seeds"], list) or not settings["seeds"]
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in settings["seeds"])
            or len(set(settings["seeds"])) != len(settings["seeds"])):
        raise ValueError("training.seeds must be nonempty and unique")
    planned_conditions(cfg)
    for key in ("dataloader_num_workers", "early_stopping_patience"):
        if type(settings[key]) is not int or settings[key] < 0:
            raise ValueError(f"training.{key} must be a nonnegative integer")
    if settings["eval_steps"] > settings["max_steps"]:
        raise ValueError("max_steps must include at least one evaluation/save interval")
    for key in ("learning_rate", "weight_decay", "warmup_ratio"):
        if not isinstance(settings[key], (int, float)) or not math.isfinite(settings[key]):
            raise ValueError(f"training.{key} must be finite")
    if settings["learning_rate"] <= 0 or settings["weight_decay"] < 0 or not 0 <= settings["warmup_ratio"] <= 1:
        raise ValueError("Invalid learning rate, weight decay or warmup ratio")
    if settings["report_to"] not in ([], "none"):
        raise ValueError("Offline runs require training.report_to: [] or none")
    for key in ("max_input_length", "max_target_length", "num_beams"):
        if type(cfg["model"][key]) is not int or cfg["model"][key] < 1:
            raise ValueError(f"model.{key} must be a positive integer")
    if not 0 < cfg["data"]["near_duplicate_threshold"] <= 1:
        raise ValueError("near_duplicate_threshold must be in (0,1]")
    if cfg["data"]["query_timeout_seconds"] <= 0 or not math.isfinite(cfg["data"]["query_timeout_seconds"]):
        raise ValueError("query_timeout_seconds must be positive and finite")
    for key, minimum in (("augmentation_budget", 1), ("diagnostic_families_per_phenomenon", 1), ("diagnostic_fixture_count", 2)):
        if type(cfg["data"][key]) is not int or cfg["data"][key] < minimum:
            raise ValueError(f"data.{key} must be an integer at least {minimum}")
    if not isinstance(cfg["data"]["train_files"], list) or not cfg["data"]["train_files"] or any(not isinstance(p, str) or not p for p in cfg["data"]["train_files"]):
        raise ValueError("data.train_files must be a nonempty list of local filenames")
    if type(cfg["data"]["query_max_rows"]) is not int or cfg["data"]["query_max_rows"] < 1:
        raise ValueError("query_max_rows must be a positive integer")
    for key in ("allow_input_truncation",):
        if type(cfg["model"][key]) is not bool:
            raise ValueError(f"model.{key} must be a boolean")
    if type(settings["gradient_checkpointing"]) is not bool:
        raise ValueError("gradient_checkpointing must be a boolean")
    additions = cfg["model"].get("added_tokens", ["<", "<=", "<>", ">=", "!="])
    if (not isinstance(additions, list) or any(not isinstance(t, str) or not t for t in additions)
            or len(set(additions)) != len(additions)):
        raise ValueError("model.added_tokens must be a list of unique nonempty ordinary tokens")
    return cfg


def normalized_question(text: str) -> str:
    # Preserve semantic signs, decimal boundaries and punctuation in exact dedup.
    return " ".join(re.findall(r"[+-]?\d+(?:\.\d+)?|\w+(?:['’]\w+)*|[<>=!]+|[^\w\s]", text.casefold()))


def question_similarity(a: str, b: str) -> float:
    left, right = set(normalized_question(a).split()), set(normalized_question(b).split())
    return len(left & right) / len(left | right) if left or right else 1.0


def require_unique(records: list[dict], key: str = "id") -> dict[str, dict]:
    result = {}
    for row in records:
        identifier = row[key]
        if not isinstance(identifier, str) or not identifier.strip():
            raise ValueError(f"{key} must be a nonempty string")
        if identifier in result:
            raise ValueError(f"Duplicate {key}: {identifier}")
        result[identifier] = row
    return result


def reviewed(record: dict) -> bool:
    review = record.get("review", {})
    if not isinstance(review, dict):
        return False
    checks = ("natural", "unambiguous", "sql_matches_question", "constraints_checked", "null_semantics_checked")
    return (record.get("semantic_review_status") == "reviewed"
            and review.get("decision") == "accept"
            and all(isinstance(review.get(key), str) and bool(review[key].strip())
                    for key in ("reviewer", "reviewed_at", "notes"))
            and all(review.get(check) is True for check in checks))


def require_gate(path: str | Path, purpose: str) -> None:
    gate = read_json(path)
    if not (isinstance(gate, dict) and gate.get("approved") is True
            and all(isinstance(gate.get(key), str) and gate[key].strip() for key in ("reviewer", "evidence", "reviewed_at"))):
        raise ValueError(f"{purpose} is pending. Complete the evidence and approval record in {path}.")


def planned_conditions(cfg: dict) -> list[str]:
    conditions = cfg["training"].get("conditions", ["A0", "A1", "A2", "A3", "A3_matched"])
    core = {"A0", "A1", "A2", "A3"}
    if (not isinstance(conditions, list) or any(not isinstance(c, str) for c in conditions)
            or len(set(conditions)) != len(conditions) or not core <= set(conditions)
            or not set(conditions) <= core | {"A3_matched"}):
        raise ValueError("training.conditions must include A0–A3 exactly once; A3_matched is optional")
    return list(conditions)


def verify_record_fingerprint(row: dict) -> None:
    if (row.get("validation_status") != "accepted" or not reviewed(row)
            or row.get("validation_fingerprint") != digest({k: v for k, v in row.items() if k != "validation_fingerprint"})):
        raise ValueError(f"Unvalidated or edited record: {row.get('id')}")


def validate_families(rows: list[dict]) -> dict[str, list[dict]]:
    """Check pair links before either training-set freeze or semantic scoring."""
    require_unique(rows)
    families = {}
    for row in rows:
        for field in ("family_id", "source_id", "db_id", "question", "gold_sql", "phenomenon"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"Missing or invalid {field}: {row.get('id')}")
        paths = row.get("database_paths")
        if (not isinstance(paths, list) or not paths or any(not isinstance(p, str) or not p for p in paths)
                or len(set(paths)) != len(paths)):
            raise ValueError(f"database_paths must be a nonempty list of unique local paths: {row['id']}")
        if row.get("transformation_type") not in {"original", "paraphrase", "counterfactual"}:
            raise ValueError(f"Invalid transformation type: {row['id']}")
        families.setdefault(row["family_id"], []).append(row)
    for family, members in families.items():
        originals = [r for r in members if r["transformation_type"] == "original"]
        if len(originals) != 1:
            raise ValueError(f"Expected one original in {family}")
        original = originals[0]
        for row in members:
            if row["source_id"] != original["id"]:
                raise ValueError(f"Incorrect source_id in {family}: {row['id']}")
            for field in ("db_id", "database_paths", "schema", "schema_serialization_version"):
                if row.get(field) != original.get(field):
                    raise ValueError(f"Mismatched {field} in {family}")
            if row["transformation_type"] != "original":
                for field, reference in (("original_sql", "gold_sql"), ("original_question", "question")):
                    if field in row and row[field] != original[reference]:
                        raise ValueError(f"Mismatched {field} in {family}")
    return families


def metadata(cfg: dict) -> dict:
    packages = {}
    for name in ("torch", "transformers", "datasets", "accelerate", "sqlglot", "numpy", "sentencepiece",
                 "PyYAML", "protobuf", "safetensors", "setuptools"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        commit = None
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
            "platform": platform.platform(), "packages": packages, "git_commit": commit,
            "config": cfg, "config_sha256": digest(cfg)}


def shuffled(records: list[dict], seed: int) -> list[dict]:
    result = list(records)
    random.Random(seed).shuffle(result)
    return result
