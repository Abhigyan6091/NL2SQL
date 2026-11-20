"""Frozen database-disjoint splits, provenance, and leakage audits."""
from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

from .common import (digest, file_hash, normalized_question, question_similarity,
                     read_json, read_jsonl, require_unique, write_json, write_jsonl)
from .schema import SCHEMA_VERSION, load_schemas, serialize
from .sql import abstract_sql


def near_pairs(left: list[dict], right: list[dict], threshold: float):
    """Indexed exact Jaccard candidates; no approximate sampling of pairs."""
    index = defaultdict(list)
    tokens = []
    for i, row in enumerate(right):
        words = set(normalized_question(row["question"]).split())
        tokens.append(words)
        for word in words:
            index[word].append(i)
    for row in left:
        words = set(normalized_question(row["question"]).split())
        candidates = {i for word in words for i in index[word]}
        for i in sorted(candidates):
            other = right[i]
            if row is other:
                continue
            if max(len(words), len(tokens[i])) * threshold > min(len(words), len(tokens[i])):
                continue
            similarity = question_similarity(row["question"], other["question"])
            if similarity >= threshold:
                yield row, other, similarity


def _load_spider(paths: list[str], schemas: dict, database_root: str, role: str) -> list[dict]:
    output = []
    seen = set()
    for path in paths:
        source_hash = file_hash(path)
        for index, item in enumerate(read_json(path)):
            db_id = item["db_id"]
            if db_id not in schemas:
                raise ValueError(f"Unknown schema {db_id} in {path}")
            sql = item.get("query", item.get("gold_sql"))
            if not isinstance(sql, str) or not sql.strip() or not item["question"].strip():
                raise ValueError(f"Missing question/SQL at {path}:{index}")
            key = digest([db_id, item["question"], sql])
            if key in seen and role != "dev":
                continue
            seen.add(key)
            # Preserve the official dev denominator, including repeated rows.
            source = f"spider-{role}-{key[:20]}" + (f"-{Path(path).stem}-{index}" if role == "dev" else "")
            db_path = (Path(database_root) / db_id / f"{db_id}.sqlite").as_posix()
            if not Path(db_path).is_file():
                raise FileNotFoundError(db_path)
            output.append({"id": source, "source_id": source, "family_id": source,
                           "db_id": db_id, "question": item["question"], "gold_sql": sql,
                           "schema": serialize(schemas[db_id]), "schema_serialization_version": SCHEMA_VERSION,
                           "database_paths": [db_path], "phenomenon": "spider",
                           "transformation_type": "original", "semantic_review_status": "benchmark_gold",
                           "abstract_structure": abstract_sql(sql),
                           "provenance": {"file": str(path), "index": index, "source_sha256": source_hash}})
    return output


def prepare(cfg: dict) -> None:
    dest = Path(cfg["paths"]["prepared"])
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Frozen split destination already contains files: {dest}")
    data = cfg["data"]
    for key in ("dataset_version", "license"):
        if not data.get(key) or "FILL_IN" in data[key]:
            raise ValueError(f"Record data.{key} in the configuration first")
    schemas = load_schemas(data["tables_file"])
    train = _load_spider(data["train_files"], schemas, data["database_root"], "train")
    dev = _load_spider([data["dev_file"]], schemas, data["database_root"], "dev")
    overlap = {r["db_id"] for r in train} & {r["db_id"] for r in dev}
    if overlap:
        raise ValueError(f"Spider train/dev database overlap: {sorted(overlap)}")
    threshold = data["near_duplicate_threshold"]
    if not 0 < threshold <= 1:
        raise ValueError("near_duplicate_threshold must be in (0,1]")
    # Quarantine obvious cross-benchmark wording duplicates before freezing splits.
    quarantine_ids, leakage = set(), []
    for left, right, similarity in near_pairs(train, dev, threshold):
        quarantine_ids.add(left["id"])
        leakage.append({"left": left["id"], "right": right["id"], "similarity": similarity})
    quarantine = [r for r in train if r["id"] in quarantine_ids]
    train = [r for r in train if r["id"] not in quarantine_ids]
    databases = sorted({r["db_id"] for r in train})
    if len(databases) < 2:
        raise ValueError("Need at least two training databases after quarantine")
    parent = {db: db for db in databases}
    def root(db):
        while db != parent[db]:
            parent[db] = parent[parent[db]]
            db = parent[db]
        return db
    # Group databases connected by highly similar questions, preserving both rules.
    for left, right, _ in near_pairs(train, train, threshold):
        a, b = root(left["db_id"]), root(right["db_id"])
        if a != b:
            parent[max(a, b)] = min(a, b)
    groups = defaultdict(list)
    for db in databases:
        groups[root(db)].append(db)
    components = list(groups.values())
    random.Random(data["split_seed"]).shuffle(components)
    if len(components) < 2:
        raise ValueError("Near-duplicate grouping connected all databases; review threshold/data")
    target = max(1, round(len(databases) * data["validation_fraction"]))
    validation_dbs = set()
    for group in components[:-1]:
        if len(validation_dbs) >= target:
            break
        validation_dbs.update(group)
    validation = [dict(r, split="validation") for r in train if r["db_id"] in validation_dbs]
    training = [dict(r, split="train") for r in train if r["db_id"] not in validation_dbs]
    dev = [dict(r, split="test") for r in dev]
    audit = audit_splits({"train": training, "validation": validation, "test": dev}, threshold)
    if audit["violations"]:
        raise ValueError(f"Split leakage detected: {audit['violations'][:5]}")
    for label, rows in (("train", training), ("validation", validation), ("test", dev), ("quarantine", quarantine)):
        write_jsonl(dest / f"{label}.jsonl", rows)
    write_json(dest / "audit.json", audit)
    write_json(dest / "quarantine_reasons.json", leakage)
    files = [*data["train_files"], data["dev_file"], data["tables_file"]]
    db_files = {db for row in training + validation + dev for db in row["database_paths"]}
    write_json(dest / "manifest.json", {"dataset": data, "schema_version": SCHEMA_VERSION,
               "source_files": {p: file_hash(p) for p in files},
               "database_files": {p: file_hash(p) for p in sorted(db_files)},
               "split_files": {name: file_hash(dest / f"{name}.jsonl") for name in ("train", "validation", "test")},
               "validation_databases": sorted(validation_dbs), "quarantine_count": len(quarantine)})


def audit_splits(splits: dict[str, list[dict]], threshold: float = .92) -> dict:
    violations, structure = [], []
    labels = list(splits)
    for label, rows in splits.items():
        require_unique(rows)
    for i, label in enumerate(labels):
        for other in labels[i + 1:]:
            left, right = splits[label], splits[other]
            for field in ("family_id", "source_id", "db_id"):
                shared = {r[field] for r in left} & {r[field] for r in right}
                if shared:
                    violations.append({"splits": [label, other], "field": field, "shared": sorted(shared)})
            for a, b, score in near_pairs(left, right, threshold):
                violations.append({"splits": [label, other], "field": "near_question", "ids": [a["id"], b["id"]], "score": score})
            shared_structures = {r.get("abstract_structure") for r in left} & {r.get("abstract_structure") for r in right}
            shared_structures.discard(None)
            structure.append({"splits": [label, other], "shared_abstract_structures": len(shared_structures)})
    return {"counts": {k: len(v) for k, v in splits.items()}, "violations": violations,
            "structure_overlap": structure,
            "structure_policy": "Reported, not excluded for ordinary Spider. Not a compositional holdout claim."}


def verify_prepared(cfg: dict) -> dict:
    root = Path(cfg["paths"]["prepared"])
    manifest = read_json(root / "manifest.json")
    for key in ("train_files", "dev_file", "tables_file", "database_root", "dataset_version",
                "license", "source", "validation_fraction", "split_seed", "near_duplicate_threshold"):
        if manifest["dataset"].get(key) != cfg["data"].get(key):
            raise ValueError(f"Data preparation configuration changed: {key}; prepare a new version")
    for label, expected in manifest["split_files"].items():
        if file_hash(root / f"{label}.jsonl") != expected:
            raise ValueError(f"Frozen split modified: {label}")
    if manifest["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Schema policy changed since preparation")
    for path, expected in {**manifest["source_files"], **manifest["database_files"]}.items():
        if file_hash(path) != expected:
            raise ValueError(f"Frozen source/database modified: {path}")
    return manifest
