"""Normalize researcher-supplied local robustness/composition benchmark records.

No benchmark-specific raw layout is guessed. Map the local benchmark into the
documented interchange JSONL first; this keeps provenance and pair links explicit.
"""
from __future__ import annotations

from pathlib import Path

from .common import file_hash, read_jsonl, require_unique, validate_families, write_json, write_jsonl
from .schema import SCHEMA_VERSION, load_schemas, serialize
from .sql import equivalent, execute


def import_benchmark(cfg: dict, source: str, tables: str, output: str, name: str,
                     benchmark_version: str, license_text: str) -> None:
    rows = read_jsonl(source)
    require_unique(rows)
    if not rows or not benchmark_version.strip() or not license_text.strip():
        raise ValueError("Need nonempty records and benchmark version/license")
    schemas = load_schemas(tables)
    source_hash, tables_hash = file_hash(source), file_hash(tables)
    normalized = []
    for row in rows:
        for key in ("id", "source_id", "family_id", "db_id", "question", "gold_sql", "database_paths", "transformation_type", "phenomenon"):
            if key not in row or not row[key]:
                raise ValueError(f"Missing {key} in {row.get('id')}")
        if row["transformation_type"] not in {"original", "paraphrase", "counterfactual"}:
            raise ValueError("External transformations must be original/paraphrase/counterfactual")
        schema = schemas[row["db_id"]]
        if not isinstance(row["database_paths"], list) or not row["database_paths"]:
            raise ValueError("database_paths must be a nonempty list of local SQLite paths")
        for db in row["database_paths"]:
            execute(row["gold_sql"], db, cfg["data"]["query_timeout_seconds"], cfg["data"]["query_max_rows"])
        normalized.append({**row, "schema": serialize(schema), "schema_serialization_version": SCHEMA_VERSION,
            "split": "test", "semantic_review_status": "benchmark_gold",
            "provenance": {**row.get("provenance", {}), "benchmark": name, "benchmark_version": benchmark_version,
                           "license": license_text, "input_sha256": source_hash, "tables_sha256": tables_hash}})
    validate_families(normalized)
    families = {}
    for row in normalized:
        families.setdefault(row["family_id"], []).append(row)
    for family, members in families.items():
        originals = [r for r in members if r["transformation_type"] == "original"]
        if len(originals) != 1:
            raise ValueError(f"Each external family needs one original: {family}")
        original = originals[0]
        for row in members:
            if row["source_id"] != original["id"]:
                raise ValueError(f"source_id must identify the original record: {row['id']}")
            if row["database_paths"] != original["database_paths"] or row["db_id"] != original["db_id"]:
                raise ValueError("This adapter supports NL question perturbations with the same schema/fixtures only")
            if row["transformation_type"] == "paraphrase":
                if not equivalent(row["gold_sql"], original["gold_sql"], row["database_paths"],
                                  cfg["data"]["query_timeout_seconds"], cfg["data"]["query_max_rows"]):
                    raise ValueError(f"Published paraphrase gold differs on supplied fixtures: {row['id']}")
            elif row["transformation_type"] == "counterfactual":
                if equivalent(row["gold_sql"], original["gold_sql"], row["database_paths"],
                              cfg["data"]["query_timeout_seconds"], cfg["data"]["query_max_rows"]):
                    raise ValueError(f"External counterfactual lacks a distinguishing fixture: {row['id']}")
    if Path(output).exists():
        raise FileExistsError(output)
    write_jsonl(output, normalized)
    write_json(output + ".manifest.json", {"benchmark": name, "version": benchmark_version,
        "license": license_text, "input_sha256": file_hash(source), "output_sha256": file_hash(output),
        "databases": {p: file_hash(p) for p in sorted({db for row in normalized for db in row["database_paths"]})},
        "relationship": "Dr.Spider derives from Spider dev; overlapping families across benchmarks are not independent evidence."})
