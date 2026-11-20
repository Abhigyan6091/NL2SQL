"""Conservative AST proposals and fail-closed human semantic validation."""
from __future__ import annotations

import copy
from pathlib import Path

from sqlglot import exp

from .common import (digest, file_hash, normalized_question, read_jsonl, require_gate,
                     require_unique, reviewed, shuffled, verify_record_fingerprint, write_json, write_jsonl)
from .data import near_pairs, verify_prepared
from .sql import abstract_sql, canonical, equal_results, execute, ordered, parse_query

REVIEW_TEMPLATE = {"decision": "pending", "reviewer": "", "reviewed_at": "", "notes": "",
                   "natural": False, "unambiguous": False, "sql_matches_question": False,
                   "constraints_checked": False, "null_semantics_checked": False}


def propose(cfg: dict) -> None:
    verify_prepared(cfg)
    output = Path(cfg["paths"]["candidates"])
    if output.exists():
        raise FileExistsError(f"Will not overwrite annotation work: {output}")
    rows = read_jsonl(Path(cfg["paths"]["prepared"]) / "train.jsonl")
    candidates = []
    for source in rows:
        tree = parse_query(source["gold_sql"])
        proposals = [("paraphrase", "paraphrase", source["gold_sql"], "Rewrite the question; preserve every constraint.")]
        # Restrict operator suggestions to a single numeric predicate in a simple WHERE.
        where = tree.args.get("where")
        if isinstance(tree, exp.Select) and where is not None and not any(tree.find_all(exp.Subquery)):
            numeric = [n for n in where.walk() if isinstance(n, (exp.GT, exp.GTE, exp.LT, exp.LTE))
                       and isinstance(n.this, exp.Column) and isinstance(n.expression, exp.Literal)
                       and not n.expression.is_string]
            if len(numeric) == 1:
                node = numeric[0]
                swap = {exp.GT: exp.LTE, exp.GTE: exp.LT, exp.LT: exp.GTE, exp.LTE: exp.GT}
                changed = tree.copy()
                original_sql = node.sql(dialect="sqlite")
                # The same expression may also occur in SELECT or a CTE. Edit
                # only the WHERE predicate that this catalogue actually selected.
                match = next(n for n in changed.args["where"].walk()
                             if isinstance(n, type(node)) and n.sql(dialect="sqlite") == original_sql)
                match.replace(swap[type(node)](this=match.this.copy(), expression=match.expression.copy()))
                proposals.append(("counterfactual", "comparison", changed.sql(dialect="sqlite"),
                                  f"Rewrite only the meaning of {original_sql}; explicitly consider NULL values."))
            if isinstance(where.this, (exp.And, exp.Or)) and not any(isinstance(n, (exp.And, exp.Or)) for n in list(where.this.walk())[1:]):
                changed = tree.copy()
                old = changed.args["where"].this
                replacement = exp.Or if isinstance(old, exp.And) else exp.And
                old.replace(replacement(this=old.this.copy(), expression=old.expression.copy()))
                proposals.append(("counterfactual", "and_or", changed.sql(dialect="sqlite"),
                                  "Rewrite conjunction/disjunction with explicit scope; keep other constraints."))
        for index, (kind, phenomenon, sql, instruction) in enumerate(proposals):
            candidate = copy.deepcopy(source)
            candidate.update({"id": source["id"] + f"-{kind}-{index}", "transformation_type": kind,
                              "phenomenon": phenomenon, "original_question": source["question"],
                              "original_sql": source["gold_sql"], "question": "", "gold_sql": sql,
                              "semantic_review_status": "pending", "review": copy.deepcopy(REVIEW_TEMPLATE),
                              "generation_instruction": instruction,
                              "provenance": {"generator": "ast-catalogue-v1", "source_id": source["id"],
                                             "source_record_sha256": digest(source)}})
            candidates.append(candidate)
    write_jsonl(output, candidates)


def validate_records(rows: list[dict], cfg: dict, sources: dict | None = None) -> tuple[list, list]:
    require_unique(rows)
    accepted, rejected = [], []
    timeout, limit = cfg["data"]["query_timeout_seconds"], cfg["data"]["query_max_rows"]
    question_keys = set()
    original_questions = {(r["db_id"], normalized_question(r["question"])) for r in (sources or {}).values()}
    for input_row in rows:
        row = copy.deepcopy(input_row)
        reasons = []
        try:
            if not reviewed(row):
                raise ValueError("Semantic review/checklist incomplete or decision is not accept")
            if not row.get("question", "").strip():
                raise ValueError("Empty question")
            if row["transformation_type"] not in {"original", "paraphrase", "counterfactual"}:
                raise ValueError("Unknown transformation type")
            if sources is not None:
                if row["transformation_type"] not in {"paraphrase", "counterfactual"}:
                    raise ValueError("Training augmentation must be a paraphrase or counterfactual")
                if row["source_id"] not in sources:
                    raise ValueError("Source is not in the frozen training partition")
                source = sources[row["source_id"]]
                for field in ("family_id", "db_id", "schema", "schema_serialization_version", "database_paths", "split"):
                    if row[field] != source[field]:
                        raise ValueError(f"Source metadata mismatch: {field}")
                if row.get("original_sql") != source["gold_sql"] or row.get("original_question") != source["question"]:
                    raise ValueError("Original SQL/question differs from frozen source")
                if normalized_question(row["question"]) == normalized_question(source["question"]):
                    raise ValueError("Unchanged question is not an augmentation")
                if row.get("provenance", {}).get("source_record_sha256") != digest(source):
                    raise ValueError("Source record provenance hash mismatch")
                if (row["db_id"], normalized_question(row["question"])) in original_questions:
                    raise ValueError("Augmentation question duplicates an original training question")
            if row.get("schema_serialization_version") != "v1":
                raise ValueError("Unsupported schema policy")
            parse_query(row["gold_sql"])
            if (not isinstance(row.get("database_paths"), list) or not row["database_paths"]
                    or any(not isinstance(p, str) or not p for p in row["database_paths"])
                    or len(set(row["database_paths"])) != len(row["database_paths"])):
                raise ValueError("database_paths must contain unique local database paths")
            for db in row["database_paths"]:
                execute(row["gold_sql"], db, timeout, limit)
            if not row["database_paths"]:
                raise ValueError("Missing database fixtures")
            kind = row["transformation_type"]
            if kind in {"paraphrase", "counterfactual"}:
                respect_order = ordered(row["gold_sql"]) or ordered(row["original_sql"])
                differences = [db for db in row["database_paths"] if not equal_results(
                    execute(row["gold_sql"], db, timeout, limit), execute(row["original_sql"], db, timeout, limit), respect_order)]
                same = not differences
                if kind == "paraphrase" and not same:
                    raise ValueError("Paraphrase gold differs on database/test suite")
                if kind == "counterfactual" and same:
                    raise ValueError("Counterfactual is not distinguished by any supplied database")
                if kind == "counterfactual" and canonical(row["gold_sql"]) == canonical(row["original_sql"]):
                    raise ValueError("Counterfactual SQL unchanged")
                row["contrast_witnesses"] = differences if kind == "counterfactual" else []
            alternatives = row.get("plausible_incorrect_sql", [])
            if not isinstance(alternatives, list) or any(not isinstance(s, str) or not s.strip() for s in alternatives):
                raise ValueError("plausible_incorrect_sql must be a list of nonempty SQL strings")
            witnesses = []
            for alternative in alternatives:
                respect = ordered(row["gold_sql"]) or ordered(alternative)
                differences = [db for db in row["database_paths"] if not equal_results(
                    execute(row["gold_sql"], db, timeout, limit), execute(alternative, db, timeout, limit), respect)]
                if not differences:
                    raise ValueError(f"Fixtures do not distinguish plausible incorrect SQL: {alternative}")
                witnesses.append({"incorrect_sql": alternative, "distinguishing_databases": differences})
            row["alternative_witnesses"] = witnesses
            key = (row["db_id"], normalized_question(row["question"]))
            if key in question_keys:
                raise ValueError("Duplicate question within this annotation batch")
            question_keys.add(key)
            row.update(sql_parse_valid=True, execution_valid=True, validation_status="accepted",
                       validation_policy="review-and-readonly-execution-v2", abstract_structure=abstract_sql(row["gold_sql"]))
            row["validation_fingerprint"] = digest({k: v for k, v in row.items() if k != "validation_fingerprint"})
            accepted.append(row)
        except Exception as error:
            reasons.append(f"{type(error).__name__}: {error}")
            row.update(validation_status="rejected", rejection_reasons=reasons)
            rejected.append(row)
    return accepted, rejected


def validate_augmentation(cfg: dict) -> None:
    verify_prepared(cfg)
    source = Path(cfg["paths"]["prepared"])
    training = read_jsonl(source / "train.jsonl")
    rows = read_jsonl(cfg["paths"]["reviewed"])
    accepted, rejected = validate_records(rows, cfg, require_unique(training))
    forbidden = read_jsonl(source / "validation.jsonl") + read_jsonl(source / "test.jsonl")
    leaky = {a["id"] for a, _, _ in near_pairs(accepted, forbidden, cfg["data"]["near_duplicate_threshold"])}
    kept = []
    for row in accepted:
        if row["id"] in leaky:
            rejected.append(dict(row, validation_status="rejected", rejection_reasons=["Near-duplicate of held-out question"]))
        else:
            kept.append(row)
    destination = Path(cfg["paths"]["validated"])
    write_jsonl(destination, kept)
    write_jsonl(destination.with_name("rejected.jsonl"), rejected)
    write_json(destination.with_name("validation_manifest.json"), {
        "input_sha256": file_hash(cfg["paths"]["reviewed"]), "accepted_sha256": file_hash(destination),
        "accepted": len(kept), "rejected": len(rejected),
        "policy": "Human semantic checklist plus parser, read-only execution, contrast witness and leakage checks"})


def build_conditions(cfg: dict) -> None:
    verify_prepared(cfg)
    require_gate(cfg["gates"]["novelty_review"], "Novelty review")
    require_gate(cfg["gates"]["baseline_review"], "Baseline feasibility review")
    require_gate(cfg["gates"]["augmentation_pilot_review"], "Augmentation pilot review")
    root = Path(cfg["paths"]["prepared"])
    originals = read_jsonl(root / "train.jsonl")
    rows = read_jsonl(cfg["paths"]["validated"])
    for row in rows:
        verify_record_fingerprint(row)
    # Revalidation also binds accepted records to current frozen sources.
    valid, invalid = validate_records(rows, cfg, require_unique(originals))
    if invalid:
        raise ValueError(f"Previously accepted rows failed revalidation: {[r['id'] for r in invalid][:10]}")
    forbidden = read_jsonl(root / "validation.jsonl") + read_jsonl(root / "test.jsonl")
    if next(near_pairs(valid, forbidden, cfg["data"]["near_duplicate_threshold"]), None) is not None:
        raise ValueError("Accepted augmentation contains a held-out near-duplicate; rerun validation")
    budget = cfg["data"]["augmentation_budget"]
    if not isinstance(budget, int) or budget < 1:
        raise ValueError("augmentation_budget must be a positive integer")
    pools = {kind: shuffled([r for r in valid if r["transformation_type"] == kind], cfg["data"]["split_seed"])
             for kind in ("paraphrase", "counterfactual")}
    if any(len(pool) < budget for pool in pools.values()):
        raise ValueError(f"Need {budget} accepted examples of each type; available: { {k: len(v) for k,v in pools.items()} }")
    para, counter = pools["paraphrase"][:budget], pools["counterfactual"][:budget]
    variants = {"A0": originals, "A1": originals + para, "A2": originals + counter,
                "A3": originals + para + counter,
                "A3_matched": originals + para[:budget // 2] + counter[:budget - budget // 2]}
    dest = Path(cfg["paths"]["conditions"])
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Conditions already frozen: {dest}")
    for name, items in variants.items():
        require_unique(items)
        write_jsonl(dest / f"{name}.jsonl", items)
    write_json(dest / "manifest.json", {"budget": budget, "seed": cfg["data"]["split_seed"],
        "files": {name: {"count": len(items), "sha256": file_hash(dest / f"{name}.jsonl")} for name, items in variants.items()},
        "prepared_manifest_sha256": file_hash(root / "manifest.json"),
        "accepted_sha256": file_hash(cfg["paths"]["validated"]),
        "fairness": "Same optimizer steps/batch/lengths. Actual nonpadding token exposure is recorded during training."})
