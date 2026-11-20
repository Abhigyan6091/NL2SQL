"""Per-item execution archives, family metrics, and local official evaluator bridge."""
from __future__ import annotations

import subprocess
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from .common import (digest, file_hash, fingerprint_tree,
                     write_json, write_jsonl)
from .provenance import verify_predictions
from .sql import canonical, equal_results, execute, ordered, parse_query


def rate(values: list[bool]) -> dict:
    return {"value": sum(values) / len(values) if values else None,
            "numerator": sum(values), "denominator": len(values)}


def _error_hint(gold: str, prediction: str, valid: bool, correct: bool) -> str:
    if not valid:
        return "invalid_sql"
    if correct:
        return "correct_on_supplied_instances"
    from sqlglot import exp
    a, b = parse_query(gold), parse_query(prediction)
    if {t.name.lower() for t in a.find_all(exp.Table)} != {t.name.lower() for t in b.find_all(exp.Table)}:
        return "schema_linking_error_candidate"
    operators = (exp.And, exp.Or, exp.Not, exp.GT, exp.GTE, exp.LT, exp.LTE, exp.EQ, exp.NEQ, exp.Exists)
    counts = lambda tree: {cls.__name__: len(list(tree.find_all(cls))) for cls in operators}
    if counts(a) != counts(b):
        return "operator_or_dropped_constraint_candidate"
    return "requires_manual_analysis"


def evaluate(cfg: dict, dataset: str, predictions: str, output: str) -> None:
    dest = Path(output)
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Evaluation output already exists: {dest}")
    rows, pred_by_id, manifest = verify_predictions(dataset, predictions, cfg)
    timeout, limit = cfg["data"]["query_timeout_seconds"], cfg["data"]["query_max_rows"]
    archived = []
    @lru_cache(maxsize=32)
    def run(sql, database):
        return execute(sql, database, timeout, limit)
    for row in rows:
        prediction = pred_by_id[row["id"]]
        sql, databases = prediction["predicted_sql"], row["database_paths"]
        if not databases:
            raise ValueError(f"No databases for {row['id']}")
        # Invalid gold is an experiment failure, never counted as a model error.
        try:
            gold_results = [run(row["gold_sql"], db) for db in databases]
        except Exception as exception:
            raise ValueError(f"Gold SQL failed for {row['id']}: {exception}") from exception
        respect_order = ordered(row["gold_sql"])
        parse_valid, syntactic = False, False
        parse_error = None
        try:
            syntactic = canonical(sql) == canonical(row["gold_sql"])
            parse_valid = True
        except Exception as exception:
            parse_error = f"{type(exception).__name__}: {exception}"
        correctness, validity, errors = [], [], []
        for db, gold_result in zip(databases, gold_results):
            valid, correct, error = False, False, parse_error
            if parse_valid:
                try:
                    result = run(sql, db)
                    valid = True
                    correct = equal_results(result, gold_result, respect_order)
                except Exception as exception:
                    error = f"{type(exception).__name__}: {exception}"
            validity.append(valid)
            correctness.append(correct)
            errors.append(error)
        archived.append({**row, **prediction, "sql_parse_valid": parse_valid,
                         "sql_valid": validity[0], "suite_sql_valid": all(validity), "instance_validity": validity,
                         "execution_correct": correctness[0],
                         "suite_correct": all(correctness), "instance_correctness": correctness,
                         "syntactic_match": syntactic, "execution_error": next((e for e in errors if e is not None), None),
                         "instance_errors": errors,
                         "error_category_suggestion": _error_hint(row["gold_sql"], sql, all(validity), all(correctness)),
                         "error_category_reviewed": None,
                         "semantic_evaluator": "strict-sqlite-bag-v2",
                         "semantic_evaluator_result": all(correctness)})
    families = defaultdict(list)
    for row in archived:
        families[row["family_id"]].append(row)
    pairs = []
    for family, members in families.items():
        originals = [r for r in members if r["transformation_type"] == "original"]
        if len(originals) != 1:
            raise ValueError(f"Expected one original in {family}")
        original = originals[0]
        paraphrases = [r for r in members if r["transformation_type"] == "paraphrase"]
        counterfactuals = [r for r in members if r["transformation_type"] == "counterfactual"]
        if not paraphrases and not counterfactuals:
            continue
        family_row = {"family_id": family, "phenomenon": original["phenomenon"],
                      "paraphrase_family_correct": all(r["suite_correct"] for r in [original, *paraphrases]) if paraphrases else None,
                      "paraphrase_consistency": [], "counterfactual_contrast": [], "gold_contrast_distinguished": [],
                      "pair_details": []}
        for sibling in [*paraphrases, *counterfactuals]:
            if original["database_paths"] != sibling["database_paths"]:
                raise ValueError(f"Family fixture mismatch in {family}")
            databases = original["database_paths"]
            # Paired equivalence respects order if either question requests ordered output.
            respect = ordered(original["gold_sql"]) or ordered(sibling["gold_sql"])
            gold_same = all(equal_results(run(original["gold_sql"], db), run(sibling["gold_sql"], db), respect) for db in databases)
            predictions_same = None
            if original["suite_sql_valid"] and sibling["suite_sql_valid"]:
                predictions_same = all(equal_results(run(original["predicted_sql"], db), run(sibling["predicted_sql"], db), respect) for db in databases)
            if sibling["transformation_type"] == "paraphrase":
                if not gold_same:
                    raise ValueError(f"Paraphrase gold mismatch in {family}")
                family_row["paraphrase_consistency"].append(predictions_same is True)
                family_row["pair_details"].append({"id": sibling["id"], "phenomenon": sibling["phenomenon"],
                    "type": "paraphrase", "consistent": predictions_same is True,
                    "correct": original["suite_correct"] and sibling["suite_correct"]})
            else:
                if gold_same:
                    raise ValueError(f"Counterfactual lacks a distinguishing fixture in {family}")
                family_row["gold_contrast_distinguished"].append(True)
                family_row["counterfactual_contrast"].append(original["suite_correct"] and sibling["suite_correct"] and predictions_same is False)
                family_row["pair_details"].append({"id": sibling["id"], "phenomenon": sibling["phenomenon"],
                    "type": "counterfactual", "correct": family_row["counterfactual_contrast"][-1]})
        pairs.append(family_row)
    def summarize(items):
        return {"execution_accuracy": rate([r["execution_correct"] for r in items]),
                "strict_suite_accuracy": rate([r["suite_correct"] for r in items]),
                "sql_validity": rate([r["sql_valid"] for r in items]),
                "suite_sql_validity": rate([r["suite_sql_valid"] for r in items]),
                "syntactic_match_diagnostic": rate([r["syntactic_match"] for r in items])}
    pair_metrics = {
        "paraphrase_family_correctness": rate([p["paraphrase_family_correct"] for p in pairs if p["paraphrase_family_correct"] is not None]),
        "paraphrase_pair_consistency": rate([v for p in pairs for v in p["paraphrase_consistency"]]),
        "counterfactual_contrast_accuracy": rate([v for p in pairs for v in p["counterfactual_contrast"]])}
    per_phenomenon = {}
    for phenomenon in sorted({r["phenomenon"] for r in archived}):
        selected = [r for r in archived if r["phenomenon"] == phenomenon]
        details = [d for p in pairs for d in p["pair_details"] if d["phenomenon"] == phenomenon]
        paraphrase_families = []
        for pair in pairs:
            category_paraphrases = [d for d in pair["pair_details"] if d["phenomenon"] == phenomenon and d["type"] == "paraphrase"]
            if category_paraphrases:
                paraphrase_families.append(all(d["correct"] for d in category_paraphrases))
        per_phenomenon[phenomenon] = {**summarize(selected),
            "paraphrase_family_correctness": rate(paraphrase_families),
            "paraphrase_pair_consistency": rate([d["consistent"] for d in details if d["type"] == "paraphrase"]),
            "counterfactual_contrast_accuracy": rate([d["correct"] for d in details if d["type"] == "counterfactual"])}
    write_jsonl(dest / "items.jsonl", archived)
    write_jsonl(dest / "pairs.jsonl", pairs)
    write_json(dest / "metrics.json", {"condition": manifest["condition"], "seed": manifest["seed"],
        "dataset_sha256": file_hash(dataset), "predictions_sha256": file_hash(predictions),
        "config_sha256": digest(cfg),
        "metrics": {**summarize(archived), **pair_metrics}, "per_phenomenon": per_phenomenon,
        "policy": {"evaluator": "strict-sqlite-bag-v2", "column_order": "strict", "duplicates": "multiset",
                   "row_order": "respected when gold has top-level ORDER BY", "float_comparison": "exact",
                   "suite_instances": sorted({len(r['database_paths']) for r in rows}),
                   "official_spider_exact_match": "not computed by this evaluator",
                   "official_test_suite_accuracy": "use the official-evaluate command separately"}})
    write_jsonl(dest / "error_review.jsonl", [dict(id=r["id"], question=r["question"], gold_sql=r["gold_sql"],
        predicted_sql=r["predicted_sql"], suggested=r["error_category_suggestion"], category="", reviewer="", notes="")
        for r in archived if not r["suite_correct"]])
    # One complete family per phenomenon, selected without looking at correctness.
    demo_families = {}
    for row in archived:
        demo_families.setdefault(row["phenomenon"], row["family_id"])
    write_jsonl(dest / "demonstration.jsonl", [r for r in archived if r["family_id"] in set(demo_families.values())])
    write_json(dest / "completion.json", {"files": {name: file_hash(dest / name)
               for name in ("metrics.json", "items.jsonl", "pairs.jsonl", "demonstration.jsonl")}})


def single_line_sql(sql: str) -> str:
    """Fold external whitespace/comments without changing quoted SQL values.

    A literal newline inside a quoted value cannot be represented by the
    official line-based protocol. Reject it rather than silently relabel it.
    Invalid predictions otherwise remain invalid predictions for official scoring.
    """
    output, index, quote, block_comment = [], 0, None, False
    while index < len(sql):
        char = sql[index]
        following = sql[index:index + 2]
        if block_comment:
            if following == "*/":
                block_comment = False
                output.append(" ")
                index += 2
            else:
                index += 1
            continue
        if quote is not None:
            if char in "\r\n\t":
                raise ValueError("Official line-based evaluator cannot preserve a newline/tab in a quoted SQL value")
            output.append(char)
            if char == quote:
                if index + 1 < len(sql) and sql[index + 1] == quote and quote != "]":
                    output.append(sql[index + 1])
                    index += 1
                else:
                    quote = None
            index += 1
            continue
        if following == "--":
            index += 2
            while index < len(sql) and sql[index] not in "\r\n":
                index += 1
            output.append(" ")
            continue
        if following == "/*":
            block_comment = True
            index += 2
            continue
        if char in "'\"`[":
            quote = "]" if char == "[" else char
        output.append(" " if char.isspace() else char)
        index += 1
    if block_comment:
        # Preserve an unterminated comment rather than silently removing it.
        output.append(" /*")
    return "".join(output).strip()


def official_evaluate(dataset: str, predictions: str, script: str, database_root: str,
                      tables: str, output: str, evaluator_kind: str, cfg: dict | None = None) -> None:
    rows, by_id, prediction_manifest = verify_predictions(dataset, predictions, cfg)
    if evaluator_kind not in {"spider", "test-suite"}:
        raise ValueError("Choose spider or test-suite")
    source = Path(script).resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    if not Path(database_root).is_dir() or not Path(tables).is_file():
        raise FileNotFoundError("Official evaluation requires a local database root and tables file")
    database_instances = fingerprint_tree(database_root, suffixes=(".sqlite", ".sqlite3", ".db"))
    dest = Path(output).resolve()
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(dest)
    gold, pred = dest / "gold.tsv", dest / "predicted.sql"
    gold_lines, pred_lines, empty_ids = [], [], []
    for row in rows:
        if any(char in row["db_id"] for char in "\r\n\t"):
            raise ValueError("Database IDs cannot contain TSV delimiters")
        gold_lines.append(single_line_sql(row["gold_sql"]) + "\t" + row["db_id"] + "\n")
        prediction = single_line_sql(by_id[row["id"]]["predicted_sql"])
        if not prediction:
            # Some official versions discard empty lines and shift all later IDs.
            prediction = "INVALID_EMPTY_PREDICTION"
            empty_ids.append(row["id"])
        pred_lines.append(prediction + "\n")
    dest.mkdir(parents=True, exist_ok=True)
    gold.write_text("".join(gold_lines), encoding="utf-8")
    pred.write_text("".join(pred_lines), encoding="utf-8")
    write_jsonl(dest / "item_order.jsonl", [{"index": i, "id": row["id"], "db_id": row["db_id"]} for i, row in enumerate(rows)])
    # Both official repositories expose these common CLI flags. Their local code
    # and SQLite instance layout must be supplied by the researcher, never fetched.
    command = [sys.executable, str(source), "--gold", str(gold), "--pred", str(pred),
               "--db", str(Path(database_root).resolve()), "--table", str(Path(tables).resolve()), "--etype", "all"]
    record = {"kind": evaluator_kind, "command": command, "script_sha256": file_hash(source),
              "dataset_sha256": file_hash(dataset), "predictions_sha256": file_hash(predictions),
              "tables_sha256": file_hash(tables),
              "database_instances": database_instances,
              "empty_prediction_ids_exported_as_invalid": empty_ids,
              "condition": prediction_manifest["condition"], "seed": prediction_manifest["seed"],
              "python_sources": {p.relative_to(source.parent).as_posix(): file_hash(p) for p in sorted(source.parent.rglob("*.py"))},
              "note": "Official evaluator stdout is authoritative; strict local metrics are kept separate."}
    try:
        completed = subprocess.run(command, cwd=source.parent, capture_output=True, text=True, check=False)
        (dest / "stdout.txt").write_text(completed.stdout, encoding="utf-8")
        (dest / "stderr.txt").write_text(completed.stderr, encoding="utf-8")
        record["returncode"] = completed.returncode
    except OSError as error:
        record["error"] = str(error)
        raise
    finally:
        write_json(dest / "invocation.json", record)
    if completed.returncode:
        raise RuntimeError(f"Official evaluator failed; inspect {dest / 'stderr.txt'}")
