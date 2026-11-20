import sqlite3

import pytest

from nl2sql.common import file_hash, read_json, read_jsonl, write_json, write_jsonl
from nl2sql.evaluate import evaluate, single_line_sql


def test_consistently_wrong_paraphrase_is_not_correct(tmp_path):
    db = tmp_path / "db.sqlite"
    conn = sqlite3.connect(db)
    conn.executescript("CREATE TABLE items(id INTEGER, price INTEGER); INSERT INTO items VALUES (1,9), (2,11);")
    conn.close()
    base = {"source_id": "o", "family_id": "family", "db_id": "db", "phenomenon": "comparison", "database_paths": [str(db)]}
    records = [dict(base, id="o", question="Above ten", transformation_type="original", gold_sql="SELECT id FROM items WHERE price > 10"),
               dict(base, id="p", question="Exceeds ten", transformation_type="paraphrase", gold_sql="SELECT id FROM items WHERE price > 10"),
               dict(base, id="c", question="At most ten", transformation_type="counterfactual", gold_sql="SELECT id FROM items WHERE price <= 10")]
    dataset = str(tmp_path / "test.jsonl")
    predictions = str(tmp_path / "pred.jsonl")
    write_jsonl(dataset, records)
    write_jsonl(predictions, [dict(id=r["id"], predicted_sql="SELECT id FROM items WHERE price <= 10", condition="A0", seed=13) for r in records])
    cfg = {"data": {"query_timeout_seconds": 2, "query_max_rows": 100}}
    write_json(predictions + ".manifest.json", {"dataset_sha256": file_hash(dataset), "prediction_sha256": file_hash(predictions),
        "databases": {str(db): file_hash(db)}, "condition": "A0", "seed": 13, "config": cfg})
    evaluate(cfg, dataset, predictions, str(tmp_path / "results"))
    metrics = read_json(tmp_path / "results" / "metrics.json")["metrics"]
    assert metrics["paraphrase_pair_consistency"]["value"] == 1
    assert metrics["paraphrase_family_correctness"]["value"] == 0
    assert metrics["counterfactual_contrast_accuracy"]["value"] == 0


def test_later_instance_failure_does_not_erase_primary_accuracy(tmp_path):
    databases = []
    for index in range(2):
        path = tmp_path / f"db_{index}.sqlite"
        conn = sqlite3.connect(path)
        conn.executescript("CREATE TABLE items(id INTEGER); INSERT INTO items VALUES (1);")
        if index == 0:
            conn.executescript("CREATE TABLE primary_only(id INTEGER); INSERT INTO primary_only VALUES (1);")
        conn.close()
        databases.append(str(path))
    row = {"id": "o", "source_id": "o", "family_id": "f", "db_id": "db", "phenomenon": "comparison",
           "database_paths": databases, "question": "List IDs", "gold_sql": "SELECT id FROM items", "transformation_type": "original"}
    dataset, predictions = str(tmp_path / "test.jsonl"), str(tmp_path / "pred.jsonl")
    write_jsonl(dataset, [row])
    write_jsonl(predictions, [{"id": "o", "predicted_sql": "SELECT id FROM primary_only", "condition": "A0", "seed": 13}])
    cfg = {"data": {"query_timeout_seconds": 2, "query_max_rows": 100}}
    write_json(predictions + ".manifest.json", {"dataset_sha256": file_hash(dataset), "prediction_sha256": file_hash(predictions),
        "databases": {p: file_hash(p) for p in databases}, "condition": "A0", "seed": 13, "config": cfg})
    evaluate(cfg, dataset, predictions, str(tmp_path / "results"))
    item = read_jsonl(tmp_path / "results" / "items.jsonl")[0]
    assert item["execution_correct"] is True
    assert item["sql_valid"] is True
    assert item["suite_correct"] is False
    assert item["instance_validity"] == [True, False]


def test_official_export_preserves_literal_whitespace_and_comment_boundaries():
    sql = "SELECT 'a  b', 'it''s -- a value' -- comment\nFROM items"
    folded = single_line_sql(sql)
    assert "'a  b'" in folded
    assert "'it''s -- a value'" in folded
    assert "FROM items" in folded
    assert "comment" not in folded


def test_official_export_rejects_unrepresentable_literal_newline():
    with pytest.raises(ValueError, match="quoted SQL"):
        single_line_sql("SELECT 'a\nb'")
