from nl2sql.augment import propose
from nl2sql.common import read_jsonl, write_jsonl
from nl2sql.sql import parse_query


def test_counterfactual_edits_selected_where_not_identical_projection(tmp_path, monkeypatch):
    prepared = tmp_path / "prepared"
    output = tmp_path / "candidates.jsonl"
    source = {"id": "o", "question": "Flags for values above ten",
              "gold_sql": "SELECT value > 10 AS flag FROM items WHERE value > 10"}
    write_jsonl(prepared / "train.jsonl", [source])
    monkeypatch.setattr("nl2sql.augment.verify_prepared", lambda cfg: None)
    propose({"paths": {"prepared": str(prepared), "candidates": str(output)}})
    counter = next(r for r in read_jsonl(output) if r["transformation_type"] == "counterfactual")
    before, after = parse_query(source["gold_sql"]), parse_query(counter["gold_sql"])
    assert before.expressions[0].sql() == after.expressions[0].sql()
    assert after.args["where"].this.sql() == "value <= 10"
