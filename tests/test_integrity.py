import copy
import sqlite3

from nl2sql.augment import validate_records
import pytest

from nl2sql.common import fingerprint_tree, normalized_question, reviewed, validate_families, write_json
from nl2sql.data import _load_spider, audit_splits, near_pairs
from nl2sql.schema import serialize


def test_split_audit_detects_family_and_near_question_leakage():
    base = {"id": "a", "source_id": "source", "family_id": "family", "db_id": "db_a", "question": "List all paid orders"}
    sibling = {**base, "id": "b", "db_id": "db_b"}
    audit = audit_splits({"train": [base], "test": [sibling]})
    fields = {v["field"] for v in audit["violations"]}
    assert {"source_id", "family_id", "near_question"} <= fields


def test_review_requires_each_semantic_check():
    row = {"semantic_review_status": "reviewed", "review": {"decision": "accept", "reviewer": "tester", "reviewed_at": "2026-11-10", "notes": "Checked"}}
    assert not reviewed(row)
    for check in ("natural", "unambiguous", "sql_matches_question", "constraints_checked", "null_semantics_checked"):
        row["review"][check] = True
    assert reviewed(row)


def test_serializer_preserves_original_identifiers_and_fk():
    schema = {"table_names_original": ["customer", "order"],
              "column_names_original": [[-1, "*"], [0, "id"], [1, "customer_id"]],
              "column_types": ["text", "number", "number"], "primary_keys": [1], "foreign_keys": [[2, 1]]}
    result = serialize(schema)
    assert '"customer"("id" number PRIMARY KEY)' in result
    assert '"order"."customer_id" = "customer"."id"' in result
    assert result == serialize(copy.deepcopy(schema))


def test_valid_sql_does_not_replace_semantic_review(tmp_path):
    db = tmp_path / "db.sqlite"
    connection = sqlite3.connect(db)
    connection.execute("CREATE TABLE items(id INTEGER)")
    connection.close()
    row = {"id": "a", "question": "Show IDs", "gold_sql": "SELECT id FROM items", "database_paths": [str(db)],
           "transformation_type": "original", "schema_serialization_version": "v1", "semantic_review_status": "pending"}
    accepted, rejected = validate_records([row], {"data": {"query_timeout_seconds": 1, "query_max_rows": 100}})
    assert not accepted and len(rejected) == 1


def test_counterfactual_requires_observable_difference(tmp_path):
    db = tmp_path / "db.sqlite"
    connection = sqlite3.connect(db)
    connection.executescript("CREATE TABLE items(id INTEGER, value INTEGER); INSERT INTO items VALUES (1, 20);")
    connection.close()
    checks = {k: True for k in ("natural", "unambiguous", "sql_matches_question", "constraints_checked", "null_semantics_checked")}
    row = {"id": "a", "question": "IDs above ten", "gold_sql": "SELECT id FROM items WHERE value > 10",
           "original_sql": "SELECT id FROM items WHERE value >= 10", "database_paths": [str(db)],
           "transformation_type": "counterfactual", "schema_serialization_version": "v1", "semantic_review_status": "reviewed",
           "review": {**checks, "decision": "accept", "reviewer": "tester", "reviewed_at": "2026-11-10", "notes": "Boundary change"}}
    accepted, rejected = validate_records([row], {"data": {"query_timeout_seconds": 1, "query_max_rows": 100}})
    assert not accepted
    assert "not distinguished" in rejected[0]["rejection_reasons"][0]


def test_signed_and_decimal_values_are_not_exact_duplicates():
    assert normalized_question("above -1.5") != normalized_question("above 1.5")
    assert normalized_question("above 1.5") != normalized_question("above 15")


def test_shared_id_in_different_objects_does_not_hide_question_leakage():
    left = {"id": "same", "question": "List paid orders"}
    right = dict(left)
    assert list(near_pairs([left], [right], .92))
    assert not list(near_pairs([left], [left], .92))


def test_source_fingerprint_excludes_bytecode_cache(tmp_path):
    (tmp_path / "module.py").write_text("value = 1\n", encoding="utf-8")
    before = fingerprint_tree(tmp_path, suffixes=(".py",))["sha256"]
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "module.pyc").write_bytes(b"cached bytecode")
    assert fingerprint_tree(tmp_path, suffixes=(".py",))["sha256"] == before


def test_family_source_id_must_point_to_original():
    base = {"family_id": "f", "db_id": "db", "question": "List IDs", "gold_sql": "SELECT 1",
            "phenomenon": "p", "database_paths": ["fixture.sqlite"]}
    with pytest.raises(ValueError, match="source_id"):
        validate_families([{**base, "id": "o", "source_id": "o", "transformation_type": "original"},
                           {**base, "id": "p", "source_id": "wrong", "transformation_type": "paraphrase"}])


def test_official_dev_duplicates_keep_the_published_denominator(tmp_path):
    db_dir = tmp_path / "database" / "db"
    db_dir.mkdir(parents=True)
    (db_dir / "db.sqlite").touch()
    schema = {"db_id": "db", "table_names_original": ["items"], "column_names_original": [[-1, "*"], [0, "id"]],
              "column_types": ["text", "number"], "primary_keys": [], "foreign_keys": []}
    path = tmp_path / "dev.json"
    entry = {"db_id": "db", "question": "Show IDs", "query": "SELECT id FROM items"}
    write_json(path, [entry, dict(entry)])
    rows = _load_spider([str(path)], {"db": schema}, str(tmp_path / "database"), "dev")
    assert len(rows) == 2
    assert len({row["id"] for row in rows}) == 2
