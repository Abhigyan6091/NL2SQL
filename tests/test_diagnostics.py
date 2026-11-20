import copy

from nl2sql.common import read_json, read_jsonl, write_jsonl
from nl2sql.diagnostics import build_diagnostics, freeze_diagnostics


def test_diagnostic_families_pass_only_after_review(tmp_path):
    cfg = {"paths": {"diagnostics": str(tmp_path / "diagnostics")},
           "data": {"diagnostic_families_per_phenomenon": 2, "diagnostic_fixture_count": 2,
                    "query_timeout_seconds": 5, "query_max_rows": 10000}}
    build_diagnostics(cfg)
    root = tmp_path / "diagnostics"
    rows = read_jsonl(root / "candidates.jsonl")
    assert len(rows) == 24
    assert {r["phenomenon"] for r in rows} == {"comparison", "and_or", "negation", "quantifier"}
    checked = copy.deepcopy(rows)
    for row in checked:
        row["semantic_review_status"] = "reviewed"
        row["review"].update(decision="accept", reviewer="unit-test", reviewed_at="2026-11-10", notes="Synthetic review fixture")
        for key in ("natural", "unambiguous", "sql_matches_question", "constraints_checked", "null_semantics_checked"):
            row["review"][key] = True
    reviewed_path = str(root / "reviewed.jsonl")
    write_jsonl(reviewed_path, checked)
    freeze_diagnostics(cfg, reviewed_path)
    assert read_json(root / "manifest.json")["families"] == 8
    assert len(read_jsonl(root / "test.jsonl")) == 24
    assert all(r["alternative_witnesses"] for r in read_jsonl(root / "test.jsonl"))
    quantifier = next(r for r in read_jsonl(root / "test.jsonl")
                      if r["phenomenon"] == "quantifier" and r["transformation_type"] == "original")
    assert any("COUNT(o.total)" in witness["incorrect_sql"] and witness["distinguishing_databases"]
               for witness in quantifier["alternative_witnesses"])
