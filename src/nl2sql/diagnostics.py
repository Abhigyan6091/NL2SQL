"""Independent controlled diagnostic families with explicit NULL and boundary rows."""
from __future__ import annotations

import copy
import sqlite3
from pathlib import Path

from .augment import REVIEW_TEMPLATE, validate_records
from .common import digest, file_hash, read_json, read_jsonl, validate_families, write_json, write_jsonl
from .schema import SCHEMA_VERSION, serialize

DDL = """
CREATE TABLE customers (customer_id INTEGER PRIMARY KEY, name TEXT NOT NULL);
CREATE TABLE orders (
  order_id INTEGER PRIMARY KEY,
  customer_id INTEGER NOT NULL REFERENCES customers(customer_id),
  total REAL,
  status TEXT
);
CREATE TABLE products (product_id INTEGER PRIMARY KEY, price REAL);
"""


def diagnostic_schema() -> dict:
    return {"db_id": "controlled_shop_v1", "table_names_original": ["customers", "orders", "products"],
            "column_names_original": [[-1, "*"], [0, "customer_id"], [0, "name"], [1, "order_id"],
                                      [1, "customer_id"], [1, "total"], [1, "status"],
                                      [2, "product_id"], [2, "price"]],
            "column_types": ["text", "number", "text", "number", "number", "number", "text", "number", "number"],
            "primary_keys": [1, 3, 7], "foreign_keys": [[4, 1]]}


def _fixture(path: Path, thresholds: list[int], counts: list[int], variant: int) -> None:
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(DDL)
        # Every numeric boundary has below/equal/above cases and an explicit NULL.
        prices = sorted({float(t + delta) for t in thresholds for delta in (-1, 0, 1)})
        connection.executemany("INSERT INTO products VALUES (?, ?)", [(i + 1, p) for i, p in enumerate(prices)] + [(len(prices) + 1, None)])
        customer_counts = sorted({0, 1, *[n + delta for n in counts for delta in (-1, 0, 1)]})
        order_rows, customer_rows = [], []
        order_id = 1
        for customer_id, count in enumerate(customer_counts, 1):
            customer_rows.append((customer_id, f"Customer {customer_id}"))
            for index in range(count):
                order_rows.append((order_id, customer_id, float(thresholds[(index + variant) % len(thresholds)]), "paid"))
                order_id += 1
        # Truth table: both, left only, right only, neither, and NULL permutations.
        extra_id = len(customer_rows) + 1
        customer_rows.append((extra_id, "Truth table customer"))
        for threshold in thresholds:
            for status in ("paid", "pending", None):
                for total in (float(threshold - 1), float(threshold), float(threshold + 1), None):
                    order_rows.append((order_id, extra_id, total, status))
                    order_id += 1
        # At each count boundary add customers whose orders have NULL totals and
        # no paid status: COUNT(total) and status-filtered counts must be wrong.
        for count in counts:
            extra_id += 1
            customer_rows.append((extra_id, f"NULL total customer {count}"))
            for index in range(count):
                order_rows.append((order_id, extra_id, None, "pending" if index % 2 else None))
                order_id += 1
        # Renumber each instance consistently so fixed-ID predictions cannot pass
        # an alleged multi-instance suite with repeated copies of the same rows.
        offset = variant * 1000000
        if offset:
            connection.execute("UPDATE products SET product_id = product_id + ?", (offset,))
            customer_rows = [(cid + offset, name) for cid, name in customer_rows]
            order_rows = [(oid + offset, cid + offset, total, status) for oid, cid, total, status in order_rows]
        if variant % 2:
            order_rows.reverse()
        connection.executemany("INSERT INTO customers VALUES (?, ?)", customer_rows)
        connection.executemany("INSERT INTO orders VALUES (?, ?, ?, ?)", order_rows)
        connection.commit()
    finally:
        connection.close()


def build_diagnostics(cfg: dict) -> None:
    dest = Path(cfg["paths"]["diagnostics"])
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"Diagnostic directory already contains files: {dest}")
    n = cfg["data"]["diagnostic_families_per_phenomenon"]
    fixture_count = cfg["data"]["diagnostic_fixture_count"]
    if n < 1 or fixture_count < 2:
        raise ValueError("Need at least one family per phenomenon and two fixture instances")
    thresholds = [10 + 5 * i for i in range(n)]
    counts = [2 + i for i in range(n)]
    databases = []
    for variant in range(fixture_count):
        path = dest / "database" / f"fixture_{variant}.sqlite"
        _fixture(path, thresholds, counts, variant)
        databases.append(path.as_posix())
    schema = diagnostic_schema()
    schema_text = serialize(schema)
    records = []
    for phenomenon in ("comparison", "and_or", "negation", "quantifier"):
        for i in range(n):
            value, count = thresholds[i], counts[i]
            if phenomenon == "comparison":
                questions = [f"List the IDs of products with a known price greater than {value}.",
                             f"Which products have a non-NULL price exceeding {value}? Return their IDs.",
                             f"List the IDs of products with a known price at most {value}."]
                gold = f"SELECT product_id FROM products WHERE price > {value}"
                contrast = f"SELECT product_id FROM products WHERE price <= {value}"
            elif phenomenon == "and_or":
                questions = [f"List order IDs where status is paid and total is greater than {value}.",
                             f"Return the IDs of orders satisfying both conditions: paid status and total above {value}.",
                             f"List order IDs where status is paid or total is greater than {value}, including orders satisfying both."]
                gold = f"SELECT order_id FROM orders WHERE status = 'paid' AND total > {value}"
                contrast = f"SELECT order_id FROM orders WHERE status = 'paid' OR total > {value}"
            elif phenomenon == "negation":
                questions = [f"List customer IDs having at least one order with a known total above {value}.",
                             f"Which customers placed an order whose non-NULL total exceeds {value}? Return customer IDs.",
                             f"List customer IDs having no order with a known total above {value}, including customers with no orders."]
                base = f"SELECT 1 FROM orders AS o WHERE o.customer_id = c.customer_id AND o.total > {value}"
                gold = f"SELECT c.customer_id FROM customers AS c WHERE EXISTS ({base})"
                contrast = f"SELECT c.customer_id FROM customers AS c WHERE NOT EXISTS ({base})"
            else:
                questions = [f"List customer IDs with at least {count} orders, counting all orders regardless of status or total.",
                             f"Return IDs of customers whose number of orders is {count} or more; count every order.",
                             f"List customer IDs with fewer than {count} orders, including customers with zero orders; count every order."]
                base = "SELECT COUNT(*) FROM orders AS o WHERE o.customer_id = c.customer_id"
                gold = f"SELECT c.customer_id FROM customers AS c WHERE ({base}) >= {count}"
                contrast = f"SELECT c.customer_id FROM customers AS c WHERE ({base}) < {count}"
            family = f"controlled-{phenomenon}-{i:04d}"
            for kind, question, sql in zip(("original", "paraphrase", "counterfactual"), questions, (gold, gold, contrast)):
                if phenomenon == "comparison":
                    alternatives = [f"SELECT product_id FROM products WHERE price {operator} {value}"
                                    for operator in (">", ">=", "<", "<=")]
                    alternatives.append("SELECT product_id FROM products")
                elif phenomenon == "and_or":
                    alternatives = [gold, contrast, f"SELECT order_id FROM orders WHERE total > {value}",
                                    "SELECT order_id FROM orders WHERE status = 'paid'"]
                elif phenomenon == "negation":
                    any_order = "SELECT 1 FROM orders AS o WHERE o.customer_id = c.customer_id"
                    alternatives = [gold, contrast, f"SELECT c.customer_id FROM customers AS c WHERE {'NOT ' if kind == 'counterfactual' else ''}EXISTS ({any_order})"]
                else:
                    comparator = "<" if kind == "counterfactual" else ">="
                    strict_comparator = "<=" if kind == "counterfactual" else ">"
                    alternatives = [gold, contrast,
                        f"SELECT c.customer_id FROM customers AS c WHERE ({base}) {strict_comparator} {count}",
                        f"SELECT c.customer_id FROM customers AS c WHERE (SELECT COUNT(o.total) FROM orders AS o WHERE o.customer_id = c.customer_id) {comparator} {count}",
                        f"SELECT c.customer_id FROM customers AS c WHERE (SELECT COUNT(*) FROM orders AS o WHERE o.customer_id = c.customer_id AND o.status = 'paid') {comparator} {count}"]
                alternatives = list(dict.fromkeys(s for s in alternatives if s != sql))
                records.append({"id": f"{family}-{kind}", "source_id": family + "-original", "family_id": family,
                                "db_id": schema["db_id"], "split": "test", "question": question, "gold_sql": sql,
                                "original_question": questions[0], "original_sql": gold,
                                "schema": schema_text, "schema_serialization_version": SCHEMA_VERSION,
                                "database_paths": databases, "phenomenon": phenomenon, "transformation_type": kind,
                                "semantic_review_status": "pending", "review": copy.deepcopy(REVIEW_TEMPLATE),
                                "plausible_incorrect_sql": alternatives,
                                "notes": "SQL predicates use SQLite three-valued logic; NULL fails numeric comparison. EXISTS/COUNT include zero-order customers as stated.",
                                "provenance": {"generator": "controlled-shop-templates-v2", "index": i,
                                               "threshold": value, "count": count}})
    write_json(dest / "tables.json", [schema])
    write_jsonl(dest / "candidates.jsonl", records)
    write_json(dest / "construction.json", {"families": n * 4, "records": len(records),
        "schema_sha256": digest(schema), "database_files": {p: file_hash(p) for p in databases},
        "limitations": "Threshold variants share templates and schema. Family bootstrap understates template-level uncertainty. Controlled evidence only; no broad transfer claim."})


def freeze_diagnostics(cfg: dict, reviewed_path: str) -> None:
    dest = Path(cfg["paths"]["diagnostics"])
    if (dest / "test.jsonl").exists():
        raise FileExistsError("Diagnostic test set is already frozen")
    rows = read_jsonl(reviewed_path)
    construction = read_json(dest / "construction.json")
    for path, expected in construction["database_files"].items():
        if file_hash(path) != expected:
            raise ValueError(f"Diagnostic construction fixture modified: {path}")
    accepted, rejected = validate_records(rows, cfg)
    write_jsonl(dest / "rejected.jsonl", rejected)
    if rejected:
        raise ValueError(f"{len(rejected)} diagnostic records failed; inspect rejected.jsonl, revise, and review")
    families = validate_families(accepted)
    for row in accepted:
        if row["split"] != "test" or row["db_id"] != diagnostic_schema()["db_id"]:
            raise ValueError("Controlled diagnostic records must remain independent test examples")
        if row["schema"] != serialize(diagnostic_schema()) or row["database_paths"] != list(construction["database_files"]):
            raise ValueError("Diagnostic schema/fixture paths must match the construction manifest")
        if not row.get("plausible_incorrect_sql"):
            raise ValueError("Controlled diagnostics require plausible incorrect alternatives and witnesses")
    if not families:
        raise ValueError("Empty diagnostic set")
    for family, members in families.items():
        if sorted(r["transformation_type"] for r in members) != ["counterfactual", "original", "paraphrase"]:
            raise ValueError(f"Expected original, paraphrase, counterfactual in {family}")
        original = next(r for r in members if r["transformation_type"] == "original")
        for member in members:
            if member["source_id"] != original["id"] or member["original_sql"] != original["gold_sql"] or member["original_question"] != original["question"]:
                raise ValueError(f"Mismatched family origin in {family}")
            for field in ("schema", "db_id", "database_paths", "phenomenon"):
                if member[field] != original[field]:
                    raise ValueError(f"Mismatched {field} in {family}")
    write_jsonl(dest / "test.jsonl", accepted)
    files = {db for row in accepted for db in row["database_paths"]}
    write_json(dest / "manifest.json", {"version": "controlled-v2", "families": len(families),
        "test_sha256": file_hash(dest / "test.jsonl"), "reviewed_source_sha256": file_hash(reviewed_path),
        "databases": {p: file_hash(p) for p in sorted(files)},
        "policy": "Every record human-reviewed, parsed, executed; each counterfactual has a distinguishing fixture"})
