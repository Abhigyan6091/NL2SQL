"""Restricted read-only SQLite execution and explicit comparison policy."""
from __future__ import annotations

import sqlite3
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import sqlglot
from sqlglot import exp


def parse_query(sql: str) -> exp.Expression:
    if not isinstance(sql, str) or not sql.strip():
        raise ValueError("SQL must be a nonempty string")
    try:
        trees = sqlglot.parse(sql, read="sqlite")
    except sqlglot.errors.ParseError as error:
        raise ValueError(f"Invalid SQLite SQL: {error}") from error
    if len(trees) != 1 or trees[0] is None:
        raise ValueError("Exactly one SQL statement is required")
    tree = trees[0]
    if not isinstance(tree, exp.Query):
        raise ValueError("Only read-only queries are accepted")
    forbidden = (exp.Insert, exp.Update, exp.Delete, exp.Create, exp.Drop, exp.Command)
    if any(isinstance(node, forbidden) for node in tree.walk()):
        raise ValueError("Mutating SQL is forbidden")
    return tree


def canonical(sql: str) -> str:
    # A syntactic diagnostic only; this is NOT official Spider exact match.
    return parse_query(sql).sql(dialect="sqlite", normalize=True, pretty=False)


def abstract_sql(sql: str) -> str:
    tree = parse_query(sql).copy()
    for node in list(tree.find_all(exp.Literal)):
        node.replace(exp.Literal.string("VALUE") if node.is_string else exp.Literal.number("0"))
    return tree.sql(dialect="sqlite", normalize=True)


def ordered(sql: str) -> bool:
    return parse_query(sql).args.get("order") is not None


@dataclass
class Result:
    rows: list[tuple]
    columns: int


def execute(sql: str, database: str | Path, timeout: float = 5, max_rows: int = 100000) -> Result:
    parse_query(sql)
    db = Path(database).resolve()
    if not db.is_file():
        raise FileNotFoundError(f"Missing SQLite database: {db}")
    if timeout <= 0 or max_rows < 1:
        raise ValueError("Execution timeout and row limit must be positive")
    # mode=ro prevents accidental DB creation; immutable suppresses sidecar writes.
    uri = db.as_uri() + "?mode=ro&immutable=1"
    connection = sqlite3.connect(uri, uri=True, timeout=timeout)
    deadline = time.monotonic() + timeout
    try:
        connection.execute("PRAGMA query_only = ON")
        allowed = {sqlite3.SQLITE_SELECT, sqlite3.SQLITE_READ, sqlite3.SQLITE_FUNCTION, sqlite3.SQLITE_RECURSIVE}
        def authorizer(action, arg1, arg2, database_name, trigger_name):
            if action not in allowed:
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION and str(arg2).lower() in {"load_extension", "writefile", "readfile"}:
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK
        connection.set_authorizer(authorizer)
        connection.set_progress_handler(lambda: int(time.monotonic() > deadline), 1000)
        cursor = connection.execute(sql)
        rows = cursor.fetchmany(max_rows + 1)
        if len(rows) > max_rows:
            raise ValueError(f"Result exceeds {max_rows} rows; increase query_max_rows explicitly")
        if time.monotonic() > deadline:
            raise TimeoutError("SQL execution deadline exceeded")
        return Result(rows, len(cursor.description or []))
    finally:
        connection.close()


def equal_results(left: Result, right: Result, order_sensitive: bool = False) -> bool:
    if left.columns != right.columns:
        return False
    return left.rows == right.rows if order_sensitive else Counter(left.rows) == Counter(right.rows)


def equivalent(left: str, right: str, databases: list[str], timeout: float = 5,
               max_rows: int = 100000, order_sensitive: bool | None = None) -> bool:
    if not databases:
        raise ValueError("At least one database is required")
    respect_order = (ordered(left) or ordered(right)) if order_sensitive is None else order_sensitive
    return all(equal_results(execute(left, db, timeout, max_rows), execute(right, db, timeout, max_rows),
                             respect_order) for db in databases)
