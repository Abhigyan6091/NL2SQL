import sqlite3

import pytest

from nl2sql.sql import Result, equal_results, equivalent, execute, parse_query


@pytest.fixture
def database(tmp_path):
    path = tmp_path / "fixture.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript("CREATE TABLE items(id INTEGER, value REAL); INSERT INTO items VALUES (1, 9), (2, 10), (3, 11), (4, NULL), (5, 11);")
    connection.close()
    return path


def test_comparison_boundary_and_null(database):
    assert execute("SELECT id FROM items WHERE value > 10", database).rows == [(3,), (5,)]
    assert execute("SELECT id FROM items WHERE value <= 10", database).rows == [(1,), (2,)]
    assert not equivalent("SELECT id FROM items WHERE value > 10", "SELECT id FROM items WHERE value <= 10", [str(database)])


def test_multiset_and_order_policy():
    assert equal_results(Result([(1,), (2,)], 1), Result([(2,), (1,)], 1))
    assert not equal_results(Result([(1,), (1,)], 1), Result([(1,)], 1))
    assert not equal_results(Result([(1,), (2,)], 1), Result([(2,), (1,)], 1), True)
    assert not equal_results(Result([], 1), Result([], 2))


@pytest.mark.parametrize("sql", ["DELETE FROM items", "SELECT 1; SELECT 2", "CREATE TABLE x(y)", "PRAGMA user_version=1", "ATTACH DATABASE 'x' AS x"])
def test_reject_non_query(sql):
    with pytest.raises(ValueError):
        parse_query(sql)


def test_missing_database_not_created(tmp_path):
    path = tmp_path / "missing.sqlite"
    with pytest.raises(FileNotFoundError):
        execute("SELECT 1", path)
    assert not path.exists()


def test_execution_resolves_schema_and_row_limit(database):
    with pytest.raises(sqlite3.OperationalError):
        execute("SELECT missing FROM items", database)
    with pytest.raises(ValueError, match="exceeds"):
        execute("SELECT * FROM items", database, max_rows=2)


def test_timeout(database):
    with pytest.raises((sqlite3.OperationalError, TimeoutError)):
        execute("WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x+1 FROM n) SELECT sum(x) FROM n", database, timeout=.01)
