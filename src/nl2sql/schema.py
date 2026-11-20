"""Stable Spider schema serialization, shared by every condition."""
from __future__ import annotations

from .common import read_json

SCHEMA_VERSION = "v1"


def load_schemas(path) -> dict:
    rows = read_json(path)
    schemas = {row["db_id"]: row for row in rows}
    if len(schemas) != len(rows):
        raise ValueError("Duplicate database IDs in tables.json")
    return schemas


def quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def serialize(schema: dict) -> str:
    tables = schema["table_names_original"]
    columns = schema["column_names_original"]
    types = schema.get("column_types", ["unknown"] * len(columns))
    if (not tables or not all(isinstance(t, str) and t for t in tables)
            or len(set(t.casefold() for t in tables)) != len(tables)):
        raise ValueError("Schema table names must be nonempty and unique")
    if len(types) != len(columns):
        raise ValueError("Schema column_types and column_names_original lengths differ")
    for table, name in columns:
        if type(table) is not int or table < -1 or table >= len(tables) or not isinstance(name, str) or not name:
            raise ValueError("Invalid schema column reference")
    for reference in schema.get("primary_keys", []):
        if type(reference) is not int or not 0 <= reference < len(columns) or columns[reference][0] < 0:
            raise ValueError("Invalid primary-key reference")
    for pair in schema.get("foreign_keys", []):
        if len(pair) != 2 or any(type(ref) is not int or not 0 <= ref < len(columns) or columns[ref][0] < 0 for ref in pair):
            raise ValueError("Invalid foreign-key reference")
    primary = set(schema.get("primary_keys", []))
    pieces = []
    for index, name in enumerate(tables):
        cols = [f"{quote(col)} {types[i]}{' PRIMARY KEY' if i in primary else ''}"
                for i, (table, col) in enumerate(columns) if table == index]
        pieces.append(f"{quote(name)}({', '.join(cols)})")
    foreign = []
    for left, right in sorted(schema.get("foreign_keys", [])):
        lt, lc = columns[left]
        rt, rc = columns[right]
        foreign.append(f"{quote(tables[lt])}.{quote(lc)} = {quote(tables[rt])}.{quote(rc)}")
    return "; ".join(pieces) + ("; foreign keys: " + ", ".join(foreign) if foreign else "")


def model_input(row: dict) -> str:
    if row.get("schema_serialization_version") != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema version for {row.get('id')}")
    return f"translate English to SQL: {row['question']}\nSchema: {row['schema']}"
