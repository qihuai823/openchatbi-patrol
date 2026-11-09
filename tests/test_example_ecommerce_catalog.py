"""Integrity checks for the `example_ecommerce` catalog.

These are cheap, LLM-free assertions that the demo catalog is actually loadable
and internally consistent. They exist because two classes of breakage are
invisible at runtime:

* a UTF-8 BOM on a CSV renames its first column (``column_name`` becomes
  ``\\ufeffcolumn_name``), so ``_load_common_columns`` silently drops every row
  and the whole catalog degrades to bare column names;
* a ``sql_example.yaml`` entry with a typo still loads fine - it is only ever
  fed to the LLM as a few-shot example, never executed.

Both are pinned below.
"""

from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest
import yaml

from openchatbi.catalog.store.file_system import FileSystemCatalogStore

DATA_DIR = Path(__file__).resolve().parent.parent / "example_ecommerce"
SQLITE_PATH = DATA_DIR / "ecommerce.sqlite"

REQUIRED_FILES = [
    "bi.yaml",
    "config.yaml",
    "table_info.yaml",
    "table_columns.csv",
    "common_columns.csv",
    "table_spec_columns.csv",
    "table_selection_example.csv",
    "sql_example.yaml",
]

EXPECTED_TABLES = {
    "Customers",
    "Invoices",
    "Order_Items",
    "Orders",
    "Products",
    "Shipment_Items",
    "Shipments",
}


@pytest.fixture(scope="module")
def store() -> FileSystemCatalogStore:
    return FileSystemCatalogStore(
        data_path=str(DATA_DIR),
        data_warehouse_config={
            "uri": f"sqlite:///{SQLITE_PATH.as_posix()}",
            "database_name": "",
        },
    )


def _parse_examples(text: str) -> list[tuple[str, str]]:
    """Mirror of FileSystemCatalogStore._parse_example_text."""
    examples: list[tuple[str, str]] = []
    question = answer = ""
    current: str | None = None
    for line in text.strip().split("\n"):
        if line.startswith("Q:"):
            if question and answer:
                examples.append((question.strip(), answer.strip()))
                question = answer = ""
            question, current = line[2:], "Q"
        elif line.startswith("A:"):
            answer, current = line[2:], "A"
        elif current == "Q":
            question += "\n" + line
        elif current == "A":
            answer += "\n" + line
    if question and answer:
        examples.append((question.strip(), answer.strip()))
    return examples


# ---------------------------------------------------------------------------
# files & encoding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("filename", REQUIRED_FILES)
def test_required_catalog_file_exists(filename: str) -> None:
    assert (DATA_DIR / filename).is_file(), f"missing catalog file: {filename}"


@pytest.mark.parametrize("filename", [f for f in REQUIRED_FILES if f.endswith(".csv")])
def test_csv_files_have_no_bom(filename: str) -> None:
    raw = (DATA_DIR / filename).read_bytes()
    assert not raw.startswith(b"\xef\xbb\xbf"), (
        f"{filename} starts with a UTF-8 BOM; csv.DictReader would rename its "
        "first column and the loader would silently drop every row"
    )


@pytest.mark.parametrize("filename", [f for f in REQUIRED_FILES if f.endswith(".csv")])
def test_csv_headers_match_expected_schema(filename: str) -> None:
    expected = {
        "table_columns.csv": ["db_name", "table_name", "column_name"],
        "common_columns.csv": [
            "column_name",
            "display_name",
            "alias",
            "type",
            "category",
            "tag",
            "description",
        ],
        "table_spec_columns.csv": [
            "db_name",
            "table_name",
            "column_name",
            "display_name",
            "alias",
            "type",
            "category",
            "tag",
            "description",
        ],
        "table_selection_example.csv": ["question", "selected_tables"],
    }[filename]
    with open(DATA_DIR / filename, encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert header[: len(expected)] == expected


# ---------------------------------------------------------------------------
# catalog content
# ---------------------------------------------------------------------------


def test_all_tables_are_registered(store: FileSystemCatalogStore) -> None:
    assert set(store.get_table_list()) == EXPECTED_TABLES


def test_every_column_has_metadata(store: FileSystemCatalogStore) -> None:
    """No column may fall through both common_columns and table_spec_columns."""
    missing = []
    for table in store.get_table_list():
        for column in store.get_column_list(table):
            if not column.get("description") or not column.get("type"):
                missing.append(f"{table}.{column['column_name']}")
    assert missing == []


def test_common_column_aliases_survive_loading(store: FileSystemCatalogStore) -> None:
    category = next(c for c in store.get_column_list("Products") if c["column_name"] == "category")
    assert category["is_common"] is True
    assert "品类" in category["alias"]


def test_unit_price_is_table_specific(store: FileSystemCatalogStore) -> None:
    """The whole point of table_spec_columns: same name, different meaning."""
    products = next(c for c in store.get_column_list("Products") if c["column_name"] == "unit_price")
    items = next(c for c in store.get_column_list("Order_Items") if c["column_name"] == "unit_price")

    assert products["is_common"] is False
    assert items["is_common"] is False
    assert products["description"] != items["description"]
    assert "List Price" in products["display_name"]
    assert "Order Time" in items["display_name"]


def test_spec_columns_do_not_shadow_untouched_columns(store: FileSystemCatalogStore) -> None:
    """Declaring unit_price for two tables must not mark other tables' unit_price."""
    shipments = store.get_column_list("Shipments")
    assert any(c["column_name"] == "order_id" and c["is_common"] is False for c in shipments)
    assert any(c["column_name"] == "shipment_date" and c["is_common"] is True for c in shipments)


def test_table_info_covers_every_table(store: FileSystemCatalogStore) -> None:
    for table in store.get_table_list():
        info = store.get_table_information(table)
        assert info.get("description"), f"{table} has no description in table_info.yaml"
        assert info.get("selection_rule"), f"{table} has no selection_rule"


# ---------------------------------------------------------------------------
# few-shot examples
# ---------------------------------------------------------------------------


def test_every_sql_example_executes_and_returns_rows() -> None:
    """A few-shot example that errors or returns nothing teaches the model wrong."""
    examples = yaml.safe_load((DATA_DIR / "sql_example.yaml").read_text(encoding="utf-8"))
    con = sqlite3.connect(SQLITE_PATH)
    failures = []
    try:
        for tables in examples.values():
            for table, text in tables.items():
                for question, sql in _parse_examples(text):
                    try:
                        rows = con.execute(sql).fetchall()
                    except Exception as exc:  # noqa: BLE001 - report every failure at once
                        failures.append(f"{table}: {type(exc).__name__}: {exc} | Q: {question}")
                        continue
                    if not rows:
                        failures.append(f"{table}: query returned no rows | Q: {question}")
    finally:
        con.close()
    assert failures == []


def test_sql_example_tables_exist() -> None:
    examples = yaml.safe_load((DATA_DIR / "sql_example.yaml").read_text(encoding="utf-8"))
    for db_name, tables in examples.items():
        assert db_name == "", "the ecommerce catalog has no database prefix"
        assert set(tables) <= EXPECTED_TABLES, f"unknown table in sql_example.yaml: {set(tables) - EXPECTED_TABLES}"


def test_table_selection_examples_reference_real_tables() -> None:
    with open(DATA_DIR / "table_selection_example.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) >= 10
    for row in rows:
        selected = set(yaml.safe_load(row["selected_tables"]))
        assert selected <= EXPECTED_TABLES, f"unknown table in {row['question']!r}: {selected - EXPECTED_TABLES}"
        assert selected, f"empty selection for {row['question']!r}"


def test_table_selection_examples_load_through_store(store: FileSystemCatalogStore) -> None:
    examples = store.get_table_selection_examples()
    assert len(examples) >= 10
    for question, tables in examples:
        assert question
        assert tables
        assert set(tables) <= EXPECTED_TABLES
