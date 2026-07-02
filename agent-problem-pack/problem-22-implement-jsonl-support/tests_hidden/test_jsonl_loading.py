"""Hidden: grade jsonl support against the loader conventions."""

import pytest

from src.datakit.errors import RecordParseError
from src.datakit.loaders import EXT_LOADERS, load_any


def test_jsonl_registered_in_ext_loaders():
    assert ".jsonl" in EXT_LOADERS


def test_loads_jsonl_records(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{"a": 2}\n', encoding="utf-8")
    assert load_any(path) == [{"a": 1}, {"a": 2}]


def test_blank_lines_skipped(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n\n   \n{"a": 2}\n', encoding="utf-8")
    assert load_any(path) == [{"a": 1}, {"a": 2}]


def test_bad_line_reports_path_and_line_number(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"a": 1}\n{broken\n', encoding="utf-8")
    with pytest.raises(RecordParseError) as excinfo:
        load_any(path)
    message = str(excinfo.value)
    assert str(path) in message
    assert "2" in message.replace(str(path), "")


def test_json_loader_unchanged(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text('[{"a": 1}]', encoding="utf-8")
    assert load_any(path) == [{"a": 1}]
