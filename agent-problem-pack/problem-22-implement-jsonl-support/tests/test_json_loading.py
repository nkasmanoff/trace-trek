import pytest

from src.datakit.errors import RecordParseError, UnsupportedFormatError
from src.datakit.loaders import load_any


def test_loads_json_list(tmp_path):
    path = tmp_path / "rows.json"
    path.write_text('[{"a": 1}, {"a": 2}]', encoding="utf-8")
    assert load_any(path) == [{"a": 1}, {"a": 2}]


def test_loads_json_object_as_single_record(tmp_path):
    path = tmp_path / "row.json"
    path.write_text('{"a": 1}', encoding="utf-8")
    assert load_any(path) == [{"a": 1}]


def test_malformed_json_raises_parse_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{nope", encoding="utf-8")
    with pytest.raises(RecordParseError):
        load_any(path)


def test_unknown_extension_raises(tmp_path):
    path = tmp_path / "rows.parquet"
    path.write_text("", encoding="utf-8")
    with pytest.raises(UnsupportedFormatError):
        load_any(path)
