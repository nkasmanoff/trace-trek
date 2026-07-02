import pytest

from src.logkit.errors import ParseError
from src.logkit.parser import parse


def test_parses_simple_assignments():
    assert parse("a=1\nb=two") == {"a": "1", "b": "two"}


def test_skips_blanks_and_comments():
    assert parse("\n# note\na=1\n\n") == {"a": "1"}


def test_last_assignment_wins():
    assert parse("a=1\na=2") == {"a": "2"}


def test_invalid_line_raises_parse_error():
    with pytest.raises(ParseError):
        parse("a=1\nnot-an-assignment")


def test_error_message_includes_line_number():
    with pytest.raises(ParseError) as excinfo:
        parse("a=1\nb=2\nbroken")
    assert str(excinfo.value) == "line 3: missing '=' in 'broken'"
