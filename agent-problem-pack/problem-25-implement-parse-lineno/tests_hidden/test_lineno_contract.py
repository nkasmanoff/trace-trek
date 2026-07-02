"""Hidden grading tests for the parse-error line number contract.

These check the exact contract documented in src/logkit/errors.py: 1-based
line numbers that count every physical line (blanks and comments included),
an int ``lineno`` attribute, an untouched ``reason`` attribute, and
first-error-wins behavior.
"""

import pytest

from src.logkit.errors import ParseError
from src.logkit.parser import parse


def test_lineno_counts_blank_and_comment_lines():
    text = "\n# header comment\n\na=1\n   \nbad key=x"
    with pytest.raises(ParseError) as excinfo:
        parse(text)
    assert excinfo.value.lineno == 6


def test_lineno_is_an_int():
    with pytest.raises(ParseError) as excinfo:
        parse("oops")
    assert type(excinfo.value.lineno) is int
    assert excinfo.value.lineno == 1


def test_reason_attribute_is_bare_reason():
    with pytest.raises(ParseError) as excinfo:
        parse("a=1\nnope")
    assert excinfo.value.reason == "missing '=' in 'nope'"


def test_message_format_is_exact():
    with pytest.raises(ParseError) as excinfo:
        parse("# c\n =1")
    assert str(excinfo.value) == "line 2: bad key in '=1'"


def test_first_invalid_line_wins():
    with pytest.raises(ParseError) as excinfo:
        parse("a=1\nbad1\nbad2")
    assert excinfo.value.lineno == 2


def test_valid_parsing_still_works_after_change():
    text = "# comment\nname=trace trek\nname=viewer\nempty=\n"
    assert parse(text) == {"name": "viewer", "empty": ""}
