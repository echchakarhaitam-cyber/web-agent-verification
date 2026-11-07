import pytest

from webverify.actions import (
    MAX_TEXT,
    Click,
    Done,
    InvalidAction,
    Select,
    Type,
    first_json_object,
    parse_action,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"action": "click", "id": 7}', Click(7)),
        ('Sure! ```json\n{"action":"type","id":3,"text":"a}b"}\n```', Type(3, "a}b")),
        ('{"action": "select", "id": 5, "option": "Blue"}', Select(5, "Blue")),
        ('{"action": "done", "success": true, "confidence": 80}', Done(True, 80)),
        ('{"action": "done", "success": false, "confidence": 12.6}', Done(False, 13)),
    ],
)
def test_valid_actions(text, expected):
    assert parse_action(text) == expected


@pytest.mark.parametrize(
    "text, reason",
    [
        ("I will click the button", "no JSON object"),
        ('{"action": "click", "id": 7', "no JSON object"),
        ('{"action": "click", "id": "7"}', "'id' must be a non-negative integer"),
        ('{"action": "click", "id": true}', "'id' must be a non-negative integer"),
        ('{"action": "click", "id": -1}', "'id' must be a non-negative integer"),
        ('{"action": "type", "id": 1}', "'text' must be a string"),
        ('{"action": "done", "success": "yes", "confidence": 50}',
         "'success' must be true or false"),
        ('{"action": "done", "success": true, "confidence": 150}', "between 0 and 100"),
        ('{"action": "done", "success": true}', "'confidence' must be a number"),
        ('{"action": "scroll", "id": 1}', "unknown action 'scroll'"),
        ("{'action': 'click'}", "invalid JSON"),
    ],
)
def test_invalid_actions(text, reason):
    out = parse_action(text)
    assert isinstance(out, InvalidAction)
    assert reason in out.reason
    assert out.raw == text


def test_text_length_is_limited():
    long = "x" * (MAX_TEXT + 1)
    assert isinstance(parse_action(f'{{"action":"type","id":1,"text":"{long}"}}'), InvalidAction)


def test_first_json_object_skips_unbalanced_and_respects_strings():
    assert first_json_object('{ broken {"a": "}"}') == '{"a": "}"}'
    assert first_json_object('x {"a": "\\"}"} y') == '{"a": "\\"}"}'
    assert first_json_object("none") is None
