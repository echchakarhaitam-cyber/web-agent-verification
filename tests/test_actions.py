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
    parse_report,
)


@pytest.mark.parametrize(
    "text, expected",
    [
        ('{"action": "click", "ref": 7}', Click(7)),
        ('Sure! ```json\n{"action":"type","ref":3,"text":"a}b"}\n```', Type(3, "a}b")),
        ('{"action": "select", "ref": 5, "option": "Blue"}', Select(5, "Blue")),
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
        ('{"action": "click", "ref": 7', "no JSON object"),
        ('{"action": "click", "ref": "7"}', "'ref' must be the number"),
        ('{"action": "click", "ref": true}', "'ref' must be the number"),
        ('{"action": "click", "ref": -1}', "'ref' must be the number"),
        ('{"action": "type", "ref": 1}', "'text' must be a string"),
        ('{"action": "done", "success": "yes", "confidence": 50}',
         "'success' must be true or false"),
        ('{"action": "done", "success": true, "confidence": 150}', "between 0 and 100"),
        ('{"action": "done", "success": true}', "'confidence' must be a number"),
        ('{"action": "scroll", "ref": 1}', "unknown action 'scroll'"),
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
    assert isinstance(parse_action(f'{{"action":"type","ref":1,"text":"{long}"}}'), InvalidAction)


def test_first_json_object_skips_unbalanced_and_respects_strings():
    assert first_json_object('{ broken {"a": "}"}') == '{"a": "}"}'
    assert first_json_object('x {"a": "\\"}"} y') == '{"a": "\\"}"}'
    assert first_json_object("none") is None


def test_parse_report_ignores_action_key_but_checks_fields():
    assert parse_report('{"action": "click", "success": true, "confidence": 90}') == Done(True, 90)
    assert parse_report('{"success": false, "confidence": 0}') == Done(False, 0)
    assert isinstance(parse_report('{"action": "click", "ref": 6}'), InvalidAction)
    assert isinstance(parse_report('{"success": "yes", "confidence": 9}'), InvalidAction)
    assert isinstance(parse_report("no"), InvalidAction)
    assert isinstance(parse_report("[1, 2]"), InvalidAction)
