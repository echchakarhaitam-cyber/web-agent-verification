import math

import pytest

from webverify.llm import Generation, ScriptedBackend, TokenInfo, binary_probability
from webverify.observation import MAX_TEXT, flatten, render

DOM = {
    "tag": "BODY", "ref": 1, "children": [
        {"tag": "DIV", "ref": 2, "children": [
            {"tag": "LABEL", "ref": 3, "text": "Username"},
            {"tag": "INPUT_text", "ref": 4, "id": "username", "value": "ke"},
            {"tag": "INPUT_checkbox", "ref": 5, "value": True},
            {"tag": "SELECT", "ref": 6},
            {"tag": "DIV", "ref": 7, "children": [{"tag": "t", "ref": -1, "text": "free text"}]},
            {"tag": "BUTTON", "ref": 8, "text": "Login", "focused": True},
        ]},
    ],
}


def test_flatten_keeps_meaningful_elements_in_order():
    els = flatten(DOM, {6: ["Red", "Blue"]})
    assert [e.ref for e in els] == [3, 4, 5, 6, -1, 8]
    assert els[2].checked is True and els[2].value is None
    assert els[3].options == ("Red", "Blue")


def test_render_lines():
    lines = render(flatten(DOM, {6: ["Red", "Blue"]})).splitlines()
    assert lines == [
        '[3] label "Username"',
        '[4] input_text (html_id=username, value="ke")',
        "[5] input_checkbox (checked)",
        "[6] select (options=Red|Blue)",
        'text "free text"',
        '[8] button "Login" (focused)',
    ]


def test_long_text_is_clipped():
    els = flatten({"tag": "SPAN", "ref": 1, "text": "a  b " * 100})
    assert len(els[0].text) == MAX_TEXT and els[0].text.endswith("...")


def test_binary_probability_from_alternatives():
    toks = [TokenInfo('{"', -0.1), TokenInfo(" true", math.log(0.6),
                                             {" true": math.log(0.6), " false": math.log(0.2),
                                              "x": math.log(0.2)})]
    assert binary_probability(toks, {"true"}, {"false"}) == pytest.approx(0.75)


def test_binary_probability_missing():
    assert binary_probability([TokenInfo("hello", -1.0)], {"yes"}, {"no"}) is None
    assert binary_probability([], {"yes"}, {"no"}) is None


def test_scripted_backend_order_and_exhaustion():
    b = ScriptedBackend(["a", Generation("b")])
    assert b.chat([], 5).text == "a" and b.chat([], 5).text == "b"
    with pytest.raises(RuntimeError):
        b.chat([], 5)
