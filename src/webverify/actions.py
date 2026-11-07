"""The agent's action language and its strict parser.

The model must answer with ONE JSON object, one of

    {"action": "click",  "id": 7}
    {"action": "type",   "id": 7, "text": "keneth"}
    {"action": "select", "id": 5, "option": "Blue"}
    {"action": "done",   "success": true, "confidence": 80}

``done`` is the agent's own report: whether it believes the task was accomplished, and how sure
it is (0-100). That report is what this project tests against the benchmark's reward.

Why a strict parser: a small model often writes extra prose, code fences or invalid fields.
Anything that does not parse into exactly one of the forms above becomes an ``InvalidAction``
with a reason. The episode loop counts it as a wasted step and never guesses what the model meant,
so every executed action is one the model actually wrote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

MAX_TEXT = 200


@dataclass(frozen=True)
class Click:
    ref: int


@dataclass(frozen=True)
class Type:
    ref: int
    text: str


@dataclass(frozen=True)
class Select:
    ref: int
    option: str


@dataclass(frozen=True)
class Done:
    success: bool
    confidence: int          # 0..100, the model's stated confidence in ``success``


@dataclass(frozen=True)
class InvalidAction:
    raw: str
    reason: str


Action = Click | Type | Select | Done


def first_json_object(text: str) -> str | None:
    """The first balanced {...} in ``text`` (strings and escapes respected), or None."""
    start = text.find("{")
    while start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[start:i + 1]
        start = text.find("{", start + 1)
    return None


def _ref(obj: dict) -> int:
    ref = obj.get("id")
    if isinstance(ref, bool) or not isinstance(ref, int) or ref < 0:
        raise ValueError("'id' must be a non-negative integer")
    return ref


def _string(obj: dict, key: str) -> str:
    val = obj.get(key)
    if not isinstance(val, str):
        raise ValueError(f"'{key}' must be a string")
    if len(val) > MAX_TEXT:
        raise ValueError(f"'{key}' longer than {MAX_TEXT} characters")
    return val


def parse_action(text: str) -> Action | InvalidAction:
    """Parse one model answer; never raises."""
    blob = first_json_object(text)
    if blob is None:
        return InvalidAction(text, "no JSON object")
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError as err:
        return InvalidAction(text, f"invalid JSON: {err.msg}")
    if not isinstance(obj, dict):
        return InvalidAction(text, "not a JSON object")
    kind = obj.get("action")
    try:
        if kind == "click":
            return Click(_ref(obj))
        if kind == "type":
            return Type(_ref(obj), _string(obj, "text"))
        if kind == "select":
            return Select(_ref(obj), _string(obj, "option"))
        if kind == "done":
            success = obj.get("success")
            if not isinstance(success, bool):
                raise ValueError("'success' must be true or false")
            conf = obj.get("confidence")
            if isinstance(conf, bool) or not isinstance(conf, int | float):
                raise ValueError("'confidence' must be a number")
            if not 0 <= conf <= 100:
                raise ValueError("'confidence' must be between 0 and 100")
            return Done(success, round(conf))
    except ValueError as err:
        return InvalidAction(text, str(err))
    return InvalidAction(text, f"unknown action {kind!r}")
