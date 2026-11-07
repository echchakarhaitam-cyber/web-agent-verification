"""Turn MiniWoB++'s DOM description into a short text the model can read.

MiniWoB++ exposes ``core.getDOMInfo()``: a tree of visible elements, each with a stable integer
``ref`` for the episode. A small model cannot read hundreds of layout <div>s, so only elements
that carry meaning are kept: elements with their own text, form controls, links and buttons.
Each kept element becomes one line ``[ref] tag "text" (attributes)``; the model refers to
elements by that ref. Container elements without text are dropped; their text children are
kept, so no visible word is lost. MiniWoB++ splits mixed text into pseudo-elements (tag "t",
negative ref); they are shown as plain text lines without a ref, since they cannot be acted on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

INTERACTIVE = {"BUTTON", "A", "SELECT", "TEXTAREA", "OPTION", "LABEL"}
MAX_TEXT = 80


@dataclass(frozen=True)
class Element:
    ref: int
    tag: str                     # e.g. BUTTON, INPUT_text, INPUT_checkbox, SPAN
    text: str = ""
    elem_id: str = ""
    value: str | None = None
    checked: bool | None = None
    focused: bool = False
    options: tuple[str, ...] = field(default_factory=tuple)

    @property
    def interactive(self) -> bool:
        return self.tag in INTERACTIVE or self.tag.startswith("INPUT")


def _clip(text: str) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 3] + "..."


def flatten(dom: dict, options: dict[int, list[str]] | None = None) -> list[Element]:
    """Keep the meaningful elements of a ``core.getDOMInfo()`` tree, in document order.

    ``options`` maps the ref of each <select> to its option labels (MiniWoB++ does not report
    them because closed options have no size on screen).
    """
    out: list[Element] = []
    options = options or {}

    def walk(node: dict) -> None:
        tag = str(node.get("tag", ""))
        text = _clip(node.get("text") or "")
        ref = int(node["ref"])
        is_box = tag in ("INPUT_checkbox", "INPUT_radio")
        value = node.get("value")
        el = Element(
            ref=ref, tag=tag, text=text, elem_id=str(node.get("id") or ""),
            value=None if value is None or is_box else _clip(value),
            checked=bool(value) if is_box else None,
            focused=bool(node.get("focused")),
            options=tuple(_clip(o) for o in options.get(ref, [])),
        )
        if el.interactive or text:
            out.append(el)
        for child in node.get("children", []) or []:
            walk(child)

    walk(dom)
    return out


def render(elements: list[Element]) -> str:
    """One line per element: ``[ref] tag "text" (id=..., value=..., checked)``."""
    lines = []
    for e in elements:
        if e.ref < 0:                          # text pseudo-element: not actionable
            lines.append(f'text "{e.text}"')
            continue
        parts = [f"[{e.ref}] {e.tag.lower()}"]
        if e.text:
            parts.append(f'"{e.text}"')
        attrs = []
        if e.elem_id:
            attrs.append(f"id={e.elem_id}")
        if e.value is not None and e.tag.startswith(("INPUT", "TEXTAREA")):
            attrs.append(f'value="{e.value}"')
        if e.checked is not None:
            attrs.append("checked" if e.checked else "unchecked")
        if e.options:
            attrs.append("options=" + "|".join(e.options))
        if e.focused:
            attrs.append("focused")
        if attrs:
            parts.append("(" + ", ".join(attrs) + ")")
        lines.append(" ".join(parts))
    return "\n".join(lines)
