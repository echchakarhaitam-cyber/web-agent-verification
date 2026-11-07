"""The MiniWoB++ tasks of the study and their step budgets, fixed before any model was run.

Selection rule: tasks whose instruction and every element needed to solve them appear in the
text observation, and that the action set (click / type / select) can solve; checked with a
scripted solution through our harness (tests/test_tasks.py). Excluded, for example:
click-collapsible (its section header has no ref in MiniWoB++'s DOM description),
email-inbox-delete (icon without text), drag and drawing tasks (need coordinates), and
use-autocomplete (needs a word list the model cannot see).

``min_steps`` is the length of the shortest solution; the default budget is 2 * min_steps + 2,
so a model can recover from a few mistakes but cannot wander indefinitely.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    name: str
    min_steps: int
    kind: str                  # click | form

    @property
    def budget(self) -> int:
        return 2 * self.min_steps + 2


TASKS: tuple[Task, ...] = (
    Task("click-button", 1, "click"),
    Task("click-link", 1, "click"),
    Task("click-tab", 1, "click"),
    Task("click-dialog", 1, "click"),
    Task("click-dialog-2", 1, "click"),
    Task("click-widget", 1, "click"),
    Task("navigate-tree", 1, "click"),
    Task("click-button-sequence", 2, "click"),
    Task("click-option", 2, "click"),
    Task("click-checkboxes", 3, "click"),
    Task("enter-text", 2, "form"),
    Task("enter-text-2", 2, "form"),
    Task("choose-list", 2, "form"),
    Task("enter-password", 3, "form"),
    Task("login-user", 3, "form"),
)
BY_NAME = {t.name: t for t in TASKS}
