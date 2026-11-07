"""A small Playwright harness around MiniWoB++ task pages (Liu et al., 2018; Farama Foundation).

Only the task pages of the ``miniwob`` package (MIT) are used; this module drives them directly
with Playwright, so that the episode loop, the observation and the action execution are ours.

* ``reset(task, seed)`` loads the page, seeds the page's random generator (``Math.seedrandom``,
  provided by MiniWoB++), replaces the 10 s episode time limit by 11 days (a CPU-only language
  model needs far longer per step; the time-independent raw reward is used; a larger value
  overflows the browser's 32-bit timer and fires at once) and starts the episode.
* ``observe()`` returns the instruction and the flattened page (see observation.py).
* ``execute(action)`` performs a click / type / select on the element with that ref.
* ``outcome()`` reads the benchmark's own verdict: whether the task page ended the episode and
  its raw reward (+1 success, -1 failure, 0 not finished). This is the ground truth; the agent
  never sees it.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from webverify.actions import Click, Select, Type
from webverify.observation import Element, flatten, render

SELECT_OPTIONS_JS = """() => {
  const out = {};
  document.querySelectorAll('select').forEach(s => {
    if (s.dataset.wob_ref !== undefined) out[s.dataset.wob_ref] =
      Array.from(s.options).map(o => o.text);
  });
  return out;
}"""


def task_dir() -> Path:
    """Directory of the MiniWoB++ task pages shipped with the ``miniwob`` package."""
    return Path(str(resources.files("miniwob") / "html" / "miniwob"))


@dataclass(frozen=True)
class Observation:
    instruction: str
    elements: list[Element]

    @property
    def text(self) -> str:
        return render(self.elements)

    def has_ref(self, ref: int) -> bool:
        return any(e.ref == ref for e in self.elements if e.ref >= 0)


@dataclass(frozen=True)
class Outcome:
    page_done: bool          # the task page ended the episode (e.g. a submit button was pressed)
    raw_reward: float        # MiniWoB++ reward without time penalty: +1 / -1 / partial / 0
    reason: str | None

    @property
    def success(self) -> bool:
        """Ground truth used throughout: the page ended the episode with a positive reward."""
        return self.page_done and self.raw_reward > 0


class MiniWobEnv:
    """Drives one Playwright page; the caller owns the browser."""

    def __init__(self, page, pages: Path | None = None) -> None:
        self.page = page
        self.pages = pages or task_dir()
        self.task: str | None = None

    def reset(self, task: str, seed: int) -> Observation:
        path = self.pages / f"{task}.html"
        if not path.exists():
            raise FileNotFoundError(f"unknown MiniWoB++ task {task!r}")
        self.task = task
        self.page.goto(path.as_uri())
        self.page.evaluate(
            f"Math.seedrandom({int(seed)}); core.EPISODE_MAX_TIME = 1e9; core.startEpisodeReal();"
        )
        self.page.wait_for_function("WOB_TASK_READY === true")
        return self.observe()

    def observe(self) -> Observation:
        dom = self.page.evaluate("core.getDOMInfo()")
        options = {int(k): v for k, v in self.page.evaluate(SELECT_OPTIONS_JS).items()}
        return Observation(self.page.evaluate("core.getUtterance()"), flatten(dom, options))

    def execute(self, action: Click | Type | Select, obs: Observation) -> str | None:
        """Perform ``action``; return an error message (shown to the agent) or None."""
        if not obs.has_ref(action.ref):
            return f"there is no element [{action.ref}] on the page"
        if isinstance(action, Click | Type):
            result = self.page.evaluate(f"core.elementClick({action.ref})")
            if result is not True:
                return f"click on [{action.ref}] failed"
            if isinstance(action, Type):
                self.page.keyboard.press("Control+A")
                self.page.keyboard.type(action.text)
            return None
        selector = f'select[data-wob_ref="{action.ref}"]'
        if self.page.locator(selector).count() == 0:
            return f"[{action.ref}] is not a drop-down list"
        try:
            self.page.select_option(selector, label=action.option, timeout=1000)
        except Exception:   # Playwright raises several error types; the message is enough
            return f'no option "{action.option}" in [{action.ref}]'
        return None

    def outcome(self) -> Outcome:
        r = self.page.evaluate(
            "({done: WOB_DONE_GLOBAL, raw: WOB_RAW_REWARD_GLOBAL, reason: WOB_REWARD_REASON})"
        )
        return Outcome(bool(r["done"]), float(r["raw"]), r["reason"])

    def page_done(self) -> bool:
        return bool(self.page.evaluate("WOB_DONE_GLOBAL"))
