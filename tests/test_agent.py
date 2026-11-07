"""The episode loop, with a fake page (unit) and with real MiniWoB++ pages (browser)."""

import math

import pytest

from webverify.actions import Click
from webverify.agent import MAX_INVALID_IN_A_ROW, run_episode
from webverify.env import Observation, Outcome
from webverify.llm import Generation, ScriptedBackend, TokenInfo
from webverify.observation import Element

DONE_TRUE = '{"action": "done", "success": true, "confidence": 90}'
JUDGE_YES = Generation("Yes", (TokenInfo("Yes", math.log(0.8), {"No": math.log(0.2)}),))


class FakeEnv:
    """A page with one button; clicking it ends the episode with the given reward."""

    def __init__(self, reward: float = 1.0) -> None:
        self.reward = reward
        self.done = False
        self.executed = []

    def reset(self, task, seed):
        self.done = False
        return self.observe()

    def observe(self):
        return Observation("Click OK", [Element(1, "BUTTON", "OK")])

    def execute(self, action, obs):
        self.executed.append(action)
        if not obs.has_ref(action.ref):
            return "no such element"
        self.done = True
        return None

    def page_done(self):
        return self.done

    def outcome(self):
        return Outcome(self.done, self.reward if self.done else 0.0, None)


def test_agent_reports_done_itself():
    b = ScriptedBackend(['{"action": "click", "ref": 1}', DONE_TRUE, JUDGE_YES])
    env = FakeEnv()
    env.page_done = lambda: False            # the page does not end on its own here
    ep = run_episode(env, b, "t", 0, budget=5)
    assert ep.stop == "done" and not ep.claim_forced
    assert ep.claim_success is True and ep.claim_confidence == 90
    assert ep.judge_p_yes == pytest.approx(0.8) and ep.judge_answer == "Yes"
    assert env.executed == [Click(1)]


def test_page_end_forces_a_report_and_reward_is_ground_truth():
    b = ScriptedBackend(['{"action": "click", "ref": 1}', DONE_TRUE, "No"])
    ep = run_episode(FakeEnv(reward=-1.0), b, "t", 0, budget=5)
    assert ep.stop == "page_done" and ep.claim_forced
    assert ep.claim_success is True and ep.success is False      # a false claim of success
    assert ep.raw_reward == -1.0
    # the forced report prompt is the second call and asks for a report
    assert "can take no more actions" in b.calls[1][0]["content"]
    assert "Final page:" in b.calls[1][1]["content"]
    assert ep.claim_raw == DONE_TRUE


def test_invalid_answers_are_not_executed_and_stop_after_limit():
    answers = ["click it"] * MAX_INVALID_IN_A_ROW + ["garbage", "Yes"]
    env = FakeEnv()
    ep = run_episode(env, ScriptedBackend(answers), "t", 0, budget=10)
    assert ep.stop == "invalid" and ep.n_invalid == MAX_INVALID_IN_A_ROW
    assert env.executed == []
    assert ep.claim_success is None and ep.success is False


def test_budget_exhaustion_and_bad_ref():
    answers = ['{"action": "click", "ref": 9}'] * 2 + [DONE_TRUE]
    env = FakeEnv()
    ep = run_episode(env, ScriptedBackend(answers), "t", 0, budget=2, use_judge=False)
    assert ep.stop == "budget" and ep.claim_forced
    assert [s.error for s in ep.steps] == ["no such element"] * 2
    assert ep.judge_p_yes is None and not ep.success


# --------------------------------------------------------------------- real pages (Chromium)


@pytest.fixture(scope="module")
def page():
    pw = pytest.importorskip("playwright.sync_api")
    with pw.sync_playwright() as p:
        browser = p.chromium.launch()
        yield browser.new_page()
        browser.close()


@pytest.mark.browser
def test_real_page_success_and_failure(page):
    from webverify.env import MiniWobEnv

    env = MiniWobEnv(page)
    obs = env.reset("click-button", seed=9)
    target = obs.instruction.split('"')[1]
    good = next(e.ref for e in obs.elements if e.tag == "BUTTON" and e.text == target)
    bad = next(e.ref for e in obs.elements if e.tag == "BUTTON" and e.text != target)
    ep = run_episode(env, ScriptedBackend([f'{{"action":"click","ref":{good}}}', DONE_TRUE, "Yes"]),
                     "click-button", 9, budget=3)
    assert ep.success and ep.stop == "page_done" and ep.raw_reward == 1.0
    ep = run_episode(env, ScriptedBackend([f'{{"action":"click","ref":{bad}}}', DONE_TRUE, "Yes"]),
                     "click-button", 9, budget=3)
    assert not ep.success and ep.raw_reward == -1.0 and ep.claim_success is True


@pytest.mark.browser
def test_real_page_typing_and_seed_determinism(page):
    from webverify.actions import Type
    from webverify.env import MiniWobEnv

    env = MiniWobEnv(page)
    first = env.reset("login-user", seed=3)
    assert env.reset("login-user", seed=3).instruction == first.instruction
    words = first.instruction.split('"')
    user, pw_ = words[1], words[3]
    refs = {e.elem_id: e.ref for e in first.elements}
    obs = env.reset("login-user", seed=3)
    assert env.execute(Type(refs["username"], user), obs) is None
    assert env.execute(Type(refs["password"], "x" + pw_), obs) is None
    obs = env.observe()
    assert env.execute(Type(refs["password"], pw_), obs) is None      # replaces, not appends
    assert env.execute(Click(refs["subbtn"]), env.observe()) is None
    out = env.outcome()
    assert out.page_done and out.success


@pytest.mark.browser
def test_real_page_errors(page):
    from webverify.actions import Select
    from webverify.env import MiniWobEnv

    env = MiniWobEnv(page)
    obs = env.reset("click-button", seed=1)
    assert "no element" in env.execute(Click(9999), obs)
    btn = next(e.ref for e in obs.elements if e.tag == "BUTTON")
    assert "not a drop-down" in env.execute(Select(btn, "x"), obs)
    with pytest.raises(FileNotFoundError):
        env.reset("no-such-task", seed=0)
