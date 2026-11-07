"""The task list, and a scripted solution of every multi-step task through the real harness."""

import re

import pytest

from webverify.actions import Click, Select, Type
from webverify.env import task_dir
from webverify.tasks import BY_NAME, TASKS


def test_task_list_is_consistent():
    assert len(BY_NAME) == len(TASKS) == 15
    for t in TASKS:
        assert (task_dir() / f"{t.name}.html").exists()
        assert t.budget == 2 * t.min_steps + 2 and t.kind in {"click", "form"}


def quoted(s):
    return re.findall(r'"([^"]*)"', s)


@pytest.fixture(scope="module")
def env():
    pw = pytest.importorskip("playwright.sync_api")
    from webverify.env import MiniWobEnv

    with pw.sync_playwright() as p:
        browser = p.chromium.launch()
        yield MiniWobEnv(browser.new_page())
        browser.close()


def by_id(obs):
    return {e.elem_id: e.ref for e in obs.elements if e.elem_id}


@pytest.mark.browser
@pytest.mark.parametrize("seed", [0, 1])
def test_scripted_solutions(env, seed):
    o = env.reset("choose-list", seed)
    name = o.instruction.split("Select ")[1].split(" from")[0]
    sel = next(e for e in o.elements if e.tag == "SELECT")
    assert env.execute(Select(sel.ref, name), o) is None
    btn = next(e for e in o.elements if e.tag == "BUTTON")
    env.execute(Click(btn.ref), env.observe())
    assert env.outcome().success

    o = env.reset("enter-password", seed)
    pw, r = quoted(o.instruction)[0], by_id(o)
    env.execute(Type(r["password"], pw), o)
    env.execute(Type(r["verify"], pw), env.observe())
    env.execute(Click(r["subbtn"]), env.observe())
    assert env.outcome().success

    o = env.reset("click-button-sequence", seed)
    r = by_id(o)
    env.execute(Click(r["subbtn"]), o)
    env.execute(Click(r["subbtn2"]), env.observe())
    assert env.outcome().success

    o = env.reset("enter-text", seed)
    r = by_id(o)
    env.execute(Type(r["tt"], quoted(o.instruction)[0]), o)
    env.execute(Click(r["subbtn"]), env.observe())
    assert env.outcome().success
