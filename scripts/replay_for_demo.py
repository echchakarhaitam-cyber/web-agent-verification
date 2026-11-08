"""Rebuild what the agent saw in every recorded episode, for the replay page.

The episode logs keep the agent's answers but not the pages. MiniWoB++ pages are generated from
a seed, so each episode can be replayed without the model: reset the same (task, seed), apply
the recorded answers in order with the same harness, and capture the page after each step.

Checks (any mismatch is reported in the output, not hidden):
* every executed action gives the same error message (or none) as in the log;
* the page's reward at the end gives the same success as recorded.

Writes demo/episodes.json (episodes + per-step observations + check results) and one small
lossless WebP frame of the task area per step: demo/frames/<id>_<k>.webp, where frame k shows the
page before step k with the element the agent acts on outlined in red, and the last frame shows
the page at the end.
Usage:  python scripts/replay_for_demo.py results/episodes_<model>.jsonl [...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

from webverify.actions import Done, InvalidAction, parse_action
from webverify.env import MiniWobEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from export_demo_episodes import export  # noqa: E402

FRAMES = ROOT / "demo" / "frames"
HIGHLIGHT_JS = """(ref) => {
  const el = document.querySelector(`[data-wob_ref=\"${ref}\"]`);
  if (!el) return false;
  const r = el.getBoundingClientRect();
  const w = document.getElementById("wrap").getBoundingClientRect();
  const left = Math.max(w.left + 1, r.left - 3), right = Math.min(w.right - 7, r.right + 3);
  const d = document.createElement("div");
  d.id = "__hl";
  Object.assign(d.style, {position: "fixed", left: left + "px", top: (r.top - 3) + "px",
    width: Math.max(4, right - left) + "px", height: (r.height + 6) + "px",
    border: "3px solid #e00000",
    borderRadius: "4px", zIndex: 99999, pointerEvents: "none"});
  document.body.appendChild(d);
  return true;
}"""


def episode_id(e: dict) -> str:
    return f"{e['model'].lower().replace('qwen2.5-', 'q')}_{e['task']}_{e['seed']}"


def shot(env: MiniWobEnv, path: Path, ref: int | None = None) -> None:
    """Save the task area as lossless WebP, optionally with element ``ref`` outlined."""
    import io

    from PIL import Image

    # once the page ends an episode MiniWoB++ covers the task with a "START" screen; hide it so
    # the last frame shows the page as the agent left it
    env.page.evaluate("""() => { const c = document.getElementById('sync-task-cover');
                                 if (c) c.style.display = 'none'; }""")
    marked = ref is not None and env.page.evaluate(HIGHLIGHT_JS, ref)
    png = env.page.locator("#wrap").screenshot()
    if marked:
        env.page.evaluate("document.getElementById('__hl').remove()")
    Image.open(io.BytesIO(png)).save(path, "WEBP", lossless=True, method=6)


def replay(env: MiniWobEnv, e: dict) -> dict:
    eid = episode_id(e)
    obs = env.reset(e["task"], e["seed"])
    observations = [obs.text]
    mismatches = []
    frame = 0
    for i, step in enumerate(e["steps"]):
        act = parse_action(step["raw"])
        target = None if isinstance(act, InvalidAction | Done) else act.ref
        shot(env, FRAMES / f"{eid}_{frame}.webp", target)
        frame += 1
        if isinstance(act, Done):
            break
        if not isinstance(act, InvalidAction):
            error = env.execute(act, obs)
            if error != step["error"]:
                mismatches.append(f"step {i + 1}: error {error!r} != logged {step['error']!r}")
        obs = env.observe()
        observations.append(obs.text)
        if env.page_done():
            break
    shot(env, FRAMES / f"{eid}_{frame}.webp")
    out = env.outcome()
    if out.success != e["success"]:
        mismatches.append(f"success {out.success} != logged {e['success']}")
    return {"id": eid, "observations": observations, "frames": frame + 1,
            "replay_ok": not mismatches, "replay_notes": mismatches}


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    data = export([Path(p) for p in argv])
    FRAMES.mkdir(parents=True, exist_ok=True)
    bad = 0
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 400, "height": 400})
        env = MiniWobEnv(page)
        for n, e in enumerate(data["episodes"], 1):
            e.update(replay(env, e))
            bad += not e["replay_ok"]
            if n % 50 == 0:
                print(f"{n}/{len(data['episodes'])} replayed, {bad} with differences", flush=True)
        browser.close()
    data["replay"] = {"episodes": len(data["episodes"]), "identical": len(data["episodes"]) - bad}
    target = ROOT / "demo" / "episodes.json"
    target.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"wrote {target} ({target.stat().st_size} bytes); {bad} episodes differ on replay")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
