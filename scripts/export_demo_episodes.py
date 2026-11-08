"""Export the main-run episodes (seeds < 100, no pilot) for the replay page (space/index.html).

Keeps what the page shows: instruction, the agent's actions and raw answers, its final report,
the judge's answer, the four signals and the page's own reward. Observations are not kept.
Usage:  python scripts/export_demo_episodes.py results/episodes_<model>.jsonl [...]
Writes demo/episodes.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KEEP = ("task", "seed", "instruction", "budget", "stop", "n_invalid", "claim_raw", "claim_forced",
        "claim_success", "claim_confidence", "claim_p_true", "judge_answer", "judge_p_yes",
        "success", "raw_reward")


def short(model: str) -> str:
    return model.split("-instruct")[0].replace("qwen2.5-", "Qwen2.5-").replace("b", "B")


def export(paths: list[Path]) -> dict:
    out: dict = {"models": [], "episodes": []}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line:
                continue
            e = json.loads(line)
            if e["seed"] >= 100:
                continue
            m = short(e["model"])
            if m not in out["models"]:
                out["models"].append(m)
            row = {k: e[k] for k in KEEP}
            row["model"] = m
            row["steps"] = [{"action": s["action"], "error": s["error"], "raw": s["raw"]}
                            for s in e["steps"]]
            out["episodes"].append(row)
    out["episodes"].sort(key=lambda r: (out["models"].index(r["model"]), r["task"], r["seed"]))
    return out


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    data = export([Path(p) for p in argv])
    (ROOT / "demo").mkdir(exist_ok=True)
    target = ROOT / "demo" / "episodes.json"
    target.write_text(json.dumps(data, separators=(",", ":"), sort_keys=True) + "\n",
                      encoding="utf-8", newline="\n")
    print(f"wrote {target}: {len(data['episodes'])} episodes, {target.stat().st_size} bytes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
