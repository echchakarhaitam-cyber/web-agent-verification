"""Run the agent on every (task, seed) pair and append one JSON line per episode.

Usage:
  python scripts/run_episodes.py --model models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
      --seeds 0-19 [--tasks click-button,login-user] [--out results/episodes_<model>.jsonl]

Resumable: pairs already present in the output file are skipped, so an interrupted run can be
restarted with the same command. Seeds of the page are the task seeds; decoding is greedy, so an
episode is determined by (model file, task, seed) up to floating-point non-determinism of the
CPU backend.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from webverify.agent import run_episode
from webverify.env import MiniWobEnv
from webverify.llm import LlamaBackend
from webverify.tasks import BY_NAME, TASKS

ROOT = Path(__file__).resolve().parents[1]


def parse_seeds(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        if "-" in part:
            a, b = part.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def done_pairs(path: Path) -> set[tuple[str, int]]:
    if not path.exists():
        return set()
    pairs = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            pairs.add((r["task"], r["seed"]))
    return pairs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seeds", default="0-19")
    ap.add_argument("--tasks", default=",".join(t.name for t in TASKS))
    ap.add_argument("--out")
    ap.add_argument("--threads", type=int)
    args = ap.parse_args()
    tasks = [BY_NAME[t] for t in args.tasks.split(",")]
    seeds = parse_seeds(args.seeds)
    backend = LlamaBackend(args.model, n_threads=args.threads)
    out = Path(args.out) if args.out else ROOT / "results" / f"episodes_{backend.name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    done = done_pairs(out)
    todo = [(t, s) for s in seeds for t in tasks if (t.name, s) not in done]
    print(f"{backend.name}: {len(todo)} episodes to run, {len(done)} already done", flush=True)

    from playwright.sync_api import sync_playwright

    t0 = time.perf_counter()
    with sync_playwright() as p:
        browser = p.chromium.launch()
        env = MiniWobEnv(browser.new_page())
        for i, (task, seed) in enumerate(todo, 1):
            ep = run_episode(env, backend, task.name, seed, task.budget)
            with out.open("a", encoding="utf-8", newline="\n") as f:
                f.write(json.dumps(ep.to_dict(), sort_keys=True) + "\n")
            el = time.perf_counter() - t0
            print(f"[{i}/{len(todo)}] {task.name} seed {seed}: success={ep.success} "
                  f"claim={ep.claim_success} conf={ep.claim_confidence} stop={ep.stop} "
                  f"{ep.seconds:.0f}s (eta {el / i * (len(todo) - i) / 60:.0f} min)", flush=True)
        browser.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
