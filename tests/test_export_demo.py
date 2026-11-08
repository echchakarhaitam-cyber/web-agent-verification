"""The replay export keeps main-run episodes only, with the fields the page shows."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _mod():
    spec = importlib.util.spec_from_file_location(
        "export_demo_episodes", ROOT / "scripts" / "export_demo_episodes.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _episode(seed, success, claim=True):
    return {"task": "click-button", "seed": seed, "instruction": "Click OK", "budget": 4,
            "stop": "done", "n_invalid": 0, "claim_raw": "{}", "claim_forced": False,
            "claim_success": claim, "claim_confidence": 90, "claim_p_true": 0.8,
            "judge_answer": "Yes", "judge_p_yes": 0.7, "success": success, "raw_reward": 1.0,
            "model": "qwen2.5-0.5b-instruct-q4_k_m", "seconds": 1.0, "page_done": True,
            "reward_reason": None,
            "steps": [{"action": "Click(ref=3)", "error": None, "raw": "{}", "seconds": 0.5}]}


def test_export_filters_pilot_and_drops_extra_fields(tmp_path):
    log = tmp_path / "episodes.jsonl"
    log.write_text("\n".join(json.dumps(e) for e in
                             [_episode(0, True), _episode(101, False), _episode(1, False)]) + "\n")
    data = _mod().export([log])
    assert data["models"] == ["Qwen2.5-0.5B"]
    assert [e["seed"] for e in data["episodes"]] == [0, 1]
    first = data["episodes"][0]
    assert "seconds" not in first and "seconds" not in first["steps"][0]
    assert first["model"] == "Qwen2.5-0.5B"


def test_committed_export_matches_reported_counts():
    data = json.loads((ROOT / "demo" / "episodes.json").read_text(encoding="utf-8"))
    summary = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf-8"))
    by_short = {m.split("-instruct")[0].replace("qwen2.5-", "Qwen2.5-").replace("b", "B"): v
                for m, v in summary["models"].items()}
    for m in data["models"]:
        eps = [e for e in data["episodes"] if e["model"] == m]
        assert len(eps) == by_short[m]["episodes"]
        assert sum(e["success"] for e in eps) == by_short[m]["successes"]
        false_claims = sum(bool(e["claim_success"]) and not e["success"] for e in eps)
        assert false_claims == by_short[m]["claims"]["false_success_claims"]
