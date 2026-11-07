"""Draw the two README figures from results/summary.json (run scripts/analyse.py first).

Writes docs/figures/claims.png and docs/figures/auroc.png.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from webverify.analysis import SIGNALS

ROOT = Path(__file__).resolve().parents[1]


def short(name: str) -> str:
    size = name.split("-instruct")[0].removeprefix("qwen2.5-")
    return f"Qwen2.5-{size.upper()}"


def main(summary_path: Path = ROOT / "results" / "summary.json",
         out: Path = ROOT / "docs" / "figures") -> int:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    models = summary["models"]
    names = sorted(models, key=lambda n: models[n]["success_rate"])
    out.mkdir(parents=True, exist_ok=True)

    # Figure 1: what the agent said vs what the page recorded.
    parts = [("true_success_claims", "says success, succeeded", "#3b7d4f"),
             ("false_success_claims", "says success, failed", "#c0504d"),
             ("missed_successes", "says failure or no report, succeeded", "#e3a33b"),
             ("other_failures", "says failure or no report, failed", "#7f8fa6")]
    fig, ax = plt.subplots(figsize=(7, 2.8))
    for i, n in enumerate(names):
        c = dict(models[n]["claims"])
        c["other_failures"] = c["n"] - c["claims_success"] - c["missed_successes"]
        left = 0
        for key, label, color in parts:
            v = c[key]
            ax.barh(i, v, left=left, color=color, label=label if i == 0 else None)
            if v >= 12:
                ax.text(left + v / 2, i, str(v), ha="center", va="center", color="white",
                        fontsize=9)
            left += v
    ax.set_yticks(range(len(names)), [short(n) for n in names])
    ax.set_xlabel("episodes (300 per model)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, fontsize=8, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "claims.png", dpi=150)
    plt.close(fig)

    # Figure 2: AUROC of each signal with the task-level bootstrap interval.
    fig, ax = plt.subplots(figsize=(7, 3))
    width = 0.38
    for j, n in enumerate(names):
        for i, sig in enumerate(SIGNALS):
            v = models[n]["signals"][sig]
            x = i + (j - 0.5) * width
            lo, hi = v["auroc_ci"] if v["auroc_ci"] else (v["auroc"], v["auroc"])
            ax.bar(x, v["auroc"], width, color=("#9db4d3", "#2f5f98")[j],
                   label=short(n) if i == 0 else None)
            ax.errorbar(x, v["auroc"], yerr=[[v["auroc"] - lo], [hi - v["auroc"]]], color="black",
                        capsize=3, lw=1)
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.text(len(SIGNALS) - 0.5, 0.51, "no information", ha="right", va="bottom", fontsize=8,
            color="grey")
    ax.set_xticks(range(len(SIGNALS)), list(SIGNALS))
    ax.set_ylim(0, 1)
    ax.set_ylabel("AUROC (success vs failure)")
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "auroc.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'claims.png'} and {out / 'auroc.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
