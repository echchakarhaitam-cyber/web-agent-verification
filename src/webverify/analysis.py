"""Metrics of the study, from episode records (dicts as written by run_episodes.py).

Ground truth y = 1 when the benchmark rewarded the episode. Four VERIFICATION SIGNALS, each a
number in [0, 1] meant as "probability that the episode succeeded", all available without the
reward:

* ``claim``   the agent's own yes/no report (1 if it claims success, else 0; no report -> 0);
* ``verbal``  its stated confidence turned into P(success): c/100 if it claims success, 1 - c/100
              if it claims failure (no report -> 0);
* ``ptrue``   the model's token probability of ``true`` at the "success" field of its report
              (falls back to ``claim`` when the report has no such token);
* ``judge``   P(Yes) of a separate judge prompt that sees only the task and the final page.

Used for
* RQ2: is the claim trustworthy? (precision of "success" claims, false-success rate);
* RQ3: which signal separates successes from failures (AUROC) and is calibrated (Brier, ECE)?
* RQ4: abstention. An episode is ACCEPTED automatically when the agent claims success and the
  signal is >= a threshold; everything else goes to a human. Risk = share of accepted episodes
  that actually failed; coverage = share of all episodes accepted. The threshold is certified on
  the calibration seeds (risk <= r* with probability >= 1 - delta, binomial test with Bonferroni
  over a fixed grid of thresholds) and evaluated on the other seeds.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.stats import binom, norm

SIGNALS = ("claim", "verbal", "ptrue", "judge")
THRESHOLD_GRID = tuple(round(0.05 * i, 2) for i in range(0, 20))       # 0.00 ... 0.95
N_BINS = 10


def signal(ep: dict, name: str) -> float:
    claim = ep.get("claim_success")
    if name == "claim":
        return 1.0 if claim else 0.0
    if name == "verbal":
        if claim is None or ep.get("claim_confidence") is None:
            return 0.0
        c = ep["claim_confidence"] / 100
        return c if claim else 1 - c
    if name == "ptrue":
        p = ep.get("claim_p_true")
        return float(p) if p is not None else (1.0 if claim else 0.0)
    if name == "judge":
        p = ep.get("judge_p_yes")
        return float(p) if p is not None else 0.5
    raise KeyError(name)


def arrays(eps: Sequence[dict], name: str) -> tuple[np.ndarray, np.ndarray]:
    s = np.array([signal(e, name) for e in eps], dtype=float)
    y = np.array([1 if e["success"] else 0 for e in eps], dtype=int)
    return s, y


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float] | None:
    """Wilson 95% interval for a proportion k/n (None when n = 0)."""
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    mid = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, mid - half), min(1.0, mid + half)


def auroc(s: np.ndarray, y: np.ndarray) -> float | None:
    """Probability that a random success scores above a random failure (ties count 1/2)."""
    pos, neg = s[y == 1], s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return None
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def bootstrap_auroc(s: np.ndarray, y: np.ndarray, groups: Sequence[str], n: int = 1000,
                    seed: int = 0) -> tuple[float, float] | None:
    """95% percentile interval, resampling whole groups (tasks) with replacement."""
    rng = np.random.default_rng(seed)
    labels = sorted(set(groups))
    idx = {g: np.flatnonzero(np.asarray(groups) == g) for g in labels}
    vals = []
    for _ in range(n):
        pick = np.concatenate([idx[labels[i]] for i in rng.integers(len(labels), size=len(labels))])
        v = auroc(s[pick], y[pick])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def brier(s: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((s - y) ** 2))


def ece(s: np.ndarray, y: np.ndarray, n_bins: int = N_BINS) -> float:
    """Expected calibration error with equal-width bins (the last bin includes 1.0)."""
    bins = np.minimum((s * n_bins).astype(int), n_bins - 1)
    total = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            total += m.sum() / len(s) * abs(s[m].mean() - y[m].mean())
    return float(total)


def claim_table(eps: Sequence[dict]) -> dict:
    """Confusion of the agent's claim against the reward."""
    claim = np.array([bool(e.get("claim_success")) for e in eps])
    y = np.array([bool(e["success"]) for e in eps])
    n_claim = int(claim.sum())
    false_success = int((claim & ~y).sum())
    return {
        "n": len(eps), "successes": int(y.sum()), "claims_success": n_claim,
        "true_success_claims": int((claim & y).sum()), "false_success_claims": false_success,
        "missed_successes": int((~claim & y).sum()),
        "no_report": int(sum(e.get("claim_success") is None for e in eps)),
        "false_success_rate": false_success / n_claim if n_claim else None,
        "false_success_ci": wilson(false_success, n_claim),
    }


def accept(s: np.ndarray, claim: np.ndarray, threshold: float) -> np.ndarray:
    return claim & (s >= threshold)


def choose_threshold(s: np.ndarray, claim: np.ndarray, y: np.ndarray, target: float,
                     delta: float = 0.05, grid: Sequence[float] = THRESHOLD_GRID) -> dict:
    """Loosest grid threshold whose risk among accepted episodes is certified <= target.

    For each threshold, accepted n with k failures; H0 "risk > target" is rejected when
    P(Binomial(n, target) <= k) <= delta / len(grid). Among rejected thresholds the one
    accepting the most episodes is returned (None if no threshold is certified).
    """
    level = delta / len(grid)
    best = None
    for t in grid:
        acc = accept(s, claim, t)
        n, k = int(acc.sum()), int((acc & (y == 0)).sum())
        p = float(binom.cdf(k, n, target)) if n else 1.0
        if p <= level and (best is None or n > best["cal_accepted"]):
            best = {"threshold": t, "cal_accepted": n, "cal_failures": k,
                    "cal_coverage": n / len(y)}
    return {"target": target, "delta": delta, "test_level": level,
            "certified": best is not None, **(best or {"threshold": None})}


def apply_threshold(s: np.ndarray, claim: np.ndarray, y: np.ndarray,
                    threshold: float | None) -> dict:
    if threshold is None:
        return {"coverage": 0.0, "accepted": 0, "failures": 0, "risk": None, "risk_ci": None}
    acc = accept(s, claim, threshold)
    n, k = int(acc.sum()), int((acc & (y == 0)).sum())
    return {"coverage": n / len(y), "accepted": n, "failures": k,
            "risk": k / n if n else None, "risk_ci": wilson(k, n)}


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float | None:
    """Two-sided p-value of the pooled z-test for p1 = p2."""
    if n1 == 0 or n2 == 0:
        return None
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k1 / n1 - k2 / n2) / se
    return float(2 * norm.sf(abs(z)))
