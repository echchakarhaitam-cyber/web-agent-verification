import numpy as np
import pytest

from webverify.analysis import (
    apply_threshold,
    arrays,
    auroc,
    bootstrap_auroc,
    brier,
    choose_threshold,
    claim_table,
    ece,
    signal,
    two_proportion_z,
    wilson,
)


def ep(success, claim=None, conf=None, p_true=None, judge=None, task="t"):
    return {"success": success, "claim_success": claim, "claim_confidence": conf,
            "claim_p_true": p_true, "judge_p_yes": judge, "task": task}


def test_signals():
    e = ep(True, True, 80, 0.9, 0.7)
    assert signal(e, "claim") == 1 and signal(e, "verbal") == pytest.approx(0.8)
    assert signal(e, "ptrue") == 0.9 and signal(e, "judge") == 0.7
    f = ep(False, False, 80)
    assert signal(f, "verbal") == pytest.approx(0.2) and signal(f, "ptrue") == 0.0
    none = ep(False)
    assert signal(none, "claim") == 0 and signal(none, "verbal") == 0
    assert signal(none, "judge") == 0.5
    with pytest.raises(KeyError):
        signal(e, "nope")


def test_wilson_known_value():
    lo, hi = wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert wilson(0, 0) is None
    assert wilson(0, 20)[0] == 0.0


def test_auroc():
    assert auroc(np.array([0.9, 0.8, 0.1, 0.2]), np.array([1, 1, 0, 0])) == 1.0
    assert auroc(np.array([0.5, 0.5]), np.array([1, 0])) == 0.5
    assert auroc(np.array([0.1, 0.9]), np.array([1, 0])) == 0.0
    assert auroc(np.array([0.1]), np.array([1])) is None


def test_bootstrap_auroc_by_group():
    s = np.array([0.9, 0.1, 0.8, 0.2])
    y = np.array([1, 0, 1, 0])
    assert bootstrap_auroc(s, y, ["a", "a", "b", "b"], n=50) == (1.0, 1.0)


def test_brier_and_ece():
    s, y = np.array([1.0, 0.0, 0.5, 0.5]), np.array([1, 0, 1, 0])
    assert brier(s, y) == pytest.approx(0.125)
    assert ece(s, y) == pytest.approx(0.0)
    assert ece(np.array([0.9, 0.9]), np.array([0, 0])) == pytest.approx(0.9)


def test_claim_table():
    eps = [ep(True, True), ep(False, True), ep(True, False), ep(False, None)]
    t = claim_table(eps)
    assert t["claims_success"] == 2 and t["false_success_claims"] == 1
    assert t["missed_successes"] == 1 and t["no_report"] == 1
    assert t["false_success_rate"] == 0.5


def test_threshold_certified_and_applied():
    rng = np.random.default_rng(0)
    n = 2000
    s = rng.random(n)
    y = (rng.random(n) < s).astype(int)            # calibrated signal
    claim = np.ones(n, bool)
    out = choose_threshold(s, claim, y, target=0.10)
    assert out["certified"] and out["threshold"] >= 0.85
    res = apply_threshold(s, claim, y, out["threshold"])
    assert res["risk"] <= 0.10 and 0 < res["coverage"] < 0.2
    assert not choose_threshold(s[:5], claim[:5], y[:5], target=0.01)["certified"]
    assert apply_threshold(s, claim, y, None)["coverage"] == 0.0


def test_two_proportion_z():
    assert two_proportion_z(50, 100, 50, 100) == pytest.approx(1.0)
    assert two_proportion_z(90, 100, 50, 100) < 1e-6
    assert two_proportion_z(1, 0, 1, 1) is None
    assert two_proportion_z(0, 10, 0, 10) == 1.0


def test_arrays():
    s, y = arrays([ep(True, True, 90), ep(False, False, 90)], "verbal")
    assert s.tolist() == pytest.approx([0.9, 0.1]) and y.tolist() == [1, 0]
