"""The figure script runs on the committed summary and writes both images."""

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "scripts" / "make_figures.py"
    spec = importlib.util.spec_from_file_location("make_figures", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_short_names():
    mod = _load()
    assert mod.short("qwen2.5-1.5b-instruct-q4_k_m") == "Qwen2.5-1.5B"


def test_figures_from_summary(tmp_path):
    mod = _load()
    assert mod.main(ROOT / "results" / "summary.json", tmp_path) == 0
    for name in ("claims.png", "auroc.png"):
        assert (tmp_path / name).stat().st_size > 1000
