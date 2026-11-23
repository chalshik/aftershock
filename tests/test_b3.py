"""analysis/b3.py (B3's gate) on synthetic runs (no run folder is read)."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import b3

WARM, DUR = 240.0, 4080.0
CFG = {"workload": {"warmup_s": WARM, "duration_s": DUR}}


def run_with(ns, seed=15, t0=WARM):
    """Gauge values at 1 s scrapes starting at t0 (inside the measured window by default)."""
    m = pd.DataFrame({"t": t0 + np.arange(len(ns), dtype=float), b3.N: ns})
    return {"metrics": m, "dir": Path(f"x_stage2_control_l1pl350cl_r2.0_w3000_s{seed}"), "cfg": CFG}


def five(peaks):
    return [b3.control_row(run_with([1, p, 2], seed=s)) for s, p in zip(b3.B3_GATE["seeds"], peaks)]


def test_row_reads_the_peak_the_share_and_the_count_at_the_peak():
    r = b3.control_row(run_with([3, 5, 16, 7, 16, 2]))
    assert r["n_max"] == 16 and r["t_max"] == WARM + 2 and r["at_max"] == 2 and r["seed"] == 15
    assert r["share_at_flag"] == pytest.approx(2 / 6) and r["n_mean"] == pytest.approx(49 / 6)


def test_scrapes_outside_the_measured_window_are_not_judged():
    """A peak in the warm-up or the drain is reported (whole frame), not judged (rule F r1, item 1)."""
    m = pd.DataFrame({"t": [-1.0, 10.0, WARM, WARM + 1, DUR, DUR + 30], b3.N: [20, 18, 4, 6, 19, 17]})
    r = b3.control_row({"metrics": m, "dir": Path("x_s15"), "cfg": CFG})
    assert r["n_max"] == 6 and r["scrapes"] == 2 and r["n_max_whole_frame"] == 20


def test_a_missing_gauge_is_not_usable():
    r = {"metrics": pd.DataFrame({"t": [WARM, WARM + 1]}), "dir": Path("x_s15"), "cfg": CFG}
    row = b3.control_row(r)
    assert row["scrapes"] == 0 and math.isnan(row["n_max"])


def test_nan_scrapes_are_skipped():
    assert b3.control_row(run_with([3, np.nan, 9]))["n_max"] == 9


def test_k_max_is_23():
    assert b3.k_max() == 23 and 23 * 4096 < 96656 <= 24 * 4096


@pytest.mark.parametrize("peaks,want", [
    ([15] * 5, "pass"),
    ([15, 15, 15, 15, 16], "fail_redraft"),
    ([10, 12, 22, 9, 11], "fail_redraft"),
    ([10, 12, 23, 9, 11], "infeasible"),
])
def test_outcome(peaks, want):
    assert b3.gate_outcome(five(peaks))["decision"] == want


def test_a_missing_or_duplicated_seed_is_undefined():
    rows = five([10] * 5)
    assert b3.gate_outcome(rows[:4])["decision"] == "undefined"
    dup = rows[:4] + [b3.control_row(run_with([1, 10], seed=15))]
    assert b3.gate_outcome(dup)["decision"] == "undefined"
    assert b3.gate_outcome(rows + [rows[0]])["decision"] == "undefined"


def test_gate_refuses_before_its_registration(monkeypatch):
    monkeypatch.setattr(b3.withheld, "lifted", lambda h, *a, **k: False)
    called = []
    import analysis.load as load
    monkeypatch.setattr(load, "find_runs", lambda *a, **k: called.append(a) or [])
    with pytest.raises(RuntimeError, match="B3's gate"):
        b3.gate()
    assert not called


def test_only_the_exact_heading_unlocks(tmp_path):
    from analysis import withheld
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-17 · [REG] B3's gate draft withdrawn\n")
    assert not withheld.lifted(b3.GATE_HEADING, d)
    d.write_text("## 2026-11-17 · [REG] B3's gate (rule G): the highest running N …\n")
    assert withheld.lifted(b3.GATE_HEADING, d)


def test_the_pattern_matches_only_reg19s_controls():
    import fnmatch
    pat = b3.B3_GATE["controls"]
    assert fnmatch.fnmatchcase("20261112-050920_stage2_control_l1pl350cl_r2.0_w3000_s15", pat)
    for other in ("x_burst_l1pl350cl_r2.0_w3000_s15", "x_stage2_control_l1pl350cl_r2.0_w3000_s20",
                  "x_stage2_control_l1pl350cl_r2.0_long_s43"):
        assert not fnmatch.fnmatchcase(other, pat)
