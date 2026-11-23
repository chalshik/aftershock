"""analysis/reg30.py: REG-30's re-warm onset reading (synthetic inputs; no run folder is read)."""
import math
import pathlib

import pytest

from analysis import reg30

WE, DE = 5580.0, 14580.0


def _series(values, t0=0.0, step=60.0):
    return [(t0 + i * step, v) for i, v in enumerate(values)]


# ---------------------------------------------------------------- onsets

def test_onset60_needs_hold_consecutive_bins():
    s = _series([0.0, 0.9, 0.9, 0.0, 0.9, 0.9, 0.9, 0.9, 0.9])
    assert reg30.onset_of(s, 0.8, 5) == 240.0
    assert reg30.onset_of(s, 0.8, 2) == 60.0


def test_a_nan_bin_breaks_the_hold():
    assert reg30.onset_of(_series([0.9, 0.9, math.nan, 0.9, 0.9, 0.9, 0.9, 0.9]), 0.8, 5) == 180.0


def test_no_onset_is_nan_and_equality_reaches_the_level():
    assert math.isnan(reg30.onset_of(_series([0.1] * 20), 0.8, 5))
    assert math.isnan(reg30.onset_of(_series([0.9] * 3), 0.8, 5))
    assert reg30.onset_of(_series([0.8] * 5), 0.8, 5) == 0.0


def _step_reuse(t_back, lo=0.0, hi=0.9):
    """Pooled reuse over [x, y) of a run whose per-turn reuse jumps from lo to hi at t_back (turns uniform in time)."""
    def f(x, y):
        if y <= t_back:
            return lo
        if x >= t_back:
            return hi
        return (lo * (t_back - x) + hi * (y - t_back)) / (y - x)
    return f


def test_pooled_onset_needs_two_consecutive_windows():
    # reuse jumps to 0.9 at 1000 s; level 0.84: [t, t+300) needs t >= 1000 - 300 x 0.06 / 0.9 = 980
    assert reg30.onset_pooled(_step_reuse(1000.0), 0.0, 5000.0, 0.84, 60.0, 300.0) == 1020.0
    assert math.isnan(reg30.onset_pooled(_step_reuse(4500.0), 0.0, 5000.0, 0.84, 60.0, 300.0))   # 2 windows don't fit


def test_pooled_onset_ignores_a_single_good_window():
    f = lambda x, y: 0.9 if 600 <= x < 900 else 0.0
    assert math.isnan(reg30.onset_pooled(f, 0.0, 3000.0, 0.84, 60.0, 300.0))


# ---------------------------------------------------------------- held_low

def test_low_run_start_is_the_unbroken_run_before_the_onset():
    n = _series([80, 30, 20, 50, 24, 22, 20, 18, 30])   # bins at 0, 60, ..., 480
    assert reg30.low_run_start(n, 480.0, 24.4) == 240.0     # 24, 22, 20, 18 (bins 240-420), broken by 50 at 180
    assert math.isnan(reg30.low_run_start(n, 240.0, 24.4))   # the bin before 240 (50) is above the bound
    assert reg30.low_run_start(_series([20, math.nan, 20, 20]), 240.0, 24.4) == 120.0   # a NaN breaks the run


@pytest.mark.parametrize("lead,n,onset,drained,lab", [
    (math.nan, math.nan, math.nan, 100.0, "no_onset"),
    (math.nan, 30.0, 10.0, math.nan, "no_drain"),
    (299.9, 30.0, 10.0, 309.9, "near_drain"),
    (-299.9, 30.0, 10.0, -289.9, "near_drain"),
    (-300.0, 30.0, 10.0, -290.0, "after_drain"),
    (300.0, 30.0, 0.0, 300.0, "before_drain"),     # the lead bound is inclusive
    (600.0, 20.0, 0.0, 600.0, "before_drain"),     # and so are both ends of the band
    (600.0, 40.0, 0.0, 600.0, "before_drain"),
    (600.0, 41.0, 0.0, 600.0, "early_other"),
    (600.0, 12.0, 0.0, 600.0, "early_other"),
])
def test_labels(lead, n, onset, drained, lab):
    assert reg30.label(lead, n, onset, drained, level=0.94) == lab


def test_bad_baseline_comes_first():
    assert reg30.label(600.0, 30.0, 0.0, 600.0, level=math.nan) == "bad_base"


def test_held_low_needs_the_full_stay_and_low_reuse():
    assert reg30.label(900.0, 20.0, 600.0, 1500.0, t_low=300.0, reuse_low=0.1, level=0.94) == "held_low"
    assert reg30.label(10.0, 20.0, 600.0, 610.0, t_low=300.0, reuse_low=0.1, level=0.94) == "held_low"
    assert reg30.label(900.0, 20.0, 599.0, 1499.0, t_low=300.0, reuse_low=0.1, level=0.94) == "before_drain"
    # reuse already at level over the stay: not hysteresis
    assert reg30.label(900.0, 20.0, 600.0, 1500.0, t_low=300.0, reuse_low=0.94, level=0.94) == "before_drain"
    assert reg30.label(900.0, 20.0, 600.0, 1500.0, t_low=300.0, reuse_low=math.nan, level=0.94) == "before_drain"


# ---------------------------------------------------------------- one run, stubbed

def _stub(monkeypatch, reuse_back_at, n_of, drained_rel=1500.0, base=1.0):
    calls = {}
    monkeypatch.setattr(reg30.confirm, "phases", lambda cfg: {"drop": (WE, DE), "baseline": (300.0, 900.0)})

    def drained_at(run, a, b, lvl):
        calls["drained"] = (a, b, lvl)
        return WE + drained_rel
    monkeypatch.setattr(reg30.confirm, "drained_at", drained_at)

    def owed(run, t):
        calls.setdefault("owed", []).append(t)
        return 28
    monkeypatch.setattr(reg30.G, "owed_backlog", owed)
    step = _step_reuse(WE + reuse_back_at, 0.0, 1.0)
    monkeypatch.setattr(reg30.k1, "reuse", lambda run, a, b: base if b <= 900.0 else step(a, b))
    monkeypatch.setattr(reg30.k1, "sessions_in_service", lambda run, a, b: n_of(a - WE))
    monkeypatch.setattr(reg30, "_level", lambda run, a, b: {"q": 5.0, "kv": 0.7})
    run = {"cfg": {"seed": 61, "verdict": {"recovered_hit_drop": 0.06}, "trigger": {"at_s": 900}},
           "dir": type("D", (), {"name": "x_confirm_l1pl350cl_r2.0_s61"})()}
    return run, calls


def test_rewarm_population_link(monkeypatch):
    """N falls from 300 to 30 at 600 s into the drop; reuse returns there; the line drains at 1500 s."""
    run, calls = _stub(monkeypatch, 600.0, lambda t: 300.0 if t < 600 else 30.0)
    out = reg30.rewarm(run)
    assert calls["drained"] == (WE, DE, 28) and calls["owed"][0] == 900.0   # the line at the trigger, over the drop
    assert out["level"] == pytest.approx(0.94)
    assert out["onset_s"] == 600.0 and out["onset60_s"] == 600.0
    assert out["lead_s"] == 900.0 and out["n_onset"] == 30.0 and out["label"] == "before_drain"


def test_rewarm_slow_fall_through_the_band_is_not_hysteresis(monkeypatch):
    """Rule F r1 B1: N passes 40 at 0 s and falls slowly, reaching 24.4 only at 600 s, where reuse returns."""
    run, _ = _stub(monkeypatch, 600.0, lambda t: 40.0 - t * (15.6 / 600.0) if t < 600 else 24.0)
    out = reg30.rewarm(run)
    assert out["label"] == "before_drain"
    assert out["readouts"]["labels_at_low_bound"][40.0] == "held_low"      # draft 1's bound would have said so


def test_rewarm_hysteresis(monkeypatch):
    """N at 15 from the drop's start; reuse returns only at 900 s."""
    run, _ = _stub(monkeypatch, 900.0, lambda t: 15.0, drained_rel=3000.0)
    out = reg30.rewarm(run)
    assert out["t_low_s"] == 0.0 and out["reuse_low"] == 0.0 and out["label"] == "held_low"


def test_rewarm_hold_sensitive(monkeypatch):
    """The two onsets fall on either side of the lead bound: the run is hold_sensitive."""
    run, _ = _stub(monkeypatch, 600.0, lambda t: 30.0 if t >= 600 else 300.0, drained_rel=900.0)
    monkeypatch.setattr(reg30, "onset_of", lambda series, level, hold: WE + 660.0)
    out = reg30.rewarm(run)
    assert out["label_pooled"] == "before_drain" and out["label_60"] == "near_drain"
    assert out["label"] == "hold_sensitive"


def test_rewarm_nan_baseline(monkeypatch):
    run, _ = _stub(monkeypatch, 600.0, lambda t: 30.0, base=math.nan)
    assert reg30.rewarm(run)["label"] == "bad_base"


# ---------------------------------------------------------------- judge and table

def _rows(labels, seeds=(43, 61, 68, 72, 79)):
    return [{"label": lab, "seed": s} for lab, s in zip(labels, seeds)]


def test_judge_outcomes():
    assert reg30.judge(_rows(["before_drain"] * 4 + ["near_drain"]))["outcome"] == "population_link"
    assert reg30.judge(_rows(["held_low"] * 4 + ["before_drain"]))["outcome"] == "hysteresis"
    assert reg30.judge(_rows(["near_drain"] * 4 + ["before_drain"]))["outcome"] == "drain_bound"
    assert reg30.judge(_rows(["early_other"] * 4 + ["before_drain"]))["outcome"] == "early_population"
    assert reg30.judge(_rows(["before_drain"] * 3 + ["hold_sensitive"] * 2))["outcome"] == "mixed"
    assert reg30.judge(_rows(["before_drain"] * 3, seeds=(61, 68, 72)))["outcome"] == "mixed"   # 4 still needed
    assert reg30.judge(_rows(["before_drain"] * 2))["outcome"] == "too_few"


def test_judge_bounds_are_clopper_pearson():
    j = reg30.judge(_rows(["before_drain"] * 5))
    assert j["bounds"]["before_drain"]["lower"] == pytest.approx(0.05 ** 0.2, abs=1e-3)   # 0.549
    assert j["bounds"]["near_drain"]["upper"] == pytest.approx(1 - 0.05 ** 0.2, abs=1e-3)   # 0.451


def test_design_seed_is_reported_apart():
    j = reg30.judge(_rows(["before_drain"] + ["near_drain"] * 4))
    assert j["without_design_seed"]["near_drain"] == 4 and j["without_design_seed"]["n"] == 4


def test_table_refuses_before_its_registration(monkeypatch):
    monkeypatch.setattr(reg30.withheld, "lifted", lambda h: False)
    with pytest.raises(RuntimeError):
        reg30.table()


def test_lifting_heading_is_reg30s_registration():
    assert reg30.REG30["heading"] == "[REG] REG-30"


def test_module_never_sets_the_unseal_variable():
    assert "AFTERSHOCK_UNSEAL" not in pathlib.Path(reg30.__file__).read_text()


def test_pinned_runs_and_controls():
    r, c = reg30.REG30["runs"], reg30.REG30["controls"]
    assert sorted(r) == sorted(c) == [43, 61, 68, 72, 79]
    assert all(v.endswith(f"_confirm_l1pl350cl_r2.0_s{s}") for s, v in r.items())
    assert all(v.endswith(f"_stage2_control_l1pl350cl_r2.0_long_s{s}") for s, v in c.items())
    assert all(c[s] < r[s] for s in r)          # each control ran before its burst run


def test_low_bound_is_the_lowest_n_star():
    assert reg30.REG30["low_bound"] == pytest.approx(24.36, abs=0.01)


def test_z2_reference_is_e1s_four_saturated_runs():
    ids = reg30.Z2["ref_runs"]
    assert len(ids) == 4 and all("_sweep_l1pl350cl_pc_off_r" in i for i in ids)
    assert [float(i.rsplit("_r", 1)[1]) for i in ids] == [2.5, 3.0, 3.5, 4.0]
