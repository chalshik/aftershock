"""analysis/pw.py (reading PW): the truncated view, the late-outcome variants, the pair checks, the counts and the
registration gate on synthetic runs; and, once PW is registered and the run folders are present, REG-19's check."""
import math

import numpy as np
import pandas as pd
import pytest

from analysis import pw as PW
from analysis.load import RUNS


def cfg(kind="burst", control_t=1080, t_max=16080.0, steps=({"at_s": 5580, "rate": 1},), **over):
    c = {"name": "x", "seed": 43, "loadgen_force_close": True,
         "trigger": {"kind": "burst", "at_s": 900, "duration_s": 180} if kind == "burst" else None,
         "verdict": {"watch_s": 4500, "control_t_s": control_t if kind != "burst" else 900},
         "workload": {"duration_s": t_max, "rate": 2.0, "steps": list(steps), "sessions": {"closed_loop": True}}}
    c.update(over)
    return c


def fake(kind="burst", **kw):
    t = pd.Series([0.0, 3000.0, 4079.9, 4080.0, 9000.0])
    return {"dir": RUNS / f"x_{kind}", "cfg": cfg(kind, **kw),
            "metrics": pd.DataFrame({"t": t}), "gpu": pd.DataFrame({"t": t}),
            "requests": pd.DataFrame({"t_planned": t, "t_end": t + 10, "t_first_token": t + 1.0,
                                      "status": ["ok"] * 5, "ttft": [0.5] * 5})}


@pytest.mark.parametrize("kind", ["burst", "control"])
def test_truncate_cuts_at_4080_and_sets_reg19s_watch(kind):
    r = fake(kind)
    v = PW.truncate(r)
    assert v["cfg"]["verdict"]["watch_s"] == 3000.0 and v["cfg"]["workload"]["duration_s"] == 4080.0
    assert v["metrics"].t.max() < 4080 and v["gpu"].t.max() < 4080
    assert v["requests"].t_planned.max() < 4080
    assert list(v["requests"].t_end) == [10.0, 3010.0, 4089.9]   # a request planned before the cut keeps its record
    assert r["cfg"]["verdict"]["watch_s"] == 4500 and len(r["metrics"]) == 5   # the input is not changed


def test_truncate_refuses_a_control_whose_watch_does_not_end_at_the_cut():
    with pytest.raises(ValueError):
        PW.truncate(fake("control", control_t=900))


def test_late_requests_are_those_planned_in_the_two_windows_without_a_first_token_before_the_cut():
    v = PW.truncate(fake())
    # planned 3000 (first token 3001: decided before the cut) and 4079.9 (first token 4080.9: after it)
    assert list(PW.late_mask(v)) == [False, False, True]
    f, m = PW.set_late(v, "fast")["requests"], PW.set_late(v, "miss")["requests"]
    assert f.ttft.iloc[2] == 0.0 and f.status.iloc[2] == "ok"
    assert m.status.iloc[2] == "unfinished" and math.isnan(m.ttft.iloc[2])
    assert (f.iloc[:2] == v["requests"].iloc[:2]).all().all()            # others untouched
    assert PW.set_late(v, None) is v


def template():
    return {"control": cfg("control", t_max=4080.0, steps=(), seed=15),
            "burst": cfg("burst", t_max=4080.0, steps=(), seed=15)}


def test_check_pair_accepts_a_pair_differing_only_in_the_allowed_keys():
    PW.check_pair(fake("control"), fake("burst"), template())


@pytest.mark.parametrize("change", [
    {"loadgen_force_close": False},
    {"slo": {"ttft_s": 2.0}},
    {"workload": {"duration_s": 16080.0, "rate": 2.0, "steps": [{"at_s": 4000, "rate": 1}],
                  "sessions": {"closed_loop": True}}},
    {"workload": {"duration_s": 16080.0, "rate": 2.0, "steps": [], "sessions": {"closed_loop": True, "cap": 40}}},
    {"workload": {"duration_s": 16080.0, "rate": 2.0, "steps": [],
                  "sessions": {"closed_loop": True, "release": "max_planned"}}},
])
def test_check_pair_refuses(change, monkeypatch):
    monkeypatch.setattr(PW.G, "closed_loop", lambda r: bool(r["cfg"]["workload"]["sessions"].get("closed_loop")))
    b = fake("burst")
    b["cfg"].update(change)
    with pytest.raises(ValueError):
        PW.check_pair(fake("control"), b, template())


def test_check_pair_refuses_unequal_durations():
    with pytest.raises(ValueError):
        PW.check_pair(fake("control", t_max=16000.0), fake("burst"), template())


def test_counts_count_only_passes_stable_under_every_variant():
    rows = [{"seed": s, "outcome": "pass_clear", "counted": True, "cut_sensitive": False} for s in range(10)]
    p = PW.counts(rows)
    assert (p["k"], p["n"]) == (10, 10) and p["cp_lower"] == pytest.approx(0.05 ** (1 / 10), abs=1e-6)
    rows[0].update(counted=False, cut_sensitive=True)
    p = PW.counts(rows)
    assert (p["k"], p["n"]) == (9, 10) and 0.60 < p["cp_lower"] < 0.61 and p["cut_sensitive"] == [0]


def test_compare_flags_a_label_and_a_rate_difference():
    reg19 = {"control_verdict": "recovered", "burst_verdict": "stuck", "outcome": "pass_clear",
             "cold_side": {"holds": True, "c": 1.75, "lam_p": 2.0}, "after_empty": {"rewarmed": None},
             "surge_control": {"tipped": False}, "exposure": {"exposed": False, "peak": 3.0},
             "control_waiting": [{"share_waiting": 0.0}] * 8, "emptyings_s": []}
    row = {"control_verdict": "recovered", "burst_verdict": "stuck", "outcome": "pass_clear", "cold_side": True,
           "rewarmed": None, "tipped": False, "exposed": False, "c": 1.75, "lam_p": 2.0, "shares": [0.0] * 8,
           "emptyings_s": [], "peak": 3.0}
    assert PW.compare(row, reg19) == []
    assert PW.compare({**row, "outcome": "drains"}, reg19) and PW.compare({**row, "c": 1.80}, reg19)


def test_table_refuses_before_registration(monkeypatch):
    monkeypatch.setattr(PW, "_registered", lambda: False)
    with pytest.raises(RuntimeError):
        PW.table()


@pytest.mark.skipif(not PW._registered() or not any(RUNS.glob("*_stage2_control_l1pl350cl_r2.0_w3000_s15")),
                    reason="PW not registered, or REG-19's run folders are not here (the release ships runs/ empty)")
def test_reg19_check_passes():
    assert len(PW.reg19_check()["rows"]) == 5
