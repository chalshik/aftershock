"""analysis/k4.py: K4 draft 1's design investigation (judges nothing)."""
import math

import numpy as np
import pandas as pd
import pytest

from analysis import k4


def _run(trigger=None, off=False, closed=True):
    args = {"--no-enable-prefix-caching": True} if off else {"--enable-prefix-caching": None}
    return {"cfg": {"trigger": trigger, "server": {"args": args},
                    "workload": {"sessions": {"closed_loop": closed}}}}


def test_natural_leaves_out_triggers_cache_off_and_open_loop():
    assert k4.natural(_run())
    assert not k4.natural(_run(trigger={"kind": "burst"}))
    assert not k4.natural(_run(trigger={"kind": "cache_wipe"}))
    assert not k4.natural(_run(off=True))
    assert not k4.natural(_run(closed=False))


def test_classify_arms():
    g = np.arange(240.0, 2700.0, 5.0)
    base = np.full(len(g), 2.5)
    surge = base.copy(); surge[(g >= 1500) & (g < 1620)] = 3.45
    assert k4.classify(g, surge) == {"surge": True, "calm": False, "t_cross": 1500.0, "peak": 3.45}
    early = surge.copy(); early[(g >= 600) & (g < 700)] = 3.25           # a 3.2 peak before the window
    assert not k4.classify(g, early)["surge"]
    late = base.copy(); late[g >= 2000] = 3.5                            # first crossing after the window
    assert not k4.classify(g, late)["surge"]
    assert k4.classify(g, base)["calm"] and not k4.classify(g, base)["surge"]
    edge = base.copy(); edge[100] = 3.1                                  # calm means strictly below 3.1
    assert not k4.classify(g, edge)["calm"]


def test_outcome_rule_order():
    r = k4.outcome_rule
    assert r(5, (5, 0), 0, 2, 2, 4) == "supports"
    assert r(5, (4, 1), 0, 2, 2, 4) == "supports"
    assert r(5, (4, 0), 1, 2, 2, 4) == "mixed"          # one calm tip: never supports
    assert r(5, (4, 0), 1, 1, 2, 4) == "refutes"
    assert r(5, (5, 0), 2, 2, 2, 4) == "refutes"
    assert r(5, (3, 2), 0, 2, 2, 4) == "refutes"
    assert r(5, (3, 1), 0, 2, 2, 4) == "mixed"


def test_power_is_a_distribution_and_matches_a_hand_case():
    ps = {"tipped": 1.0, "warm": 0.0, "unclear": 0.0}
    pc = {"tipped": 0.1, "warm": 0.9, "unclear": 0.0}
    p = k4.power(4, ps, pc, 1, 2, 3)
    assert math.isclose(sum(p.values()), 1.0)
    assert p["supports"] == pytest.approx(0.9 ** 4) and p["refutes"] == pytest.approx(1 - 0.9 ** 4)


def test_crossings_and_surgeless_on_rows():
    lv = k4.K4["levels"]
    def row(tipped, peak, onset=2000.0, warm=1000.0):
        return {"run": f"r{peak}", "tipped": tipped, "warm_s": warm,
                "cross": {l: {"t": 1500.0 if peak >= l else None,
                              "lag_to_onset": (onset - 1500.0) if (peak >= l and tipped) else None} for l in lv}}
    rows = [row(True, 3.45), row(False, 3.25), row(True, 3.05)]
    c = k4.crossings(rows)
    assert c[3.4]["n_crossed"] == 1 and c[3.4]["n_tipped"] == 1 and c[3.4]["lags_s"] == [500.0]
    assert c[3.2]["n_crossed"] == 2 and c[3.2]["runs_crossed_not_tipped"] == ["r3.25"]
    s = k4.surgeless(rows, 3.1)
    assert s["n_tips_below"] == 1 and s["warm_h"] == pytest.approx(3000 / 3600)


def test_design_and_judged_seed_ranges_are_apart():
    assert not set(k4.K4["design_seeds"]) & set(k4.K4["judged_seeds"])
    assert min(k4.K4["judged_seeds"]) > 44                     # above every seed run so far (REG-25's 44)


def test_predicted_is_a_distribution():
    p = k4.predicted()
    assert math.isclose(sum(p.values()), 1.0) and set(p) == {"supports", "refutes", "mixed"}


# ---------------------------------------------------------------- the registration's functions

import fnmatch

from analysis import withheld as W


def test_config_is_reg14s_control_with_seed_name_and_duration_alone():
    from analysis import d1
    from analysis.gate import _burst_cfg
    base = _burst_cfg(k4.K4["base_cfg"])
    k = k4.resolved(123, "surge")
    assert d1._diff_keys(base, k) == ["name", "seed", "workload.duration_s"]
    assert k["seed"] == 123 and k["workload"]["duration_s"] == 2700 and k["workload"]["rate"] == 2.2
    assert k.get("trigger") is None and k["workload"]["sessions"]["closed_loop"] is True
    with pytest.raises(ValueError):
        k4.config_text(123, "other")


def test_the_run_schedule_is_the_scanned_schedule():
    from harness.workload import build_schedule
    from analysis import gate
    k = k4.resolved(123, "calm")
    s, _ = build_schedule(k["workload"], k["seed"], k.get("trigger"))
    g = np.arange(gate.SURGE["start_s"], k4.K4["duration_s"], gate.SURGE["grid_s"])
    x = gate.rolling_rate(s.t_planned.to_numpy(float), g, gate.SURGE["window_s"])
    g2, x2 = gate.schedule_peak_series(gate._burst_cfg(k4.K4["base_cfg"]), 2.2, 123, k4.K4["duration_s"])
    assert np.array_equal(g, g2) and np.allclose(x, x2)


def test_pick_takes_the_lowest_qualifying_seeds_and_alternates(monkeypatch):
    fake = {"n": 9, "surge": [3, 5, 8, 9, 11, 12], "calm": [1, 2, 4, 6, 7, 10], "t_cross": [1300.0, 1400, 1500, 1600,
                                                                                            1700, 1750]}
    monkeypatch.setattr(k4, "scan", lambda lam, seeds: fake)
    pk = k4.pick(range(9))
    assert pk["surge"] == [3, 5, 8, 9, 11] and pk["calm"] == [1, 2, 4, 6, 7]
    assert pk["order"] == [3, 1, 5, 2, 8, 4, 9, 6, 11, 7] and pk["t_cross"][11] == 1700
    fake2 = {**fake, "surge": [3, 5], "t_cross": [1300.0, 1400.0]}
    monkeypatch.setattr(k4, "scan", lambda lam, seeds: fake2)
    assert not k4.feasibility()["passes"]


def _r(lab, on_time=None):
    return {"label": lab, "on_time": on_time}


def test_k4_seal_covers_its_folders_and_no_other():
    seal = [p for reg, p, _ in W.SEALS if reg == "K4"]
    assert seal == ["*_k4_l1pl350cl_r2.2_s[1-5][0-9][0-9]"]
    for sd in (100, 345, 599):
        assert fnmatch.fnmatchcase(f"20261118-000000_{k4.config_name(sd)}", seal[0])
        assert not [p for reg, p, _ in W.SEALS if reg != "K4"
                    and fnmatch.fnmatchcase(f"20261118-000000_{k4.config_name(sd)}", p)]
    from analysis.load import RUNS
    own = {k4.config_name(sd) for sd in k4.REGISTERED["surge"] + k4.REGISTERED["calm"]}
    # Matches no earlier folder: every folder the seal matches is one of K4's ten registered runs ([CORR] 2026-11-17:
    # this line asserted that the seal matched no folder at all, true only until K4's first run was copied back).
    assert not [p.name for p in RUNS.glob("*") if fnmatch.fnmatchcase(p.name, seal[0])
                and p.name.split("_", 1)[1] not in own]


def test_the_committed_configs_are_the_written_text_for_the_picked_seeds():
    import json
    from pathlib import Path
    pk = json.loads((Path(__file__).resolve().parent.parent / "docs/probe/k4_draft/feasibility_92acfbd.txt").read_text())
    for arm in ("surge", "calm"):
        for sd in pk[arm]:
            assert (k4._configs() / f"{k4.config_name(sd)}.yaml").read_text() == k4.config_text(sd, arm)
    assert pk["surge"] == [133, 137, 175, 196, 287] and pk["calm"] == [102, 103, 105, 106, 107] and pk["passes"]


def test_outcome_rules():
    T, Wm = "tipped", "warm"
    surge5 = [_r(T, True)] * 5
    calm5 = [_r(Wm)] * 5
    assert k4.outcome(surge5, calm5)["decision"] == "supports"
    assert k4.outcome([_r(T, True)] * 4 + [_r(Wm)], calm5)["decision"] == "supports"
    assert k4.outcome([_r(T, True)] * 3 + [_r(T, False)] * 2, calm5)["decision"] == "supports_untimed"
    # persistence (rule F r1, item 2): a tip that recovered is not a surge tip
    assert k4.outcome([_r(T, True)] * 3 + [_r("tipped_recovered")] * 2, calm5)["decision"] == "mixed"
    # any onset counts against the calm arm (items 1, 2)
    for kind in k4.CALM_TIPS:
        assert k4.outcome(surge5, [_r(kind)] + [_r(Wm)] * 4)["decision"] == "mixed"
        assert k4.outcome(surge5, [_r(kind)] * 2 + [_r(Wm)] * 3)["decision"] == "refutes"
    # supports needs the calm arm shown warm (item 3)
    assert k4.outcome(surge5, [_r("unclear")] * 2 + [_r(Wm)] * 3)["decision"] == "mixed"
    # bad_baseline leaves n (item 3)
    assert k4.outcome(surge5, [_r("bad_baseline")] * 2 + [_r(Wm)] * 3)["decision"] == "too_few"
    o = k4.outcome([_r(T, True)] * 3 + [dict(_r(Wm), seed=175), dict(_r(Wm), seed=287)], calm5, {175, 287})
    assert o["decision"] == "refutes" and o["refutes_source"] == "surge" and o["surge_warm_marginal_only"] is True
    o = k4.outcome([_r(T, True)] * 3 + [dict(_r(Wm), seed=133), dict(_r(Wm), seed=287)], calm5, {175, 287})
    assert o["surge_warm_marginal_only"] is False
    assert k4.outcome(surge5[:3], calm5)["decision"] == "too_few"
    o = k4.outcome(surge5, calm5)
    assert o["fisher_p"] == pytest.approx(1 / 252) and o["cp_upper_calm_tipped"] == pytest.approx(1 - 0.05 ** 0.2)


def _lab_run(monkeypatch, onset, recovered, pf=1.0):
    import analysis.k1 as K1
    from analysis import gate
    monkeypatch.setattr(gate, "kv_onset", lambda run: onset)
    monkeypatch.setattr(K1, "cold_backlogged", lambda run, win, v: {"recovered": recovered, "base": {"pf": pf}})
    return {"cfg": {"verdict": {"min_baseline_pf": 0.98}, "workload": {"duration_s": 2700.0}}}


def test_label(monkeypatch):
    r = _lab_run(monkeypatch, 1600.0, False)
    out = k4.label(r, 1500.0)
    assert out["label"] == "tipped" and out["on_time"] is True
    assert k4.label(r, 1300.0)["on_time"] is False                     # 300 s after the crossing: late
    assert k4.label(r, 1700.0)["on_time"] is False                     # before the crossing
    assert k4.label(r, None)["on_time"] is None
    # rule F r1 T1: an onset is read before the baseline, so a bad baseline cannot hide a tip
    assert k4.label(_lab_run(monkeypatch, 650.0, False, pf=0.5), None)["label"] == "tipped"
    assert k4.label(_lab_run(monkeypatch, 650.0, None, pf=math.nan), None)["label"] == "tipped_unclear"
    assert k4.label(_lab_run(monkeypatch, 500.0, False, pf=0.5), None)["label"] == "tipped_early"
    # T2: a tip that recovered by the end window
    out = k4.label(_lab_run(monkeypatch, 1815.0, True), 1760.0)
    assert out["label"] == "tipped_recovered" and out["on_time"] is None
    assert k4.label(_lab_run(monkeypatch, math.nan, True), None)["label"] == "warm"
    assert k4.label(_lab_run(monkeypatch, math.nan, False), None)["label"] == "unclear"
    assert k4.label(_lab_run(monkeypatch, math.nan, True, pf=0.9), None)["label"] == "bad_baseline"
    assert k4.REG["base"][1] == k4.REG["tip_from_s"]                   # the baseline ends where tips start counting


def test_table_uses_the_registered_pick_and_refuses_while_sealed(monkeypatch):
    reg = k4.REGISTERED
    monkeypatch.setattr(k4, "pick", lambda seeds=None: {**reg, "t_cross": dict(reg["t_cross"])})
    monkeypatch.setattr(k4, "_sealed", lambda seeds: ["x"])
    with pytest.raises(W.WithheldError):
        k4.table()
    monkeypatch.setattr(k4, "pick", lambda seeds=None: {**reg, "surge": [133, 137, 175, 196, 288]})
    with pytest.raises(AssertionError):
        k4.table()


def test_registered_pick_is_the_feasibility_output_and_the_schedule_load_is_fixed():
    import json
    from pathlib import Path
    fz = json.loads((Path(__file__).resolve().parent.parent / "docs/probe/k4_draft/feasibility_92acfbd.txt").read_text())
    assert fz["surge"] == k4.REGISTERED["surge"] and fz["calm"] == k4.REGISTERED["calm"]
    assert {int(k): v for k, v in fz["t_cross"].items()} == k4.REGISTERED["t_cross"]
    ld = k4.schedule_load()
    assert sorted(sd for sd, v in ld.items() if v["marginal"]) == [175, 287]
    assert all(ld[sd]["peak"] < 3.1 for sd in k4.REGISTERED["calm"])
    assert all(ld[sd]["peak"] >= 3.4 for sd in k4.REGISTERED["surge"])


def test_label_and_readouts_run_on_a_real_run_folder(tmp_path):
    from analysis.load import load_run
    from tests.test_e2e_fake import run_fake
    from tests.test_session_cap import WL
    run = load_run(run_fake(tmp_path, workload=WL))
    run["cfg"].setdefault("verdict", k4.resolved(123, "surge")["verdict"])   # the fake config has none
    lab = k4.label(run, 10.0)
    assert lab["label"] == "bad_baseline" and lab["onset"] is None      # no onset; the windows lie past this short run
    ro = k4.readouts(run, 10.0)
    assert math.isnan(ro["planned_peak"]) and math.isnan(ro["planned_rate"])     # shorter than 240 s: no grid


def test_predicted_r1_is_a_distribution_and_needs_the_calm_arm_warm():
    p = k4.predicted_r1()
    assert math.isclose(sum(p.values()), 1.0)
    ps = {"tipped": 1.0, "warm": 0.0, "unclear": 0.0}
    pc = {"tipped": 0.0, "warm": 0.5, "unclear": 0.5}
    assert k4.power(5, ps, pc, 2, 2, 4)["supports"] == pytest.approx(1.0)            # draft 2's rule
    assert k4.power(5, ps, pc, 2, 2, 4, need_cw=4)["supports"] == pytest.approx(6 / 32)   # >= 4 of 5 warm

