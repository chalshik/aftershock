"""analysis/confirm.py (REG-24 draft 3): configs, schedules and synthetic runs only; no run folder is read."""
import fnmatch
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis.confirm as C
import analysis.gate as G
from harness.workload import build_schedule

FLAT = C._flat


# ---------------------------------------------------------------- design (no candidate seed is read here)

def test_settings_are_the_earlier_registrations_values():
    """Every number the rule uses comes from an earlier registration's code or result; none is new except the drop
    candidates (REG-16's plus 3600 s) and the seed range (outside REG-20b's 20-39)."""
    assert C.REG24["window_tol"] == G.REG15["window_tol"] and C.REG24["surge_tip"] == G.REG17["surge_tip"]
    assert C.REG24["warm_watch_s"] == G.REG16["warm_watch_s"] and C.REG24["c_pessimism"] == G.REG16["c_pessimism"]
    assert C.REG24["grid_s"] == G.REG16["grid_s"] and C.REG24["return_end_s"] == G.REG15["return_end_s"]
    assert C.REG24["watch_s"] == G.REG15["watch_s"] == G.REG16["watch_s"] and C.REG24["drop_rate"] == 1.0
    assert C.REG24["return_s"] == G.REG15["return_s"] and C.REG24["baseline"] == G.REG16["baseline"]
    assert C.REG24["timeline_s"] == G.REG15["timeline_s"] == G.REG16["timeline_s"]
    assert set(G.REG16["drop_candidates"]) < set(C.REG24["drop_candidates"])
    assert C.REG24["lam"] == G.REG19["lam"] and C.REG24["reg19_seeds"] == G.REG19["seeds"]
    # REG-15's calibration as REG-16 registered it (decisions.md:3156): owed 28; c 1.758 / 1.672; lowest 1.728 / 1.637
    cal = C.REG24["cal15"]
    assert cal["owed_at_trigger"] == 28 and [round(cal[k], 3) for k in ("c_watch", "c_drop", "c_watch_min",
                                                                        "c_drop_min")] == [1.758, 1.672, 1.728, 1.637]
    assert round(C.REG24["u_off"], 2) == 5164.76 and round(C.REG24["r_expected"], 3) == 1.021


def test_seed_range_is_unused_and_outside_reg20b():
    """The candidates start at 40, outside REG-20b's 20-39. Rule F round 1, RE-O3: a later config may set a seed in
    40-199 without breaking this test; the guard that matters is that no config other than REG-24's two sets seed 43."""
    used = C.used_seeds()
    assert {0, 1, 2, 3, 4, 5, 6, 7, 12, 14, 15, 16, 17, 18, 19} <= used
    assert min(C.REG24["seed_candidates"]) == 40 and not set(range(20, 40)) & set(C.REG24["seed_candidates"])
    assert C.REG24["seed"] not in used and C.REG24["seed"] in C.candidate_seeds()


def test_burst_and_control_cfg_change_only_name_seed_duration_steps_watch():
    """Against each of REG-19's configs: only name, seed, workload.duration_s, workload.steps and verdict.watch_s
    differ (as REG-15's config against REG-14's burst run)."""
    for sd in G.REG19["seeds"]:
        for mine, theirs in ((C.burst_cfg(41, 5400.0), f"burst_l1pl350cl_r2.0_w3000_s{sd}"),
                             (C.control_cfg(41, 5400.0), f"stage2_control_l1pl350cl_r2.0_w3000_s{sd}")):
            a, b = FLAT(mine), FLAT(G._burst_cfg(theirs))
            assert sorted(k for k in a.keys() | b.keys() if a.get(k) != b.get(k)) == \
                ["name", "seed", "verdict.watch_s", "workload.duration_s", "workload.steps"]
    b = C.burst_cfg(41, 5400.0)
    assert b["workload"]["steps"] == [{"at_s": 5580.0, "rate": 1.0}, {"at_s": 10980.0, "rate": 2.0}]
    assert b["workload"]["duration_s"] == 12480.0 and b["verdict"]["watch_s"] == 4500.0
    assert b["trigger"] == {"kind": "burst", "at_s": 900, "duration_s": 180, "rate": 2.4, "prompt_tokens": 2944}
    assert b["workload"]["sessions"]["closed_loop"] is True and b["loadgen_force_close"] is True
    assert b["checks"]["expected_power_limit_w"] == 350


def test_control_shares_the_burst_runs_session_schedule():
    b, c = C.burst_cfg(15, 3600.0), C.control_cfg(15, 3600.0)     # seed 15: used by REG-19, not a candidate
    sb, _ = build_schedule(b["workload"], b["seed"], b["trigger"])
    sc, _ = build_schedule(c["workload"], c["seed"], c.get("trigger"))
    cols = ["t_planned", "session_id", "turn", "prompt_len"]
    assert sc[cols].equals(sb[~sb.burst][cols].reset_index(drop=True))
    assert c.get("trigger") is None and c["verdict"]["control_t_s"] == 1080


def test_phases_from_the_config_and_refusals():
    b, c = C.burst_cfg(15, 5400.0), C.control_cfg(15, 5400.0)
    for cfg in (b, c):
        ph = C.phases(cfg)
        assert ph["burst_end"] == 1080 and ph["watch_end"] == 5580 and ph["drop"] == (5580, 10980)
        assert ph["return"] == (10980, 12480) and ph["return_end"] == (11880, 12480) and ph["drop_s"] == 5400
        assert ph["baseline"] == (300.0, 900.0)
    t = float(b["trigger"]["at_s"])
    assert C.REG24["baseline"] == (t - b["verdict"]["baseline_s"], t)
    w = C.judged_windows(b)
    assert w["previous"] == (4380, 4980) and w["end"] == (4980, 5580) and w["return_end"] == (11880, 12480)
    bad = [{**b, "workload": {**b["workload"], "steps": b["workload"]["steps"][:1]}},
           {**b, "workload": {**b["workload"], "duration_s": 12000.0}},
           {**b, "workload": {**b["workload"], "rate": 2.2}},
           {**b, "verdict": {**b["verdict"], "watch_s": 3000.0}},
           {**b, "workload": {**b["workload"], "steps": [{"at_s": 5580.0, "rate": 0.5}, {"at_s": 10980.0, "rate": 2.0}]}},
           {**b, "workload": {**b["workload"], "steps": [{"at_s": 5580.0, "rate": 1.0}, {"at_s": 10000.0, "rate": 2.0}],
                              "duration_s": 11500.0}},                                        # a drop of 4420 s
           {**b, "trigger": {"kind": "wipe", "at_s": 900}}]
    for cfg in bad:
        with pytest.raises(ValueError):
            C.phases(cfg)


def test_load_verdict_judges_the_registered_windows(monkeypatch):
    """`load.verdict` itself, on a run with the burst config and with the control config, judges the end window
    4980-5580 against 4380-4980: the windows `judged_windows` names (REG-15's round-2 point 6)."""
    import analysis.load as L
    for cfg, base in ((C.burst_cfg(15, 5400.0), (300.0, 900.0)), (C.control_cfg(15, 5400.0), (480.0, 1080.0))):
        seen = []
        monkeypatch.setattr(L, "_level", lambda run, a, b, seen=seen: seen.append((a, b)) or
                            {"pf": 1.0, "hit": 0.8, "q": 0.0, "q_slope": 0.0, "n": 5.0, "kv": 0.2})
        L.verdict({"cfg": cfg, "metrics": pd.DataFrame(), "requests": pd.DataFrame(), "trigger": pd.DataFrame()})
        w = C.judged_windows(cfg)
        assert seen == [base, w["previous"], w["end"]]


def test_request_model_is_reg16s_and_reproduces_its_published_numbers():
    """`empty_time_requests` is `gate.reg16_empty_time` on a schedule built once: on REG-16's own design (seed 14,
    9000 s drop, λ = 2.2) it gives REG-16's registered 4340 s and 5540 s (decisions.md:3174, 3179)."""
    cfg, cal = G.reg16_burst_cfg(14, 9000), C.REG24["cal15"]
    s, _ = build_schedule(cfg["workload"], cfg["seed"], cfg["trigger"])
    for pess in (False, True):
        assert C.empty_time_requests(s, cfg, cal, pess) == G.reg16_empty_time(cfg, cal, pess)
    assert (C.empty_time_requests(s, cfg, cal), C.empty_time_requests(s, cfg, cal, True)) == (4340.0, 5540.0)
    # the owed backlog at the drop's start in the same arithmetic
    we = 5580.0
    tp = np.sort(s.t_planned.to_numpy(float))
    want = 28 + np.searchsorted(tp, we) - np.searchsorted(tp, 900.0) - cal["c_watch"] * (we - 900.0)
    assert C.owed_requests_model(s, cfg, cal, we) == pytest.approx(want)


def test_token_model_counts_planned_prompt_tokens_against_the_rate():
    s = pd.DataFrame({"t_planned": [905.0, 910.0, 5600.0, 5700.0], "prompt_len": [3000, 3000, 2000, 2000],
                      "burst": False})
    cfg = {"trigger": {"at_s": 900}, "workload": {"steps": [{"at_s": 5580.0}, {"at_s": 9180.0}]}}
    assert C.excess_tokens(s, 900.0, 5580.0, 1.0) == 6000 - 4680
    assert C.excess_tokens(s, 900.0, 5800.0, 0.0) == 10000
    assert C.empty_time_tokens(s, cfg, 10.0) == 0.0                      # 6000 tokens done long before the drop
    assert math.isnan(C.empty_time_tokens(s, cfg, 0.0))
    assert math.isnan(C.empty_time_tokens(s, cfg, 1.0))                  # 10000 tokens need 10000 s: past 9180 s
    assert C.empty_time_tokens(s, cfg, 1.25) == 8900.0 - 5580.0          # 10000 tokens at 1.25/s: done at 8900 s
    assert C.u_rate("pessimistic") == pytest.approx(0.95 * C.REG24["u_off"])
    assert C.u_rate("optimistic") > C.u_rate("expected") > C.u_rate("pessimistic")
    # a slower engine never empties sooner
    cfg2, cal = G.reg16_burst_cfg(14, 9000), C.REG24["cal15"]
    s2, _ = build_schedule(cfg2["workload"], cfg2["seed"], cfg2["trigger"])
    times = [C.empty_time_tokens(s2, cfg2, C.u_rate(k)) for k in ("optimistic", "expected", "pessimistic")]
    assert times[0] <= times[1] <= times[2]


def test_seed_rule_takes_the_lowest_seed_then_its_shortest_drop(monkeypatch):
    """Seeds first, then drops (REG-16 review N5): seed 41 passes only at 7200 s, seed 42 at 3600 s; 41 is taken."""
    monkeypatch.setattr(C, "candidate_seeds", lambda: (40, 41, 42))
    passes = {(41, 7200.0), (41, 9000.0), (42, 3600.0)}
    r = C.seed_rule(lambda sd, d: {"seed": sd, "drop_s": d, "ok": (sd, d) in passes})
    assert r["picked"] == {"seed": 41, "drop_s": 7200.0}
    assert [(x["seed"], x["drop_s"]) for x in r["rows"]] == \
        [(40, 3600.0), (40, 5400.0), (40, 7200.0), (40, 9000.0), (41, 3600.0), (41, 5400.0), (41, 7200.0)]
    assert C.seed_rule(lambda sd, d: {"seed": sd, "drop_s": d, "ok": False})["picked"] is None


def test_design_row_conditions(monkeypatch):
    """W, E and F on seed 15 (a used seed, not a candidate): each condition is what the row says it is."""
    r = C.design_row(15, 9000.0)
    tol = C.REG24["window_tol"] * C.REG24["lam"]
    assert r["windows_ok"] == all(abs(v - 2.0) <= tol + 1e-12 for v in r["lam_p"].values())
    assert r["not_exposed"] == (r["peak_120s"] < 3.2)
    worst = max(r["empty_req_pess"], r["empty_tok_pess"])
    assert r["drop_ok"] == bool(np.isfinite(worst) and worst + 1200.0 <= 9000.0)
    assert r["ok"] == (r["windows_ok"] and r["not_exposed"] and r["drop_ok"])
    assert r["empty_req"] <= r["empty_req_pess"] and r["empty_tok"] <= r["empty_tok_pess"]


# ---------------------------------------------------------------- outcome (synthetic runs)

def test_emptyings_need_30_s_of_zeros_inside_the_window():
    q = [5.0] * 100 + [0.0] * 40 + [3.0] * 20 + [0.0] * 25 + [2.0] * 15 + [0.0] * 50
    m = pd.DataFrame({"t": [float(i) for i in range(len(q))], "vllm:num_requests_waiting": q})
    run = {"metrics": m}
    assert C.emptyings(run, 0, 250) == [100.0, 200.0]          # the 25 s stretch at 160 s is too short
    assert C.emptyings(run, 0, 225) == [100.0]                  # 200 + 30 > 225: not inside the window
    assert C.emptyings(run, 110, 250) == [110.0, 200.0]        # a window starting inside a zero stretch
    assert C.emptyings(run, 0, 100) == []


@pytest.mark.parametrize("v5,broken,ew,ed,back,cold,want", [
    ("bad_baseline", False, False, True, True, True, "bad_baseline"),
    ("recovered", True, True, True, True, False, "not_persistent"),
    ("slow_drain", False, False, True, True, True, "not_persistent"),
    ("unclear", False, False, True, True, True, "inconclusive"),
    ("stuck", True, True, True, True, False, "reformed"),
    ("stuck", False, True, True, True, False, "emptied_in_watch"),
    ("stuck", False, False, True, True, None, "inconclusive"),
    ("stuck", False, False, True, True, False, "stuck_not_cold"),
    ("stuck", False, False, False, True, False, "stuck_not_cold"),
    ("stuck", False, False, False, True, True, "drop_inconclusive"),
    ("stuck", False, False, True, None, True, "inconclusive"),
    ("stuck", False, False, True, False, True, "no_return"),
    ("stuck", False, False, True, True, True, "confirms"),
])
def test_burst_outcome_order(v5, broken, ew, ed, back, cold, want):
    assert C.burst_outcome(v5, broken, ew, ed, back, cold=cold) == want
    assert want in C.BURST_LABELS


def test_burst_outcome_keeps_reg15s_order_where_the_tests_agree():
    """With no emptying in the watch, REG-24's labels are REG-15's for the same inputs (REG-15's line_reached_zero
    standing for the drop's emptying)."""
    for v5 in ("bad_baseline", "recovered", "slow_drain", "unclear", "stuck"):
        for broken in (False, True):
            for ed in (False, True):
                for back in (None, False, True):
                    assert C.burst_outcome(v5, broken, False, ed, back, cold=True) == G.reg15_outcome(v5, ed, back, broken)


@pytest.mark.parametrize("ctl,burst,want", [
    ("control_warm", "confirms", "confirms"),
    ("control_warm", "no_return", "no_return"),
    ("control_warm", "drop_inconclusive", "drop_inconclusive"),
    ("control_warm", None, "burst_missing"),
    ("control_cold_persistent", None, "control_not_warm"),
    ("control_unclear", "confirms", "control_not_warm"),
    ("bad_baseline", "confirms", "control_not_warm"),
    ("control_cold_transient", "not_persistent", "not_persistent"),
    ("control_warm", "reformed", "reformed"),
    (None, "emptied_in_watch", "emptied_in_watch"),
    ("control_warm", "stuck_not_cold", "stuck_not_cold"),
    (None, "confirms", "control_missing"),
])
def test_pair_outcome(ctl, burst, want):
    assert C.pair_outcome(ctl, burst) == want


def test_k3_band_labels():
    assert C.k3_band(2500.0, 2000.0, 3800.0) == "inside" and C.k3_band(2000.0, 2000.0, 3800.0) == "inside"
    assert C.k3_band(1500.0, 2000.0, 3800.0) == "outside_early"
    assert C.k3_band(4000.0, 2000.0, 3800.0) == "outside_late"
    assert C.k3_band(math.nan, 2000.0, 3800.0) == "undefined" and C.k3_band(2500.0, 2000.0, math.nan) == "undefined"


def test_check_pins_the_registered_configs(monkeypatch):
    monkeypatch.setitem(C.REG24, "seed", 15)
    cfgs = {"burst": C.burst_cfg(15, 5400.0), "control": C.control_cfg(15, 5400.0)}
    monkeypatch.setattr(C, "_resolved", lambda name: next(c for c in cfgs.values() if c["name"] == name)
                        if name in (C.REG24["burst_name"].format(seed=15), C.REG24["control_name"].format(seed=15))
                        else G._burst_cfg(name))
    run = lambda c, ok=True: {"dir": Path("x"), "checks": {"valid": ok}, "cfg": c}
    for kind, cfg in cfgs.items():
        C.check(run({**cfg, "run_id": "x", "source_config": "y"}), kind)       # run-time extras are fine
        for bad in ({**cfg, "seed": 16}, {**cfg, "workload": {**cfg["workload"], "rate": 2.2}},
                    {**cfg, "loadgen_force_close": False}):
            with pytest.raises(ValueError):
                C.check(run(bad), kind)
        with pytest.raises(ValueError):
            C.check(run(cfg, ok=False), kind)
    with pytest.raises(ValueError):                                           # the control without its no-trigger
        C.check(run({**cfgs["control"], "trigger": cfgs["burst"]["trigger"]}), "control")
    with pytest.raises(ValueError):
        C.check(run(cfgs["burst"]), "control")


def test_step_continues_only_after_a_warm_control(monkeypatch):
    monkeypatch.setattr(C, "_one", lambda kind: {"kind": kind})
    for label, want in (("control_warm", True), ("control_cold_transient", False), ("control_unclear", False),
                        ("bad_baseline", False), ("inconclusive", False)):
        monkeypatch.setattr(C, "control_outcome", lambda run, label=label: {"outcome": label, "baseline": {"pf": 1.0}})
        out = C.step()
        assert out == {"control_done": True, "continue": want}            # no label, level or count printed


def test_step_computes_no_readout(monkeypatch):
    """Rule F round 1, RE-O8: `step` computes only the control's outcome; a readout that fails cannot stop the queue."""
    cfg = C.control_cfg(15, 5400.0)
    run = _synthetic_run(cfg, lambda t: 0.0, lambda t: 0.84, kind="control")
    monkeypatch.setattr(C, "_one", lambda kind: run)

    def boom(*a, **k):
        raise RuntimeError("a readout ran")
    for name in ("surge_run", "window_mix", "reg15_timeline"):
        monkeypatch.setattr(G, name, boom)
    monkeypatch.setattr(C, "cotenant", boom)
    assert C.step() == {"control_done": True, "continue": True}
    with pytest.raises(RuntimeError):
        C.control_table(run)


def _synthetic_run(cfg, q_fn, hit_fn, kind="burst"):
    """A run on the REG-24 timeline: 1 s scrapes with waiting q_fn(t) and a hit share hit_fn(t); one session request
    per second, fast (ttft 0.2 s) where hit_fn(t) is high, slow (5 s) otherwise."""
    dur = float(cfg["workload"]["duration_s"])
    t = np.arange(0.0, dur + 1.0, 1.0)
    h = np.array([hit_fn(x) for x in t])
    m = pd.DataFrame({"t": t, "vllm:prefix_cache_queries_total": 100.0 * t,
                      "vllm:prefix_cache_hits_total": np.concatenate([[0.0], np.cumsum(100.0 * h[1:])]),
                      "vllm:num_requests_waiting": [q_fn(x) for x in t], "vllm:num_requests_running": 10.0,
                      "vllm:kv_cache_usage_perc": [0.2 if hit_fn(x) > 0.5 else 0.95 for x in t]})
    tp = np.arange(0.5, dur, 1.0)
    fast = np.array([hit_fn(x) > 0.5 for x in tp])
    r = pd.DataFrame({"t_planned": tp, "t_sent": tp, "t_released": tp, "turn": 1, "session_id": np.arange(len(tp)),
                      "status": "ok", "prompt_tokens_planned": 3000, "t_first_token": tp + np.where(fast, 0.2, 5.0)})
    r["t_end"] = r.t_first_token + 1.0
    r["ttft"] = r.t_first_token - r.t_released
    return {"dir": Path("x"), "cfg": cfg, "metrics": m, "requests": r, "trigger": pd.DataFrame(),
            "checks": {"valid": True}}


def test_burst_table_on_a_synthetic_confirming_run(monkeypatch):
    """A run cold from the burst to 6600 s, its line growing in the watch, emptying at 6600 s in the drop, warm after:
    `confirms`; the first emptying 1020 s into the drop; with the line emptying in the watch instead: emptied_in_watch;
    with no emptying in the drop: drop_inconclusive."""
    cfg = C.burst_cfg(15, 5400.0)
    monkeypatch.setitem(C.REG24, "seed", 15)
    monkeypatch.setitem(C.REG24, "drop_s", 5400.0)
    monkeypatch.setattr(C, "inputs", lambda seed, drop: {"empty_s": {"tokens": 900.0, "tokens_optimistic": 700.0,
                                                                     "tokens_pessimistic": 1500.0}})
    cold = lambda t: 0.84 if t < 900 or t >= 6600 else 0.0
    q_ok = lambda t: 0.0 if t < 900 or t >= 6600 else 50.0 + 0.01 * t
    grow = lambda r: _with_backlog(r, 900.0, 5580.0, 0.9, 1.0 + 468.0 / 1020.0)      # the total line grows, drains at 6600 s
    out = C.burst_table(grow(_synthetic_run(cfg, q_ok, cold)))
    assert out["v5"] == "stuck" and not out["watch_broken"]["broken"] and out["emptyings_watch"] == []
    assert out["cold_side"]["holds"] is True and out["cold_side"]["window"] == (1680.0, 5580.0)
    assert out["emptyings_drop"][0] == 6600.0 and out["first_empty_drop_s"] == 1020.0
    assert abs(out["drained_drop_s"] - 1020.0) <= 20.0
    assert out["return_recovered"] is True and out["outcome"] == "confirms" and out["k3"]["label"] == "inside"
    # readouts of draft 2 are present and never decisive (RE-O1, RE-O2, RE-O4)
    assert set(out["end_window_mix"]) == {"released_in_drop", "pf_by_sent_previous", "pf_by_sent_end"}
    assert out["end_window_mix"]["released_in_drop"] == 0.0                 # synthetic: released when planned
    assert set(out["cotenant"]) == {"mem_range_mib", "processes_at_launch", "processes_at_end"}
    assert "implied_r" in out["k3"]
    q_dip = lambda t: 0.0 if t < 900 or t >= 6600 or 3000 <= t < 3040 else 50.0 + 0.01 * t
    assert C.burst_table(grow(_synthetic_run(cfg, q_dip, cold)))["outcome"] == "emptied_in_watch"
    q_full = lambda t: 0.0 if t < 900 else 50.0
    always = lambda t: 0.84 if t < 900 else 0.0
    assert C.burst_table(grow(_synthetic_run(cfg, q_full, always)))["outcome"] == "drop_inconclusive"


def _with_backlog(run, a, we, c_watch, c_drop):
    """The synthetic run's session requests served first come first served from `a`: at c_watch per second up to `we`,
    then at c_drop until the line is gone, then at once (1.2 s). The total line (`gate.owed_backlog`) grows in the
    watch when c_watch is below the 1 req/s planned rate; the waiting gauge (q_fn) is left as it is."""
    r = run["requests"].copy()
    tp, te, free = r.t_planned.to_numpy(float), r.t_end.to_numpy(float).copy(), a
    for i in np.flatnonzero(tp >= a):
        c = c_watch if max(free, tp[i]) < we else c_drop
        free = max(free, tp[i]) + 1.0 / c
        te[i] = free if free > tp[i] + 1.2 else tp[i] + 1.2
    r["t_end"] = te
    return {**run, "requests": r}


def test_burst_table_rules_of_rule_f_round_1(monkeypatch):
    """RE1: a `stuck` verdict on a line that is not growing (completions keep up with the planned rate) is
    `stuck_not_cold`. RE2: a line that empties 20 s before the watch's end and stays empty into the drop emptied in the
    watch. RE3: a 40 s zero of the waiting gauge early in the drop while the total line is still owed is not the drain
    K3 measures."""
    cfg = C.burst_cfg(15, 5400.0)
    monkeypatch.setitem(C.REG24, "seed", 15)
    monkeypatch.setitem(C.REG24, "drop_s", 5400.0)
    monkeypatch.setattr(C, "inputs", lambda seed, drop: {"empty_s": {"tokens": 900.0, "tokens_optimistic": 700.0,
                                                                     "tokens_pessimistic": 1500.0}})
    cold = lambda t: 0.84 if t < 900 or t >= 6600 else 0.0
    q_ok = lambda t: 0.0 if t < 900 or t >= 6600 else 50.0 + 0.01 * t
    flat = C.burst_table(_synthetic_run(cfg, q_ok, cold))              # 1 completion per second against 1 planned
    assert flat["v5"] == "stuck" and flat["cold_side"]["holds"] is False and flat["outcome"] == "stuck_not_cold"
    grow = lambda r: _with_backlog(r, 900.0, 5580.0, 0.9, 1.0 + 468.0 / 1020.0)
    q_edge = lambda t: 0.0 if t < 900 or 5560 <= t < 5700 or t >= 6600 else 50.0 + 0.01 * t
    edge = C.burst_table(grow(_synthetic_run(cfg, q_edge, cold)))
    assert edge["v5"] == "stuck" and edge["emptyings_watch"] == [5560.0] and edge["outcome"] == "emptied_in_watch"
    q_flick = lambda t: 0.0 if t < 900 or 5800 <= t < 5840 or t >= 6600 else 50.0 + 0.01 * t
    fl = C.burst_table(grow(_synthetic_run(cfg, q_flick, cold)))
    assert fl["outcome"] == "confirms" and fl["first_empty_drop_s"] == 220.0
    assert abs(fl["drained_drop_s"] - 1020.0) <= 20.0 and fl["k3"]["measured_s"] == fl["drained_drop_s"]
    assert fl["k3"]["label"] == "inside" and fl["k3"]["label_first_empty"] == "outside_early"
    # a 40 s zero of the gauge in the drop while the total line never drains there (completions only keep up with the
    # planned rate), and a warm return: the draft's rule gave `confirms`; (g) now also needs the drain
    stay = lambda r: _with_backlog(r, 900.0, 5580.0, 0.9, 1.0)
    q_no = lambda t: 0.0 if t < 900 or 5800 <= t < 5840 or t >= 10980 else 50.0
    cold_no = lambda t: 0.84 if t < 900 or t >= 10980 else 0.0
    nd = C.burst_table(stay(_synthetic_run(cfg, q_no, cold_no)))
    assert nd["emptyings_drop"] == [5800.0] and math.isnan(nd["drained_drop_s"]) and nd["return_recovered"] is True
    assert nd["outcome"] == "drop_inconclusive"


def test_owed_series_is_gate_owed_backlog():
    cfg = C.burst_cfg(15, 5400.0)
    run = _with_backlog(_synthetic_run(cfg, lambda t: 0.0, lambda t: 0.84), 900.0, 5580.0, 0.9, 1.4)
    grid = np.array([0.0, 899.0, 1000.0, 3000.5, 5580.0, 6000.0, 7000.0, 12479.0])
    assert list(C.owed_series(run, grid)) == [G.owed_backlog(run, t) for t in grid]


def test_table_refuses_a_pair_from_two_builds(monkeypatch):
    import analysis.load as L
    runs = {"control": {"env": {"git_commit": "a" * 40}}, "burst": {"env": {"git_commit": "b" * 40}}}
    monkeypatch.setattr(C, "_one", lambda kind: runs[kind])
    monkeypatch.setattr(L, "find_runs", lambda pat: [runs["burst"]])
    monkeypatch.setattr(C, "control_table", lambda run: {"outcome": "control_warm"})
    monkeypatch.setattr(C, "burst_table", lambda run, ctl: {"outcome": "confirms"})
    monkeypatch.setattr(G, "reg19_baseline_pair", lambda c, b: {})
    assert C.table()["outcome"] == "not_same_build"
    runs["burst"]["env"]["git_commit"] = "a" * 40
    assert C.table()["outcome"] == "confirms"


def test_tilt_scan_on_the_pick_and_seed_61():
    """RE9 readout: the pick's end window carries 0.165 req/s more planned load than its return's end (the bound
    under W is 0.2); seed 61's two W-passing rows carry less."""
    d = C.tilt_scan(seeds=(43, 61))
    assert d["n_windows_ok"] == 3 and round(d["pick_tilt"], 3) == 0.165 and d["n_larger"] == 0
    assert round(d["max_possible"], 3) == 0.2
    assert [(s, dr, round(x, 3)) for s, dr, x in d["tilts"]] == [(43, 9000.0, 0.165), (61, 5400.0, 0.107),
                                                                 (61, 9000.0, 0.083)]
    assert d["n_positive"] == 3 and round(d["median"], 3) == 0.107


def test_released_after_counts_late_and_unsent_requests():
    r = pd.DataFrame({"t_planned": [10.0, 20.0, 30.0, 40.0, 99.0], "t_sent": [11.0, 60.0, np.nan, 41.0, 99.0],
                      "t_released": [11.0, 60.0, np.nan, np.nan, 99.0]})
    run = {"requests": r}
    assert C.released_after(run, 0.0, 50.0, 50.0) == 0.5     # 20 released at 60; 30 never sent; 10 and 40 before 50
    assert math.isnan(C.released_after(run, 200.0, 300.0, 250.0))


def test_cotenant_readout():
    cfg = C.burst_cfg(15, 5400.0)
    g = pd.DataFrame({"t": [400.0, 800.0, 2000.0, 3000.0, 6000.0, 7000.0],
                      "mem_used_mib": [20000.0, 20010.0, 20000.0, 21500.0, 20000.0, 20000.0]})
    ev = pd.DataFrame({"event": ["start", "gpu_processes_end"], "processes": [None, "123, python, 20000 MiB"]})
    out = C.cotenant({"cfg": cfg, "gpu": g, "events": ev, "env": {"gpu_processes": ""}})
    assert out["mem_range_mib"]["baseline"] == 10.0 and out["mem_range_mib"]["watch"] == 1500.0
    assert out["mem_range_mib"]["drop"] == 0.0 and math.isnan(out["mem_range_mib"]["return"])
    assert out["processes_at_end"] == "123, python, 20000 MiB" and out["processes_at_launch"] == ""
    empty = C.cotenant({"cfg": cfg})
    assert all(math.isnan(v) for v in empty["mem_range_mib"].values()) and empty["processes_at_end"] is None


def test_implied_r_inverts_the_token_model():
    """RE-O4: tokens planned at 905 s worth exactly 6680 s of work at r = 1 drain 2000 s into a drop starting at 5580 s;
    the implied r of a 2000 s drain is 1, of a later drain lower, and outside the bracket NaN."""
    u = C.REG24["u_off"]
    s = pd.DataFrame({"t_planned": [905.0], "prompt_len": [u * 6680.0], "burst": False})
    cfg = {"trigger": {"at_s": 900}, "workload": {"steps": [{"at_s": 5580.0}, {"at_s": 14580.0}]}}
    assert C.empty_time_tokens(s, cfg, u) == 2000.0
    assert C.implied_r(s, cfg, 2000.0) == pytest.approx(1.0, abs=1e-9)
    assert C.implied_r(s, cfg, 3000.0) == pytest.approx(6680.0 / 7680.0, abs=1e-6)     # 0.8698
    assert C.implied_r(s, cfg, 0.0) == pytest.approx(6680.0 / 4680.0, abs=1e-6)        # 1.427: drained at once
    assert math.isnan(C.implied_r(s, cfg, math.nan))
    assert math.isnan(C.implied_r(s, cfg, 9500.0))             # later than r = 0.5 predicts (8680 s): outside
    assert C.READOUT["r_bracket"] == (0.5, 1.5)


def test_readout_constants_are_the_studys():
    from analysis import k2
    assert C.READOUT["floor_8"] == k2.E0["floor_8"]
    assert "floor_8" not in C.REG24            # REG24 keeps the frozen rule's settings (RE7)


def test_control_table_on_synthetic_runs(monkeypatch):
    """REG-16's control outcome on REG-24's phases: warm throughout gives `control_warm`; a cold stretch from 4000 s
    to the watch's end gives `control_cold_persistent`."""
    cfg = C.control_cfg(15, 5400.0)
    monkeypatch.setattr(G, "surge_run", lambda run: {})
    warm = C.control_table(_synthetic_run(cfg, lambda t: 0.0, lambda t: 0.84, kind="control"))
    assert warm["outcome"] == "control_warm" and warm["v_watch"] == "recovered"
    tip = lambda t: 0.0 if 4000 <= t < 6000 else 0.84
    cold = C.control_table(_synthetic_run(cfg, lambda t: 40.0 if 4000 <= t < 6000 else 0.0, tip, kind="control"))
    assert cold["outcome"] == "control_cold_persistent"


# ---------------------------------------------------------------- the pick (after the rule ran once)

def test_registered_configs_are_the_design_at_the_pick():
    """The two config files resolve to `burst_cfg` / `control_cfg` at the rule's pick, and against each of REG-19's
    configs differ only in name, seed, workload.duration_s, workload.steps and verdict.watch_s."""
    import yaml
    sd, d = C.REG24["seed"], C.REG24["drop_s"]
    assert (sd, d) == (43, 9000.0)
    b = G._burst_cfg(C.REG24["burst_name"].format(seed=sd))
    c = G._burst_cfg(C.REG24["control_name"].format(seed=sd))
    assert b == C.burst_cfg(sd, d) and c == C.control_cfg(sd, d)
    for s19 in C.REG24["reg19_seeds"]:
        for mine, theirs in ((b, f"burst_l1pl350cl_r2.0_w3000_s{s19}"),
                             (c, f"stage2_control_l1pl350cl_r2.0_w3000_s{s19}")):
            x, y = FLAT(mine), FLAT(G._burst_cfg(theirs))
            assert sorted(k for k in x.keys() | y.keys() if x.get(k) != y.get(k)) == \
                ["name", "seed", "verdict.watch_s", "workload.duration_s", "workload.steps"]
    ph = C.phases(b)
    assert ph["drop"] == (5580, 14580) and ph["return_end"] == (15480, 16080) and C.phases(c) == ph
    assert sd not in C.used_seeds()                                  # REG-24's own configs are left out
    own = [f.name for f in C.CONFIGS.glob("*.yaml") if (yaml.safe_load(f.read_text()) or {}).get("seed") == sd]
    assert sorted(own) == sorted([C.REG24["burst_name"].format(seed=sd) + ".yaml",
                                  C.REG24["control_name"].format(seed=sd) + ".yaml"])


def test_seed_rule_picks_seed_43_at_9000_s(monkeypatch):
    """The rule as run once (2026-11-13 23:23:45 UTC): seeds 40-42 fail W at every drop; seed 43 fails W at 3600, 5400
    and 7200 s and passes W, E and F at 9000 s. The candidates are pinned to the ones the rule saw (40-199, all unused
    then), so that a later config setting a seed in 40-42 does not break this test (rule F round 1, RE-O3)."""
    monkeypatch.setattr(C, "candidate_seeds", lambda: tuple(range(40, 200)))
    r = C.seed_rule()
    assert r["picked"] == {"seed": 43, "drop_s": 9000.0} and len(r["rows"]) == 16
    assert [x["seed"] for x in r["rows"]] == [40] * 4 + [41] * 4 + [42] * 4 + [43] * 4
    assert not any(x["windows_ok"] for x in r["rows"][:-1]) and r["rows"][-1]["ok"]


def test_inputs_schedule_facts_at_the_pick():
    d = C.inputs()
    assert d["control_schedule_same"] and d["seeds"] == (43, 43)
    assert (d["n_session"], d["n_burst"]) == (23194, 463)
    lam = d["judged_lam_p"]
    assert all(abs(v - 2.0) <= 0.1 for v in lam.values())
    assert round(lam["end"], 3) == 2.068 and round(lam["return_end"], 3) == 1.903
    assert max(d["peak_120s"].values()) < C.REG24["surge_tip"]
    e = d["empty_s"]
    assert (e["requests"], e["requests_pessimistic"]) == (2490.0, 3550.0)
    assert (e["tokens_optimistic"], e["tokens"], e["tokens_pessimistic"]) == (2130.0, 2560.0, 4080.0)
    assert max(e["requests_pessimistic"], e["tokens_pessimistic"]) + C.REG24["warm_watch_s"] <= d["drop_s"]
    assert len(d["drop_by_600"]) == 15 and len(d["watch_spans_300"]) == 15
    pk = {k: round(v, 3) for k, v in d["peak_120s_inside_judged"].items()}
    assert (pk["end"], pk["return_end"]) == (2.408, 2.292)             # the heavier side is the one judged `stuck`
    assert round(d["lam_p_control_own_baseline"], 3) == 2.017
    # draft 2's readouts (rule F round 1): the cold side's window [1680, 5580) s (RE1) and the factor-two length (RE-O5)
    assert round(d["lam_p_cold_side"], 3) == 2.045
    assert round(d["lam_p"]["drop"], 5) == 0.99367 and round(d["factor_two_s"]) == 4859


def test_request_model_on_reg19s_schedules():
    """Readout pinned, not a gate (no tolerance is judged; the numbers were seen before this test was written): REG-15's
    calibration (λ = 2.2) on REG-19's five λ = 2.0 schedules, the owed backlog at 1680 and 4080 s (expected). The draft
    sets them beside the total line REG-19's [RESULT] reports (decisions.md 2026-11-13, cited, not read from runs/)."""
    rows = C.model_check_reg19()
    assert [r["seed"] for r in rows] == [15, 16, 17, 18, 19]
    assert [round(r[4080.0]["expected"]) for r in rows] == [1342, 1271, 1539, 1049, 1171]
    assert [round(r[1680.0]["expected"]) for r in rows] == [577, 601, 860, 655, 466]
    assert [round(r[4080.0]["pessimistic"]) for r in rows] == [1596, 1525, 1793, 1303, 1425]


# ---------------------------------------------------------------- rule F round 2 on draft 2

def _late_rewarm_run(cfg, t_rewarm, c_cold=0.85, c_warm=1.15, c_drop=1.4):
    """Rule F round 2, trace T-A: cold from the burst to `t_rewarm`, then partly warm (hit 0.72 against 0.84) and
    completing faster than the planned rate (1 req/s) to the watch's end, so that the total line shrinks over the
    watch's last windows while the closed-loop waiting gauge falls only slowly (it counts sessions in service, not
    owed turns); then the drop drains the line and the return is warm. Requests served behind a queue are slow."""
    we = 5580.0
    dur = float(cfg["workload"]["duration_s"])
    tp = np.arange(0.5, dur, 1.0)
    te, free = tp + 1.2, 900.0
    for i in np.flatnonzero(tp >= 900.0):
        s = max(free, tp[i])
        free = s + 1.0 / (c_cold if s < t_rewarm else c_warm if s < we else c_drop)
        te[i] = free if free > tp[i] + 1.2 else tp[i] + 1.2
    r = pd.DataFrame({"t_planned": tp, "t_sent": tp, "t_released": tp, "turn": 1, "session_id": np.arange(len(tp)),
                      "status": "ok", "prompt_tokens_planned": 3000, "t_end": te})
    owed0 = G.owed_backlog({"requests": r}, 900.0)
    t_drain = next(t for t in np.arange(we, dur, 10.0) if G.owed_backlog({"requests": r}, t) <= owed0)
    r["t_first_token"] = np.where(r.t_end - r.t_planned > 3.0, r.t_end - 1.0, r.t_planned + 0.2)
    r["ttft"] = r.t_first_token - r.t_released
    t = np.arange(0.0, dur + 1.0, 1.0)
    hit = lambda x: 0.84 if x < 900 or x >= t_drain else (0.0 if x < t_rewarm else 0.72 if x < we else 0.3)
    q = lambda x: (0.0 if x < 900 or x >= t_drain else 50.0 + 150.0 * (x - 900) / (t_rewarm - 900) if x < t_rewarm
                   else 200.0 - 0.033 * (x - t_rewarm) if x < we else max(1.0, 160.0 * (t_drain - x) / (t_drain - we)))
    h = np.array([hit(x) for x in t])
    m = pd.DataFrame({"t": t, "vllm:prefix_cache_queries_total": 100.0 * t,
                      "vllm:prefix_cache_hits_total": np.concatenate([[0.0], np.cumsum(100.0 * h[1:])]),
                      "vllm:num_requests_waiting": [q(x) for x in t], "vllm:num_requests_running": 10.0,
                      "vllm:kv_cache_usage_perc": 0.9})
    return {"dir": Path("x"), "cfg": cfg, "metrics": m, "requests": r, "trigger": pd.DataFrame(),
            "checks": {"valid": True}}


def test_end_window_cold_side_catches_a_late_rewarm(monkeypatch):
    """Rule F round 2, DK1 (trace T-A): a partial re-warm at λ from 4380 s. `load.verdict` stays `stuck` (no hit climb
    between its windows, pf 0 in both, the gauge's fall too slow for the 3600 s horizon) and the whole-window cold side
    holds (0.942 < 1), but the total line shrinks over the last 1200 s at a rate that would empty it within 3600 s.
    Draft 2 gave `confirms`; with the end window's cold side the label is `stuck_not_cold`. A run cold to the watch's
    end is unchanged."""
    cfg = C.burst_cfg(15, 5400.0)
    monkeypatch.setitem(C.REG24, "seed", 15)
    monkeypatch.setitem(C.REG24, "drop_s", 5400.0)
    monkeypatch.setattr(C, "inputs", lambda seed, drop: {"empty_s": {"tokens": 900.0, "tokens_optimistic": 700.0,
                                                                     "tokens_pessimistic": 1500.0}})
    run = _late_rewarm_run(cfg, 4380.0)
    out = C.burst_table(run)
    v = C.verdict(run)["end"]
    assert out["v5"] == "stuck" and v["q"] / -v["q_slope"] > 3600.0 and not out["watch_broken"]["broken"]
    assert out["cold_side"]["holds"] is True and round(out["cold_side"]["c"], 3) == 0.942
    assert out["cold_side_end"]["window"] == (4980.0, 5580.0) and out["cold_side_end"]["holds"] is False
    owed = lambda t: G.owed_backlog(run, t)
    shrink = (owed(4380.0) - owed(5580.0)) / 1200.0
    assert shrink > 0 and (owed(5580.0) - owed(900.0)) / shrink < 3600.0      # would empty at λ within the horizon
    assert out["outcome"] == "stuck_not_cold"
    cold_to_end = C.burst_table(_late_rewarm_run(cfg, 5580.0))
    assert cold_to_end["cold_side_end"]["holds"] is True and cold_to_end["outcome"] == "confirms"
    assert set(cold_to_end["owed"]) >= {"drop_end", "min_drop", "t_min_drop_s"}                         # DK2
    assert cold_to_end["owed"]["min_drop"] <= cold_to_end["owed"]["trigger"]
    assert cold_to_end["owed"]["t_min_drop_s"] >= cold_to_end["drained_drop_s"]


def test_emptyings_are_reg19s_on_reg19s_window():
    """Rule J (draft 2's ⚠ on "emptying"): `confirm.emptyings` over [burst end, end of arrivals) is `gate.reg19_emptyings`
    shifted to absolute time, so the two codes are one rule with the window as a parameter."""
    cfg = C.burst_cfg(15, 5400.0)
    zeros = [(1500, 1540), (1600, 1620), (2000, 2100), (2110, 2150), (12440, 12480), (12470, 12480)]
    q = lambda t: 0.0 if t < 900 or any(a <= t < b for a, b in zeros) else 5.0
    run = _synthetic_run(cfg, q, lambda t: 0.84)
    end, dur = G.burst_end(run), float(cfg["workload"]["duration_s"])
    mine = C.emptyings(run, end, dur)
    assert mine == [end + x for x in G.reg19_emptyings(run)] and mine == [1500.0, 2000.0, 2110.0, 12440.0]
    # the one difference in use: REG-24's pass ends at the drop's end, so a zero whose 30 s hold would cross it is left out
    assert C.emptyings(run, end, 12460.0) == [1500.0, 2000.0, 2110.0]


def test_table_gives_control_missing_without_a_valid_control(monkeypatch):
    import analysis.load as L
    monkeypatch.setattr(L, "find_runs", lambda pat: [])
    assert C.table()["outcome"] == "control_missing"


def test_table_gives_burst_missing_after_a_valid_control(monkeypatch):
    """Rule F round 2, DK-O1 (draft 3): a valid control and no valid burst run (its three attempts spent) give the
    control's table and `burst_missing` (or `control_not_warm`); no burst run is read."""
    import analysis.load as L
    ctl = {"env": {"git_commit": "a" * 40}}
    monkeypatch.setattr(L, "find_runs", lambda pat: [ctl] if "stage2_control" in pat else [])
    monkeypatch.setattr(C, "_one", lambda kind: ctl if kind == "control" else pytest.fail("a burst run was read"))
    monkeypatch.setattr(C, "burst_table", lambda run, c: pytest.fail("a burst table was computed"))
    monkeypatch.setattr(C, "control_table", lambda run: {"outcome": "control_warm"})
    out = C.table()
    assert out == {"control": {"outcome": "control_warm"}, "burst": None, "outcome": "burst_missing"}
    monkeypatch.setattr(C, "control_table", lambda run: {"outcome": "control_cold_persistent"})
    assert C.table()["outcome"] == "control_not_warm"
