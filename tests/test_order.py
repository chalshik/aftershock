"""The literal order test (analysis/order.py, draft) on hand-made runs: no run folder is read."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis.c2 as C
import analysis.order as O

VERDICT = {"baseline_s": 600.0, "watch_s": 1500.0, "end_s": 600.0, "stuck_hit_drop": 0.06}
BURST = {"trigger": {"kind": "burst", "at_s": 900.0, "duration_s": 180.0, "prompt_tokens": 2944}, "verdict": VERDICT}
WIPE = {"trigger": {"kind": "cache_wipe", "at_s": 900.0, "mode": "preempt"}, "verdict": VERDICT}
NEW = "20261115-120000_x"


@pytest.fixture(autouse=True)
def registered(monkeypatch):
    monkeypatch.setitem(O.ORDER, "registered_after", "20261115-000000")


def fake(cfg=BURST, cold_from=900.0, n_from=1000.0, kv_from=1100.0, rewarm_from=None, preempt_per_s=0.0,
         kv_before=0.3, kv_after=0.95, name=NEW, wipe_at=900.0):
    """8-turn sessions, one request every 0.5 s; a later turn reuses its previous prompt until `cold_from` (and again
    from `rewarm_from`), none after. N and KV step at n_from and kv_from. A burst config adds 40 one-off requests."""
    end = (900.0 + 180.0 if cfg["trigger"]["kind"] == "burst" else wipe_at + 0.5) + 1500.0
    rows, sid, t = [], 0, 0.0
    while t < end:
        for k in range(8):
            warm = t < cold_from or (rewarm_from is not None and t >= rewarm_from)
            p = 2176 + 256 * k
            rows.append({"session_id": sid, "turn": k, "t_planned": t, "t_first_token": t, "t_end": t + 0.3,
                         "prompt_tokens": p, "prompt_tokens_planned": p,
                         "cached_tokens": (p - 256) if (warm and k > 0) else 0, "status": "ok"})
            t += 0.5
        sid += 1
    if cfg["trigger"]["kind"] == "burst":
        for j, tb in enumerate(np.linspace(900.0, 1079.0, 40)):
            rows.append({"session_id": 10_000 + j, "turn": 0, "t_planned": tb, "t_first_token": tb + 1.0,
                         "t_end": tb + 5.0, "prompt_tokens": 2944, "prompt_tokens_planned": 2944, "cached_tokens": 0,
                         "status": "ok"})
    ts = np.arange(0.0, end, 1.0)
    m = pd.DataFrame({"t": ts, C.N: np.where(ts < n_from, 5.0, 60.0), C.KV: np.where(ts < kv_from, kv_before, kv_after),
                      O.PREEMPT: preempt_per_s * ts})
    trig = pd.DataFrame([{"t_start": wipe_at, "t_end": wipe_at + 0.5, "success": True}]) \
        if cfg["trigger"]["kind"] == "cache_wipe" else pd.DataFrame()
    return {"cfg": cfg, "requests": pd.DataFrame(rows), "metrics": m, "dir": Path(name), "trigger": trig}


def test_the_plans_order_follows():
    r = O.judge(fake())
    assert r["label"] == "follows" and r["failed"] == []
    on = r["onsets"]
    assert on["reuse"] <= on["u"] <= on["n"] <= on["kv"]


def test_the_burst_runs_order_on_record_does_not_follow():
    # as REG-22's onsets: N first, then the hits, then KV; the uncached rate last
    # REG-22's onsets on record: N with or before the hits, KV 30-40 s after, the uncached rate ~290 s after the hits
    run = fake(cold_from=930.0, n_from=925.0, kv_from=965.0)
    q = run["requests"]   # the burst's queue holds session first tokens back: 1 session in 4 is served in [930, 1210)
    held = (q.t_first_token >= 930.0) & (q.t_first_token < 1210.0) & (q.session_id < 10_000) & (q.session_id % 4 != 0)
    run["requests"] = q[~held].reset_index(drop=True)
    r = O.judge(run)
    assert r["label"] == "does_not_follow" and "u>n" in r["failed"] and r["u_only"], r


def test_a_wipe_has_no_duration_and_its_watch_starts_at_the_wipe():
    w = O.windows(fake(cfg=WIPE))
    assert w["at"] == 900.0 and w["end"] == 900.5 and w["watch_end"] == 2400.5 and w["attempt"] == (900.0, 900.5)
    assert O.judge(fake(cfg=WIPE))["label"] == "follows"


@pytest.mark.parametrize("t0", [900.0, 903.0, 905.0, 917.0, 930.0, 1200.0])
def test_a_step_of_all_four_at_once_is_not_contradicted(t0):
    # the documented limit: without tol_s, a step at one instant reads 'u>n' at some phases (900, 930)
    r = O.judge(fake(cold_from=t0, n_from=t0, kv_from=t0))
    assert r["label"] == "follows", r


def test_a_reversal_beyond_the_resolution_is_caught():
    # N and KV step 80 s before the hits drop: more than tol_s
    r = O.judge(fake(cold_from=1000.0, n_from=920.0, kv_from=920.0))
    assert r["label"] == "does_not_follow" and "u>n" in r["failed"]


def test_hits_that_come_back_fail_item_5():
    r = O.judge(fake(rewarm_from=1600.0))
    assert r["label"] == "does_not_follow" and r["failed"] == ["stays_low"]


def test_frequent_preemptions_fail():
    r = O.judge(fake(preempt_per_s=2.0))
    assert r["label"] == "does_not_follow" and r["failed"] == ["preemptions"]


def test_a_counter_that_goes_down_is_undefined_not_low():
    run = fake()
    run["metrics"].loc[2000, O.PREEMPT] = -1.0
    assert O.judge(run)["label"] == "undefined"


def test_a_signal_already_at_its_level_before_the_trigger_is_undefined():
    r = O.judge(fake(kv_before=0.95))
    assert r["label"] == "undefined" and r["onsets"]["pre"] == ["kv"]


def test_a_signal_that_never_crosses_is_undefined():
    assert O.judge(fake(kv_from=1e9))["label"] == "undefined"


def test_a_run_made_before_the_registration_is_refused():
    for name in ("20261114-235959_x", "fake", "2026111-120000_x"):
        with pytest.raises(ValueError, match="before the order test"):
            O.judge(fake(name=name))


def test_another_trigger_kind_is_refused():
    with pytest.raises(ValueError, match="burst and cache_wipe only"):
        O.windows({"cfg": {"trigger": {"kind": "ramp", "at_s": 0}, "verdict": VERDICT}})


def test_the_registered_constants_are_c2s():
    assert O.ORDER["plan"] == C.PLAN_ORDER and O.ORDER["max_preempt_frac"] == 0.01
    assert O.ORDER["tol_s"] == C.C2["roll_s"] + C.C2["step_s"]
    assert not math.isnan(C.C2["kv_onset"])


def test_every_pair_is_checked_not_only_neighbours():
    # the hits drop 60 s after N, and N 110 s after KV: only the pair (reuse, KV) is beyond tol_s
    r = O.judge(fake(cold_from=1120.0, n_from=1060.0, kv_from=950.0))
    assert r["label"] == "does_not_follow" and "reuse>kv" in r["failed"] and "reuse>n" not in r["failed"], r


def test_without_tol_a_simultaneous_step_reads_out_of_order_at_4_of_6_phases(monkeypatch):
    monkeypatch.setitem(O.ORDER, "tol_s", 0.0)
    labels = [O.judge(fake(cold_from=t, n_from=t, kv_from=t))["label"] for t in (900.0, 903.0, 905.0, 917.0, 930.0, 1200.0)]
    assert labels.count("does_not_follow") == 4


def test_a_wipe_is_timed_from_its_attempt_not_from_at_s():
    # rule F T1: the wipe ran at 1100 though at_s says 900; the cache went cold at 1100
    r = O.judge(fake(cfg=WIPE, wipe_at=1100.0, cold_from=1100.0, n_from=1200.0, kv_from=1300.0))
    assert O.windows(fake(cfg=WIPE, wipe_at=1100.0))["at"] == 1100.0 and r["label"] == "follows", r


def test_a_wipe_without_a_successful_attempt_has_no_origin():
    run = fake(cfg=WIPE)
    run["trigger"]["success"] = False
    with pytest.raises(ValueError, match="no origin"):
        O.windows(run)


def test_a_wipes_own_preemptions_are_left_out():
    # rule F T2: a jump of 31 at the wipe itself, nothing else
    run = fake(cfg=WIPE)
    run["metrics"].loc[run["metrics"].t >= 901.0, O.PREEMPT] = 31.0
    r = O.judge(run)
    assert r["preempt"]["excluded"] == 31.0 and r["preempt"]["count"] == 0.0 and r["label"] == "follows", r


def test_a_scrape_gap_before_an_onset_is_undefined_not_follows():
    # rule F T3: a 60 s hole in the scrapes before N's onset
    run = fake(cold_from=1000.0, n_from=1000.0, kv_from=1000.0)
    run["metrics"] = run["metrics"][(run["metrics"].t < 940.0) | (run["metrics"].t >= 1000.0)].reset_index(drop=True)
    r = O.judge(run)
    assert r["label"] == "undefined" and set(r["onsets"]["unresolved"]) >= {"n", "kv"}, r


def test_kv_between_080_and_090_has_an_onset():
    # rule F T6 / R9: the onset level is condition 4's min_kv
    r = O.judge(fake(kv_after=0.85))
    assert not math.isnan(r["onsets"]["kv"]) and r["label"] == "follows", r


def _metrics_with_idle(tmp_path, base_mean, end_mean, n_each=30):
    rows, cs, cn = [], 0.0, 0.0
    for t in np.arange(0.0, 2581.0, 1.0):
        if 300.0 <= t < 900.0:
            cs, cn = cs + base_mean * n_each / 600.0, cn + n_each / 600.0
        if 1980.0 <= t < 2580.0:
            cs, cn = cs + end_mean * n_each / 600.0, cn + n_each / 600.0
        rows.append({"t": float(t), "values": {O.IDLE + "_sum": cs, O.IDLE + "_count": cn}})
    (tmp_path / "metrics.jsonl").write_text("\n".join(json.dumps(r) for r in rows))


@pytest.mark.parametrize("end_mean,n_each,sooner,label", [(5.0, 30, True, "follows"), (90.0, 30, False, "does_not_follow"),
                                                          (5.0, 10, None, "follows")])
def test_evicted_sooner_is_judged_only_when_measurable(tmp_path, end_mean, n_each, sooner, label):
    d = tmp_path / NEW
    d.mkdir()
    _metrics_with_idle(d, 60.0, end_mean, n_each)
    run = fake()
    run["dir"] = d
    r = O.judge(run)
    assert r["evictions"]["sooner"] is sooner and r["label"] == label, r
    if sooner is False:
        assert r["failed"] == ["evictions"]


def test_table_judges_only_valid_stuck_runs(monkeypatch):
    runs = {"20261115-120000_a": ("stuck", True), "20261115-120001_b": ("recovered", True), "20261115-120002_c": ("stuck", False)}
    def load(p):
        r = fake(name=Path(p).name)
        r["checks"] = {"valid": runs[Path(p).name][1]}
        return r
    monkeypatch.setattr(O, "load_run", load)
    monkeypatch.setattr(O, "verdict", lambda r: {"verdict": runs[r["dir"].name][0]})
    labels = [x["label"] for x in O.table(list(runs))]
    assert labels == ["follows", "not_judged (recovered)", "invalid"]


def test_a_reversal_of_n_and_kv_is_not_u_only():
    r = O.judge(fake(cold_from=1000.0, n_from=1200.0, kv_from=920.0))
    assert "n>kv" in r["failed"] and not r["u_only"], r


def test_n_or_kv_that_fall_back_fail_stays_high():
    run = fake()
    m = run["metrics"]
    m.loc[m.t >= 2000.0, C.N] = 5.0
    r = O.judge(run)
    assert r["label"] == "does_not_follow" and r["failed"] == ["stays_high"] and not r["u_only"], r


@pytest.mark.parametrize("times,failed,u_only", [((1100.0, 900.0, 1150.0, 1160.0), ["reuse>u"], False),
                                                    ((900.0, 1300.0, 950.0, 960.0), ["u>n", "u>kv"], True)])
def test_u_only_is_u_crossing_late_not_early(monkeypatch, times, failed, u_only):
    # rule F round 2 O1: the held-first-token artefact makes u late; u crossing early is not it
    on = dict(zip(O.ORDER["plan"], times), pre=[], unresolved=[])
    monkeypatch.setattr(O, "onsets", lambda run, base, w: on)
    r = O.judge(fake())
    assert r["failed"] == failed and r["u_only"] is u_only, r


def test_a_sustained_wipe_is_timed_from_its_first_call_to_its_last():
    run = fake(cfg=WIPE, wipe_at=900.0, cold_from=900.0, n_from=1000.0, kv_from=1100.0)
    run["trigger"] = pd.DataFrame({"t_start": [900.0, 905.0, 1195.0], "t_end": [900.5, 905.5, 1195.5],
                                   "success": [True, True, True]})
    w = O.windows(run)
    assert w["at"] == 900.0 and w["end"] == 1195.5 and w["attempt"] == (900.0, 1195.5)
    m = run["metrics"]
    m[O.PREEMPT] = np.where(m.t >= 901.0, 40.0, 0.0) + np.where(m.t >= 1196.0, 40.0, 0.0)   # each call preempts 40
    pe = O.preemptions(run, w)
    assert pe["excluded"] == 80.0 and pe["count"] == 0.0
