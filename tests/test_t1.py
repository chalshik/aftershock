"""analysis/t1.py: T1 drafts 3-6, arms U (uncapped twin) and P (pause cap 22, switched at the gate's from_s). Synthetic runs
and configs only; nothing here reads a run folder, and neither the gate nor any readout is computed on a real run."""
import fnmatch
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import d1 as D
from analysis import d1r as R
from analysis import t1 as T
from analysis import withheld as W

ARMS = ("U", "P")
FS = 2580.0   # a candidate used by the synthetic traces; not a choice (the gate is never computed here)


# ---------------------------------------------------------------- constants

def test_constants_are_draft_3s():
    t = T.T1
    assert t["cap"] == 22 and t["from_s_candidates"] == (1680.0, 1980.0, 2280.0, 2580.0, 2880.0)
    assert (t["span_s"], t["hold_spans"], t["prompt_s"], t["lead_min_s"]) == (120.0, 2, 240.0, 300.0)
    assert (t["x_hi"], t["x_cold"], t["gate_k"], t["most"], t["ws_margin_per_slot"]) == (3.01, 1.75, 3, 3, 256)
    from analysis.reg30 import REG30
    assert t["lead_min_s"] == REG30["lead_min_s"] and t["seeds"] == (15, 16, 17, 18, 19)
    assert t["from_s"] == 1980.0                                      # set in the [REG], from the gate (once)
    assert t["p_seeds"] == (15, 17, 19)                               # P where the gate separates (author 2026-11-23)
    assert T.hold_s() == 240.0 and T.cold_fit_by(4080.0) == 4080.0 - 480.0
    assert t["band_tol"] == 0.15                                      # draft 5, Band
    assert "fit_wait_s" not in t                                      # draft 6, B1: t_fit's waiting clause is gone
    assert t["x_cold"] == D.D1["x_cold"] == 1.75                      # REG-19's rate over W, d1's constant


def _turn_prompts(cfg):
    """Each session turn's planned prompt from the workload config: system + (k + 1) user + k reply tokens."""
    s = cfg["workload"]["sessions"]
    return [s["system_tokens"] + (k + 1) * s["user_tokens"] + k * s["reply_tokens"] for k in range(s["turns"])]


def test_cap_22_is_the_largest_whose_turn_7_working_set_fits_the_pool():
    from harness.workload import build_schedule
    cfg = T.resolved("P", 15, FS)
    c = cfg["checks"]["expected_kv_tokens"]
    assert c == 96656
    last = max(_turn_prompts(cfg))                                     # turn 7's prompt, from the workload config
    sched, _ = build_schedule(cfg["workload"], cfg["seed"], cfg["trigger"])
    sess = sched[~sched.burst.astype(bool)]
    assert sess.groupby("turn").prompt_len.max().tolist() == _turn_prompts(cfg)   # the harness agrees, turn by turn
    assert last == 3968                                                # the draft's number, now derived
    per = last + T.T1["ws_margin_per_slot"]                            # turn 7's prompt + the margin (draft 3 §1)
    assert T.T1["cap"] == math.floor(c / per) and T.T1["cap"] * per <= c < (T.T1["cap"] + 1) * per


def test_verdict_thresholds_come_from_the_run_config():
    v = T.resolved("P", 15, FS)["verdict"]
    assert (v["recovered_hit_drop"], v["recovered_q_excess"], v["stuck_hit_drop"], v["min_baseline_pf"]) == \
        (0.06, 1.0, 0.06, 0.98)


# ---------------------------------------------------------------- configs

def test_u_is_reg19s_burst_config_renamed_and_p_differs_from_u_only_in_the_three_cap_keys():
    for sd in T.T1["seeds"]:
        reg19 = D._burst_cfg(sd)
        u, p = T.resolved("U", sd), T.resolved("P", sd, FS)
        assert D._diff_keys(reg19, u) == ["name"] and u["name"] == T.config_name("U", sd)
        assert D._diff_keys(u, p) == ["name", "workload.sessions.cap", "workload.sessions.cap_from_s",
                                      "workload.sessions.cap_mode"]
        s = p["workload"]["sessions"]
        assert (s["cap"], s["cap_from_s"], s["cap_mode"]) == (22, FS, "pause")


def test_u_p_and_reg19_give_the_same_schedule_row_for_row():
    from harness.workload import build_schedule
    for sd in T.T1["seeds"]:
        r = D._burst_cfg(sd)
        sr, pr = build_schedule(r["workload"], r["seed"], r["trigger"])
        for k in (T.resolved("U", sd), T.resolved("P", sd, FS)):
            sk, pk = build_schedule(k["workload"], k["seed"], k["trigger"])
            pd.testing.assert_frame_equal(sk, sr)
            assert all((a == b).all() for a, b in zip(pk.tokens, pr.tokens))


def test_the_committed_u_configs_are_the_written_text_and_resolve_as_the_harness_reads_them():
    from harness.config import load_config, with_defaults
    from harness.run import expand
    for sd in T.T1["seeds"]:
        p = T.CONFIGS / f"{T.config_name('U', sd)}.yaml"
        assert p.read_text() == T.config_text("U", sd)
        (got,) = expand(with_defaults(load_config(p)))
        assert D._diff_keys(got, T.resolved("U", sd)) == [] and got["name"] == p.stem


def test_p_needs_a_candidate_switch_time_and_u_none():
    for bad in (None, 1700.0, 0.0):
        with pytest.raises(ValueError):
            T.config_text("P", 15, bad)
    with pytest.raises(ValueError):
        T.config_text("U", 15, FS)
    with pytest.raises(ValueError):
        T.config_text("A", 15)                                         # arm A is gone
    with pytest.raises(ValueError):
        T.config_text("U", 20)
    assert "cap_from_s: 2880\n" in T.config_text("P", 17, 2880)


def test_write_configs_writes_new_files_only(tmp_path):
    u = T.write_configs("U", configs_dir=tmp_path)
    p = T.write_configs("P", FS, configs_dir=tmp_path)
    assert len(u) == 5 and len(p) == 3
    assert {x.name for x in u} == {f"{T.config_name("U", s)}.yaml" for s in T.T1["seeds"]}
    assert {x.name for x in p} == {f"{T.config_name("P", s)}.yaml" for s in T.T1["p_seeds"]}
    with pytest.raises(FileExistsError):
        T.write_configs("U", configs_dir=tmp_path)
    with pytest.raises(ValueError):
        T.write_configs("P", configs_dir=tmp_path / "x")


def test_no_arm_a_left():
    src = (T.REPO / "analysis" / "t1.py").read_text()
    for gone in ("rescue_adm", "def decision", "def paired", "is_fast", "min_n", "PREDICTED", "def feasibility"):
        assert gone not in src, gone
    assert not (T.CONFIGS / "burst_l1pl350cl_r2.0_w3000_s15_rescue_adm.yaml").exists() or \
        "rescue_adm" not in __import__("subprocess").run(["git", "ls-files", "configs"], cwd=T.REPO,
                                                          capture_output=True, text=True).stdout


# ---------------------------------------------------------------- run order

def test_run_order_is_the_rotation():
    o = T.T1["order"]
    assert o == ((15, "P"), (15, "U"), (16, "U"), (17, "P"), (17, "U"), (18, "U"), (19, "P"), (19, "U"))
    assert [sd for sd, a in o if a == "U"] == list(T.T1["seeds"]) and [sd for sd, a in o if a == "P"] == list(T.T1["p_seeds"])
    for sd in T.T1["p_seeds"]:                                       # each P adjacent to its own U
        i = o.index((sd, "P"))
        assert (sd, "U") in (o[i - 1] if i else None, o[i + 1] if i + 1 < len(o) else None)


# ---------------------------------------------------------------- seal and patterns

def _names(arm):
    return [f"20261123-000000_{T.config_name(arm, sd)}" for sd in T.T1["seeds"]]


def test_seal_rows_are_draft_3s():
    rows = [r for r in W.SEALS if r[0] == "T1"]
    assert rows == [("T1", "*_burst_l1pl350cl_r2.0_w3000_s1[5-9]_t1p", T.T1["seal"]),
                    ("T1", "*_burst_l1pl350cl_r2.0_w3000_s1[5-9]_t1u", T.T1["seal"])]
    assert ("D1r", "*_burst_l1pl350cl_r2.0_w3000_s1[5-9]_rescue", "[RESULT] D1r (C2, cap as a rescue)") in W.SEALS


def test_seal_covers_every_t1_run_and_matches_no_other_registrations_names():
    from analysis import d1b, gate, k1, k3, k3w
    t1_names = _names("U") + _names("P")
    for n in t1_names:
        assert [r[0] for r in W.SEALS if fnmatch.fnmatchcase(n, r[1])] == ["T1"]
    others = ([f"20261123-000000_{x}" for sd in T.T1["seeds"] for x in
               (f"burst_l1pl350cl_r2.0_w3000_s{sd}", f"stage2_control_l1pl350cl_r2.0_w3000_s{sd}", D.config_name(sd),
                R.config_name(sd), f"burst_l1pl350ol_r2.0_w3000_s{sd}")]
              + [f"20261123-000000_{d1b.config_name(sd)}" for sd in (15, 16, 17)]
              + ["20261123-000000_burst_l1pl350cl_r2.0_w3000_s15_drift"])
    t1_pats = [r[1] for r in W.SEALS if r[0] == "T1"]
    assert not [(n, p) for n in others for p in t1_pats if fnmatch.fnmatchcase(n, p)]
    reg19 = next(r[1] for r in W.SEALS if r[0] == "REG-19")
    assert not any(fnmatch.fnmatchcase(n, reg19) for n in t1_names)       # REG-19's pattern stays off `..._t1u`
    selecting = ([gate.reg19_spec(s)[k] for s in T.T1["seeds"] for k in ("control", "burst")]
                 + [k1.closed_pattern(s) for s in T.T1["seeds"]] + [k1.open_pattern(2.0, s) for s in T.T1["seeds"]]
                 + [k3.REG21["batch2_pattern"], k3w.REG21["batch2_pattern"], D.D1["drift_pattern"]]
                 + [D.run_pattern(s) for s in T.T1["seeds"]] + [R.run_pattern(s) for s in T.T1["seeds"]]
                 + [d1b.run_pattern(s) for s in (15, 16, 17)] + [p for reg, p, _ in W.SEALS if reg != "T1"])
    assert not [(n, p) for n in t1_names for p in selecting if fnmatch.fnmatchcase(n, p)]
    for arm in ARMS:
        assert all(fnmatch.fnmatchcase(n, T.run_pattern(arm, sd)) for n, sd in zip(_names(arm), T.T1["seeds"]))


def test_which_broad_globs_take_in_t1_folders_once_the_seal_lifts():
    """As D1's test: `gate.SURGE["pattern"]` and k3's batch-1 glob match T1's names; the seal keeps them out until
    [RESULT] T1. Not a pattern that selects a registration's runs by name."""
    from analysis import gate
    for n in _names("U") + _names("P"):
        assert fnmatch.fnmatchcase(n, gate.SURGE["pattern"]) and fnmatch.fnmatchcase(n, "*_l1pl350cl_*")


def test_only_its_result_lifts_it(tmp_path, monkeypatch):
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-23 · [REG] T1 (rescue, two arms): waits for [RESULT] T1 (rescue, two arms)\n")
    monkeypatch.setattr(W, "DECISIONS", d)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    names = _names("U") + _names("P")
    assert [W.sealed_by(n) for n in names] == ["T1"] * 10
    d.write_text(d.read_text() + "## 2026-11-24 · [RESULT] T1 (rescue, two arms): x\n")
    assert [W.sealed_by(n) for n in names] == [None] * 10


def test_table_refuses_without_from_s_and_while_sealed(tmp_path, monkeypatch):
    import analysis.load as L
    monkeypatch.setitem(T.T1, "from_s", None)                       # the code base before the [REG]
    with pytest.raises(ValueError):
        T.table()
    (tmp_path / _names("U")[0]).mkdir()
    (tmp_path / f"20261121-000000_{R.config_name(15)}").mkdir()        # D1r's folder is not T1's
    monkeypatch.setattr(L, "RUNS", tmp_path)
    monkeypatch.setattr(W, "lifted", lambda *a, **k: False)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    monkeypatch.setitem(T.T1, "from_s", FS)
    assert T.sealed() == [_names("U")[0]]
    with pytest.raises(W.WithheldError):
        T.table()


def test_no_environment_access():
    src = (T.REPO / "analysis" / "t1.py").read_text()
    assert "os.environ" not in src and "putenv" not in src and "import os" not in src and "getenv" not in src


# ---------------------------------------------------------------- slots held and the pause invariant (draft 2's)

COLS = ["session_id", "turn", "t_arrived", "t_admitted", "t_sent", "t_end", "t_paused", "t_resumed", "slots_at_pause",
        "slots_at_resume"]
N = math.nan
# cap 1, switch at 10: sessions 0 and 1 admitted before it; 0 pauses at its turn 1's end (11 s, two holding a slot);
# 1 ends at 15 and hands its slot to 0 (resume first); 2 arrives at 11 and is admitted when 0 ends at 18.
GOOD = [(0, 0, 1, 1, 1, 5, N, N, N, N), (0, 1, N, N, 6, 11, N, N, N, N), (0, 2, N, N, 15, 18, 11, 15, 1, 1),
        (1, 0, 2, 2, 2, 12, N, N, N, N), (1, 1, N, N, 13, 15, N, N, N, N), (2, 0, 11, 18, 18, 20, N, N, N, N)]


def _run(rows, pause=True, dur=40.0):
    q = pd.DataFrame(rows, columns=COLS)
    if pause:
        q["resume_queue_at_pause"] = np.where(q.t_paused.notna(), 1, np.nan)
        q["resume_queue_at_resume"] = np.where(q.t_resumed.notna(), 0, np.nan)
    else:
        q = q.drop(columns=COLS[6:])
    q["request_id"] = range(len(q))
    q["prompt_tokens_planned"] = 2176
    return {"requests": q, "cfg": {"workload": {"duration_s": dur}, "trigger": {"kind": "burst", "prompt_tokens": 64}}}


def _with(i, **kw):
    rows = [list(r) for r in GOOD]
    for k, v in kw.items():
        rows[i][COLS.index(k)] = v
    return [tuple(r) for r in rows]


def test_slots_held_and_the_resume_queue_on_a_correct_pause_run():
    run = _run(GOOD)
    t = np.array([0.5, 3.0, 10.5, 11.0, 12.5, 14.9, 15.0, 17.0, 18.0, 19.0, 21.0])
    assert T.slots_held(run, t).tolist() == [0, 2, 2, 1, 1, 1, 1, 1, 1, 1, 0]
    assert R.in_service(run, t).tolist() == [0, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0]   # the first computation counts 0's pause
    assert T.resume_queue(run, t).tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0]
    sl = T.slots_paused(run, 1, 10.0)
    assert sl["fields"] and sl["pauses"] == 1 and sl["resumes"] == 1 and sl["handovers"] == 2
    assert (sl["pauses_before"], sl["pauses_under"], sl["handovers_over"], sl["resumes_reported_over"]) == (0, 0, 0, 0)
    assert sl["idle_share"] == 0.0 and sl["max_wait_before_s"] == 0.0 and sl["t_at_cap"] == 11.0
    assert not T.pause_fault(sl)
    h = T.held_spans(run)
    assert h.values.tolist() == [[0, 1.0, 11.0], [0, 15.0, 18.0], [1, 2.0, 15.0], [2, 18.0, 20.0]]


def test_t_cap_is_slots_paused_t_at_cap():
    run = _run(GOOD)
    run["cfg"]["workload"]["duration_s"] = 40.0
    import unittest.mock as m
    with m.patch.dict(T.T1, {"cap": 1}):
        assert T.t_cap(run, 10.0) == T.slots_paused(run, 1, 10.0)["t_at_cap"] == 11.0


def test_without_pauses_slots_held_is_in_service():
    run = _run([(0, 0, 1, 1, 1, 5, N, N, N, N), (0, 1, N, N, 6, N, N, N, N, N), (1, 0, 2, 2, 2, 12, N, N, N, N)])
    t = np.arange(0.0, 30.0, 0.5)
    assert T.slots_held(run, t).tolist() == R.in_service(run, t).tolist()
    bare = _run([(0, 0, 1, 1, 1, 5, N, N, N, N), (1, 0, 2, 2, 2, 12, N, N, N, N)], pause=False)
    assert T.slots_held(bare, t).tolist() == R.in_service(bare, t).tolist()
    assert (T.resume_queue(bare, t) == 0).all()


@pytest.mark.parametrize("rows,key", [
    (_with(2, t_paused=9.0), "pauses_before"),
    (_with(2, slots_at_pause=0.0), "pauses_under"),
    (_with(2, slots_at_resume=2.0), "resumes_reported_over"),
    (_with(5, t_admitted=13.0, t_sent=13.0, t_end=14.0), "handovers_over"),
    (_with(2, t_resumed=12.5, t_sent=12.5), "handovers_over"),
])
def test_each_pause_fault_is_flagged(rows, key):
    sl = T.slots_paused(_run(rows), 1, 10.0)
    assert sl[key] >= 1 and T.pause_fault(sl), sl


def test_pause_fault_on_missing_fields_a_wait_before_the_switch_and_an_idle_slot():
    assert T.pause_fault(T.slots_paused(_run(GOOD, pause=False), 1, 10.0))
    waited = [(1, 0, 2, 6, 6, 12, N, N, N, N) if r[:2] == (1, 0) else r for r in GOOD]
    sl = T.slots_paused(_run(waited), 1, 10.0)
    assert sl["max_wait_before_s"] == 4.0 and T.pause_fault(sl)
    idle = _with(2, t_resumed=30.0, t_sent=30.0, t_end=31.0)
    idle = [(2, 0, 11, 32, 32, 33, N, N, N, N) if r[:2] == (2, 0) else r for r in idle]
    sl = T.slots_paused(_run(idle), 1, 10.0)
    assert sl["idle_share"] > T.T1["max_idle_share"] and T.pause_fault(sl)


def test_u_fault_on_cap_or_pause_fields():
    assert T.u_fault(_run(GOOD)) and T.u_fault(_run(GOOD, pause=False))       # t_arrived alone is a cap field
    bare = _run(GOOD, pause=False)
    bare["requests"] = bare["requests"].drop(columns=["t_arrived", "t_admitted"])
    assert not T.u_fault(bare)


def test_slots_held_matches_loadgens_own_count_on_a_virtual_clock_run(monkeypatch, tmp_path):
    from tests import test_cap_pause as CP
    q, log = CP._run_synthetic(monkeypatch, tmp_path)
    run = {"requests": q.reset_index(drop=True), "cfg": {"workload": {"duration_s": 60.0}, "trigger": None}}
    times = sorted({t for t, *_ in log})
    mids = np.array([(a + b) / 2 for a, b in zip(times, times[1:])])
    used = [next(u for t, u, *_ in reversed(log) if t <= m) for m in mids]
    queued = [next(r for t, _, r, *_ in reversed(log) if t <= m) for m in mids]
    assert T.slots_held(run, mids).tolist() == used
    assert T.resume_queue(run, mids).tolist() == queued
    sl = T.slots_paused(run, CP.CAP, CP.FROM)
    assert sl["pauses"] == 4 and sl["resumes"] == 4 and not T.pause_fault(sl), sl


# ---------------------------------------------------------------- ws_slot on a hand-built session set

def _ws_run(over=None):
    run = _run(GOOD)
    q = run["requests"]
    q["prompt_tokens"] = [100.0, 200.0, 300.0, 1000.0, 2000.0, 10000.0]
    for (i, col), v in (over or {}).items():
        q.loc[i, col] = v
    return run


def test_ws_slot_sums_the_latest_sent_prompt_of_each_session_holding_a_slot():
    run = _ws_run()
    t = np.array([0.5, 1.5, 3.0, 7.0, 11.5, 13.5, 15.5, 18.5, 21.0])
    # 0's turn 0; + 1's turn 0; 0's turn 1; 0 paused; 1's turn 1; 0 resumed (turn 2 sent at 15); 2; none
    assert T.ws_slot(run, t).tolist() == [0, 100, 1100, 1200, 1000, 2000, 300, 10000, 0]


def test_ws_slot_counts_turn_0_planned_before_the_first_send_and_planned_when_prompt_tokens_is_null():
    run = _ws_run({(5, "t_sent"): 19.0})                            # 2 admitted at 18, sends at 19
    assert T.ws_slot(run, np.array([18.5, 19.5])).tolist() == [2176, 10000]
    run = _ws_run({(4, "prompt_tokens"): None})                     # 1's turn 1 cut off before its usage chunk
    assert T.ws_slot(run, np.array([13.5])).tolist() == [2176]


# ---------------------------------------------------------------- reuse_unpaused

def _turns_run(n, paused_miss=True, pause_cols=True):
    rows = []
    for s in range(n):
        rows += [dict(session_id=s, turn=0, t_first_token=100.0 + s, t_end=100.5 + s, prompt_tokens=1000,
                      cached_tokens=0, t_paused=N),
                 dict(session_id=s, turn=1, t_first_token=150.0 + s, t_end=150.5 + s, prompt_tokens=1300,
                      cached_tokens=1000, t_paused=N)]
    if paused_miss:                                                    # a resumed turn that missed
        rows += [dict(session_id=n, turn=0, t_first_token=110.0, t_end=110.5, prompt_tokens=1000, cached_tokens=0,
                      t_paused=N),
                 dict(session_id=n, turn=1, t_first_token=160.0, t_end=160.5, prompt_tokens=1300, cached_tokens=0,
                      t_paused=120.0)]
    q = pd.DataFrame(rows).assign(status="ok", prompt_tokens_planned=lambda d: d.prompt_tokens)
    if not pause_cols:
        q = q.drop(columns=["t_paused"])
    return {"requests": q, "cfg": {"trigger": {"kind": "burst", "prompt_tokens": 64}}}


def test_a_resumed_turn_miss_counts_in_k1_reuse_and_not_in_reuse_unpaused():
    from analysis.k1 import reuse
    run = _turns_run(25)
    assert reuse(run, 140.0, 200.0) == pytest.approx(25 / 26)
    assert T.reuse_unpaused(run, 140.0, 200.0) == 1.0


def test_without_pause_fields_reuse_unpaused_is_k1_reuse_only_with_c2s_minimum_of_turns():
    from analysis.c2 import C2
    from analysis.k1 import reuse
    big = _turns_run(30, paused_miss=False, pause_cols=False)
    assert T.reuse_unpaused(big, 140.0, 200.0) == reuse(big, 140.0, 200.0) == 1.0
    small = _turns_run(C2["min_turns"] - 1, paused_miss=False, pause_cols=False)
    assert math.isnan(T.reuse_unpaused(small, 140.0, 200.0)) and reuse(small, 140.0, 200.0) == 1.0


# ---------------------------------------------------------------- the label, on hand-built evidence

def _ev(**kw):
    ev = {"min_baseline_pf": 0.98, "base_p": {"pf": 1.0}, "base_u": {"pf": 1.0},
          "established": {"cold_backlogged": True}, "u_ok": {"ok": True}, "u_fault": False, "pause_fault": False,
          "nan_input": False,
          "t_w": 2700.0, "t_fit": 2660.0, "relapse": False, "lead": 540.0, "t_end": 4080.0}
    ev.update(kw)
    return ev


@pytest.mark.parametrize("kw,sep,want", [
    ({}, True, "prompt_warm"),
    ({"base_p": {"pf": 0.97}}, True, "bad_baseline"),
    ({"base_u": {"pf": 0.97}}, True, "bad_baseline"),                 # either arm
    ({"base_u": {"pf": 0.97}, "established": {"cold_backlogged": False}}, False, "bad_baseline"),   # first match
    ({"established": {"cold_backlogged": False}}, True, "not_established"),
    ({"established": {"cold_backlogged": None}}, True, "not_established"),
    ({}, False, "unseparable"),
    ({"u_ok": {"ok": False}}, True, "control_failed"),
    ({"pause_fault": True}, True, "unclear"),
    ({"nan_input": True}, True, "unclear"),
    ({"relapse": True}, True, "relapse"),
    ({"relapse": True, "lead": 10.0}, True, "relapse"),               # relapse before drain_bound
    ({"t_w": None, "t_fit": 3600.0}, True, "cold_fit"),
    ({"t_w": None, "t_fit": 3601.0}, True, "unclear"),
    ({"t_w": None, "t_fit": None}, True, "ws_never_fit"),           # draft 6: was cold_over_pool
    ({"lead": 299.0}, True, "drain_bound"),
    ({"lead": 301.0}, True, "prompt_warm"),
    ({"lead": 300.0}, True, "prompt_warm"),
    ({"t_fit": None}, True, "unclear"),                               # label 9: a t_w with no t_fit, in n
    ({"u_fault": True}, True, "control_failed"),                      # draft 4, gap 10
    ({"u_fault": True, "pause_fault": True}, True, "control_failed"),
    ({"t_w": 2660.0 + 239, "lead": 1e9}, True, "prompt_warm"),
    ({"t_w": 2660.0 + 240, "lead": 1e9}, True, "prompt_warm"),
    ({"t_w": 2660.0 + 241, "lead": 1e9}, True, "delayed"),
    ({"t_w": 2580.0, "t_fit": 2660.0}, True, "warm_before_fit"),      # draft 6: a negative lead is no prompt evidence
    ({"t_w": 2659.0, "t_fit": 2660.0, "lead": 1e9}, True, "warm_before_fit"),
    ({"t_w": 2660.0, "t_fit": 2660.0, "lead": 1e9}, True, "prompt_warm"),           # t_w = t_fit: lead 0
    ({"t_w": 2580.0, "t_fit": 2660.0, "lead": 10.0}, True, "drain_bound"),          # drain_bound first
    ({"t_w": 2580.0, "t_fit": None}, True, "unclear"),                              # rule 9 before rule 10
    ({"base_p": {"pf": math.nan}, "nan_input": True}, True, "unclear"),
])
def test_label_from_each_branch(kw, sep, want):
    assert T.label_from(_ev(**kw), sep) == want


def test_labels_partition_into_out_of_n_and_counted():
    assert set(T.OUT_OF_N) | set(T.COUNTED) == set(T.LABELS) and not set(T.OUT_OF_N) & set(T.COUNTED)
    assert T.COUNTED == ("unclear", "relapse", "cold_fit", "ws_never_fit", "drain_bound", "warm_before_fit",
                         "prompt_warm", "delayed") and "cold_over_pool" not in T.LABELS
    assert T.MEMORY == ("cold_fit", "delayed") and set(T.MEMORY) <= set(T.COUNTED)


# ---------------------------------------------------------------- outcome and prediction

@pytest.mark.parametrize("labels,want", [
    (["prompt_warm"] * 2 + ["unseparable"] * 3, "too_few"),
    (["prompt_warm"] * 3 + ["delayed"] * 2, "no_memory"),
    (["prompt_warm"] * 3 + ["cold_fit"] * 2, "no_memory"),
    (["cold_fit", "delayed", "delayed", "prompt_warm", "prompt_warm"], "memory"),
    (["drain_bound"] * 3 + ["prompt_warm"] * 2, "drain_bound"),
    (["ws_never_fit"] * 3 + ["prompt_warm"] * 2, "mixed"),
    (["warm_before_fit"] * 3 + ["prompt_warm"] * 2, "mixed"),
    (["unclear", "relapse", "ws_never_fit", "prompt_warm", "delayed"], "mixed"),
    (["ws_never_fit", "warm_before_fit", "prompt_warm"], "mixed"),          # both count in n
    (["prompt_warm"] * 3 + ["bad_baseline", "control_failed"], "no_memory"),
])
def test_outcome_order(labels, want):
    o = T.outcome(labels)
    assert o["decision"] == want == T.decide(labels)
    assert o["n"] == sum(x in T.COUNTED for x in labels)


def test_outcome_bounds_and_refusal():
    from analysis.stats import cp_upper
    o = T.outcome(["prompt_warm"] * 3 + ["delayed", "cold_fit"])
    assert o["cells"]["prompt_warm"] == {"k": 3, "n": 5, "upper": cp_upper(3, 5)}
    assert o["cells"]["memory"]["k"] == 2
    with pytest.raises(ValueError):
        T.outcome(["stuck"])


def test_predicted_reproduces_draft_5s_numbers():
    """draft 5 §Prediction's row. Pinned at the four decimals the draft's script printed (0.6425, 0.2785, 0.0744,
    0.0031, 0.0015); the draft's three-decimal 0.643 and 0.279 round those 4-decimal prints again (0.64247 and
    0.27850 to three decimals are 0.642 and 0.278)."""
    p = T.predicted()
    g = p["given_gate"]
    assert math.isclose(sum(g.values()), 1.0)
    assert round(g["no_memory"], 4) == 0.6425 and round(g["mixed"], 4) == 0.2785 and round(g["too_few"], 4) == 0.0744
    assert round(g["memory"], 4) == 0.0031 and round(g["drain_bound"], 4) == 0.0015
    assert round(p["no_memory_unconditional"], 3) == 0.514 and round(p["e_k_prompt_warm"], 3) == 2.886
    assert round(p["p_control_failed_most"], 5) == 0.00116
    assert T.T1["prediction"]["counted"] == {"prompt_warm": 0.74, "delayed": 0.07, "cold_fit": 0.02,
                                             "drain_bound": 0.07, "relapse": 0.02, "unclear": 0.08}
    assert math.isclose(sum(T.T1["prediction"]["counted"].values()), 1.0)
    assert math.isclose(sum(T.T1["prediction"]["out_of_n"].values()), 0.22)


DRAFT4_COUNTED = {"prompt_warm": 0.65, "delayed": 0.12, "cold_fit": 0.04, "drain_bound": 0.07, "relapse": 0.04,
                  "unclear": 0.08}


def test_predicted_failure_control_reproduces_draft_4s_table_from_draft_4s_inputs(monkeypatch):
    """draft 5 Appendix E1's failure control, in the repo: `predicted` on draft 4's per-seed inputs prints draft 4's
    published row (0.513, 0.395, 0.074, 0.016, 0.0015; unconditional 0.410; E[k] 2.535; 0.00116). It can fail: a
    change to `decide` or `predicted` that moves draft 5's row moves this one too, against numbers fixed elsewhere."""
    monkeypatch.setitem(T.T1, "prediction", {**T.T1["prediction"], "counted": DRAFT4_COUNTED})
    p = T.predicted()
    g = p["given_gate"]
    want = {"no_memory": 0.513, "mixed": 0.395, "too_few": 0.074, "memory": 0.016, "drain_bound": 0.0015}
    assert all(abs(g[k] - v) < 0.0006 for k, v in want.items()), g
    assert round(p["no_memory_unconditional"], 3) == 0.410 and round(p["e_k_prompt_warm"], 3) == 2.535
    assert round(p["p_control_failed_most"], 5) == 0.00116


def test_predicted_band_reproduces_the_draft_and_the_review():
    """draft 5 E3: P(inside | label) averaged over the counted prediction is 0.3435 (the draft's "0.34"), giving the
    draft's band_inside 0.123, band_miss 0.802, band_too_few 0.074. With the rounded 0.34 the same arithmetic gives
    the review's 0.120 and 0.805 (rule F round 2, §4b): the two differ only by that rounding."""
    b = T.predicted_band()
    assert b["p_inside"] == pytest.approx(0.3435) and math.isclose(sum(b["given_gate"].values()), 1.0)
    g = b["given_gate"]
    assert (round(g["band_inside"], 3), round(g["band_miss"], 3), round(g["band_too_few"], 3)) == (0.123, 0.802, 0.074)
    r = T.predicted_band(p_inside=0.34)["given_gate"]
    assert (round(r["band_inside"], 3), round(r["band_miss"], 3), round(r["band_too_few"], 3)) == (0.120, 0.805, 0.074)
    assert g["band_too_few"] == pytest.approx(T.predicted()["given_gate"]["too_few"])  # the same n < 3 event


# ---------------------------------------------------------------- the separation gate (synthetic only)

def _uncapped(n_in=30, n_after=100, t_after=None, fs=FS):
    """n_in sessions open at the switch (turn 0 planned at 1000, never ended), n_after requests planned at fs + 10.
    Nothing is owed at the trigger (900 s), so the healthy level is 0."""
    t_after = fs + 10.0 if t_after is None else t_after
    rows = [(s, 0, 1000.0, N) for s in range(n_in)] + [(n_in + i, 0, t_after, N) for i in range(n_after)]
    q = pd.DataFrame(rows, columns=["session_id", "turn", "t_planned", "t_end"])
    q["prompt_tokens_planned"] = 2176
    return {"requests": q, "dir": Path("toy_uncapped"),
            "cfg": {"workload": {"duration_s": 4080.0}, "verdict": {"baseline_s": 600, "watch_s": 3000, "end_s": 600},
                    "trigger": {"kind": "burst", "at_s": 900.0, "duration_s": 180.0, "prompt_tokens": 64}}}


def _expected_t_d(l0, n_after, fs=FS, n_in=30):
    """The fluid by hand: 1.75/s for the grid seconds before t_cap_est, 3.01/s after, n_after added in [fs+10, fs+11)."""
    tc = fs + max(0, n_in - 22) / 1.75
    b, t = float(l0), fs
    while True:
        b = max(0.0, b + (n_after if t == fs + 10 else 0) - (3.01 if t >= tc else 1.75))
        t += 1.0
        if b <= 0:
            return t, tc


@pytest.mark.parametrize("n_after,separates", [(100, False), (2000, True)])
def test_gate_seed_on_a_known_line(n_after, separates):
    r = T.gate_seed(_uncapped(n_after=n_after), FS)
    td, tc = _expected_t_d(30, n_after)
    assert r["n_in"] == 30 and r["owed_at_from_s"] == 30 and r["healthy_level"] == 0
    assert r["t_cap_est"] == pytest.approx(FS + 8 / 1.75) == pytest.approx(tc)
    assert r["t_d_est"] == td and r["separates"] is separates
    assert r["separates"] == (td - tc >= 540.0)


def test_gate_seed_margin_boundary():
    """Find n_after giving a margin just under and just over 540 s; the flag follows the registered >= 540."""
    got = {}
    for n in range(1500, 1700):
        td, tc = _expected_t_d(30, n)
        got[n] = td - tc
    lo = max(n for n, m in got.items() if m < 540)
    hi = min(n for n, m in got.items() if m >= 540)
    assert not T.gate_seed(_uncapped(n_after=lo), FS)["separates"]
    assert T.gate_seed(_uncapped(n_after=hi), FS)["separates"]


def test_fluid_drain_starts_drained_and_under_the_cap_has_no_phase_a():
    assert T.fluid_drain(0, 0, np.array([]), FS, FS) == FS
    assert T.t_cap_est(20, FS) == FS and T.t_cap_est(22, FS) == FS


def test_separation_gate_and_choose_from_s_on_toy_runs(monkeypatch):
    import analysis.load as L
    seen = []

    def fake(p):
        seen.append(Path(p).name)
        return _uncapped(n_after=2000)
    monkeypatch.setattr(L, "load_run", fake)
    g = T.separation_gate(FS)
    assert g["k_separate"] == 5 and g["passes"] and set(g["rows"]) == set(T.T1["seeds"])
    assert set(seen) == set(D.D1["uncapped"].values()) | set(D.D1["controls"].values())
    assert all(r["control_mean_owed_base"] == 0.0 for r in g["rows"].values())
    with pytest.raises(ValueError):
        T.separation_gate(1700.0)
    calls = []

    def gate(fs):
        calls.append(fs)
        return {"passes": fs >= 2580.0, "from_s": fs}
    c = T.choose_from_s(gate)
    assert c["from_s"] == 2580.0 and calls == [1680.0, 1980.0, 2280.0, 2580.0]   # stops at the first pass
    assert T.choose_from_s(lambda fs: {"passes": False})["from_s"] is None


# ---------------------------------------------------------------- end to end on synthetic P and U runs

def _step(*pairs):
    """A step function: pairs (t_from, value), in order; the value of the last t_from <= t."""
    def f(t):
        v = pairs[0][1]
        for a, x in pairs:
            if t >= a:
                v = x
        return v
    return f


def _synth(arm, reuse_at, q_at, kv_at, fill_end=None, n_fill=60, ttft_base=0.1):
    """A synthetic run on T1's resolved config: one 2-turn session every 4 s (turn 1's first token at t, its cache
    hit reuse_at(t) of its previous prompt); `n_fill` turn-0 requests planned at 1000 s and ended at fill_end (the total
    line); metrics every 1 s with waiting q_at, KV kv_at, and a preemption counter rising 0.1 per s. Each session holds
    a slot ~3.4 s, so slots_held <= 1."""
    cfg = T.resolved("U", 15) if arm == "U" else T.resolved("P", 15, FS)
    rows = []
    for sid, t1 in enumerate(np.arange(10.0, 4080.0, 4.0)):
        t0 = t1 - 3.0
        rows.append(dict(session_id=sid, turn=0, t_planned=t0, t_sent=t0, t_first_token=t0 + 0.1, t_end=t0 + 0.5,
                         prompt_tokens=1000.0, prompt_tokens_planned=1000, cached_tokens=0.0, ttft=ttft_base))
        rows.append(dict(session_id=sid, turn=1, t_planned=t1 - 0.1, t_sent=t1 - 0.1, t_first_token=t1,
                         t_end=t1 + 0.4, prompt_tokens=1300.0, prompt_tokens_planned=1300,
                         cached_tokens=float(round(reuse_at(t1) * 1000)), ttft=ttft_base))
    last = len(rows) // 2
    if fill_end is not None:
        for i in range(n_fill):
            rows.append(dict(session_id=last + i, turn=0, t_planned=1000.0, t_sent=N, t_first_token=N, t_end=fill_end,
                             prompt_tokens=N, prompt_tokens_planned=1000, cached_tokens=N, ttft=N))
    q = pd.DataFrame(rows).assign(status="ok")
    q["request_id"] = range(len(q))
    if arm == "P":
        q["t_arrived"] = np.where(q.turn == 0, q.t_sent, np.nan)
        q["t_admitted"] = q["t_arrived"]
        from harness.loadgen import PAUSE_FIELDS
        for k in PAUSE_FIELDS:
            q[k] = np.nan
        q["pause_s"], q["pause_s_cum"] = 0.0, 0.0
    t = np.arange(0.0, 4101.0, 1.0)
    m = pd.DataFrame({"t": t, "vllm:num_requests_waiting": [q_at(x) for x in t],
                      "vllm:num_requests_running": 5.0, "vllm:kv_cache_usage_perc": [kv_at(x) for x in t],
                      "vllm:prefix_cache_hits_total": 500.0 * t, "vllm:prefix_cache_queries_total": 1000.0 * t,
                      "vllm:num_preemptions_total": 0.1 * t})
    return {"requests": q, "metrics": m, "cfg": cfg, "dir": Path(f"toy_{arm}")}


COLD_Q, COLD_KV = 5.0, 0.95


def _u():
    return _synth("U", _step((0, 0.95), (1080, 0.5)), _step((0, 0.0), (1080, COLD_Q)), _step((0, 0.5), (1080, COLD_KV)))


def _coupled(reuse_at, cold, warm, warm_reuse=0.95):
    """A gauge coupled to reuse (review B1, fact 2): `cold` whenever reuse is below its warm level, `warm` otherwise.
    A cold capped server keeps a queue, so its waiting falls only at the re-warm."""
    return lambda t: cold if reuse_at(t) < warm_reuse else warm


def _p(reuse_at, fill_end=3295.0, ttft_base=0.1):
    """Arm P: waiting and KV follow reuse (high exactly while reuse is cold, back at the re-warm), not a fixed time."""
    return _synth("P", reuse_at, _coupled(reuse_at, COLD_Q, 0.0), _coupled(reuse_at, COLD_KV, 0.5),
                  fill_end=fill_end, ttft_base=ttft_base)


def _cold_then(*pairs):
    return _step((0, 0.95), (1080, 0.5), *pairs)


def test_end_to_end_prompt_warm_trace():
    """draft 3 §3's H0 trace in miniature: cap binds at once (t_cap = t_fit = from_s, draft 6), reuse and waiting are
    back from 2640, spans [2700, 2820) and [2820, 2940) warm (t_w 2700), the total line drains at 3300 (lead 600), U
    stays cold."""
    lab, ev = T.seed_label(_p(_cold_then((2640, 0.95))), _u(), True, FS)
    assert ev["t_cap"] == FS and ev["t_fit"] == FS                        # the working set fits at the cap
    assert ev["t_w"] == 2700.0 and ev["t_d"] == 3300.0 and ev["lead"] == 600.0
    assert ev["u_ok"]["ok"] and ev["established"]["cold_backlogged"] is True and not ev["pause_fault"]
    assert not ev["nan_input"] and lab == "prompt_warm"
    r = T.readouts(_p(_cold_then((2640, 0.95))), _u(), ev)
    assert r["t_w_minus_t_cap"] == 120.0 and r["warm_hold"]["resume_queue_mean"] == 0.0
    assert r["u_on_judged_span"]["span"] == (2700.0, 2940.0) and r["u_on_judged_span"]["reuse_unpaused"] == 0.5
    assert D.cap_queue is T.d1.cap_queue and r["d1_label_with_resume"] in D.LABELS


def test_readouts_report_kv_preemptions_hit_rate_and_k1_reuse_before_t_w_beside_reuse_unpaused():
    """draft 6, review M3: over [from_s, t_w) = [2580, 2700) on the H0 trace: KV 0.95 while cold (to 2640), 0.5 after;
    preemptions 0.1 per s from the first to the last scrape inside (2580 to 2699); the server's hit share 0.5; 15
    turns at reuse 0.5 and 15 at 0.95. Beside them the baseline's; every warm span carries the same three."""
    p = _p(_cold_then((2640, 0.95)))
    _, ev = T.seed_label(p, _u(), True, FS)
    r = T.readouts(p, _u(), ev)["before_t_w"]
    assert r["span"] == (FS, 2700.0) and r["kv"] == pytest.approx(0.725) and r["preemptions"] == pytest.approx(11.9)
    assert r["hit_server"] == pytest.approx(0.5)
    assert r["reuse_k1"] == pytest.approx(0.725) and r["reuse_unpaused"] == pytest.approx(0.725)
    b = T.readouts(p, _u(), ev)["baseline"]
    assert b["reuse"] == b["reuse_k1"] == pytest.approx(0.95) and b["hit_server"] == pytest.approx(0.5)
    assert b["kv"] == pytest.approx(0.5)
    sp = ev["warm"]["spans"][1]
    assert sp["span"] == (2700.0, 2820.0) and sp["kv"] == 0.5 and sp["hit_server"] == pytest.approx(0.5)
    assert sp["reuse_k1"] == sp["reuse_unpaused"] == pytest.approx(0.95)
    cold = _p(_cold_then(), fill_end=N)
    _, ev = T.seed_label(cold, _u(), True, FS)
    assert T.readouts(cold, _u(), ev)["before_t_w"]["span"] == (FS, 4080.0)          # no t_w: to the end


def test_drained_split_counts_never_drained_seeds_apart():
    rows = [{"seed": 15, "label": "prompt_warm", "t_d": 3300.0}, {"seed": 16, "label": "prompt_warm", "t_d": math.inf},
            {"seed": 17, "label": "prompt_warm", "t_d": 3400.0}, {"seed": 18, "label": "cold_fit", "t_d": math.inf},
            {"seed": 19, "label": "unseparable", "t_d": math.inf}]
    d = T.drained_split(rows)
    assert (d["prompt_warm_drained"], d["prompt_warm_never_drained"]) == (2, 1)
    assert d["by_label"]["prompt_warm"]["never_drained"] == [16] and d["by_label"]["cold_fit"]["never_drained"] == [18]
    assert "unseparable" not in d["by_label"]                                       # out of n: not counted
    assert T.drained_split([])["prompt_warm_never_drained"] == 0


def test_end_to_end_one_warm_span_then_cold_is_no_t_w():
    lab, ev = T.seed_label(_p(_cold_then((2700, 0.95), (2820, 0.5))), _u(), True, FS)
    assert [s["warm"] for s in ev["warm"]["spans"][:3]] == [False, True, False]
    assert ev["t_w"] is None and lab == "cold_fit"


def test_end_to_end_cold_with_a_fitting_working_set_is_cold_fit():
    lab, ev = T.seed_label(_p(_cold_then()), _u(), True, FS)
    assert ev["t_w"] is None and ev["t_fit"] <= T.cold_fit_by(4080.0) and lab == "cold_fit"


def test_b1_waiting_coupled_to_reuse_a_late_rewarm_is_delayed_and_a_fitting_cold_is_cold_fit():
    """Review B1 (draft 6): waiting stays high exactly while reuse is cold and falls only at the re-warm, as on a cold
    capped server (fact 2). A re-warm 900 s after t_cap must read `delayed`, and cold to the end with a fitting working
    set `cold_fit`. Draft 5's t_fit waited for waiting to fall: it read the first `prompt_warm` (t_fit 3478, t_w -
    t_fit 62) and the second `cold_over_pool` (no t_fit). Fails on that code (checked: draft 5's t1.py at 4e5bca7)."""
    from analysis.load import _level
    late = _p(_cold_then((FS + 900.0, 0.95)), fill_end=3995.0)
    assert _level(late, FS, FS + 900.0)["q"] == COLD_Q and _level(late, FS + 900.0, 4080.0)["q"] == 0.0   # coupled
    lab, ev = T.seed_label(late, _u(), True, FS)
    assert ev["t_cap"] == ev["t_fit"] == FS and ev["t_w"] == FS + 960.0 and ev["t_w"] - ev["t_cap"] >= 900.0
    assert ev["t_d"] == 4000.0 and ev["lead"] >= T.T1["lead_min_s"] and lab == "delayed"
    cold = _p(_cold_then(), fill_end=N)                                # the owed fill never ends
    assert _level(cold, FS, 4080.0)["q"] == COLD_Q                                              # never back
    lab, ev = T.seed_label(cold, _u(), True, FS)
    assert ev["t_w"] is None and ev["t_fit"] == FS and ev["t_d"] == math.inf and lab == "cold_fit"
    assert T.outcome([lab, lab, "delayed", "prompt_warm", "prompt_warm"])["decision"] == "memory"


def _fit_run(c):
    run = _ws_run()
    run["cfg"]["checks"] = {"expected_kv_tokens": c}
    return run


@pytest.mark.parametrize("limit,want", [(1500, 11.0), (1000, 11.0), (999, 15.0), (300, 15.0), (299, 20.0),
                                        (-1, None)])
def test_t_fit_is_the_working_set_clause_alone(limit, want):
    """t_fit = t_cap (11 s at cap 1) if ws_slot <= C - cap x 256 there, else the first grid time after it where it
    holds (ws_slot from 11 s: 1000 to 13, 2000 to 15, 300 to 18, 10000 to 20, then 0), None if never."""
    import unittest.mock as m
    with m.patch.dict(T.T1, {"cap": 1}):
        run = _fit_run(limit + T.T1["ws_margin_per_slot"])
        assert T.t_cap(run, 10.0) == 11.0
        assert T.t_fit(run, T.t_cap(run, 10.0)) == want
        assert T.t_fit(run, None) is None
    assert T.t_fit(_ws_run(), 11.0) is None                            # no C in the config


def test_end_to_end_delayed_and_relapse_and_drain_bound():
    lab, ev = T.seed_label(_p(_cold_then((3000, 0.95)), fill_end=3695.0), _u(), True, FS)
    assert ev["t_w"] == 3060.0 and ev["t_w"] - ev["t_fit"] > 240 and ev["lead"] >= 300 and lab == "delayed"
    lab, ev = T.seed_label(_p(_cold_then((2640, 0.95), (3000, 0.5))), _u(), True, FS)
    assert ev["t_w"] == 2700.0 and ev["warm"]["t_relapse"] == 2940.0 and lab == "relapse"
    lab, ev = T.seed_label(_p(_cold_then((2640, 0.95)), fill_end=2895.0), _u(), True, FS)
    assert ev["lead"] == 200.0 and lab == "drain_bound"


def test_end_to_end_bad_baseline_on_either_arm_and_control_failed():
    lab, _ = T.seed_label(_p(_cold_then((2640, 0.95)), ttft_base=5.0), _u(), True, FS)
    assert lab == "bad_baseline"
    u = _u()
    u["requests"]["ttft"] = 5.0
    assert T.seed_label(_p(_cold_then((2640, 0.95))), u, True, FS)[0] == "bad_baseline"
    warm_u = _synth("U", _step((0, 0.95)), _step((0, 0.0)), _step((0, 0.5)))
    lab, ev = T.seed_label(_p(_cold_then((2640, 0.95))), warm_u, True, FS)
    assert not ev["u_ok"]["ok"] and lab == "control_failed"


def test_end_to_end_established_is_tied_to_the_switch():
    """review R1: cold before 2580 but warm on [1380, 1680): established at 2580, not on D1r's fixed window."""
    p = _synth("P", _step((0, 0.95), (1800, 0.5), (2640, 0.95)),
               _step((0, 0.0), (1800, COLD_Q), (2600, 0.0)), _step((0, 0.5), (1800, COLD_KV), (2600, 0.5)),
               fill_end=3295.0)
    assert T.established(p, FS)["cold_backlogged"] is True
    assert T.established(p, 1680.0)["cold_backlogged"] is False
    assert R.established(p)["cold_backlogged"] is False                  # D1r's window says otherwise


def test_p_is_named_t1p_and_the_old_rescue_pause_names_are_not_t1s():
    assert [T.config_name("P", sd) for sd in T.T1["seeds"]] == [f"burst_l1pl350cl_r2.0_w3000_s{sd}_t1p"
                                                               for sd in T.T1["seeds"]]
    old = [f"20261123-000000_burst_l1pl350cl_r2.0_w3000_s{sd}_rescue_pause" for sd in T.T1["seeds"]]
    assert not any(fnmatch.fnmatchcase(n, T.run_pattern(a, sd)) for n in old for a in ARMS for sd in T.T1["seeds"])
    assert not any(r[0] == "T1" and fnmatch.fnmatchcase(n, r[1]) for n in old for r in W.SEALS)


def test_end_to_end_a_u_fault_is_control_failed_and_reported():
    """draft 4, gap 10: a U run whose records carry a cap field is control_failed (out of n), with u_fault set."""
    u = _u()
    u["requests"]["t_arrived"] = np.where(u["requests"].turn == 0, u["requests"].t_sent, np.nan)
    lab, ev = T.seed_label(_p(_cold_then((2640, 0.95))), u, True, FS)
    assert ev["u_fault"] and ev["u_ok"]["ok"] and lab == "control_failed"
    lab, ev = T.seed_label(_p(_cold_then((2640, 0.95))), _u(), True, FS)
    assert not ev["u_fault"] and lab == "prompt_warm"


# ---------------------------------------------------------------- table's gate check (draft 4, gap 16)

def _gate(sep, from_s=FS, t_d_est=FS + 700.0, t_cap_est=FS + 60.0):
    rows = {sd: {"separates": s, "t_d_est": t_d_est, "t_cap_est": t_cap_est, "owed_at_from_s": 400.0}
            for sd, s in zip(T.T1["seeds"], sep)}
    return {"from_s": from_s, "rows": rows, "k_separate": sum(sep), "passes": sum(sep) >= T.T1["gate_k"]}


def _probe(tmp_path, sep, from_s=FS, name="choose_test.json", **kw):
    import json
    tmp_path.mkdir(parents=True, exist_ok=True)
    g = _gate(sep, from_s, **kw)
    (tmp_path / name).write_text(json.dumps({"from_s": from_s if g["passes"] else None, "gate": g, "tried": [g]},
                                            default=str))
    return tmp_path


@pytest.fixture
def table_env(tmp_path, monkeypatch):
    """table() on no run data: T1's from_s set, nothing sealed, arm_run finds nothing (every seed not_run)."""
    monkeypatch.setitem(T.T1, "from_s", FS)
    monkeypatch.setattr(T, "sealed", lambda: [])
    monkeypatch.setattr(T, "arm_run", lambda arm, sd, fs: (None, []))
    monkeypatch.setattr(T, "reference_platforms", lambda: {"k1_arm_a": {"a15": "Linux-A"}, "d1": {"c15": "Linux-D"}})
    return tmp_path


SEP = (True, True, False, True, False)


def test_table_accepts_a_recomputed_gate_equal_to_the_committed_one(table_env, monkeypatch):
    monkeypatch.setattr(T, "separation_gate", lambda fs: _gate(SEP, fs))
    out = T.table(_probe(table_env / "probe", SEP))
    assert out["not_run"] == list(T.T1["seeds"]) and out["outcome"]["decision"] == "too_few"
    assert {sd: r["separates"] for sd, r in out["gate"]["rows"].items()} == dict(zip(T.T1["seeds"], SEP))


@pytest.mark.parametrize("recomputed,committed,from_s", [
    ((True, True, False, False, False), SEP, FS),                                 # the recomputed gate fails
    ((True, True, True, True, False), SEP, FS),                                   # a seed's separates differs
    (SEP, SEP, 2880.0),                                                           # probe at another from_s
])
def test_table_refuses_a_gate_that_fails_or_differs(table_env, monkeypatch, recomputed, committed, from_s):
    monkeypatch.setattr(T, "separation_gate", lambda fs: _gate(recomputed, fs))
    d = _probe(table_env / "probe", committed, from_s)
    with pytest.raises(ValueError, match="does not pass|differs|chose"):
        T.table(d)


def test_table_refuses_without_exactly_one_committed_gate(table_env, monkeypatch):
    monkeypatch.setattr(T, "separation_gate", lambda fs: _gate(SEP, fs))
    with pytest.raises(ValueError):
        T.table(table_env / "none")                                # no probe directory
    d = _probe(table_env / "two", SEP)
    _probe(d, SEP, name="choose_other.json")
    with pytest.raises(ValueError):
        T.table(d)
    with pytest.raises(ValueError):
        T.table(_probe(table_env / "nopass", (True, False, False, False, False)))   # names no passing from_s


def test_table_labels_a_u_fault_control_failed_and_does_not_raise(table_env, monkeypatch):
    monkeypatch.setattr(T, "separation_gate", lambda fs: _gate((True,) * 5, fs))
    u = _u()
    u["requests"]["t_arrived"] = np.nan                             # a cap field present: u_fault
    p = _p(_cold_then((2640, 0.95)))
    monkeypatch.setattr(T, "arm_run", lambda arm, sd, fs: ((p if arm == "P" else u), []) if sd == 15 else (None, []))
    out = T.table(_probe(table_env / "probe", (True,) * 5))
    assert out["rows"][15]["label"] == "control_failed" and out["rows"][15]["u_fault"] is True
    assert out["u_faults"] == [15] and out["not_run"] == [16, 17, 18, 19]


# ---------------------------------------------------------------- draft 5: the drain band (judged)

def _bev(t_d, t_cap=FS, t_end=4080.0, from_s=FS):
    return {"from_s": from_s, "t_d": t_d, "t_cap": t_cap, "t_end": t_end, "owed_at_from_s": 410.0}


def _brow(t_d_est, t_cap_est=FS):
    return {"t_d_est": t_d_est, "t_cap_est": t_cap_est, "owed_at_from_s": 400.0}


@pytest.mark.parametrize("t_d,t_d_est,want", [
    (FS + 600.0, FS + 600.0, "inside"),
    (FS + 690.0, FS + 600.0, "inside"),            # D / D_est - 1 = +0.15: inside (<=)
    (FS + 691.0, FS + 600.0, "late"),
    (FS + 510.0, FS + 600.0, "inside"),            # -0.15: inside (the ratio form would round this out)
    (FS + 509.0, FS + 600.0, "early"),
    (math.inf, FS + 600.0, "late"),                # never drained, from_s + 1.15 D_est = 3270 <= 4080
    (math.inf, FS + 1300.0, "late"),               # 2580 + 1495 = 4075 <= 4080
    (math.inf, FS + 1310.0, "unjudged"),           # 2580 + 1506.5 > 4080
    (3000.0, FS, "unjudged"),                      # D_est = 0 (the line drained at from_s): no ratio
    (None, FS + 600.0, "unjudged"),
])
def test_drain_band_per_seed(t_d, t_d_est, want):
    b = T.drain_band(_bev(t_d), _brow(t_d_est))
    assert b["band"] == want and b["band"] in T.BAND
    assert b["d_est"] == t_d_est - FS and (b["owed_at_from_s"], b["gate_l0"]) == (410.0, 400.0)


def test_drain_band_never_drained_at_the_exact_edge_is_late():
    """from_s + 1.15 D_est = t_end exactly (2580 + 1150 = 3730): `late` (<=); a second later the seed is unjudged."""
    assert T.drain_band(_bev(math.inf, t_end=3730.0), _brow(FS + 1000.0))["band"] == "late"
    assert T.drain_band(_bev(math.inf, t_end=3729.0), _brow(FS + 1000.0))["band"] == "unjudged"


@pytest.mark.parametrize("t_cap,t_cap_est,want", [
    (FS + 200.0, FS + 60.0, "early"),     # review Trace C: D on target, but a late cap offset by a fast drain
    (FS, FS + 200.0, "late"),             # an early cap and a slow drain
    (FS + 13.5, FS + 90.0, "inside"),     # (600 - 13.5) / 510 - 1 = +0.15: inside (<=)
    (FS + 13.0, FS + 90.0, "late"),
    (None, FS + 60.0, "unjudged"),        # inside from the switch, no t_cap: the second clause cannot be judged
])
def test_drain_band_inside_needs_the_deviation_from_t_cap_too(t_cap, t_cap_est, want):
    """draft 6, M2: D = D_est exactly (deviation 0 from the switch) in every row; the cap's clause decides."""
    b = T.drain_band(_bev(FS + 600.0, t_cap=t_cap), _brow(FS + 600.0, t_cap_est))
    assert b["deviation"] == 0.0 and b["band"] == want


def test_drain_band_each_clause_uses_its_own_denominator():
    """D_est 600 and t_d_est - t_cap_est 570 differ: +0.15 from the switch (690 against 600) is inside, though 90 s is
    more than 0.15 x 570; the cap's clause is +0.105 (630 against 570)."""
    assert T.drain_band(_bev(FS + 690.0, t_cap=FS + 60.0), _brow(FS + 600.0, FS + 30.0))["band"] == "inside"
    assert T.drain_band(_bev(FS + 691.0, t_cap=FS + 60.0), _brow(FS + 600.0, FS + 30.0))["band"] == "late"


def test_drain_band_outside_from_the_switch_is_judged_without_t_cap():
    assert T.drain_band(_bev(FS + 700.0, t_cap=None), _brow(FS + 600.0))["band"] == "late"
    assert T.drain_band(_bev(FS + 400.0, t_cap=None), _brow(FS + 600.0))["band"] == "early"


def test_drain_band_readouts_are_signed_and_measured_from_the_switch():
    b = T.drain_band(_bev(FS + 660.0, t_cap=FS + 80.0), _brow(FS + 600.0, FS + 60.0))
    assert b["d"] == 660.0 and b["deviation"] == pytest.approx(0.1)
    assert b["deviation_from_t_cap"] == pytest.approx(580.0 / 540.0 - 1)        # (t_d - t_cap) / (t_d_est - t_cap_est)
    assert T.drain_band(_bev(math.inf), _brow(FS + 600.0))["deviation"] is None


def _br(label, band):
    return {"label": label, "band": None if band is None else {"band": band, "deviation": 0.0}}


@pytest.mark.parametrize("rows,want,k,n", [
    ([("prompt_warm", "inside")] * 3 + [("delayed", "late")] * 2, "band_inside", 3, 5),
    ([("prompt_warm", "inside")] * 3, "band_inside", 3, 3),
    ([("prompt_warm", "inside")] * 2 + [("cold_fit", "late")], "band_miss", 2, 3),          # n_band = 3, k < 3
    ([("prompt_warm", "inside")] * 2 + [("cold_fit", "late")] * 3, "band_miss", 2, 5),
    ([("prompt_warm", "inside")] * 2, "band_too_few", 2, 2),
    ([("prompt_warm", "inside")] * 2 + [("cold_fit", "unjudged")] * 3, "band_too_few", 2, 2),  # unjudged leave n
    ([("prompt_warm", "inside")] * 2 + [("unseparable", "inside"), ("control_failed", "inside"),
                                        ("bad_baseline", "inside")], "band_too_few", 2, 2),  # out of n never judged
    ([("prompt_warm", "early"), ("relapse", "late"), ("unclear", "inside"), ("drain_bound", "inside"),
      ("delayed", "inside")], "band_inside", 3, 5),                                          # every counted label
    ([("prompt_warm", "early")] * 3, "band_miss", 0, 3),
    ([("prompt_warm", "inside")] * 2 + [("not_established", None)], "band_too_few", 2, 2),
])
def test_band_outcome(rows, want, k, n):
    from analysis.stats import cp_upper
    o = T.band_outcome([_br(*r) for r in rows])
    assert (o["decision"], o["k_inside"], o["n_band"]) == (want, k, n) and o["decision"] in T.BAND_OUTCOMES
    assert o["upper"] == cp_upper(k, n)


def test_band_outcome_reports_the_prompt_warm_subset():
    o = T.band_outcome([_br("prompt_warm", "inside"), _br("prompt_warm", "early"), _br("delayed", "inside")])
    assert o["prompt_warm"] == {"k_inside": 1, "n": 2} and o["counts"] == {"inside": 2, "early": 1}


def test_end_to_end_band_on_the_prompt_warm_trace():
    """The H0 trace drains at 3300 (D = 720 s): inside a fluid line at 2580 + 700, late against 2580 + 620."""
    _, ev = T.seed_label(_p(_cold_then((2640, 0.95))), _u(), True, FS)
    assert ev["t_d"] == 3300.0 and ev["owed_at_from_s"] is not None
    assert T.drain_band(ev, _brow(FS + 700.0))["band"] == "inside"
    assert T.drain_band(ev, _brow(FS + 620.0))["band"] == "late"


# ---------------------------------------------------------------- draft 5: U's end window (readout)

def _env(run, platform):
    run["env"] = {"platform": platform}
    return run


def _warm_u():
    return _synth("U", _step((0, 0.95)), _step((0, 0.0)), _step((0, 0.5)))


def test_u_end_readout_counts_cold_backlogged_u_runs_and_lists_what_is_left_out():
    from analysis.stats import cp_upper
    nan_u = _u()
    m = nan_u["metrics"]
    m.loc[m.t >= 3480, "vllm:num_requests_waiting"] = np.nan                 # a None on the end window
    fault = _u()
    fault["requests"]["t_arrived"] = np.nan
    bad = _u()
    bad["requests"]["ttft"] = 5.0
    u_runs = {15: _env(_u(), "Linux-6.8.0-149"), 16: _env(_warm_u(), "Linux-6.8.0-149"), 17: None, 18: fault,
              19: bad, 20: _env(_u(), "Linux-6.8.0-147"), 21: nan_u}
    r = T.u_end_readout(u_runs, {"k1_arm_a": {}, "d1": {}})
    assert (r["k"], r["n"]) == (2, 3) and r["none"] == [21]
    assert set(r["excluded"]) == {17, 18, 19} and r["excluded"][18] == "u_fault"
    assert r["seeds"][16]["cold_backlogged"] is False and r["seeds"][15]["cold_backlogged"] is True
    assert r["upper"] == cp_upper(2, 3) and r["lower"] == pytest.approx(1 - cp_upper(1, 3))
    assert r["platforms"] == ["Linux-6.8.0-147", "Linux-6.8.0-149"] and r["reference"] == {"k1_arm_a": {}, "d1": {}}
    assert r["platforms_match"] is False and r["blocks_appendix_f_rewrite"] is True


A, B = "Linux-6.8.0-149", "Linux-6.8.0-147"


@pytest.mark.parametrize("u_plat,arm_a,caps,want", [
    ((A, A), {"a15": A, "a16": A}, {"c15": A}, True),
    ((A, B), {"a15": A, "a16": A}, {"c15": A}, False),          # one U run on another stack
    ((A, A), {"a15": A, "a16": B}, {"c15": A}, False),          # arm A itself on two strings
    ((A, A), {"a15": A}, {"c15": B}, False),                    # D1 differs from arm A
    ((A, None), {"a15": A}, {"c15": A}, False),                 # a U run without env.json's platform
    ((A, A), {}, {}, False),                                    # nothing to compare against
])
def test_u_end_readout_platforms_match_every_run_against_both_reference_arms(u_plat, arm_a, caps, want):
    us = {15 + i: (_env(_u(), p) if p is not None else _u()) for i, p in enumerate(u_plat)}
    r = T.u_end_readout(us, {"k1_arm_a": arm_a, "d1": caps})
    assert r["n"] == 2 and r["platforms_match"] is want and r["blocks_appendix_f_rewrite"] is (not want)


def test_u_end_readout_without_references_or_runs_compares_nothing():
    r = T.u_end_readout({15: _env(_u(), A)})
    assert r["platforms_match"] is None and r["blocks_appendix_f_rewrite"] is True
    r = T.u_end_readout({15: None}, {"k1_arm_a": {"a": A}, "d1": {"c": A}})
    assert r["n"] == 0 and r["platforms_match"] is None and r["blocks_appendix_f_rewrite"] is True


def test_u_end_readout_judges_the_end_window_only():
    """Warm through [3480, 4080) after a cold middle is not counted: the window is the end one, not U's whole run."""
    u = _synth("U", _step((0, 0.95), (1080, 0.5), (3400, 0.95)), _step((0, 0.0), (1080, COLD_Q), (3400, 0.0)),
               _step((0, 0.5), (1080, COLD_KV), (3400, 0.5)))
    r = T.u_end_readout({15: u, 16: _u()})
    assert (r["k"], r["n"]) == (1, 2) and r["seeds"][15]["cold_backlogged"] is False


def test_reference_platforms_reads_k1_arm_a_and_d1s_capped_runs(monkeypatch):
    import analysis.load as L
    from analysis.k1 import open_pattern
    seen, caps = [], []

    def find_runs(pattern):
        seen.append(pattern)
        return [{"dir": Path(pattern.replace("*", "x")), "env": {"platform": "Linux-A"}}]

    def capped_run(seed, cap):
        caps.append((seed, cap))
        return {"dir": Path(f"cap_{seed}"), "env": {"platform": "Linux-D"}}, []

    monkeypatch.setattr(L, "find_runs", find_runs)
    monkeypatch.setattr(T.d1, "capped_run", capped_run)
    r = T.reference_platforms()
    assert seen == [open_pattern(2.0, sd) for sd in (15, 16, 17, 18, 19)]
    assert caps == [(sd, R.d1_cap(sd)) for sd in (15, 16, 17, 18, 19)]
    assert set(r["k1_arm_a"].values()) == {"Linux-A"} and r["d1"] == {f"cap_{sd}": "Linux-D" for sd in range(15, 20)}


# ---------------------------------------------------------------- draft 5: table prints both

def test_table_prints_the_band_and_the_u_readout(table_env, monkeypatch):
    monkeypatch.setattr(T, "separation_gate", lambda fs: _gate((True,) * 5, fs))
    p = _p(_cold_then((2640, 0.95)))
    us = {15: _env(_u(), "Linux-6.8.0-149"), 16: _env(_warm_u(), "Linux-6.8.0-149")}

    def arm_run(arm, sd, fs):
        if arm == "P":
            return (p, []) if sd == 15 else (None, [])
        return (us.get(sd), [])

    monkeypatch.setattr(T, "arm_run", arm_run)
    out = T.table(_probe(table_env / "probe", (True,) * 5, t_d_est=FS + 700.0))
    assert out["rows"][15]["label"] == "prompt_warm" and out["rows"][15]["band"]["band"] == "inside"
    assert out["band"]["decision"] == "band_too_few" and (out["band"]["k_inside"], out["band"]["n_band"]) == (1, 1)
    assert out["not_run"] == [16, 17, 18, 19]
    assert out["drained_split"]["prompt_warm_drained"] == 1 and out["drained_split"]["prompt_warm_never_drained"] == 0
    u = out["u_end"]                                    # seed 16 counts though its P is not run (whatever P's label)
    assert (u["k"], u["n"]) == (1, 2) and set(u["excluded"]) == {17, 18, 19}
    assert u["reference"] == {"k1_arm_a": {"a15": "Linux-A"}, "d1": {"c15": "Linux-D"}}


def test_predicted_after_gate_is_the_regs_numbers():
    """The [REG]'s prediction (author 2026-11-23): three judgeable seeds, unseparable no longer drawn. Pinned to the
    values the [REG] states; the failure control is predicted()'s own five-seed numbers, which must differ."""
    a = T.predicted_after_gate()
    assert a["n_judgeable"] == 3 and abs(a["p_out_other"] - 0.12) < 1e-12
    got = {k: round(v, 3) for k, v in a["outcome"].items()}
    assert got == {"too_few": 0.319, "no_memory": 0.276, "memory": 0.0, "drain_bound": 0.0, "mixed": 0.405}
    assert abs(sum(a["outcome"].values()) - 1) < 1e-9 and abs(sum(a["band"].values()) - 1) < 1e-9
    assert round(a["band"]["band_inside"], 3) == 0.028
    assert round(T.predicted()["given_gate"]["no_memory"], 3) == 0.642        # five seeds: not the [REG]'s number
