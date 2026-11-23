"""analysis/d1r.py: D1r, the session cap switched on at 1680 s as a rescue (C2), and its seal row."""
import fnmatch
import json
import math

import numpy as np
import pandas as pd
import pytest

from analysis import d1 as D
from analysis import d1r as R
from analysis import withheld as W


def test_rescue_config_is_d1s_capped_config_with_the_switch_alone():
    for sd in R.D1R["seeds"]:
        k = R.resolved(sd)
        cap = R.d1_cap(sd)
        assert D._diff_keys(D.resolved(sd, cap), k) == ["name", "workload.sessions.cap_from_s"]
        assert D._diff_keys(D._burst_cfg(sd), k) == ["name", "workload.sessions.cap", "workload.sessions.cap_from_s"]
        assert k["workload"]["sessions"]["cap_from_s"] == 1680 and k["workload"]["sessions"]["cap"] == cap
        assert k["seed"] == sd and k["workload"]["duration_s"] == D.D1["duration_s"]
    assert [R.d1_cap(s) for s in R.D1R["seeds"]] == [29, 25, 29, 26, 25]     # [RESULT] D1's caps


def test_the_rescue_schedule_is_reg19s_and_d1s_row_for_row():
    from harness.workload import build_schedule
    for sd in R.D1R["seeds"]:
        k, b, c = R.resolved(sd), D._burst_cfg(sd), D.resolved(sd, R.d1_cap(sd))
        sk, pk = build_schedule(k["workload"], k["seed"], k["trigger"])
        for other in (b, c):
            so, po = build_schedule(other["workload"], other["seed"], other["trigger"])
            pd.testing.assert_frame_equal(sk, so)
            pd.testing.assert_frame_equal(pk, po)


def test_the_switch_is_the_start_of_reg19s_window_and_established_is_the_span_before():
    b = D._burst_cfg(15)
    end = b["trigger"]["at_s"] + b["trigger"]["duration_s"]
    assert R.D1R["from_s"] == end + 600 and R.D1R["established"] == (end + 300, end + 600)


def test_config_text_refuses_other_seeds_and_writes_new_files_only(tmp_path):
    with pytest.raises(ValueError):
        R.config_text(20)
    paths = R.write_configs(tmp_path)
    assert [p.name for p in paths] == [f"{R.config_name(s)}.yaml" for s in R.D1R["seeds"]]
    with pytest.raises(FileExistsError):
        R.write_configs(tmp_path)


def test_the_rescue_folders_match_no_other_pattern_and_no_other_folder_matches_the_seal():
    from analysis import gate, k1, k3, k3w
    names = [f"20261121-000000_{R.config_name(s)}" for s in R.D1R["seeds"]]
    others = ([gate.reg19_spec(s)[k] for s in D.D1["seeds"] for k in ("control", "burst")]
              + [k1.closed_pattern(s) for s in D.D1["seeds"]] + [k1.open_pattern(2.0, s) for s in D.D1["seeds"]]
              + [k3.REG21["batch2_pattern"], k3w.REG21["batch2_pattern"]] + [D.run_pattern(s) for s in D.D1["seeds"]]
              + [p for reg, p, _ in W.SEALS if reg != "D1r"] + [D.D1["drift_pattern"]])
    assert not [(n, p) for n in names for p in others if fnmatch.fnmatchcase(n, p)]
    assert all(fnmatch.fnmatchcase(n, R.run_pattern(s)) for n, s in zip(names, R.D1R["seeds"]))
    seal = [p for reg, p, _ in W.SEALS if reg == "D1r"]
    assert seal == ["*_burst_l1pl350cl_r2.0_w3000_s1[5-9]_rescue"]
    existing = [p.name for p in (R.REPO / "runs").glob("*")] if (R.REPO / "runs").exists() else []
    existing += [f"20261120-000000_{D.config_name(s)}" for s in D.D1["seeds"]]
    existing += [f"20261120-000000_burst_l1pl350cl_r2.0_w3000_s{s}_cap40" for s in (15, 16, 17)]
    assert not [n for n in existing if fnmatch.fnmatchcase(n, seal[0])]


def test_d1r_seal_lifts_only_on_its_result(tmp_path, monkeypatch):
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-17 · [REG] D1r (C2, cap as a rescue): x\n## 2026-11-17 · [RESULT] D1b (C2, cap above "
                 "the cold batch): x\n## 2026-11-16 · [RESULT] D1 (C2, session cap): x\n")
    monkeypatch.setattr(W, "DECISIONS", d)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    names = [f"20261121-000000_{R.config_name(s)}" for s in R.D1R["seeds"]]
    assert [W.sealed_by(n) for n in names] == ["D1r"] * 5
    d.write_text(d.read_text() + "## 2026-11-18 · [RESULT] D1r (C2, cap as a rescue): x\n")
    assert [W.sealed_by(n) for n in names] == [None] * 5


def test_outcome_is_d1s_rule_with_not_established_left_out():
    cut, st = "recovered", "stuck"
    assert R.outcome([cut, cut, "recovered_cap_draining", "unclear", "not_established"])["decision"] == "rescues"
    o = R.outcome([cut, cut, cut, "not_established", "not_established"])
    assert o["decision"] == "rescues" and o["n"] == 3 and o["n_not_established"] == 2 and o["n_seeds"] == 5
    assert R.outcome([cut, cut, "not_established", "not_established", "not_established"])["decision"] == "too_few"
    assert R.outcome([st, st, st, cut, cut])["decision"] == "refutes"
    assert R.outcome([cut, cut, cut, st, "unclear"])["decision"] == "mixed"         # one stuck blocks rescues
    assert R.outcome(["server_only"] * 3 + [cut, cut])["decision"] == "admission_only"
    with pytest.raises(ValueError):
        R.outcome(["harness_fault", cut, cut])


def test_predicted_matches_draft_1():
    p = R.predicted()
    assert math.isclose(sum(p.values()), 1.0)
    assert p["rescues"] == pytest.approx(0.557, abs=5e-4) and p["mixed"] == pytest.approx(0.430, abs=5e-4)
    assert p["admission_only"] == pytest.approx(0.009, abs=5e-4) and p["refutes"] == pytest.approx(0.005, abs=5e-4)


def test_empty_time_two_phases():
    x = R.D1R["x_after"]
    # nothing to drain: phase B starts at the switch; 100 turns planned at 1700 s are served by about 1700 + 100 / x
    e = R.empty_time(0.0, 25, np.full(100, 1700.0), 1680.0, 4080.0, 3600.0)
    assert e["t_a"] == 1680.0 and e["backlog_peak"] == pytest.approx(100 - x, abs=1e-9)
    assert e["t_empty"] == pytest.approx(1700.0 + math.ceil(100 / x), abs=1.0) and e["meets"] and e["backlog_at_by"] == 0
    # phase A: 25 caps x 4 owed turns stay with the cap; the rest is served at x_cold before the queue is touched
    e = R.empty_time(100.0 + 25 * R.D1R["held_turns"], 25, np.full(10, 1700.0), 1680.0, 4080.0, 3600.0)
    assert e["t_a"] == pytest.approx(1680.0 + 100.0 / R.D1R["x_cold"]) and e["t_empty"] > e["t_a"]
    # a phase A that runs past the end window's start, with the queue fed faster than x_after: never meets
    late = np.linspace(1680.0, 4080.0, int(3.0 * 2400))
    e = R.empty_time(1e5, 25, late, 1680.0, 4080.0, 3600.0)
    assert e["t_a"] > R.D1R["feasible_by"] and not e["meets"]
    # a backlog at 3480 s that falls at x_after - 2.0 req/s meets iff it empties within the horizon
    steady = np.arange(1680.0, 4080.0, 0.5)                                  # 2.0 req/s planned
    e = R.empty_time(0.0, 25, np.concatenate([steady, np.full(1000, 3400.0)]), 1680.0, 4080.0, 3600.0)
    assert e["backlog_at_by"] > 0 and e["meets"] == (e["backlog_at_by"] / (x - e["rate_after"]) <= 3600.0)


# ---------------------------------------------------------------- the switched cap's invariant from the records

def _sess_run(rows, from_s=10.0, dur=40.0):
    q = pd.DataFrame(rows, columns=["session_id", "turn", "t_arrived", "t_admitted", "t_sent", "t_end"])
    q["request_id"] = range(len(q))
    q["prompt_tokens_planned"] = 2176
    return {"requests": q, "cfg": {"workload": {"duration_s": dur}, "trigger": {"kind": "burst", "prompt_tokens": 64}}}


def test_slots_switched_reads_a_correct_run_as_no_fault():
    # cap 1, switch at 10: sessions 0 and 1 admitted at once before it; session 2 arrives at 11 and waits for both
    run = _sess_run([(0, 0, 1.0, 1.0, 1.0, 12.0), (1, 0, 2.0, 2.0, 2.0, 14.0), (2, 0, 11.0, 14.0, 14.0, 20.0)])
    sl = R.slots_switched(run, 1, 10.0)
    assert sl["max_wait_before_s"] == 0.0 and sl["late_admissions"] == 1 and sl["late_admissions_over"] == 0
    assert sl["t_at_cap"] == 12.0 and sl["idle_share"] == 0.0 and not R.harness_fault(sl)


def test_slots_switched_flags_each_fault():
    over = _sess_run([(0, 0, 1.0, 1.0, 1.0, 12.0), (1, 0, 2.0, 2.0, 2.0, 14.0), (2, 0, 11.0, 12.0, 12.0, 20.0)])
    assert R.slots_switched(over, 1, 10.0)["late_admissions_over"] == 1 and R.harness_fault(R.slots_switched(over, 1, 10.0))
    waited = _sess_run([(0, 0, 1.0, 5.0, 5.0, 8.0)])
    assert R.harness_fault(R.slots_switched(waited, 1, 10.0))
    idle = _sess_run([(0, 0, 11.0, 30.0, 30.0, 31.0)])          # waited 19 s with no one in service
    sl = R.slots_switched(idle, 1, 10.0)
    assert sl["idle_share"] > 0.9 and R.harness_fault(sl)


def test_a_switched_fake_run_has_no_harness_fault(tmp_path):
    from analysis.load import load_run
    from tests.test_e2e_fake import run_fake
    from tests.test_session_cap import WL
    run_dir = run_fake(tmp_path, workload={**WL, "sessions": {**WL["sessions"], "cap": 1, "cap_from_s": 10}},
                       trigger={"kind": "burst", "at_s": 8, "duration_s": 2, "rate": 3.0, "prompt_tokens": 64})
    assert json.loads((run_dir / "checks.json").read_text())["valid"]
    sl = R.slots_switched(load_run(run_dir), 1, 10.0)
    assert not R.harness_fault(sl), sl


def test_table_refuses_while_sealed_or_with_d1_unsealed(monkeypatch):
    monkeypatch.setattr(R, "_sealed", lambda: ["x_rescue"])
    with pytest.raises(W.WithheldError):
        R.table()
    monkeypatch.setattr(R, "_sealed", lambda: [])
    monkeypatch.setenv("AFTERSHOCK_UNSEAL", "D1r,D1")
    with pytest.raises(W.WithheldError):
        R.table()


def test_the_committed_configs_are_the_written_text():
    for sd in R.D1R["seeds"]:
        assert (R.CONFIGS / f"{R.config_name(sd)}.yaml").read_text() == R.config_text(sd)
