"""analysis/aiperf_arm.py (AIPerf default-release arm, draft 2; not registered): the verdict on synthetic rows with a
failure control per branch, the release audit, the pairing with T1's U arm, the prediction against draft 1's numbers,
the configs and the seal. No run folder is read."""
import fnmatch
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from analysis import aiperf_arm as A
from analysis import k1, t1
from analysis import withheld as W
from harness.config import load_config

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
NAMES = sorted(f"{k}_l1pl350dr_r2.0_w3000_s{s}" for s in range(15, 20) for k in ("burst", "stage2_control"))


@pytest.fixture
def no_result(tmp_path, monkeypatch):
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-24 · [REG] TR (three turn-release arms): x\n- waits for [RESULT] TR (three turn-release arms)\n")
    monkeypatch.setattr(W, "DECISIONS", d)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    return d


# ---------------------------------------------------------------- constants and labels

def test_constants_come_from_their_sources():
    assert A.AIPERF["seeds"] == (15, 16, 17, 18, 19) == t1.T1["seeds"]
    assert A.AIPERF["min_n"] == 3 == t1.T1["most"]
    assert A.AIPERF["sees_k"] == 3 == k1.REG20["refute_k"]
    assert A.AIPERF["max_attempts"] == 3 and A.AIPERF["release"] == "max_planned"
    p = A.AIPERF["prediction"]
    assert p["burst"] == {"stuck": 0.50, "recovered": 0.45, "partial": 0.05} and p["control_cold_given_cold"] == 0.05
    assert (p["unclear"], p["lost"]) == (0.02, 0.005) and A.AIPERF["id"] == "TR"


def test_labels_are_k1s_renamed():
    k1_labels = ("recovered_open", "partial_open", "stuck_open", "overload_open", "pending_control", "cold_control",
                 "unclear", "lost_open", "not_run")
    assert {A.RENAME.get(x, x) for x in k1_labels} | {"release_fault"} == set(A.LABELS)
    assert set(A.DEFAULT_COLD) == {"stuck_default", "overload_default", "pending_control", "cold_control",
                                   "lost_default"}


# ---------------------------------------------------------------- the decision, one branch at a time

R, P, S, O = "recovered_default", "partial_default", "stuck_default", "overload_default"


@pytest.mark.parametrize("labels, want", [
    # too_few: fewer than 3 counted (unclear and not_run leave n) ...
    ([S, S, "unclear", "unclear", "not_run"], "too_few"),
    ([R, R, "not_run", "not_run", "not_run"], "too_few"),
    # ... failure control: a third counted seed is enough
    ([S, S, S, "unclear", "not_run"], "default_sees"),
    # pending_controls before every reading
    (["pending_control", S, S, S, O], "pending_controls"),
    # failure control: no pending control, the same seeds read on (an overload seed is out of n, TR M6)
    ([R, S, S, S, O], "default_sees"),
    ([O, O, O, S, R], "too_few"),
    ([O, S, S, R, R], "mixed"),
    (["release_fault", R, R, R, R], "default_blind"),                # TR M7: out of n
    # default_sees: stuck + lost + cold_control >= 3
    ([S, "lost_default", "cold_control", R, R], "default_sees"),
    (["lost_default", "lost_default", "lost_default"], "default_sees"),
    # failure control: two of them only
    ([S, "lost_default", R, R, R], "mixed"),
    # default_blind: every counted seed recovered
    ([R, R, R, R, R], "default_blind"),
    ([R, R, R, "unclear", "not_run"], "default_blind"),
    # failure control: one partial, or one stuck
    ([R, R, R, R, P], "mixed"),
    ([R, R, R, R, S], "mixed"),
    # mixed
    ([S, S, R, R, P], "mixed"),
])
def test_decide(labels, want):
    assert A.decide(labels) == want
    assert A.outcome(labels)["decision"] == want


def test_outcome_counts_and_bounds():
    out = A.outcome([S, "lost_default", "cold_control", R, "unclear"])
    assert out["n"] == 4 and out["k_sees"] == 3 and out["counts"]["unclear"] == 1
    from analysis.stats import cp_upper
    assert out["cells"]["sees"] == {"k": 3, "n": 4, "upper": cp_upper(3, 4)}
    assert out["cells"][R] == {"k": 1, "n": 4, "upper": cp_upper(1, 4)}
    assert "unclear" not in out["cells"] and "not_run" not in out["cells"]
    assert A.outcome(["not_run"] * 5)["cells"][S] == {"k": 0, "n": 0, "upper": None}
    with pytest.raises(ValueError):
        A.outcome(["stuck_open", R, R])                          # a k1 label not renamed is refused


# ---------------------------------------------------------------- the per-seed row

def test_default_seed_renames_k1s_labels(monkeypatch):
    for k1_label, want in [*A.RENAME.items(), ("pending_control", "pending_control"), ("unclear", "unclear"),
                           ("cold_control", "cold_control")]:
        monkeypatch.setattr(k1, "open_seed", lambda b, c, lost=False, lab=k1_label: {"label": lab, "mechanism": None})
        assert A.default_seed(None, None)["label"] == want


def test_a_lost_control_beside_a_cold_burst_run_is_cold_control(monkeypatch):
    """TR rule F r1 B1: three invalid control attempts do not leave the seed pending."""
    from tests.test_k1 import make_run
    monkeypatch.setattr(k1, "mechanism", lambda run: {"shown": True, "overlap": 0.0})
    monkeypatch.setattr(k1, "readouts", lambda run: {})
    monkeypatch.setattr(A, "release_audit", lambda run: {"mismatch": 0, "before_prev_end": 0})
    monkeypatch.setattr("analysis.gate.completion_rate", lambda run, a, b: 2.0)
    cold = make_run()
    cold["cfg"] = {**cold["cfg"], "workload": {"rate": 2.0}}
    assert A.default_seed(cold, None, control_lost=True)["label"] == "cold_control"
    assert A.default_seed(cold, None)["label"] == "pending_control"          # control: fewer than 3 attempts
    monkeypatch.setattr(A, "release_audit", lambda run: {"mismatch": 1, "before_prev_end": 0})
    row = A.default_seed(cold, None, control_lost=True)                    # M7: a release fault is out of n
    assert row["label"] == "release_fault" and row["label_before_fault"] == "cold_control"


def test_default_seed_without_a_burst_run_is_not_run_or_lost():
    assert A.default_seed(None, None)["label"] == "not_run"
    assert A.default_seed(None, None, lost=True)["label"] == "lost_default"


def test_an_always_run_control_beside_a_warm_burst_run_is_a_readout_only(monkeypatch):
    """TR: the control always runs. Beside a warm burst run it changes no label; its reading is `control_end`."""
    from tests.test_k1 import WARM, make_run
    monkeypatch.setattr(k1, "mechanism", lambda run: {"shown": True, "overlap": 0.0})
    monkeypatch.setattr(k1, "readouts", lambda run: {})
    monkeypatch.setattr(A, "release_audit", lambda run: {"mismatch": 0, "before_prev_end": 0})
    monkeypatch.setattr("analysis.gate.completion_rate", lambda run, a, b: 2.0)
    warm, cold = make_run(**WARM), make_run()
    for r, n in ((warm, "w"), (cold, "c")):
        r["dir"], r["cfg"] = Path(n), {**r["cfg"], "workload": {"rate": 2.0}}
    row = A.default_seed(warm, cold)
    assert row["label"] == R and row["control"] is None
    assert row["control_end"] == {"run": "c", "cold_backlogged": True, "recovered": False}
    row = A.default_seed(cold, warm)                             # control: a cold burst run reads its control
    assert row["label"] == S and "control_end" not in row


def _run(rows, trigger=None):
    q = pd.DataFrame(rows)
    q["prompt_tokens_planned"] = 1
    return {"requests": q, "cfg": {"trigger": trigger}}


def _clean_rows():
    """Two sessions under max_planned: s0's turn 1 late (released at the previous end), s1's on time."""
    return [dict(session_id=0, turn=0, t_planned=0.0, t_released=0.0, t_sent=0.0, t_end=5.0),
            dict(session_id=0, turn=1, t_planned=3.0, t_released=5.0, t_sent=5.0, t_end=6.0),
            dict(session_id=1, turn=0, t_planned=1.0, t_released=1.0, t_sent=1.0, t_end=2.0),
            dict(session_id=1, turn=1, t_planned=4.0, t_released=4.0, t_sent=4.0, t_end=5.0),
            dict(session_id=1, turn=2, t_planned=9.0, t_released=None, t_sent=None, t_end=None)]


def test_release_audit_on_a_clean_chain():
    a = A.release_audit(_run(_clean_rows()))
    assert a["later_sent"] == 2 and a["mismatch"] == 0 and a["before_prev_end"] == 0
    assert a["share_at_prev_end"] == 0.5 and a["lateness_mean_s"] == 1.0
    assert set(a["lateness_q_s"]) == {"0.5", "0.9", "0.99"}
    assert not A.release_fault(a, {"overlap": 0.0}) and not A.release_fault(a, {"overlap": math.nan})


def test_release_audit_catches_a_wrong_release_and_an_early_send():
    rows = _clean_rows()
    rows[1]["t_released"] = 5.0 + 3.0                           # the closed-loop rule: end + planned gap
    a = A.release_audit(_run(rows))
    assert a["mismatch"] == 1 and a["before_prev_end"] == 0 and A.release_fault(a, None)
    rows = _clean_rows()
    rows[1].update(t_released=3.0, t_sent=3.0)                   # open loop: sent while turn 0 still ran
    a = A.release_audit(_run(rows))
    assert a["mismatch"] == 1 and a["before_prev_end"] == 1 and A.release_fault(a, None)
    clean = A.release_audit(_run(_clean_rows()))
    assert A.release_fault(clean, {"overlap": 0.2})              # a held prefix cannot happen under the rule


def test_release_audit_leaves_burst_rows_out():
    burst = {"kind": "burst", "prompt_tokens": 99}
    rows = _clean_rows() + [dict(session_id=7, turn=0, t_planned=2.0, t_released=None, t_sent=2.0, t_end=3.0)]
    run = _run(rows, burst)
    run["requests"].loc[run["requests"].session_id == 7, "prompt_tokens_planned"] = 99
    assert A.release_audit(run)["later_sent"] == 2


# ---------------------------------------------------------------- the pairing with T1's U arm

def test_paired_u():
    d = {15: S, 16: R, 17: "unclear", 18: "pending_control", 19: R}
    u = {15: True, 16: True, 17: True, 18: False, 19: None}
    out = A.paired_u(d, u)
    assert sorted(out["pairs"]) == [15, 16, 18]                 # 17 out of n, 19 no U reading
    assert (out["b"], out["c"]) == (1, 1)                        # 16: cold only in U; 18: cold only in D
    assert out["p_one_sided"] == k1.mcnemar_one_sided(1, 1)
    none = A.paired_u({15: R}, {15: False})
    assert (none["b"], none["c"], none["p_one_sided"]) == (0, 0, 1.0)


# ---------------------------------------------------------------- the prediction

def test_predicted_with_draft_1s_inputs_under_trs_rule():
    """Draft 1's inputs (no out-of-n terms) through TR's rule (overload out of n, M6): default_sees 0.453,
    mixed 0.522, default_blind 0.024. With TR's out-of-n terms: the [REG]'s numbers (tests/test_tr.py)."""
    p = A.predicted(pred={**A.AIPERF["prediction"], "unclear": 0.0, "lost": 0.0})
    assert p["seed"] == pytest.approx({S: 0.475, O: 0.025, R: 0.45, P: 0.05})
    o = p["outcome"]
    assert {k: round(v, 3) for k, v in o.items()} == {"too_few": 0.0, "pending_controls": 0.0,
                                                     "default_sees": 0.453, "default_blind": 0.024, "mixed": 0.522}
    assert o["default_blind"] == pytest.approx(sum(math.comb(5, j) * 0.45 ** j * 0.025 ** (5 - j)
                                                   for j in range(3, 6)))
    assert sum(o.values()) == pytest.approx(1.0)


def test_predicted_moves_with_its_out_of_n_inputs():
    """Failure control: an unclear or lost mass is not ignored (the draft's computation sets both to 0)."""
    pred = json.loads(json.dumps(A.AIPERF["prediction"]))
    pred.update(unclear=0.0, lost=0.0)
    base = A.predicted(pred=pred)["outcome"]
    pred["unclear"] = 0.3
    o = A.predicted(pred=pred)["outcome"]
    assert o["too_few"] > 0 and sum(o.values()) == pytest.approx(1.0)
    pred = {**pred, "unclear": 0.0, "lost": 0.2}
    assert A.predicted(pred=pred)["outcome"]["default_sees"] > base["default_sees"]
    with pytest.raises(ValueError):
        A.predicted(pred={**pred, "burst": {"stuck": 0.5, "recovered": 0.5, "partial": 0.1}})


# ---------------------------------------------------------------- configs

def test_committed_configs_are_the_generated_ones():
    for sd in A.AIPERF["seeds"]:
        for kind in A.AIPERF["kinds"]:
            assert (CONFIGS / f"{A.config_name(kind, sd)}.yaml").read_text() == A.config_text(kind, sd)
    with pytest.raises(FileExistsError):
        A.write_configs()


def test_resolved_configs_change_only_the_name_and_the_release():
    from analysis.d1 import _diff_keys
    from harness.loadgen import release_mode
    from harness.config import with_defaults
    from harness.run import expand
    for sd in A.AIPERF["seeds"]:
        for kind in A.AIPERF["kinds"]:
            c = A.resolved(kind, sd)
            (parent,) = expand(with_defaults(load_config(CONFIGS / f"{A.parent_name(kind, sd)}.yaml")))
            assert _diff_keys(c, parent) == ["name", "workload.sessions.release"]
            assert release_mode(c) == "max_planned" and c["seed"] == sd


def test_write_configs_writes_ten_new_files(tmp_path):
    paths = A.write_configs(tmp_path)
    assert sorted(p.stem for p in paths) == NAMES


def test_names_and_seeds_are_checked():
    with pytest.raises(ValueError):
        A.config_name("burst", 20)
    with pytest.raises(ValueError):
        A.config_name("confirm", 15)


# ---------------------------------------------------------------- the seal

def test_seal_covers_exactly_the_arms_ten_configs(no_result):
    names = {load_config(c)["name"] for c in CONFIGS.glob("*.yaml") if c.name != "defaults.yaml"}
    sealed = sorted(n for n in names if W.sealed_by(f"20261124-000000_{n}") == "TR" and "_l1pl350dr_" in n)
    assert sealed == NAMES
    for n in NAMES:   # no other seal matches them, so the first match is the arm's
        assert [r[0] for r in W.SEALS if fnmatch.fnmatchcase(f"20261124-000000_{n}", r[1])] == ["TR"]


def test_seal_does_not_collide_with_t1_reg19_or_reg20(no_result):
    (pat,) = [r[1] for r in W.SEALS if r[0] == "TR" and "_l1pl350dr_" in r[1]]
    others = ([f"burst_l1pl350cl_r2.0_w3000_s{s}_{x}" for s in range(15, 20) for x in ("t1u", "t1p", "cap", "rescue")]
              + [f"{k}_l1pl350cl_r2.0_w3000_s{s}" for s in range(15, 20) for k in ("burst", "stage2_control")]
              + [f"{k}_l1pl350ol_r2.0_w3000_s{s}" for s in range(15, 20) for k in ("burst", "stage2_control")]
              + ["burst_l1pl350cl_r2.0_w3000_s15_drift", "burst_l1pl350dr_r2.0_w3000_s150",
                 "burst_l1pl350dr_r2.0_w3000_s15_t1u"])
    assert not any(fnmatch.fnmatchcase(f"20261124-000000_{n}", pat) for n in others)
    assert W.sealed_by("20261124-000000_burst_l1pl350cl_r2.0_w3000_s15_t1u") == "T1"


def test_every_run_folder_matching_the_seal_is_a_registered_d_run(no_result):
    """Written before TR ran as "no folder matches"; since TR's queue the matching folders exist, so the guard is
    that every one of them is a registered D run (a run id is `<stamp>_<config name>`) and no other folder matches."""
    import analysis.load as L
    if not L.RUNS.exists():
        pytest.skip("no runs/ in this checkout (the Mac's main checkout holds them; names checked there by hand)")
    (pat,) = [r[1] for r in W.SEALS if r[0] == "TR" and "_l1pl350dr_" in r[1]]
    names = {A.config_name(k, sd) for k in A.AIPERF["kinds"] for sd in A.AIPERF["seeds"]}
    found = [d.name for d in L.RUNS.iterdir() if fnmatch.fnmatchcase(d.name, pat)]
    assert [f for f in found if f.split("_", 1)[1] not in names] == []


def test_seal_lifts_only_on_its_result_heading(no_result, monkeypatch):
    name = "20261124-000000_burst_l1pl350dr_r2.0_w3000_s17"
    assert W.sealed_by(name) == "TR"                             # the [REG] heading and the body line lift nothing
    monkeypatch.setenv("AFTERSHOCK_UNSEAL", "TR")
    assert W.sealed_by(name) is None
    monkeypatch.delenv("AFTERSHOCK_UNSEAL")
    no_result.write_text(no_result.read_text() + "## 2026-11-30 · [RESULT] TR (three turn-release arms): mixed\n")
    assert W.sealed_by(name) is None
    assert A.AIPERF["seal"] == "[RESULT] TR (three turn-release arms)"


def test_table_refuses_while_a_folder_is_sealed(no_result, tmp_path, monkeypatch):
    import analysis.load as L
    runs = tmp_path / "runs"
    (runs / "20261124-000000_burst_l1pl350dr_r2.0_w3000_s15").mkdir(parents=True)
    monkeypatch.setattr(L, "RUNS", runs)
    assert A.sealed() == ["20261124-000000_burst_l1pl350dr_r2.0_w3000_s15"]
    with pytest.raises(W.WithheldError):
        A.table()
