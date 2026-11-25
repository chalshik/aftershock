"""analysis/armo.py (TR arm O, after T1; not registered), on synthetic rows, synthetic runs and the configs only (no run
folder is read)."""
import fnmatch
from pathlib import Path

import pytest

import analysis.armo as A
import analysis.k1 as K
import analysis.t1 as T
from analysis import withheld as W
from harness.run import load_config
from tests.test_k1 import WARM, make_run
from tests.test_t1 import _u, _warm_u

CONFIGS = Path(__file__).resolve().parents[1] / "configs"
SEEDS = (15, 16, 17, 18, 19)
REC = "recovered_open"


def P(o, u=True, seed=0):
    """One seed's pair from an O label and U's reading (None: U out of n)."""
    return A.pair(seed, {"label": o}, {"cold": u, "out": None if u is not None else "u_fault"})


def pairs(spec):
    return [P(o, u, i) for i, (o, u) in enumerate(spec)]


# ---------------------------------------------------------------- constants

def test_constants_come_from_their_sources():
    assert A.ARMO["seeds"] == T.SEEDS == K.REG20["closed_seeds"] == SEEDS
    assert A.ARMO["lam"] == K.REG20["a_lam"] == 2.0
    assert (A.ARMO["fail_k"], A.ARMO["confounded_k"], A.ARMO["same_stack_k"], A.ARMO["min_n"]) == (3, 2, 4, 3)
    assert A.ARMO["fail_k"] == K.REG20["refute_k"] and A.ARMO["colleague_k"] == 2
    assert A.ARMO["id"] == "TR" and A.ARMO["seal"] == "[RESULT] TR (three turn-release arms)"
    assert A.ARMO["max_attempts"] == 3 and A.ARMO["suffix"] == "t1o"
    assert set(A.O_LABELS) >= set(K.OPEN_COLD) | {REC, "partial_open", "unclear", "not_run"}


# ---------------------------------------------------------------- configs

@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", A.KINDS)
def test_each_config_is_reg20_arm_as_config_renamed(kind, seed):
    f = CONFIGS / f"{A.config_name(kind, seed)}.yaml"
    assert f.read_text() == A.config_text(kind, seed)
    mine, theirs = load_config(f), load_config(CONFIGS / f"{A.parent_name(kind, seed)}.yaml")
    assert mine["name"] == f"{kind}_l1pl350ol_r2.0_w3000_s{seed}_t1o"
    assert {**mine, "name": None} == {**theirs, "name": None}
    assert mine["workload"]["sessions"]["closed_loop"] is False and mine["seed"] == seed


def test_parent_is_k1s_open_pattern_and_write_refuses_existing(tmp_path):
    assert A.parent_name("burst", 15) == "burst_l1pl350ol_r2.0_w3000_s15"
    assert A.parent_name("stage2_control", 19) == "stage2_control_l1pl350ol_r2.0_w3000_s19"
    assert len(A.write_configs(tmp_path)) == 10
    with pytest.raises(FileExistsError):
        A.write_configs(tmp_path)
    with pytest.raises(ValueError):
        A.config_name("burst", 20)


def test_check_config_accepts_the_resolved_config_and_refuses_another():
    cfg = A.resolved("burst", 16)
    A.check_config({"cfg": {**cfg, "run_id": "x"}, "dir": Path("a")}, "burst", 16)
    with pytest.raises(ValueError, match="differs"):
        A.check_config({"cfg": A.resolved("burst", 17), "dir": Path("a")}, "burst", 16)


# ---------------------------------------------------------------- seal

def _o_names():
    return [f"20261124-000000_{A.config_name(k, sd)}" for sd in SEEDS for k in A.KINDS]


def _others():
    out = []
    for sd in SEEDS:
        out += [f"20261124-000000_{x}" for x in
                (f"burst_l1pl350ol_r2.0_w3000_s{sd}", f"stage2_control_l1pl350ol_r2.0_w3000_s{sd}",
                 f"burst_l1pl350cl_r2.0_w3000_s{sd}", f"stage2_control_l1pl350cl_r2.0_w3000_s{sd}",
                 T.config_name("U", sd), T.config_name("P", sd))]
    out += [f"20261124-000000_{k}_l1pl350ol_r2.5_w3000_s{sd}" for sd in (15, 16, 17) for k in A.KINDS]
    return out + ["20261124-000000_burst_l1pl350cl_r2.0_w3000_s15_drift"]


@pytest.fixture
def no_result(tmp_path, monkeypatch):
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-24 · [REG] TR (three turn-release arms): mentions [RESULT] TR (three turn-release arms)\n")
    monkeypatch.setattr(W, "DECISIONS", d)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    return d


def test_seal_rows_match_every_o_name_and_no_other(no_result):
    rows = [r for r in W.SEALS if r[0] == A.ARMO["id"] and r[1].endswith("_t1o")]   # TR's O rows
    assert sorted(r[1] for r in rows) == sorted({A.run_pattern(k, 15).replace("s15", "s1[5-9]") for k in A.KINDS})
    assert all(r[2] == A.ARMO["seal"] for r in rows)
    for n in _o_names():
        assert [r[0] for r in W.SEALS if fnmatch.fnmatchcase(n, r[1])] == ["TR"]   # no other seal, T1's included
        assert W.sealed_by(n) == "TR"
        assert not fnmatch.fnmatch(n, K.open_pattern(2.0, int(n.split("_s")[-1][:2])))   # REG-20's lookup
        assert not fnmatch.fnmatch(n, K.open_pattern(2.0, int(n.split("_s")[-1][:2]), control=True))
    assert not [(n, r[1]) for n in _others() for r in rows if fnmatch.fnmatchcase(n, r[1])]


def test_seal_lifts_on_its_own_result_only(no_result, monkeypatch):
    n = _o_names()[0]
    no_result.write_text(no_result.read_text() + "## 2026-11-25 · [RESULT] T1 (rescue, two arms): no_memory\n")
    assert W.sealed_by(n) == "TR"                               # T1's [RESULT] lifts T1 only
    assert W.sealed_by(f"20261124-000000_{T.config_name('U', 15)}") is None
    monkeypatch.setenv("AFTERSHOCK_UNSEAL", "T1")
    assert W.sealed_by(n) == "TR"                               # T1's unseal id is not TR's
    monkeypatch.setenv("AFTERSHOCK_UNSEAL", "TR")
    assert W.sealed_by(n) is None
    monkeypatch.delenv("AFTERSHOCK_UNSEAL")
    no_result.write_text(no_result.read_text() + "## 2026-11-26 · [RESULT] TR (three turn-release arms): x\n")
    assert W.sealed_by(n) is None


def test_every_run_folder_with_the_suffix_is_a_registered_o_run():
    """Written before TR ran as "no folder carries `_t1o`"; since TR's queue O's folders exist, so the guard is that
    every `_t1o` folder is a registered O run (a run id is `<stamp>_<config name>`)."""
    runs = Path(__file__).resolve().parents[1] / "runs"
    names = [p.name for p in runs.iterdir()] if runs.exists() else []
    o = {A.config_name(k, sd) for k in A.KINDS for sd in A.ARMO["seeds"]}
    assert [x for x in names if x.endswith("_t1o") and x.split("_", 1)[1] not in o] == []


def test_reg20s_lookup_still_finds_one_arm_a_run_beside_an_o_run(monkeypatch):
    folders = {"20261115-064753_burst_l1pl350ol_r2.0_w3000_s15": True,
               "20261124-000000_burst_l1pl350ol_r2.0_w3000_s15_t1o": True,
               "20261124-010000_stage2_control_l1pl350ol_r2.0_w3000_s15_t1o": True}
    monkeypatch.setattr(K, "find_runs", lambda pattern="*", valid_only=True: [
        {"dir": Path(n), "checks": {"valid": ok}} for n, ok in folders.items() if fnmatch.fnmatch(n, pattern)])
    a, bad = K._runs(K.open_pattern(2.0, 15))
    assert a["dir"].name == "20261115-064753_burst_l1pl350ol_r2.0_w3000_s15" and bad == 0
    assert K._runs(K.open_pattern(2.0, 15, control=True)) == (None, 0)
    assert K._runs(A.run_pattern("burst", 15))[0]["dir"].name.endswith("_t1o")


# ---------------------------------------------------------------- O's verdict, one branch at a time (with its control)

def D(labels, **kw):
    return A.decide_o(labels, **kw)


def test_o_verdict_too_few_below_three_counted_seeds():
    assert D([REC] * 2 + ["unclear", "not_run"]) == "too_few"
    assert D([REC] * 3) != "too_few"                                             # control: n = 3


def test_o_verdict_pending_controls_first_after_too_few():
    spec = [REC] * 3 + ["pending_control", "stuck_open"]
    assert D(spec) == "pending_controls"
    spec[3] = REC                                                               # control: no pending seed
    assert D(spec) != "pending_controls"


def test_o_verdict_overload_seeds_are_out_of_n():
    spec = [REC] + ["overload_open"] + ["stuck_open"] * 3
    assert D(spec) == "lead_fails" and A.outcome_o(spec)["k_overload"] == 1 and A.outcome_o(spec)["n"] == 4
    assert D(["overload_open"] * 3 + [REC] * 2) == "too_few"


def test_consequence_needs_same_stack_for_appendix_f():
    assert A.consequence("replicates", "same_stack") == "appendix_f_same_stack"
    for u in ("stack_confounded", "weak", "too_few", None):
        assert A.consequence("replicates", u) == "arm_a_repeats_today"
    assert A.consequence("lead_fails", "same_stack") == "lead_withdrawn"
    assert A.consequence("mixed", "same_stack") == "no_sentence"


def test_o_row_lost_control_beside_a_cold_burst_is_cold_control(monkeypatch, quiet):
    cold = make_run()
    folders = {"20261124-000000_" + A.config_name("burst", 15): (cold, True)}
    folders.update({f"2026112{i}-100000_{A.config_name('stage2_control', 15)}": (cold, False) for i in range(3)})
    folders.update({"20261124-000000_" + A.config_name("burst", 16): (cold, True),
                    "20261124-100000_" + A.config_name("stage2_control", 16): (cold, False)})
    _fake(monkeypatch, folders)
    assert A.o_row(15)["label"] == "cold_control" and A.o_row(15)["control_lost"]
    assert A.o_row(16)["label"] == "pending_control"                            # control: one invalid attempt


@pytest.mark.parametrize("cold", ["stuck_open", "cold_control", "lost_open"])
def test_o_verdict_lead_fails_at_fail_k_three(cold):
    spec = [REC] * 2 + [cold, "stuck_open", "stuck_open"]
    assert D(spec) == "lead_fails"
    spec[4] = REC                                                               # control: two failing seeds
    assert D(spec) == "mixed"


def test_o_verdict_colleague_k_two_is_a_readout():
    spec = [REC] * 3 + ["stuck_open", "lost_open"]
    assert D(spec) == "mixed" and D(spec, fail_k=2) == "lead_fails"
    out = A.outcome_o(spec)
    assert out["decision"] == "mixed" and out["decision_at_colleague_k"] == "lead_fails"


def test_o_verdict_partial_does_not_count_toward_lead_fails():
    assert D([REC] * 3 + ["partial_open"] * 2) == "mixed"


def test_o_verdict_replicates_only_when_every_counted_seed_recovered():
    assert D([REC] * 4 + ["unclear"]) == "replicates"
    assert D([REC] * 4 + ["partial_open"]) == "mixed"                          # control


def test_o_verdict_out_of_n_and_unknown_labels():
    assert D([REC] * 3 + ["not_run", "unclear"]) == "replicates"
    with pytest.raises(ValueError):
        D(["cold"] * 3)


def test_o_verdict_never_reads_t1(monkeypatch):
    """TR: O's verdict must not depend on T1's runs; `outcome_o` takes labels only and t1 is not touched."""
    monkeypatch.setattr(T, "arm_run", lambda *a, **k: pytest.fail("O's verdict read T1"))
    monkeypatch.setattr(T, "u_end_readout", lambda *a, **k: pytest.fail("O's verdict read T1"))
    assert A.outcome_o([REC] * 5)["decision"] == "replicates"


def test_outcome_o_reports_k_of_n_with_upper_bounds():
    from analysis.stats import cp_upper
    out = A.outcome_o([REC] * 4 + ["unclear"])
    assert out["decision"] == "replicates" and out["n"] == 4 and out["k_overload"] == 0
    assert out["cells"]["recovered_open"] == {"k": 4, "n": 4, "upper": 1.0}
    assert out["cells"]["fail"] == {"k": 0, "n": 4, "upper": cp_upper(0, 4)}


# ---------------------------------------------------------------- the U readout (after T1's [RESULT])

def test_u_reading_too_few_and_stack_confounded():
    assert A.u_reading(pairs([(REC, True)] * 2 + [(REC, None)] * 3)) == "too_few"
    spec = [(REC, True)] * 2 + [(REC, False)] * 3
    assert A.u_reading(pairs(spec)) == "stack_confounded"
    spec[2] = (REC, True)                                                       # control: U cold in 3
    assert A.u_reading(pairs(spec)) != "stack_confounded"


def test_u_reading_same_stack_at_four_pairs():
    spec = [(REC, True)] * 4 + [(REC, False)]
    assert A.u_reading(pairs(spec)) == "same_stack"
    spec[3] = ("partial_open", True)                                            # control: three pairs
    assert A.u_reading(pairs(spec)) == "weak"


def test_u_reading_weak_and_out_of_n():
    assert A.u_reading(pairs([(REC, True)] * 3)) == "weak"                      # n = 3 cannot reach same_stack_k
    ps = pairs([("unclear", True), ("not_run", True), (REC, None), ("lost_open", True), (REC, True)])
    assert [p["counted"] for p in ps] == [False, False, False, True, True]
    assert ps[0]["out"] == ["o_unclear"] and ps[2]["out"] == ["u_fault"]
    with pytest.raises(ValueError):
        P("cold")


def test_u_ready_only_on_t1s_result_heading(tmp_path):
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-24 · [REG] TR (three turn-release arms): waits for [RESULT] T1 (rescue, two arms)\n")
    assert A.u_ready(d) is False
    d.write_text(d.read_text() + "## 2026-11-24 · [RESULT] T1 (rescue, two arms): mixed\n")
    assert A.u_ready(d) is True


def test_u_status_reads_t1s_u_end_readout():
    nan_u = _u()
    nan_u["metrics"].loc[nan_u["metrics"].t >= 3480, "vllm:num_requests_waiting"] = float("nan")
    fault = _u()
    fault["requests"]["t_arrived"] = float("nan")
    u_end = T.u_end_readout({15: _u(), 16: _warm_u(), 17: None, 18: fault, 19: nan_u})
    got = {sd: A.u_status(sd, u_end) for sd in SEEDS}
    assert got[15] == {"cold": True, "out": None} and got[16] == {"cold": False, "out": None}
    assert got[17]["out"] == "no valid U run" and got[18]["out"] == "u_fault" and got[19]["out"] == "u_nan"


# ---------------------------------------------------------------- O rows on fake folders

@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(K, "readouts", lambda run: {})
    monkeypatch.setattr(K, "mechanism", lambda run: {"shown": True})
    monkeypatch.setattr(A, "check_config", lambda run, kind, seed: None)


def _fake(monkeypatch, folders):
    def find_runs(pattern="*", valid_only=True):
        out = [{**run, "dir": Path(n), "checks": {"valid": ok}} for n, (run, ok) in sorted(folders.items())
               if fnmatch.fnmatch(n, pattern)]
        return [r for r in out if not valid_only or r["checks"]["valid"]]
    monkeypatch.setattr(K, "find_runs", find_runs)


def test_o_row_lost_only_after_three_invalid_attempts(monkeypatch, quiet):
    cold, warm = make_run(), make_run(**WARM)
    o = lambda sd, i: f"2026112{i}-000000_{A.config_name('burst', sd)}"
    folders = {o(15, i): (warm, False) for i in range(3)}
    folders.update({o(16, i): (warm, False) for i in range(2)})
    folders.update({o(17, 0): (warm, False), o(17, 1): (cold, True),
                    f"20261124-100000_{A.config_name('stage2_control', 17)}": (warm, True), o(18, 0): (warm, True)})
    _fake(monkeypatch, folders)
    assert A.o_row(15)["label"] == "lost_open" and A.o_row(15)["invalid_attempts"] == 3
    assert A.o_row(16)["label"] == "not_run"                                     # control: two invalid attempts
    assert A.o_row(17)["label"] == "stuck_open" and A.o_row(18)["label"] == REC
    assert A.o_row(19)["label"] == "not_run"


def test_table_refuses_while_an_o_folder_is_sealed(monkeypatch):
    monkeypatch.setattr(A, "sealed", lambda: ["x_t1o"])
    with pytest.raises(W.WithheldError):
        A.table()


def _o_folders():
    warm = make_run(**WARM)
    folders = {f"20261124-0000{sd}_{A.config_name('burst', sd)}": (warm, True) for sd in SEEDS}
    folders.update({f"20261115-0000{sd}_burst_l1pl350ol_r2.0_w3000_s{sd}": (warm, True) for sd in SEEDS})
    return folders


def test_table_judges_o_without_t1_before_t1s_result(monkeypatch, quiet):
    _fake(monkeypatch, _o_folders())
    monkeypatch.setattr(A, "sealed", lambda: [])
    monkeypatch.setattr(A, "u_ready", lambda decisions=None: False)
    monkeypatch.setattr(T, "arm_run", lambda *a, **k: pytest.fail("T1 read before its [RESULT]"))
    t = A.table()
    assert t["outcome"]["decision"] == "replicates" and t["u"] == {"withheld": f"computed only after '{T.T1['seal']}'"}
    assert t["twins"][15]["label"] == {"arm_a": REC, "o": REC}


def test_table_adds_the_u_readout_after_t1s_result(monkeypatch, quiet):
    _fake(monkeypatch, _o_folders())
    us = {15: _u(), 16: _u(), 17: _u(), 18: _u(), 19: _warm_u()}
    monkeypatch.setattr(T, "arm_run", lambda arm, sd, fs: (us[sd], []))
    monkeypatch.setattr(T, "sealed", lambda: [])
    monkeypatch.setattr(A, "sealed", lambda: [])
    monkeypatch.setattr(A, "u_ready", lambda decisions=None: True)
    t = A.table()
    assert t["outcome"]["decision"] == "replicates"
    u = t["u"]
    assert u["reading"]["label"] == "same_stack" and u["reading"]["cells"]["k_same_stack"]["k"] == 4
    assert [p["u_cold"] for p in u["pairs"]] == [True] * 4 + [False]
    assert u["reg20_style"]["k_closed"] == 4 and u["reg20_style"]["k_open"] == 0
    assert u["platforms"]["one_stack"] is False                                   # no env.json platform here


def test_platforms_one_stack_only_when_every_run_shares_one_string():
    r = lambda p: {"env": {"platform": p}}
    a, b = "Linux-6.8.0-149", "Linux-6.8.0-147"
    assert A.platforms({15: r(a)}, {15: r(a)}, {15: r(a)})["one_stack"] is True
    assert A.platforms({15: r(a)}, {15: r(b)}, {15: r(a)})["one_stack"] is False
    assert A.platforms({15: r(a)}, {15: {"env": {}}}, {15: r(a)})["one_stack"] is False
    assert A.platforms({15: None}, {15: None}, {})["one_stack"] is False


# ---------------------------------------------------------------- prediction

def test_predicted_is_exact_and_sums_to_one():
    p = A.predicted()
    assert p["p_o_recovered_per_seed"] == pytest.approx(0.85 ** 0.2)
    for key in ("outcome", "outcome_at_colleague_k", "u_reading"):
        assert sum(p[key].values()) == pytest.approx(1.0)
    o = p["outcome"]
    assert o["pending_controls"] == 0.0                                          # every cold seed has its control
    # failure control: with every seed recovered and nothing out of n, `replicates` is certain
    q = A.predicted({"p_o_recovered_all": 1.0, "p_out_o": 0.0})["outcome"]
    assert q["replicates"] == pytest.approx(1.0)
    assert o["lead_fails"] <= p["outcome_at_colleague_k"]["lead_fails"]           # 3 fails no earlier than 2


def test_predicted_reproduces_draft_1_without_out_of_n_terms():
    """Draft 1's inputs (no out-of-n, no overload) give draft 1's U-side numbers through the U readout."""
    p = A.predicted({"p_out_o": 0.0, "p_out_u": 0.0, "p_control_cold_given_cold": 0.0})
    assert round(p["u_reading"]["same_stack"], 3) == 0.945 and round(p["p_mcnemar_le_alpha"], 2) == 0.66
    assert round(p["outcome_at_colleague_k"]["lead_fails"], 3) == 0.010


def test_predicted_out_of_n_terms_reach_too_few():
    q = A.predicted({"p_out_o": 0.5})["outcome"]
    assert q["too_few"] > 0.4 and sum(q.values()) == pytest.approx(1.0)
