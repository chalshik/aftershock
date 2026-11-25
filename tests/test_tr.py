"""analysis/tr.py (TR: arms O, H, D in one queue after T1; not registered): H's per-seed labels with a failure control
each, out-of-n rules, H's verdict and the joint label, the exact prediction, the configs, the seals, glob separation,
the queue order and the launcher plan. Synthetic runs only (tests/test_k1.make_run); no run folder is read."""
import fnmatch
import importlib.util
import json
import math
import sys
from pathlib import Path

import pytest

from analysis import aiperf_arm, armo, b1, gate, k1, t1
from analysis import tr as TR
from analysis import withheld as W
from harness.config import load_config
from tests.test_k1 import WARM, make_run

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
SEEDS = (15, 16, 17, 18, 19)
REC = "recovered_open"


@pytest.fixture
def quiet(monkeypatch):
    monkeypatch.setattr(k1, "readouts", lambda run: {})
    monkeypatch.setattr(k1, "mechanism", lambda run: {"shown": True})


@pytest.fixture
def no_result(tmp_path, monkeypatch):
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-24 · [REG] TR (three turn-release arms): waits for [RESULT] TR (three turn-release arms)\n")
    monkeypatch.setattr(W, "DECISIONS", d)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    return d


# ---------------------------------------------------------------- constants

def test_constants_and_their_sources():
    assert TR.TR["h_lam"] == round(2.0 * 3.0 / 2.2, 2) == 2.73
    assert TR.TR["fail_k"] == k1.REG20["refute_k"] == armo.ARMO["fail_k"] == 3
    assert TR.TR["seeds"] == SEEDS == armo.ARMO["seeds"] == aiperf_arm.AIPERF["seeds"]
    assert TR.TR["h_seeds"] == 5 and TR.TR["min_n"] == 3 and TR.TR["max_attempts"] == 3
    assert TR.TR["gb_per_run"] == round((12430552 - 10755812) / 7 / 1e6, 3)
    assert TR.TR["free_after_t1_kb"] == 10755812 - round((12430552 - 10755812) / 7)
    assert armo.ARMO["seal"] == aiperf_arm.AIPERF["seal"] == TR.TR["seal"]


def test_drain_margin_fits_the_inherited_end_window():
    d = TR.drain_margin()
    assert d["outstanding"] == pytest.approx(383.4) and d["drain_s"] == pytest.approx(1420.0)
    assert d["drain_s"] < d["gap_s"] == 2400
    assert TR.drain_margin(capacity=2.9)["drain_s"] == pytest.approx(2361.2, abs=0.1)   # not assured at <= 2.9
    assert TR.drain_margin(capacity=2.73)["drain_s"] == math.inf                      # failure control


# ---------------------------------------------------------------- H per seed

def test_h_seed_counts_only_beside_a_recovered_control(quiet):
    warm, cold = make_run(**WARM, name="w"), make_run(name="c")
    assert TR.h_seed(warm, warm)["label"] == REC
    assert TR.h_seed(cold, warm)["label"] == "stuck_open"
    assert TR.h_seed(make_run(reuse_after=0.96, q_after=200.0), warm)["label"] == "partial_open"
    # the control decides first: cold, not recovered, unreadable
    assert TR.h_seed(warm, cold)["label"] == "control_cold"                         # a warm burst run is out of n too
    assert TR.h_seed(cold, cold)["label"] == "control_cold"                         # never overload_open
    assert TR.h_seed(warm, make_run(reuse_after=0.96, q_after=200.0))["label"] == "control_partial"
    assert TR.h_seed(warm, make_run(gap_at=3600))["label"] == "control_unclear"
    assert TR.h_seed(make_run(gap_at=3600), warm)["label"] == "unclear"


def test_h_seed_missing_runs():
    warm = make_run(**WARM)
    assert TR.h_seed(None, None)["label"] == "not_run"
    assert TR.h_seed(None, warm, lost=True)["label"] == "lost_open"
    assert TR.h_seed(warm, None)["label"] == "control_not_run"
    assert TR.h_seed(warm, None, control_lost=True)["label"] == "control_lost"


def test_h_seed_reads_the_control_on_the_burst_runs_windows(quiet, monkeypatch):
    seen = []
    real = k1.cold_backlogged
    monkeypatch.setattr(k1, "cold_backlogged", lambda r, win=None, rules=None: seen.append(win) or real(r, win, rules))
    warm = make_run(**WARM)
    TR.h_seed(warm, warm)
    assert seen[0] == k1.windows(warm["cfg"]) == {"base": (300.0, 900.0), "end": (3480.0, 4080.0)}


def test_h_row_lost_after_three_invalid_attempts(monkeypatch, quiet):
    warm = make_run(**WARM)
    b = lambda sd, i: f"2026112{i}-000000_{TR.config_name('burst', sd)}"
    c = lambda sd, i: f"2026112{i}-100000_{TR.config_name('stage2_control', sd)}"
    folders = {b(15, i): (warm, False) for i in range(3)}
    folders.update({b(16, 0): (warm, True), **{c(16, i): (warm, False) for i in range(3)}})
    folders.update({b(17, 0): (warm, True), c(17, 0): (warm, True), b(18, 0): (warm, False), b(18, 1): (warm, False)})

    def find_runs(pattern="*", valid_only=True):
        out = [{**r, "dir": Path(n), "checks": {"valid": ok}} for n, (r, ok) in sorted(folders.items())
               if fnmatch.fnmatch(n, pattern)]
        return [r for r in out if not valid_only or r["checks"]["valid"]]
    monkeypatch.setattr(k1, "find_runs", find_runs)
    monkeypatch.setattr(TR, "check_config", lambda run, kind, seed: None)
    got = {sd: TR.h_row(sd)["label"] for sd in SEEDS}
    assert got == {15: "lost_open", 16: "control_lost", 17: REC, 18: "not_run", 19: "not_run"}


# ---------------------------------------------------------------- H's verdict, one branch at a time

def test_decide_h_cell_unstable_first():
    assert TR.decide_h(["control_cold", "control_partial", "control_cold", "stuck_open", "stuck_open"]) == \
        "cell_unstable"
    assert TR.decide_h(["control_cold", "control_partial", "control_lost", "stuck_open", "stuck_open"]) == \
        "too_few"                                                                  # control: a lost control is not
    out = TR.outcome_h(["control_cold"] * 3 + [REC] * 2)
    assert out["decision"] == "cell_unstable" and out["k_unstable"] == 3


def test_decide_h_too_few():
    assert TR.decide_h([REC, REC, "control_cold", "control_partial", "unclear"]) == "too_few"
    assert TR.decide_h([REC, REC, REC, "control_cold", "not_run"]) == "headroom_excluded"    # control: n = 3


def test_decide_h_headroom_explains_at_three():
    spec = [REC, REC, "stuck_open", "stuck_open", "lost_open"]
    assert TR.decide_h(spec) == "headroom_explains"
    spec[4] = "partial_open"                                                       # control: two failing seeds
    assert TR.decide_h(spec) == "mixed"


def test_decide_h_excluded_needs_no_stuck_seed_and_allows_partial():
    assert TR.decide_h([REC] * 4 + ["control_cold"]) == "headroom_excluded"
    assert TR.decide_h([REC] * 3 + ["partial_open"] * 2) == "headroom_excluded"           # B5: partial allowed
    # rule F r2 N1: partial does not count toward the 3 recovered
    assert TR.decide_h(["partial_open"] * 5) != "headroom_excluded"
    assert TR.decide_h(["partial_open"] * 5) == "mixed"
    assert TR.decide_h([REC] * 2 + ["partial_open"] * 3) == "mixed"
    assert TR.decide_h(["partial_open"] * 3 + ["control_cold"] * 2) == "mixed"
    assert TR.decide_h([REC] * 4 + ["stuck_open"]) == "mixed"                            # control: one stuck
    assert TR.decide_h([REC] * 4 + ["lost_open"]) == "mixed"
    with pytest.raises(ValueError):
        TR.decide_h(["overload_open"] * 3)


def test_outcome_h_cells_and_consequence():
    out = TR.outcome_h([REC] * 3 + ["stuck_open", "control_cold"])
    assert out["n"] == 4 and out["cells"]["stuck_open"]["k"] == 1 and out["decision"] == "mixed"
    assert TR.h_consequence("headroom_explains") == "author_words_k1_with_h_counts"     # B5: H never withdraws alone
    assert TR.h_consequence("headroom_excluded") == "s44_headroom_excluded"
    assert TR.h_consequence("mixed") == TR.h_consequence("too_few") == "no_sentence"


@pytest.mark.parametrize("o, h, want", [
    ("lead_fails", "headroom_excluded", "lead_fails"),
    ("replicates", "headroom_explains", "lead_fails"),
    ("replicates", "headroom_excluded", "lead_holds"),
    ("replicates", "mixed", "lead_unsettled"),
    ("mixed", "headroom_excluded", "lead_unsettled"),
    ("too_few", "too_few", "lead_unsettled"),
    ("pending_controls", "cell_unstable", "lead_unsettled"),
])
def test_joint(o, h, want):
    assert TR.joint(o, h) == want


def test_joint_refuses_unknown_labels():
    with pytest.raises(ValueError):
        TR.joint("concurrent", "mixed")


# ---------------------------------------------------------------- prediction

def test_h_seed_dist_sums_and_refuses_bad_inputs():
    s = TR.h_seed_dist()
    assert sum(s.values()) == pytest.approx(1.0)
    assert sum(v for x, v in s.items() if x in TR.H_OUT) == pytest.approx(0.32)
    with pytest.raises(ValueError):
        TR.h_seed_dist({"given_counted": {REC: 0.5}})


def test_predicted_h_is_exact():
    p = TR.predicted_h()
    assert sum(p["outcome"].values()) == pytest.approx(1.0)
    q = 0.32
    assert p["p_n_lt_min"] == pytest.approx(sum(math.comb(5, j) * (1 - q) ** j * q ** (5 - j) for j in range(3)))
    assert p["outcome"]["cell_unstable"] == pytest.approx(p["p_unstable"])
    assert p["p_unstable"] == pytest.approx(sum(math.comb(5, j) * 0.3 ** j * 0.7 ** (5 - j) for j in range(3, 6)))
    # failure control: every seed counted and recovered gives headroom_excluded for certain
    sure = {"control_cold": 0, "control_partial": 0, "control_lost": 0, "control_unclear": 0, "burst_unclear": 0,
            "given_counted": {REC: 1.0, "partial_open": 0, "stuck_open": 0, "lost_open": 0}}
    assert TR.predicted_h(sure)["outcome"]["headroom_excluded"] == pytest.approx(1.0)
    assert TR.predicted_h(n_seeds=3)["p_n_lt_min"] > p["p_n_lt_min"]


def test_predicted_reproduces_the_registration():
    """The numbers in docs/drafts/tr_draft2.md's Prediction section."""
    p = TR.predicted()
    r = lambda d: {k: round(v, 3) for k, v in d.items()}
    assert r(p["o"]["outcome"]) == {"too_few": 0.0, "pending_controls": 0.0, "lead_fails": 0.0,
                                    "replicates": 0.86, "mixed": 0.14}
    assert round(p["o"]["outcome_at_colleague_k"]["lead_fails"], 3) == 0.008
    assert r(p["o"]["consequence"]) == {"appendix_f_same_stack": 0.779, "no_sentence": 0.14,
                                        "arm_a_repeats_today": 0.081, "lead_withdrawn": 0.0}
    assert r(p["h"]["outcome"]) == {"cell_unstable": 0.163, "too_few": 0.027, "headroom_explains": 0.015,
                                    "headroom_excluded": 0.301, "mixed": 0.493}
    assert round(p["h_3_seeds"]["p_n_lt_min"], 3) == 0.686
    assert r(p["d"]["outcome"]) == {"too_few": 0.001, "pending_controls": 0.0, "default_sees": 0.44,
                                    "default_blind": 0.026, "mixed": 0.533}
    assert r(p["joint"]) == {"lead_fails": 0.015, "lead_holds": 0.259, "lead_unsettled": 0.725}
    assert sum(p["joint"].values()) == pytest.approx(1.0)
    c = TR.cost()
    assert (c["runs"], c["gpu_h"], c["gpu_h_max"], c["queue_h_at_wait"]) == (30, 37.5, 112.5, 55.0)
    assert c["disk_gb"] == pytest.approx(7.17) and c["slack_gb"] == pytest.approx(1.249, abs=1e-3)


# ---------------------------------------------------------------- configs

def test_h_configs_are_the_generated_ones_and_change_rate_and_name_only():
    from analysis.d1 import _diff_keys
    from harness.config import with_defaults
    from harness.run import expand
    for sd in SEEDS:
        for kind in TR.KINDS:
            assert (CONFIGS / f"{TR.config_name(kind, sd)}.yaml").read_text() == TR.config_text(kind, sd)
            c = TR.resolved(kind, sd)
            (parent,) = expand(with_defaults(load_config(CONFIGS / f"{TR.parent_name(kind, sd)}.yaml")))
            assert _diff_keys(c, parent) == ["name", "workload.rate"]
            assert c["workload"]["rate"] == 2.73 and c["seed"] == sd
            assert c["workload"]["sessions"]["closed_loop"] is False
            if kind == "burst":
                assert c["trigger"] == parent["trigger"] and c["trigger"]["rate"] == 2.4   # the burst is unchanged
    with pytest.raises(FileExistsError):
        TR.write_configs()


def test_write_configs_and_names(tmp_path):
    assert len(TR.write_configs(tmp_path)) == 10
    with pytest.raises(ValueError):
        TR.config_name("burst", 20)


# ---------------------------------------------------------------- seals and glob separation

def _tr_names():
    return [TR.arm_config(a, k, sd) for a in "OHD" for k in TR.KINDS for sd in SEEDS]


def test_every_tr_name_is_sealed_by_tr_only(no_result):
    names = _tr_names()
    assert len(set(names)) == 30
    for n in names:
        f = f"20261125-000000_{n}"
        assert [r[0] for r in W.SEALS if fnmatch.fnmatchcase(f, r[1])] == ["TR"], n
        assert W.sealed_by(f) == "TR"


def test_no_other_committed_config_is_sealed_by_tr(no_result):
    names = {load_config(c)["name"] for c in CONFIGS.glob("*.yaml") if c.name != "defaults.yaml"}
    assert sorted(n for n in names if W.sealed_by(f"20261125-000000_{n}") == "TR") == sorted(_tr_names())


def test_tr_lifts_only_on_its_result(no_result, monkeypatch):
    f = f"20261125-000000_{TR.config_name('burst', 17)}"
    no_result.write_text(no_result.read_text() + "## 2026-11-25 · [RESULT] T1 (rescue, two arms): x\n")
    assert W.sealed_by(f) == "TR"
    monkeypatch.setenv("AFTERSHOCK_UNSEAL", "T1")
    assert W.sealed_by(f) == "TR"
    monkeypatch.setenv("AFTERSHOCK_UNSEAL", "TR")
    assert W.sealed_by(f) is None
    monkeypatch.delenv("AFTERSHOCK_UNSEAL")
    no_result.write_text(no_result.read_text() + "## 2026-11-27 · [RESULT] TR (three turn-release arms): x\n")
    assert W.sealed_by(f) is None


def test_no_existing_pattern_picks_up_a_tr_run():
    pats = [r[1] for r in W.SEALS if r[0] != "TR"] + [gate.SURGE["pattern"], *b1.ARMS.values()]
    pats += [k1.open_pattern(lam, sd, control=c) for lam in (2.0, 2.5, 2.7, 2.73) for sd in SEEDS for c in (0, 1)]
    pats += [b1.BURST_OF_CONTROL.format(seed=sd) for sd in SEEDS]
    for n in _tr_names():
        f = f"20261125-000000_{n}"
        assert not [p for p in pats if fnmatch.fnmatchcase(f, p)], n
    # failure control: REG-20 arm A's own names are matched by its own pattern
    assert fnmatch.fnmatchcase("20261115-000000_burst_l1pl350ol_r2.0_w3000_s15", k1.open_pattern(2.0, 15))


def test_tr_sealed_lists_all_three_arms(no_result, tmp_path, monkeypatch):
    import analysis.load as L
    runs = tmp_path / "runs"
    for n in (TR.config_name("burst", 15), armo.config_name("burst", 15), aiperf_arm.config_name("burst", 15)):
        (runs / f"20261125-000000_{n}").mkdir(parents=True)
    (runs / "20261115-000000_burst_l1pl350ol_r2.0_w3000_s15").mkdir()
    monkeypatch.setattr(L, "RUNS", runs)
    assert len(TR.sealed()) == 3
    with pytest.raises(W.WithheldError):
        TR.table()


# ---------------------------------------------------------------- queue and plan

def test_queue_order():
    q = TR.queue()
    assert len(q) == 30 and len({i["run_name"] for i in q}) == 30
    keys = [i["key"] for i in q]
    assert keys[:6] == ["O-15", "Oc-15", "H-15", "Hc-15", "D-15", "Dc-15"]
    assert keys[6:12] == ["Hc-16", "H-16", "Dc-16", "D-16", "Oc-16", "O-16"]
    for sd in SEEDS:                                    # each seed's six runs are consecutive: a stop keeps seeds whole
        block = keys[(sd - 15) * 6:(sd - 14) * 6]
        assert all(k.endswith(f"-{sd}") for k in block)
        assert "".join(k[0] for k in block[::2]) == TR.ARM_ORDER[sd]
    # each arm takes each place (first, second, third in the seed) at least once
    for arm in "OHD":
        assert {TR.ARM_ORDER[sd].index(arm) for sd in SEEDS} == {0, 1, 2}
    for i in q:
        assert i["config"] == f"configs/{i['run_name']}.yaml" and (REPO / i["config"]).exists()
        assert load_config(REPO / i["config"])["name"] == i["run_name"] and i["wait_min"] == 110


def test_committed_plan_is_the_generated_one_and_passes_the_launcher():
    if not TR.PLAN.exists():
        pytest.skip("the launcher plan (docs/probe/tr_ops/plan.json, with the GPU host's address) is not in the "
                    "public release")
    assert TR.PLAN.read_text() == TR.plan_text()
    p = json.loads(TR.PLAN.read_text())
    t1p = json.loads((REPO / "docs/probe/t1_ops/plan.json").read_text())
    assert p["prelaunch"] == t1p["prelaunch"] and p["stop"] == t1p["stop"] and p["vm"] == t1p["vm"]
    assert set(t1p["locks"]) < set(p["locks"]) and "docs/probe/t1_ops/state/queue.lock" in p["locks"]
    assert set(t1p["after_pats"]) < set(p["after_pats"])
    import re
    heads = {"reg_pat": "## 2026-11-24 · [REG] TR (three turn-release arms): arms O, H and D",
             "approve_pat": "## 2026-11-24 · [DECISION] (author, pre-approved) TR starts: 30 runs"}
    for k, line in heads.items():
        assert re.search(p[k], line)
    assert not re.search(p["approve_pat"], "## 2026-11-24 · [DECISION] (author) TR starts")   # the pre-approved heading only
    assert re.search(p["after_pats"][-1], "## 2026-11-24 · [RESULT] T1 (rescue, two arms): x")
    assert not re.search(p["after_pats"][-1], "## 2026-11-24 · [REG] T1 (rescue, two arms): x")
    spec = importlib.util.spec_from_file_location("launcher", REPO / "tools/launcher.py")
    L = sys.modules.setdefault("launcher", importlib.util.module_from_spec(spec))
    if not hasattr(L, "load_plan"):
        spec.loader.exec_module(L)
    assert L.load_plan(TR.PLAN, REPO)["id"] == "TR"
