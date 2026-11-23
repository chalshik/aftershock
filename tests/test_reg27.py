"""analysis/reg27.py (REG-24's design on four more seeds): configs, picks, the count, seals. No run folder is read."""
import fnmatch
import math

import pytest

from analysis import confirm as C
from analysis import reg27 as R
from analysis import withheld

SEEDS = R.REG27["seeds"]


def _flat(d, p=""):
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, f"{p}{k}."))
        else:
            out[f"{p}{k}"] = v
    return out


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", ["burst", "control"])
def test_configs_are_reg24s_but_for_name_and_seed(seed, kind):
    """Each committed config resolves to REG-24's seed-43 config except `name` and `seed`, and to the design function's
    config at (seed, 9000 s) on every key."""
    name = C.REG24[f"{kind}_name"]
    got, ref = _flat(C._resolved(name.format(seed=seed))), _flat(C._resolved(name.format(seed=43)))
    assert sorted(k for k in set(got) | set(ref) if got.get(k) != ref.get(k)) == ["name", "seed"]
    design = (C.burst_cfg if kind == "burst" else C.control_cfg)(seed, R.REG27["drop_s"])
    d = _flat(design)
    assert sorted(k for k in set(got) | set(d) if got.get(k) != d.get(k)) == []
    assert C.phases(C._resolved(name.format(seed=seed)))["drop_s"] == 9000.0


@pytest.mark.parametrize("seed", SEEDS)
def test_no_key_reaches_the_harness_changes_since_reg24(seed):
    """`harness/` changed since REG-24's 1355be3 only behind `workload.sessions.cap(_from_s)` and
    `recorder.gpu_processes_every_s`; REG-27's configs set neither."""
    for kind in ("burst", "control"):
        f = _flat(C._resolved(C.REG24[f"{kind}_name"].format(seed=seed)))
        assert not any(k.startswith("workload.sessions.cap") for k in f)
        assert "recorder.gpu_processes_every_s" not in f


def test_the_rule_gives_the_committed_picks():
    assert tuple(R.picks()) == SEEDS


def test_picks_skip_43_and_keep_seed_order():
    ok = {41, 43, 50, 52, 60, 99}
    got = R.picks(n=4, rows_fn=lambda sd, d: {"ok": sd in ok})
    assert got == [41, 50, 52, 60]


@pytest.mark.parametrize("seed", SEEDS)
def test_control_and_burst_share_the_session_schedule(seed):
    assert C.inputs(seed, R.REG27["drop_s"])["control_schedule_same"]


W = "control_warm"


def rows(**kw):
    """seed -> (pair label, control label); a bare string means a warm control."""
    return {int(k[1:]): ({"pair": v, "control": W} if isinstance(v, str) else {"pair": v[0], "control": v[1]})
            for k, v in kw.items()}


@pytest.mark.parametrize("given,want", [
    (rows(s61="confirms", s68="confirms", s72="confirms", s79="confirms"), "replicates"),
    (rows(s61="confirms", s68="no_return", s72="confirms", s79=("confirms", "control_cold_transient")), "mixed"),
    (rows(s61="not_persistent", s68="stuck_not_cold", s72="emptied_in_watch", s79="confirms"), "mixed"),
    (rows(s61="not_persistent", s68="emptied_in_watch", s72="reformed", s79="bad_baseline"), "does_not_replicate"),
    (rows(s61="confirms", s68="confirms", s72=("confirms", "control_not_recovered"), s79="inconclusive"), "too_few"),
    (rows(s61="control_missing", s68="burst_missing", s72="not_same_build", s79="drop_inconclusive"), "too_few"),
])
def test_outcome(given, want):
    assert R.outcome(given) == want


def test_a_pair_behind_a_control_not_warm_is_never_counted():
    """Rule F r1 B1: behind a control that is not warm, a failing burst label must not count when a confirming one
    would not."""
    base = dict(s61="confirms", s68="confirms", s72="confirms")
    fail = R.count(rows(**base, s79=("not_persistent", "control_cold_transient")))
    ok = R.count(rows(**base, s79=("control_not_warm", "control_cold_transient")))
    assert (fail["k"], fail["n"]) == (ok["k"], ok["n"]) == (3, 3)
    assert fail["labels"][79] == ok["labels"][79] == "control_not_warm" and fail["not_counted"] == [79]
    for lab in ("no_return", "drop_inconclusive", "reformed", "bad_baseline"):
        assert R.count(rows(s61=(lab, "control_cold_persistent")))["n"] == 0
    assert R.count({61: {"pair": "control_missing", "control": None}})["labels"][61] == "control_missing"


def test_drop_inconclusive_is_reported_not_counted():
    """Rule F r1 B3: condition 6 inconclusive (it also covers a drop too short for the seed's drain)."""
    c = R.count(rows(s61="drop_inconclusive", s68="confirms", s72="confirms", s79="confirms"))
    assert (c["k"], c["n"], c["not_counted"]) == (3, 3, [61])


def test_count_and_bounds():
    c = R.count(rows(s61="confirms", s68="confirms", s72="no_return", s79=("confirms", "control_cold_persistent")))
    assert (c["k"], c["n"], c["not_counted"]) == (2, 3, [79])
    from analysis.stats import cp_upper
    assert c["upper"] == pytest.approx(cp_upper(2, 3)) and c["lower"] == pytest.approx(1 - cp_upper(1, 3))
    allc = R.count({s: {"pair": "confirms", "control": W} for s in (43, *SEEDS)})
    assert (allc["k"], allc["n"], allc["upper"]) == (5, 5, 1.0) and allc["lower"] == pytest.approx(0.549, abs=5e-4)
    assert math.isnan(R.count({61: {"pair": "control_missing", "control": None}})["upper"])


def test_every_burst_label_is_classed():
    """A label the map can give is counted or excluded on purpose; an unknown one raises."""
    for lab in (*C.BURST_LABELS, "control_not_warm", "control_missing", "burst_missing", "not_same_build"):
        R.count({61: {"pair": lab, "control": W}})
    with pytest.raises(ValueError):
        R.count({61: {"pair": "something_new", "control": W}})


def test_table_refuses_while_a_folder_is_sealed(tmp_path, monkeypatch):
    """Rule F r1 B2: a sealed folder would read as missing and give `too_few`; the table raises instead."""
    import analysis.load as L
    (tmp_path / "20261118-000000_stage2_control_l1pl350cl_r2.0_long_s61").mkdir()
    monkeypatch.setattr(L, "RUNS", tmp_path)
    monkeypatch.setattr(withheld, "lifted", lambda *a, **k: False)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    called = []
    monkeypatch.setattr(C, "table", lambda *a: called.append(a))
    with pytest.raises(RuntimeError, match="sealed"):
        R.table()
    assert not called


def test_a_failing_readout_does_not_stop_the_count(monkeypatch):
    import analysis.load as L
    monkeypatch.setattr(R, "_sealed_present", lambda: [])
    t = lambda sd: {"outcome": "confirms", "control": {"outcome": W}, "burst": {"run": f"x_s{sd}",
                                                                              "outcome": "confirms"}}
    monkeypatch.setattr(C, "table", t)
    monkeypatch.setattr(R, "tilt", lambda sd: 1 / 0)
    import analysis.order as O
    monkeypatch.setattr(O, "table", lambda names: 1 / 0)
    out = R.table()
    assert out["outcome"] == "replicates" and out["count"]["k"] == 4
    assert "error" in out["rows"][61]["tilt"] and "error" in out["rows"][61]["order"]
    assert out["with_43"]["k"] == 5


def test_surge_flag():
    assert R.surge_flag({"outcome": "control_cold_persistent"}) is True
    assert R.surge_flag({"outcome": "control_cold_transient"}) is True
    assert R.surge_flag({"outcome": "control_warm"}) is False and R.surge_flag(None) is None


def test_seals_cover_reg27s_eight_folders_and_lift_only_on_its_result(tmp_path):
    names = [f"20261118-000000_{fam}_s{sd}" for sd in SEEDS
             for fam in ("confirm_l1pl350cl_r2.0", "stage2_control_l1pl350cl_r2.0_long")]
    rows = [r for r in withheld.SEALS if r[0] == "REG-27"]
    assert len(rows) == 8
    for n in names:
        assert [r[0] for r in withheld.SEALS if fnmatch.fnmatchcase(n, r[1])] == ["REG-27"]
    for other in ("x_confirm_l1pl350cl_r2.0_s43", "x_confirm_l1pl350cl_r2.0_s62", "x_confirm_l1pl350cl_r2.0_s610",
                  "x_stage2_control_l1pl350cl_r2.0_long_s71"):
        assert not any(fnmatch.fnmatchcase(other, r[1]) for r in rows)
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-18 · [REG] REG-27: …\n## 2026-11-18 · [OPS] REG-27: the queue ended\n")
    assert not withheld.lifted("[RESULT] REG-27", d)
    d.write_text("## 2026-11-20 · [RESULT] REG-27: replicates …\n")
    assert withheld.lifted("[RESULT] REG-27", d)


def test_confirm_check_takes_the_seed(monkeypatch):
    """REG-24's `check` at seed 61 accepts the seed-61 config and refuses it as seed 43's."""
    cfg = C._resolved(C.REG24["burst_name"].format(seed=61))
    run = {"checks": {"valid": True}, "cfg": cfg, "dir": type("D", (), {"name": "x"})()}
    C.check(run, "burst", 61)
    with pytest.raises(ValueError):
        C.check(run, "burst")
