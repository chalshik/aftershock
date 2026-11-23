"""analysis/d1b.py: D1b, the cap-40 dose check beside D1 (C2)."""
import fnmatch
import math

import pytest

from analysis import d1 as D
from analysis import d1b as B
from analysis import withheld as W


def test_cap40_config_is_reg19s_burst_config_with_the_cap_alone():
    for sd in B.D1B["seeds"]:
        k = B.resolved(sd)
        assert D._diff_keys(D._burst_cfg(sd), k) == ["name", "workload.sessions.cap"]
        assert k["name"] == B.config_name(sd) and k["workload"]["sessions"]["cap"] == 40 and k["seed"] == sd
        assert k["workload"]["sessions"]["closed_loop"] is True and k["trigger"]["kind"] == "burst"
        assert D._diff_keys(D.resolved(sd, 27), k) == ["name", "workload.sessions.cap"]   # D1's run but for the cap


def test_cap40_is_above_d1s_pool_fill_point():
    fo = B.first_order()
    assert fo["cap_kv_095"] < B.D1B["cap"] and fo["kv_max"] > 0.95


def test_the_committed_configs_are_the_written_text():
    for sd in B.D1B["seeds"]:
        assert (B.CONFIGS / f"{B.config_name(sd)}.yaml").read_text() == B.config_text(sd)


def test_config_text_refuses_other_seeds():
    with pytest.raises(ValueError):
        B.config_text(18)


def test_write_configs_writes_new_files_only(tmp_path):
    paths = B.write_configs(tmp_path)
    assert [p.name for p in paths] == [f"{B.config_name(s)}.yaml" for s in B.D1B["seeds"]]
    with pytest.raises(FileExistsError):
        B.write_configs(tmp_path)


def test_the_cap40_folders_match_no_other_registrations_pattern():
    from analysis import gate, k1, k3, k3w
    names = [f"20261121-000000_{B.config_name(s)}" for s in B.D1B["seeds"]]
    others = ([gate.reg19_spec(s)[k] for s in D.D1["seeds"] for k in ("control", "burst")]
              + [k1.closed_pattern(s) for s in D.D1["seeds"]] + [k1.open_pattern(2.0, s) for s in D.D1["seeds"]]
              + [k3.REG21["batch2_pattern"], k3w.REG21["batch2_pattern"]] + [D.run_pattern(s) for s in D.D1["seeds"]]
              + [p for reg, p, _ in W.SEALS if reg != "D1b"] + [D.D1["drift_pattern"]])
    assert not [(n, p) for n in names for p in others if fnmatch.fnmatchcase(n, p)]
    assert all(fnmatch.fnmatchcase(n, B.run_pattern(s)) for n, s in zip(names, B.D1B["seeds"]))
    d1b_seal = [p for reg, p, _ in W.SEALS if reg == "D1b"]
    assert len(d1b_seal) == 1
    for sd in D.D1["seeds"]:                      # none of D1's five capped folders is D1b's, by pattern or by seal
        name = f"20261120-000000_{D.config_name(sd)}"
        assert not any(fnmatch.fnmatchcase(name, B.run_pattern(s)) for s in B.D1B["seeds"])
        assert not fnmatch.fnmatchcase(name, d1b_seal[0])


def test_d1b_seal_covers_its_three_names_and_lifts_only_on_its_result(tmp_path, monkeypatch):
    d = tmp_path / "decisions.md"
    d.write_text("\n".join([
        "## 2026-11-15 · [REG] D1b (C2, cap above the cold batch): x",
        "## 2026-11-16 · [RESULT] D1 (C2, session cap): x",          # D1's result lifts D1's seal, not D1b's
        "## 2026-11-16 · [RESULT] D1b: x", "## 2026-11-16 · [CORR] [RESULT] D1b (C2, cap above the cold batch)"]) + "\n")
    monkeypatch.setattr(W, "DECISIONS", d)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    names = [f"20261121-000000_{B.config_name(s)}" for s in B.D1B["seeds"]]
    assert [W.sealed_by(n) for n in names] == ["D1b"] * 3
    assert W.sealed_by("20261121-000000_burst_l1pl350cl_r2.0_w3000_s18_cap40") is None
    assert W.sealed_by(f"20261120-000000_{D.config_name(15)}") is None   # D1's seal lifted above
    d.write_text(d.read_text() + "## 2026-11-17 · [RESULT] D1b (C2, cap above the cold batch): batch_bound\n")
    assert [W.sealed_by(n) for n in names] == [None] * 3


def test_outcome_rules():
    cut, st = "recovered", "stuck"
    assert B.outcome({15: st}, {15: cut})["decision"] == "too_few"
    assert B.outcome({15: st, 16: st, 17: cut}, {15: cut, 16: "recovered_cap_draining", 17: cut})["decision"] == "batch_bound"
    # stuck at 40 without D1 cut on those seeds is no bracket (rule F d1b r1, U1)
    o = B.outcome({15: st, 16: st, 17: cut}, {15: "unclear", 16: "partial", 17: cut})
    assert o["decision"] == "stuck_without_d1_cut" and o["k_bracket"] == 0
    assert B.outcome({15: st, 16: st}, {15: cut})["decision"] == "stuck_without_d1_cut"   # seed 16 has no D1 run
    assert B.outcome({15: cut, 16: "recovered_cap_draining", 17: st}, {15: cut, 16: cut, 17: cut})["decision"] == "cut_at_40"
    assert B.outcome({15: st, 16: cut, 17: "unclear"}, {15: cut, 16: cut, 17: cut})["decision"] == "mixed"
    o = B.outcome({15: st, 16: st, 17: st}, {15: cut, 16: cut, 17: cut})
    assert o["k_bracket"] == 3 and o["cp_upper_stuck"] == 1.0 and [p["bracket"] for p in o["pairs"]] == [True] * 3
    with pytest.raises(ValueError):
        B.outcome({15: "harness_fault", 16: st}, {})


def test_predicted_is_an_exact_distribution():
    p = B.predicted()
    assert math.isclose(sum(p.values()), 1.0)
    assert set(p) <= {"too_few", "batch_bound", "stuck_without_d1_cut", "cut_at_40", "mixed"}
    assert p["batch_bound"] == pytest.approx(0.4910, abs=1e-4) and p["cut_at_40"] == pytest.approx(0.0397, abs=1e-4)


def _fake_label(monkeypatch, d1_label, q_base, q_end, slope):
    lab = {"label": d1_label, "windows": {"base": (300.0, 900.0), "end": (3480.0, 4080.0)}}
    monkeypatch.setattr(D, "label", lambda run: dict(lab))
    import analysis.load as L
    vals = {(300.0, 900.0): {"q": q_base, "q_slope": 0.0}, (3480.0, 4080.0): {"q": q_end, "q_slope": slope}}
    monkeypatch.setattr(L, "_level", lambda run, a, b: vals[(a, b)])
    v = {"recovered_q_excess": 1.0, "drain_horizon_s": 3600.0}
    return {"cfg": {"verdict": v}}


def test_label40_counts_a_backlog_at_the_server_as_stuck(monkeypatch):
    """Rule F d1b r1, U4: at cap 40 the cap need not bind, so a cold run's backlog can sit at the server."""
    run = _fake_label(monkeypatch, "partial", 0.0, 120.0, 0.01)
    out = B.label40(run)
    assert out["label"] == "stuck" and out["label_d1"] == "partial" and out["server_backlogged"] is True
    run = _fake_label(monkeypatch, "partial", 0.0, 0.5, 0.0)          # no line anywhere: stays partial
    assert B.label40(run)["label"] == "partial"
    run = _fake_label(monkeypatch, "partial", 0.0, 120.0, -0.1)       # draining within the horizon: stays partial
    assert B.label40(run)["label"] == "partial"
    run = _fake_label(monkeypatch, "recovered", 0.0, 120.0, 0.01)     # only partial changes
    assert B.label40(run)["label"] == "recovered"


def test_table_refuses_while_sealed_or_before_d1s_result(monkeypatch, tmp_path):
    monkeypatch.setattr(B, "_sealed", lambda: ["x_cap40"])
    with pytest.raises(W.WithheldError):
        B.table()
    monkeypatch.setattr(B, "_sealed", lambda: [])
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-15 · [REG] D1 (C2, session cap): x\n")
    monkeypatch.setattr(W, "DECISIONS", d)
    with pytest.raises(W.WithheldError):
        B.table()
    d.write_text("## 2026-11-16 · [RESULT] D1 (C2, session cap): supports\n")
    monkeypatch.setenv("AFTERSHOCK_UNSEAL", "D1b,D1")
    with pytest.raises(W.WithheldError):
        B.table()


REG19_15 = B.REPO / "runs" / "20261112-061951_burst_l1pl350cl_r2.0_w3000_s15"


@pytest.mark.skipif(not REG19_15.exists(), reason="REG-19's run folders are on the Mac only")
def test_feasibility_on_the_uncapped_runs():
    f = B.feasibility()
    assert f["passes"] and f["k_above"] == 3 and all(r["in_service_end"] > 200 for r in f["rows"])
