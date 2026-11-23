"""analysis/g1.py: G1's decision table, day flag and checks on synthetic inputs (no run folder is read)."""
import math
from pathlib import Path

import pytest

from analysis import g1, g2, t4, withheld

OK_DRIFT = {3.0: 0.004, 4.0: -0.002}


def _f(mu, status="ok"):
    return {"mu_sat_off": mu, "status": status, "lam_sat": 2.5, "n": 4}


def _out(f0, f1, w0=2.4, w1=2.4, drift=OK_DRIFT, s1_status="ok"):
    return g1.outcome({"a": _f(f0), "b": _f(f1), "from": 2.5}, {"s0": _f(f0), "s1": _f(f1, s1_status)},
                      {"s0": w0, "s1": w1}, drift)


@pytest.mark.parametrize("f0,f1,w0,w1,dec", [
    (1.80, 1.80, 2.4, 2.4, "seed_stable"),
    (1.80, 1.854, 2.4, 2.4, "seed_stable"),      # +3.0 %: the band is inclusive
    (1.80, 1.86, 2.4, 2.4, "seed_varies"),       # +3.3 %, cell still holds (2.0 >= 1.05 x 1.86 = 1.953)
    (1.80, 1.80, 2.4, 2.6, "seed_varies"),       # warm capacities differ
    (1.80, 1.92, 2.4, 2.4, "cell_seed_bound"),   # +6.7 % and 1.05 x 1.92 = 2.016 > 2.0
    (1.80, 1.80, 2.4, 2.0, "cell_seed_bound"),   # 2.0 not below the seed-1 warm capacity
    (1.88, 1.91, 2.4, 2.4, "cell_edge"),         # rule F r1 G-1 trace 1: +1.6 % crosses the 1.905 edge
    (1.92, 1.92, 2.4, 2.4, "reference_moved"),   # T4's own 2048 cell fails first
    (1.80, math.nan, 2.4, 2.4, "no_floor"),
    (1.80, 1.80, 2.4, math.nan, "no_warm"),
])
def test_decision_table(f0, f1, w0, w1, dec):
    assert _out(f0, f1, w0, w1)["decision"] == dec


def test_day_drift_blocks_cell_seed_bound():
    # rule F r1 G-1 trace 2: drift +3.5 % at one twin rate; the move is not attributed to the seed
    o = _out(1.80, 1.92, drift={3.0: 0.035, 4.0: 0.01})
    assert o["day_confounded"] and o["decision"] == "cell_edge"


@pytest.mark.parametrize("drift,delta,conf", [
    ({3.0: 0.004, 4.0: -0.002}, 0.01, False),
    ({3.0: 0.031, 4.0: 0.0}, 0.0, True),          # beyond the band
    ({3.0: math.nan, 4.0: 0.0}, 0.0, True),       # a twin missing
    ({4.0: 0.0}, 0.0, True),
    ({3.0: 0.029, 4.0: 0.0}, 0.031, True),        # rule F r1 G-10: drift >= half a delta beyond the band
    ({3.0: 0.010, 4.0: 0.0}, 0.031, False),
])
def test_day_confounded(drift, delta, conf):
    assert g1.day_confounded(drift, delta) is conf


def test_above_grid_seed1_floor_fails_its_cell_as_t4():
    o = _out(1.80, math.nan, s1_status="above_grid")
    assert o["seed_label"] == "seed_up" and o["cell_s1"] == "cell_moves" and o["decision"] == "cell_seed_bound"


def test_window_edge_is_t4s_cell_margin():
    assert t4.T4["cell_lambda"] / t4.T4["cell_margin"] == pytest.approx(1.905, abs=1e-3)
    assert _out(1.80, 1.85)["cell_s1"] == "cell_holds" and _out(1.80, 1.906)["cell_s1"] == "cell_moves"


def test_margin_is_lambda_over_each_floor():
    o = _out(1.80, 1.75)
    assert o["margin"]["s0"] == pytest.approx(2.0 / 1.80) and o["margin"]["s1"] == pytest.approx(2.0 / 1.75)


def test_parents_reference_and_twins():
    p = g1.G1["parents"]
    assert p["off_s0"] == t4.T4["parents"]["off_2048"] and p["on_s0"] == t4.T4["parents"]["on_2048"]
    assert g1.G1["twin_rates"] == g2.G2["rates"] and g1.G1["seed"] == 1


def test_configs_change_only_the_workload_seed():
    from harness.config import load_config
    for k in ("off", "on"):
        a = load_config(Path("configs") / f"sweep_t4_l1pl350cl_pc_{k}_b2048.yaml")
        b = load_config(Path("configs") / f"sweep_g1_l1pl350cl_pc_{k}_b2048_s1.yaml")
        assert b["seed"] == 1 and a["seed"] == 0 and b["server"]["args"]["--seed"] == a["server"]["args"]["--seed"] == 0
        assert {x: y for x, y in b.items() if x not in ("seed", "name")} == \
               {x: y for x, y in a.items() if x not in ("seed", "name")}


def test_the_seed_feeds_the_schedule():
    # harness/run.py:145 passes cfg["seed"] to workload.build_schedule, whose RNG it seeds
    from harness import workload
    from harness.config import load_config
    wl = load_config(Path("configs") / "sweep_g1_l1pl350cl_pc_off_b2048_s1_only_r2.5.yaml")["workload"]
    a, _ = workload.build_schedule(wl, 0)
    b, _ = workload.build_schedule(wl, 1)
    assert not a.equals(b)
    assert "build_schedule(wl, cfg[\"seed\"]" in Path("harness/run.py").read_text()


def test_run_checks_reads_seed_and_t4_settings(monkeypatch):
    import analysis.load as L
    monkeypatch.setattr(L, "load_run", lambda d: {"cfg": {"seed": 0}})
    monkeypatch.setattr(t4, "server_readouts", lambda d: {"override_blocks": 6041, "pool_tokens": 96656,
                                                         "non_default_args": {"max_num_batched_tokens": 8192}})
    bad = g1.run_checks("x", "off_2048", 1)
    assert any("seed" in b for b in bad) and any("max_num_batched_tokens" in b for b in bad)


def test_table_refuses_before_t4_and_before_g2(monkeypatch):
    monkeypatch.setattr(withheld, "lifted", lambda h, decisions=None: False)
    with pytest.raises(withheld.WithheldError):
        g1.table()
    monkeypatch.setattr(withheld, "lifted", lambda h, decisions=None: h == t4.T4["seal"])   # T4 in, G2 not
    with pytest.raises(withheld.WithheldError, match="G2"):
        g1.table()


def test_seal_row():
    assert withheld.sealed_by("20261122-000000_sweep_g1_l1pl350cl_pc_off_b2048_s1_r2.5") == \
        (None if withheld.lifted(g1.G1["seal"]) else "G1")
    assert any(r[0] == "G1" and r[2] == g1.G1["seal"] for r in withheld.SEALS)
