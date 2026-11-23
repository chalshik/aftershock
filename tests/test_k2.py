"""analysis/k2.py: the E0 prediction and gate (draft)."""
import math
from pathlib import Path

import pytest

from analysis import k2
from harness.run import load_config
from tests.test_workload import CONFIGS

S = k2.shape(load_config(CONFIGS / "sweep_l1pl350cl_pc_on.yaml")["workload"]["sessions"])


def test_token_gap_matches_the_design_note():
    assert [round(k2.token_gap(S, T), 2) for T in (4, 8, 16)] == [3.48, 6.19, 10.89]   # the note rounds 10.89 to 10.9


def test_mean_tokens_of_l1():
    assert k2.cold_tokens(S, 8) == 3072 and k2.warm_tokens(S, 8) == 496
    assert k2.cold_tokens(S, 16) == 4096 and k2.warm_tokens(S, 16) == 376


def test_fixed_model_reproduces_reg8_and_the_m5_constants():
    a, b = k2.calibrate_fixed(S)
    assert 1 / (a + b * 3072) == pytest.approx(1.681) and 1 / (a + b * 496) == pytest.approx(4.047)
    assert a == pytest.approx(0.180, abs=0.001) and b == pytest.approx(0.000135, abs=1e-6)


def test_prediction_at_16_turns():
    p = k2.predict(S, 16)
    assert p["prop"]["floor_ratio"] == pytest.approx(0.75) and p["prop"]["R"] == pytest.approx(1.758, abs=0.001)
    assert p["fixed"]["R"] == pytest.approx(1.319, abs=0.002)
    assert k2.predict(S, 8)["fixed"]["R"] == pytest.approx(1.0)


NOMINAL8 = {r: r for r in k2.E0["warm_grid_8"]}      # planned = nominal: the old reading, for the label logic alone
NOMINAL16 = {r: r for r in k2.E0["warm_grid_16"]}


def dec(w8, f8, w16, f16, p8=NOMINAL8, p16=NOMINAL16, **kw):
    return k2.decide(w8, f8, w16, f16, p8, p16, **kw)


def test_decide_labels():
    g = k2.gate_factor()
    assert dec(2.4, 1.744, 2.8, 1.431)["label"] == "widens"
    assert dec(2.4, 1.744, 2.4, 1.431)["label"] == "floor_only"          # R 1.22 >= g, warm side flat
    assert dec(2.4, 1.744, 2.0, 1.431)["label"] == "unresolved"          # R 1.016
    below = dec(2.4, 1.744, math.nan, 1.431)
    assert below["label"] == "unresolved" and below["R_upper"] == pytest.approx((2.0 / 1.431) / (2.4 / 1.744))
    assert dec(2.4, 1.70, math.nan, 1.70)["label"] == "does_not_widen"   # R_upper 0.83
    assert dec(2.4, 1.744, math.nan, 1.431, low16_measured=False)["label"] == "missing"
    assert dec(2.4, math.nan, 2.6, 1.431)["label"] == "missing"
    d = dec(2.4, 1.744, 2.6, 1.431)
    assert d["warm_ratio"] == pytest.approx(2.6 / 2.4) and d["floor_ratio"] == pytest.approx(1.431 / 1.744)
    assert d["g"] == pytest.approx(g)


def test_the_warm_clause_is_in_realized_planned_rate():
    """Rule F e0e1 r2 T1 (author 2026-11-14): seed 0's 16-turn 2.6 (planned 2.674) is inside the 8-turn capacity's
    planned interval [2.529, 2.758), so it is no gain; 2.8 (2.856) is."""
    p8 = {r: v["planned"] for r, v in k2.planned_table(k2.E0["on_8"]).items()}
    p16 = {r: v["planned"] for r, v in k2.planned_table(k2.E0["on_16"]).items()}
    assert dec(2.4, 1.744, 2.6, 1.431, p8, p16)["label"] == "floor_only"
    assert dec(2.4, 1.744, 2.6, 1.431, p8, {**p16, 2.6: 2.76})["label"] == "widens"   # a 2.6 planned above 2.758 would
    assert dec(2.4, 1.744, 2.8, 1.431, p8, p16)["label"] == "widens"
    assert dec(3.0, 1.744, 3.2, 1.431, p8, p16)["label"] == "unresolved"   # warm8 at the 8-turn top: g is inf, never widens


def test_does_not_widen_needs_r_at_most_1_with_warm16_rounded_up():
    """Rule F e0e1 r2 T3 (author 2026-11-14): a true 2.19 reads 2.0; R 0.992 but 0.992 x 2.2/2.0 > 1: unresolved."""
    d = dec(2.4, 1.70, 2.0, 1.70 * 0.84 / (2.4 / 2.4))
    assert d["R"] < 1 and d["R_round_up"] == pytest.approx(d["R"] * 2.2 / 2.0) and d["label"] == "unresolved"
    assert dec(2.4, 1.70, 2.0, 1.70)["label"] == "does_not_widen"        # R 0.833, x 1.1 = 0.917
    assert dec(2.4, 1.70, 3.2, 1.70 * 3.2)["label"] == "unresolved"      # at the 16-turn grid's top: no guard bound


def test_planned_table_is_the_schedule_seen_by_summarize():
    t8, t16 = k2.planned_table(k2.E0["on_8"]), k2.planned_table(k2.E0["on_16"])
    assert tuple(t8) == k2.E0["warm_grid_8"] and tuple(t16) == k2.E0["warm_grid_16"]
    assert t8[2.4]["planned"] == pytest.approx(2.529, abs=0.001) and t8[2.6]["planned"] == pytest.approx(2.758, abs=0.001)
    assert t16[2.6]["planned"] == pytest.approx(2.674, abs=0.001) and t16[2.8]["planned"] == pytest.approx(2.856, abs=0.001)


def test_prediction_on_the_gates_scale():
    p = k2.predict_measured(S)
    for name in ("fixed", "prop"):
        assert set(p[name]) == {"lo", "hi"}
        for end in ("lo", "hi"):
            assert p[name][end]["label"] in ("widens", "floor_only", "unresolved", "does_not_widen", "missing")


def test_gate_factor_and_labels():
    assert k2.gate_factor() == pytest.approx(1 + 0.2 / 2.4)
    assert k2.gate_factor(3.0, (2.0, 3.0)) == math.inf
    assert k2.gate(1.2) == "widens" and k2.gate(1.05) == "unresolved" and k2.gate(0.98) == "does_not_widen"
    assert k2.gate(math.nan) == "nan"
    assert k2.measured_R(2.4, 1.7, 2.6, 1.3) == pytest.approx((2.6 / 1.3) / (2.4 / 1.7))


def test_e2_configs_have_16_turns_and_the_registered_grids():
    on = load_config(CONFIGS / "sweep_l1t16pl350cl_pc_on.yaml")
    off = load_config(CONFIGS / "sweep_l1t16pl350cl_pc_off.yaml")
    e1 = load_config(CONFIGS / "sweep_l1pl350cl_pc_off.yaml")
    assert on["workload"]["sessions"]["turns"] == off["workload"]["sessions"]["turns"] == 16
    assert tuple(on["sweep"]["rates"]) == k2.E0["warm_grid_16"]
    assert e1["workload"]["sessions"]["turns"] == 8 and e1["workload"]["sessions"]["closed_loop"]
    assert e1["server"]["args"]["--no-enable-prefix-caching"] and off["server"]["args"]["--no-enable-prefix-caching"]
    for c in (on, off, e1):
        assert c["workload"]["sessions"]["closed_loop"] and c["loadgen_force_close"] and c["seed"] == 0


def test_closed_loop_floors_lower_R_a_little():
    f = k2.closed_loop_floors(S)
    p = k2.predict(S, 16)
    for name in ("prop", "fixed"):
        assert f[name]["floor_cl_8"] > 1.681 and f[name]["floor_ratio_cl"] > p[name]["floor_ratio"]
        assert 1.0 < f[name]["R_cl"] < p[name]["R"]
    assert f["fixed"]["R_cl"] == pytest.approx(1.30, abs=0.01) and f["prop"]["R_cl"] == pytest.approx(1.71, abs=0.01)


def test_a_hole_where_the_16_turn_chain_ended_is_missing():
    """Rule F e0e1 r2 T4: 2.2 has no valid run, so the monotone chain stops at 2.0; that is no measurement."""
    assert dec(2.4, 1.744, 2.0, 1.431, hole16=True)["label"] == "missing"
    assert dec(2.4, 1.744, 2.0, 1.431)["label"] == "unresolved"


def _rows(spec):
    import pandas as pd
    return pd.DataFrame([{"rate": r, "throughput": t, "planned_rate": r, "kv_mean": kv} for r, t, kv in spec])


def test_both_floors_use_the_saturated_rates_the_sweeps_share():
    """Author 2026-11-14 (rule F e0e1 r2 item 2): 8 turns saturate from 2.5, 16 turns from 2.0; both floors from 2.5."""
    r8 = _rows([(2.0, 2.0, 0.5), (2.5, 1.80, 0.95), (3.0, 1.76, 0.97)])
    r16 = _rows([(1.5, 1.4, 0.95), (2.0, 1.50, 0.95), (2.5, 1.40, 0.96), (3.0, 1.36, 0.97)])
    f = k2.shared_floors(k2.common_rows(r8), k2.common_rows(r16))
    assert f["from"] == 2.5 and f["off_8"]["mu_sat_off"] == pytest.approx(1.78)
    assert f["off_16"]["mu_sat_off"] == pytest.approx(1.38) and f["off_16"]["n"] == 2   # 1.5 and 2.0 left out
    hole = k2.shared_floors(k2.common_rows(r8.iloc[[0, 2]]), k2.common_rows(r16))          # 8 turns lost 2.5
    assert math.isnan(hole["off_8"]["mu_sat_off"]) and math.isnan(hole["from"])


@pytest.mark.parametrize("kind", ["off", "on"])
def test_each_e2_single_rate_config_is_the_sweeps_own_run_at_that_rate(kind):
    """E2's reruns and never-started rates (E1's stopping rule, applied to E2 by E0's registration)."""
    from analysis import e1
    drop = lambda c: {k: v for k, v in c.items() if k != "sweep"}
    sweep = e1._cfgs(f"configs/sweep_l1t16pl350cl_pc_{kind}.yaml")
    for c in sweep:
        only = e1._cfgs(f"configs/sweep_l1t16pl350cl_pc_{kind}_only_r{c['workload']['rate']:g}.yaml")
        assert len(only) == 1 and drop(only[0]) == drop(c)


# The tip rule (author, 2026-11-15: E0's warm capacities by whether the warm state held, not by the keep-up ratio).

import pandas as pd


def _tip(rate, onset=math.nan, slope=0.0, valid=True, end=900.0):
    return {"rate": rate, "valid": valid, "q_slope_per_s": slope, "max_q_slope": 0.01, "kv_onset": onset, "end_s": end}


def test_tip_pass_needs_valid_a_flat_queue_and_no_onset_anywhere_in_the_run():
    assert k2.tip_pass(_tip(2.0))
    assert not k2.tip_pass(_tip(2.0, onset=900.0))       # rule F e0tip r1, T3 (i): an onset in the drain is a tip too
    assert not k2.tip_pass(_tip(2.0, onset=1200.0))
    assert not k2.tip_pass(_tip(2.0, onset=899.9))
    assert not k2.tip_pass(_tip(2.0, onset=100.0))       # an onset inside the warm-up fails too
    assert not k2.tip_pass(_tip(2.0, slope=0.0101))
    assert k2.tip_pass(_tip(2.0, slope=0.01))            # the registered slope bound is inclusive, as load.capacity
    assert not k2.tip_pass(_tip(2.0, slope=math.nan))    # a NaN slope fails, as load.capacity
    assert not k2.tip_pass(_tip(2.0, valid=False))


def test_tip_capacity_is_the_monotone_chain_over_the_swept_rates():
    sw = [2.0, 2.2, 2.4, 2.6]
    rows = pd.DataFrame([_tip(2.0), _tip(2.2), _tip(2.4, onset=500.0), _tip(2.6)])
    assert k2.tip_capacity(rows, sw) == 2.2                # 2.6 passes above a failing 2.4 and cannot set it
    assert math.isnan(k2.tip_capacity(pd.DataFrame([_tip(2.0, onset=300.0)]), sw))
    hole = pd.DataFrame([_tip(2.0), _tip(2.4)])            # no valid run at 2.2 ends the chain
    assert k2.tip_capacity(hole, sw) == 2.0
    with pytest.raises(ValueError):
        k2.tip_capacity(pd.DataFrame([_tip(2.0), _tip(2.0)]), sw)
    with pytest.raises(ValueError):
        k2.tip_capacity(pd.DataFrame([_tip(2.1)]), sw)


def test_tip_rule_ignores_the_keep_up_ratio():
    """A run the ratio rule fails for release lag alone (REG-13's 2.4 read 0.9503 with an empty queue) passes here."""
    row = {**_tip(2.4), "keep_up_ratio": 0.90}
    assert k2.tip_pass(row)


REG13 = Path(__file__).resolve().parent.parent / "runs" / "20261109-191845_sweep_l1pl350cl_pc_on_r2.4"


@pytest.mark.skipif(not REG13.exists(), reason="REG-13's run folders are on the Mac only")
def test_reg13_warm_capacity_is_2_4_under_both_rules():
    from analysis.tables import sweep_table
    t = k2.tip_warm(k2.E0_PARENTS["warm_8"], sweep_table())
    assert t["tip"] == 2.4 and t["ratio_rule"] == 2.4
    assert list(t["rows"].tip_pass) == [True, True, False, False, False]


def test_widens_needs_the_released_rate_above_the_8_turn_tip_runs_released_rate():
    """Rule F e0tip r1, T1 (draft 2): a 16-turn run whose release lag held the load it offered at or below what the
    8-turn run offered where it tipped (REG-13's 2.6 released 2.470) reads floor_only, not widens."""
    p8 = {r: v["planned"] for r, v in k2.planned_table(k2.E0["on_8"]).items()}
    p16 = {r: v["planned"] for r, v in k2.planned_table(k2.E0["on_16"]).items()}
    rel = lambda r16, r8=2.470: dec(2.4, 1.744, 2.8, 1.431, p8, p16, released16=r16, released_top8=r8)
    assert dec(2.4, 1.744, 2.8, 1.431, p8, p16)["label"] == "widens"
    assert rel(2.60)["label"] == "widens"
    d = rel(2.45)
    assert d["label"] == "floor_only" and d["released_warm16"] == 2.45 and d["released_top8"] == 2.470
    assert rel(2.470)["label"] == "floor_only"                               # strict, as the planned clause
    assert dec(2.4, 1.744, 2.6, 1.431, p8, p16, released16=2.6, released_top8=2.47)["label"] == "floor_only"  # planned
    with pytest.raises(ValueError):
        dec(2.4, 1.744, 2.8, 1.431, p8, p16, released16=2.6)


def test_kv_readout_refuses_a_missing_or_thin_gauge(tmp_path):
    """Rule F e0tip r1, T4: a missing gauge must never read as "never tipped"."""
    import numpy as np
    cfg = {"workload": {"warmup_s": 0.0, "duration_s": 100.0}}
    name = "vllm:kv_cache_usage_perc"
    with pytest.raises(ValueError):
        k2.kv_readout({"cfg": cfg, "dir": tmp_path, "metrics": pd.DataFrame({"t": np.arange(50.0)})})
    with pytest.raises(ValueError):
        k2.kv_readout({"cfg": cfg, "dir": tmp_path, "metrics": pd.DataFrame({"t": [1.0, 2.0], name: [0.1, 0.2]})})
    t = np.arange(0.0, 100.0, 1.0)
    kv = np.where(t >= 90, 0.95, 0.5)
    r = k2.kv_readout({"cfg": cfg, "dir": tmp_path, "metrics": pd.DataFrame({"t": t, name: kv})})
    assert r["kv_scrapes"] == 100 and r["kv_peak"] == 0.95 and r["kv_s_above"] == pytest.approx(9.0)


def test_observed_refuses_a_parent_with_no_readable_run(monkeypatch):
    """Rule F e0tip r1, T5: a forgotten AFTERSHOCK_UNSEAL=E0 is an error, not a 'missing' label."""
    import analysis.tables as T
    sw = pd.DataFrame({"parent": [k2.E0_PARENTS["warm_8"]], "rate": [2.0], "valid": [True]})
    monkeypatch.setattr(T, "sweep_table", lambda *a, **k: sw)
    with pytest.raises(ValueError, match="AFTERSHOCK_UNSEAL"):
        k2.observed()
