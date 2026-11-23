"""analysis/t4.py (T4, the step budget): configs, seal, labels and the refusal; no run folder is read."""
import fnmatch
import json
import math
from pathlib import Path

import pandas as pd
import pytest

from analysis import t4 as T
from analysis import withheld

REPO = Path(__file__).resolve().parent.parent
P = T.T4["parents"]


def _resolved(path: str) -> list[dict]:
    from harness.config import with_defaults
    from harness.run import expand, load_config
    return expand(with_defaults(load_config(REPO / path)))


def _strip(c: dict, drop_args=()) -> dict:
    c = json.loads(json.dumps(c))
    for k in ("name", "sweep_parent", "sweep", "source_config", "run_id"):
        c.pop(k, None)
    for a in drop_args:
        c["server"]["args"].pop(a, None)
    return c


@pytest.mark.parametrize("kind", ["off", "on"])
def test_the_8192_arm_differs_from_its_twin_by_the_budget_alone(kind):
    a = {c["workload"]["rate"]: c for c in _resolved(f"configs/{P[kind + '_2048']}.yaml")}
    b = {c["workload"]["rate"]: c for c in _resolved(f"configs/{P[kind + '_8192']}.yaml")}
    assert sorted(a) == sorted(b)
    for r in a:
        assert b[r]["server"]["args"]["--max-num-batched-tokens"] == 8192
        assert "--max-num-batched-tokens" not in a[r]["server"]["args"]
        assert _strip(a[r]) == _strip(b[r], drop_args=("--max-num-batched-tokens",))


@pytest.mark.parametrize("kind,old", [("off", "configs/sweep_l1pl350cl_pc_off.yaml"), ("on", "configs/sweep_l1pl350cl_pc_on.yaml")])
def test_the_2048_arm_is_e1s_or_reg13s_sweep_but_for_the_pinned_pool(kind, old):
    """Same workload, seed and server at each rate both share (so the same schedule), plus the pool pin alone."""
    new = {c["workload"]["rate"]: c for c in _resolved(f"configs/{P[kind + '_2048']}.yaml")}
    was = {c["workload"]["rate"]: c for c in _resolved(old)}
    shared = set(new) & set(was)
    assert len(shared) >= 5
    for r in shared:
        assert new[r]["server"]["args"]["--num-gpu-blocks-override"] == T.T4["pool_blocks"]
        assert _strip(new[r], drop_args=("--num-gpu-blocks-override",)) == _strip(was[r])


def test_pool_pin_is_the_logged_default_pool():
    cfg = _resolved(f"configs/{P['on_2048']}.yaml")[0]
    assert T.T4["pool_blocks"] * 16 == cfg["checks"]["expected_kv_tokens"] == 96656


@pytest.mark.parametrize("arm", sorted(P))
def test_each_rerun_config_is_its_sweep_at_one_rate(arm):
    full = {c["workload"]["rate"]: c for c in _resolved(f"configs/{P[arm]}.yaml")}
    for r, c in full.items():
        tag = f"{r:g}"
        one = _resolved(f"configs/{P[arm]}_only_r{tag}.yaml")
        assert len(one) == 1 and one[0]["name"] == c["name"] and one[0].get("sweep_parent") == P[arm]
        assert _strip(one[0]) == _strip(c)


def test_seal_covers_every_t4_run_and_nothing_else():
    rows = [r for r in withheld.SEALS if r[0] == "T4"]
    assert len(rows) == 1
    for arm in P:
        for c in _resolved(f"configs/{P[arm]}.yaml"):
            n = f"20261121-000000_{c['name']}"
            assert [r[0] for r in withheld.SEALS if fnmatch.fnmatchcase(n, r[1])] == ["T4"]
    for other in ("x_sweep_l1pl350cl_pc_off_r2", "x_sweep_l1pl350cl_pc_on_r2.4", "x_sweep_l1t16pl350cl_pc_off_r2"):
        assert not any(r[0] == "T4" for r in withheld.SEALS if fnmatch.fnmatchcase(other, r[1]))
    runs = REPO / "runs"
    if runs.exists():
        assert not [d.name for d in runs.iterdir() if fnmatch.fnmatchcase(d.name, rows[0][1]) and "_sweep_t4_" not in d.name]


def test_only_its_result_lifts_it(tmp_path):
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-21 · [REG] T4 (step budget): …\n## 2026-11-21 · [CORR] about [RESULT] T4 (step budget)\n")
    assert not withheld.lifted(T.T4["seal"], d)
    d.write_text(d.read_text() + "## 2026-11-22 · [RESULT] T4 (step budget): x\n")
    assert withheld.lifted(T.T4["seal"], d)


def test_table_refuses_while_sealed(tmp_path, monkeypatch):
    import analysis.load as L
    (tmp_path / f"20261121-000000_{P['off_8192']}_r2").mkdir()
    monkeypatch.setattr(L, "RUNS", tmp_path)
    monkeypatch.setattr(withheld, "lifted", lambda *a, **k: False)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    with pytest.raises(withheld.WithheldError):
        T.table()


def test_no_environment_access():
    src = (REPO / "analysis" / "t4.py").read_text()
    assert "os.environ" not in src and "putenv" not in src and "import os" not in src


def _rows(pairs, kv=0.95):
    """Synthetic sweep rows: (rate, throughput); saturated (KV high, throughput < 0.95 planned) where tp < 0.95 rate."""
    return pd.DataFrame([{"run": f"r{r}", "rate": float(r), "throughput": tp, "planned_rate": float(r),
                          "kv_mean": kv if tp < 0.95 * r else 0.5, "pf": math.nan} for r, tp in pairs])


SW = (1.5, 2.0, 2.5, 3.0, 3.5, 4.0)


def test_shared_floors_use_the_rates_both_saturate():
    a = _rows([(1.5, 1.45), (2.0, 1.75), (2.5, 1.78), (3.0, 1.80), (3.5, 1.82), (4.0, 1.84)])
    b = _rows([(1.5, 1.48), (2.0, 1.97), (2.5, 1.86), (3.0, 1.88), (3.5, 1.90), (4.0, 1.92)])   # 2.0 keeps up
    f = T.shared_floors(a, b, SW, SW)
    assert f["from"] == 2.5
    assert f["a"]["mu_sat_off"] == pytest.approx((1.78 + 1.80 + 1.82 + 1.84) / 4)
    assert f["b"]["mu_sat_off"] == pytest.approx((1.86 + 1.88 + 1.90 + 1.92) / 4)


def test_labels():
    assert T.label_floor(0.03, 0.03) == "neutral" and T.label_floor(0.031, 0.03) == "floor_up"
    assert T.label_floor(-0.05, 0.03) == "floor_down" and T.label_floor(math.nan, 0.03) == "undefined"
    assert T.label_warm(2.4, 2.4) == "warm_same" and T.label_warm(2.4, 2.6) == "warm_up"
    assert T.label_warm(2.4, 0.0) == "warm_down" and T.label_warm(2.4, math.nan) == "warm_undefined"
    f = lambda x: {"mu_sat_off": x, "status": "ok"}
    assert T.label_cell(f(1.80), 2.4) == "cell_holds"
    assert T.label_cell(f(1.91), 2.4) == "cell_moves"          # 2.0 < 1.05 x 1.91
    assert T.label_cell(f(1.80), 2.0) == "cell_moves"          # 2.0 is not below the warm capacity
    assert T.label_cell(f(1.80), 0.0) == "cell_moves"          # capacity below the grid
    assert T.label_cell({"mu_sat_off": math.nan, "status": "above_grid"}, 2.4) == "cell_moves"
    assert T.label_cell(f(math.nan), 2.4) == "cell_undefined" and T.label_cell(f(1.8), math.nan) == "cell_undefined"


def _f(x, status="ok"):
    return {"mu_sat_off": x, "status": status, "lam_sat": 2.5}


def _outcome(fa, fb, own_a, own_b, wa, wb, status_b="ok"):
    floors = {"a": {"mu_sat_off": fa}, "b": {"mu_sat_off": fb}, "from": 2.5}
    return T.outcome(floors, {"2048": _f(own_a), "8192": _f(own_b, status_b)}, {"on_2048": wa, "on_8192": wb})


def test_decisions():
    assert _outcome(1.80, 1.82, 1.80, 1.82, 2.4, 2.4)["decision"] == "budget_neutral"
    assert _outcome(1.80, 1.89, 1.80, 1.89, 2.4, 2.4)["decision"] == "budget_moves"     # +5 %, cell still in
    assert _outcome(1.80, 1.81, 1.80, 1.81, 2.4, 2.6)["decision"] == "budget_moves"     # warm moved
    assert _outcome(1.80, 1.95, 1.80, 1.95, 2.4, 2.4)["decision"] == "cell_moves"
    assert _outcome(1.80, 1.81, 1.80, 1.81, 2.4, 0.0)["decision"] == "cell_moves"       # capacity below the grid
    assert _outcome(math.nan, 1.8, math.nan, 1.8, 2.4, 2.4)["decision"] == "no_floor"
    assert _outcome(1.8, 1.8, 1.8, 1.8, 2.4, math.nan)["decision"] == "no_warm"
    assert _outcome(1.8, 1.8, 1.8, 1.8, math.nan, 2.4)["decision"] == "no_warm"        # rule F r1, T4-3


def test_a_moved_baseline_is_not_blamed_on_the_budget():
    """Rule F r1, T4-1: 2048 floor 1.89 today (cell fails at 2048 too), 8192 at 1.92 (+1.6 %, neutral)."""
    o = _outcome(1.89, 1.92, 1.92, 1.92, 2.4, 2.4)
    assert o["floor_label"] == "neutral" and o["cell_2048"] == "cell_moves" and o["decision"] == "baseline_moved"


def test_cell_edge_when_the_budget_did_not_move_anything():
    o = _outcome(1.88, 1.91, 1.88, 1.91, 2.4, 2.4)    # +1.6 %; 2.0 >= 1.05 x 1.88 = 1.974, but < 1.05 x 1.91
    assert o["cell_2048"] == "cell_holds" and o["cell_8192"] == "cell_moves" and o["decision"] == "cell_edge"


def test_a_capacity_rise_never_blames_the_budget_for_the_cell():
    """Rule F r2, T4-R2-1: delta +1.06 % (neutral), warm up; the 8192 cell fails on its floor at 1.91."""
    o = _outcome(1.89, 1.91, 1.89, 1.91, 2.4, 2.6)
    assert o["cell_2048"] == "cell_holds" and o["cell_8192"] == "cell_moves" and o["decision"] == "cell_edge"
    assert _outcome(1.80, 1.80, 1.80, 1.80, 2.4, 2.0)["decision"] == "cell_moves"     # warm_down broke it


def test_a_floor_above_the_grid_moves_the_cell():
    o = _outcome(1.80, math.nan, 1.80, math.nan, 2.4, 2.4, status_b="above_grid")
    assert o["floor_label"] == "floor_up" and o["cell_8192"] == "cell_moves" and o["decision"] == "cell_moves"


def test_floor_throughput_fallback():
    r = _rows([(1.5, 1.45), (2.0, 1.75), (2.5, 1.78), (3.0, 1.80), (3.5, 1.82), (4.0, 1.84)], kv=0.6)   # never full
    f = T.floor_throughput(r)
    assert f["status"] == "throughput_only" and f["lam_sat"] == 2.0 and f["mu_sat_off"] == pytest.approx((1.75 + 1.78 + 1.80 + 1.82 + 1.84) / 5)
    up = _rows([(1.5, 1.49), (2.0, 1.99), (2.5, 2.49), (3.0, 2.99), (3.5, 3.49), (4.0, 3.5)])
    assert T.floor_throughput(up, SW)["status"] == "above_grid"
    noisy = _rows([(1.5, 1.49), (2.0, 1.99), (2.5, 2.30), (3.0, 2.99), (3.5, 3.49), (4.0, 3.5)])   # a dip, then up
    assert T.floor_throughput(noisy, SW)["status"] == "fallback_unclear"
    gap = _rows([(1.5, 1.45), (2.0, 1.75), (2.5, 1.78), (3.0, 1.80), (3.5, 1.82)], kv=0.6)          # 4.0 missing
    assert T.floor_throughput(gap, SW)["status"] == "holes"                                        # rule F r2, T4-R2-2


def test_shared_floors_use_one_rule_for_both_arms():
    a = _rows([(1.5, 1.45), (2.0, 1.75), (2.5, 1.78), (3.0, 1.80), (3.5, 1.82), (4.0, 1.84)])
    b = _rows([(1.5, 1.45), (2.0, 1.75), (2.5, 1.86), (3.0, 1.88), (3.5, 1.90), (4.0, 1.92)], kv=0.6)
    f = T.shared_floors(a, b, SW, SW)
    assert f["a"]["status"] == f["b"]["status"] == "throughput_only"


def _sweep(parent, rows):
    rule = json.dumps({"against": "planned_rate", "max_q_slope": 0.01, "min_throughput_frac": 0.95, "monotone": True}, sort_keys=True)
    return pd.DataFrame([{"run": f"2026112{i}-000000_{parent}_r{r:g}", "parent": parent, "valid": v, "rate": r,
                          "throughput": tp, "planned_rate": r, "q_slope_per_s": qs, "capacity_rule": rule}
                         for i, (r, tp, qs, v) in enumerate(rows)])


def test_warm_capacity_cases(monkeypatch):
    par = P["on_8192"]
    ok = lambda r: (r, r * 0.99, 0.0, True)
    bad = lambda r: (r, r * 0.80, 0.05, True)
    rates = [2.0, 2.2, 2.4, 2.6, 2.8, 3.0]
    w = T.warm_capacity(par, _sweep(par, [ok(2.0), ok(2.2), ok(2.4), bad(2.6), bad(2.8), bad(3.0)]))
    assert w["mu"] == 2.4 and not w["censored"]
    assert T.warm_capacity(par, _sweep(par, [ok(r) for r in rates]))["censored"] is True
    assert T.warm_capacity(par, _sweep(par, [bad(r) for r in rates]))["mu"] == 0.0           # fails at 2.0
    hole = [ok(2.0), (2.2, 2.2 * 0.99, 0.0, False), ok(2.4), ok(2.6), ok(2.8), ok(3.0)]
    w = T.warm_capacity(par, _sweep(par, hole))
    assert math.isnan(w["mu"]) and w["at_least"] == 2.0 and w["hole"] == 2.2                   # rule F r1, T4-2
    assert math.isnan(T.warm_capacity(par, _sweep(par, [(2.0, 1.98, 0.0, False)] + [ok(r) for r in rates[1:]]))["mu"])


def test_check_settings():
    good = {"override_blocks": 6041, "pool_tokens": 96656, "non_default_args": {"max_num_batched_tokens": 8192}}
    assert T.check_settings("off_8192", good) == []
    assert T.check_settings("off_2048", good)                       # flag present in the 2048 arm
    assert T.check_settings("on_2048", {**good, "non_default_args": {}}) == []
    assert T.check_settings("on_8192", {**good, "pool_tokens": 90000})
    assert T.check_settings("on_8192", {**good, "non_default_args": {"max_num_batched_tokens": "8192"}}) == []
    assert T.check_settings("on_2048", {**good, "non_default_args": None})        # rule F r2, T4-R2-5


def test_sweep_rows_honour_the_seal(tmp_path, monkeypatch):
    """Rule F r1, T4-12: a sealed T4 folder is never loaded by the readers table() uses."""
    import analysis.load as L
    from analysis.gate import sweep_rows
    d = tmp_path / f"20261121-000000_{P['off_8192']}_r2"
    d.mkdir()
    (d / "config.yaml").write_text("name: x\n")
    monkeypatch.setattr(L, "RUNS", tmp_path)
    monkeypatch.setattr(withheld, "lifted", lambda *a, **k: False)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    monkeypatch.setattr(L, "load_run", lambda p: (_ for _ in ()).throw(AssertionError(f"read {p}")))
    from analysis import tables
    assert L.find_runs("*") == [] and not len(sweep_rows(P["off_8192"])) and not len(tables.sweep_table())


def test_prediction_sums_to_one():
    for k, v in T.PREDICTED.items():
        assert sum(v.values()) == pytest.approx(1.0), k


def test_order_interleaves_the_budgets_per_rate():
    o = T.T4["order"]
    assert len(o) == len(set(o)) == 24
    for kind in ("off", "on"):
        for i, r in enumerate(T.T4["rates"][kind]):
            k = o.index((f"{kind}_8192", r)), o.index((f"{kind}_2048", r))
            assert abs(k[0] - k[1]) == 1 and ((k[0] < k[1]) == (i % 2 == 0))


@pytest.mark.skipif(not (REPO / "tools/launcher.py").exists(), reason="the GPU queue launcher and its plans are not in this tree")
def test_plan_runs_every_item_from_its_own_config_in_order():
    plan = json.loads((REPO / "docs/probe/t4_ops/plan.json").read_text())
    assert "launches" not in plan or not plan["launches"]
    assert [i["key"] for i in plan["items"]] == [f"{a}_r{r}".replace("_", "-") for a, r in T.T4["order"]]
    import importlib.util
    spec = importlib.util.spec_from_file_location("launcher", REPO / "tools/launcher.py")
    import sys
    L = sys.modules.setdefault("launcher", importlib.util.module_from_spec(spec))
    if not hasattr(L, "load_plan"):
        spec.loader.exec_module(L)
    assert L.load_plan(REPO / "docs/probe/t4_ops/plan.json", REPO)["id"] == "T4"    # the launcher's own plan check
    for i in plan["items"]:
        arm, tag = i["key"].rsplit("-r", 1)
        arm = arm.replace("-", "_")
        assert i["run_name"] == f"{P[arm]}_r{tag}" and i["config"] == f"configs/{P[arm]}_only_r{tag}.yaml"
        assert (REPO / i["config"]).exists()
        assert _resolved(i["config"])[0]["name"] == i["run_name"]
    assert plan["stop"]["max_attempts"] == 3 and plan["prelaunch"]["limit_w"] == "350.00"
