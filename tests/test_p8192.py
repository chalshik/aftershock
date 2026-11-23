"""analysis/p8192.py (P0 at an 8192-token step budget): configs, seed rule, seal, order, futility, refusals, settings
check, plan, and the cold side against REG-19's at REG-19's lambda (the only test that reads a run folder)."""
import fnmatch
import math
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from analysis import gate, withheld
from analysis import p8192 as P

REPO = Path(__file__).resolve().parent.parent
SEEDS = P.P8192["seeds"]
REG19 = {"burst": "burst_l1pl350cl_r2.0_w3000_s15", "control": "stage2_control_l1pl350cl_r2.0_w3000_s15"}


def _resolved(name: str) -> dict:
    from harness.config import with_defaults
    from harness.run import expand, load_config
    out = expand(with_defaults(load_config(REPO / f"configs/{name}.yaml")))
    assert len(out) == 1
    return json.loads(json.dumps(out[0]))


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("kind", ["burst", "control"])
def test_configs_are_reg19s_but_for_budget_pool_rate_and_seed(kind, seed):
    new, old = _resolved(P.config_name(kind, seed)), _resolved(REG19[kind])
    assert new["name"] == P.config_name(kind, seed) and new["seed"] == seed
    assert new["workload"]["rate"] == 2.2 and old["workload"]["rate"] == 2.0
    assert new["server"]["args"].pop("--max-num-batched-tokens") == 8192
    assert new["server"]["args"].pop("--num-gpu-blocks-override") == 6041
    for c in (new, old):
        c.pop("name"), c.pop("seed"), c["workload"].pop("rate"), c.pop("source_config", None)
    assert new == old                       # 4080 s arrivals, 3000 s watch, the burst, the verdict: REG-19's
    assert (new.get("trigger") or {}).get("rate") == (2.4 if kind == "burst" else None)


def test_seed_rule_on_a_fake_schedule():
    peaks = {1000: 3.2, 1001: 3.0, 1002: 3.1, 1003: 2.9, 1004: 3.09, 1005: 1.0, 1006: 2.0}
    r = P.choose_seeds({**P.P8192, "n_seeds": 3}, used={1003}, peak_fn=lambda s: {"seed": s, "peak": peaks[s]})
    assert r["seeds"] == (1001, 1004, 1005)                     # 3.1 is not calm; a used seed is skipped
    assert [c["seed"] for c in r["candidates"]] == [1000, 1001, 1002, 1004, 1005]


def test_the_seeds_are_the_rules_and_used_nowhere_else():
    own = set(SEEDS)
    r = P.choose_seeds(used=P.used_seeds() - own)               # the rule as applied before these configs existed
    assert r["seeds"] == SEEDS
    assert all(c["peak"] < 3.1 for c in r["candidates"] if c["seed"] in own)
    for f in (REPO / "configs").glob("*.yaml"):                 # no other config uses them
        if "_b8192_s100" not in f.name:
            assert not any(f"seed: {s}\n" in f.read_text() for s in own), f.name


def test_no_seed_is_exposed_and_the_order_is_ascending():
    assert all(not P.exposed(s)["exposed"] for s in SEEDS)
    assert P.order() == tuple(sorted(SEEDS))


def test_seal_covers_exactly_the_ten():
    want = {P.config_name(k, s) for k in ("burst", "control") for s in SEEDS}
    rows = [r for r in withheld.SEALS if r[0] == "P8192"]
    assert len(rows) == 1 and rows[0][2] == P.P8192["seal"]
    for f in (REPO / "configs").glob("*.yaml"):
        n = f"20261122-000000_{f.stem}"
        hit = fnmatch.fnmatchcase(n, rows[0][1])
        assert hit == (f.stem in want), f.stem
    for n in ("20261113-000000_burst_l1pl350cl_r2.0_w3000_s15", "20261117-000000_k4_l1pl350cl_r2.2_s137",
              "20261121-000000_sweep_t4_l1pl350cl_pc_on_b8192_r2.2"):   # neighbours on record stay readable
        assert withheld.sealed_by(n) != "P8192"


@pytest.mark.skipif(not (REPO / "tools/launcher.py").exists(), reason="the GPU queue launcher and its plans are not in this tree")
def test_plan_runs_control_then_burst_per_seed_from_their_configs():
    plan = json.loads((REPO / "docs/probe/p8192_ops/plan.json").read_text())
    assert [i["key"] for i in plan["items"]] == [f"{k}-{s}" for s in SEEDS for k in ("control", "burst")]
    for i in plan["items"]:
        kind, s = i["key"].split("-")
        assert i["run_name"] == P.config_name(kind, int(s)) and i["config"] == f"configs/{i['run_name']}.yaml"
        assert _resolved(i["run_name"])["name"] == i["run_name"]
    assert "docs/probe/g1g2_ops/state/queue.lock" in plan["locks"]
    spec = importlib.util.spec_from_file_location("launcher", REPO / "tools/launcher.py")
    L = sys.modules.setdefault("launcher", importlib.util.module_from_spec(spec))
    if not hasattr(L, "load_plan"):
        spec.loader.exec_module(L)
    assert L.load_plan(REPO / "docs/probe/p8192_ops/plan.json", REPO)["id"] == "P8192"


def test_step_never_skips_a_seed_and_stops_on_futility():
    p = {**P.P8192, "seeds": (1, 2, 3, 4, 5)}
    ok = lambda s: {"outcome": "pass_clear", "control_verdict": "recovered", "exposed": False}
    bad = lambda s: {"outcome": "drains", "control_verdict": "recovered", "exposed": False}
    import analysis.p8192 as M
    orig = M.order
    M.order = lambda q=p: (1, 2, 3, 4, 5)
    try:
        assert P.step(p, pair_fn=ok, has_fn=lambda s: (s <= 2, s <= 2)) == {"done": [1, 2], "continue": True}
        assert P.step(p, pair_fn=bad, has_fn=lambda s: (s <= 3, s <= 3))["continue"] is False   # 0 + 2 left < 3
        with pytest.raises(ValueError, match="never skips"):
            P.step(p, pair_fn=ok, has_fn=lambda s: (s != 2, s != 2))
    finally:
        M.order = orig


def test_check_run_flags_a_wrong_budget_or_pool(monkeypatch):
    from analysis import t4
    good = {"override_blocks": 6041, "pool_tokens": 96656, "non_default_args": {"max_num_batched_tokens": 8192}}
    r = {"cfg": _resolved(P.config_name("burst", SEEDS[0])), "dir": Path("x")}
    monkeypatch.setattr(t4, "server_readouts", lambda d: good)
    assert P.check_run(r, "burst", SEEDS[0]) == []
    assert P.check_run(r, "control", SEEDS[0]) == ["trigger does not match the kind"]
    monkeypatch.setattr(t4, "server_readouts", lambda d: {**good, "non_default_args": {}})
    assert any("8192" in b for b in P.check_run(r, "burst", SEEDS[0]))
    monkeypatch.setattr(t4, "server_readouts", lambda d: {**good, "override_blocks": 5850})
    assert any("pool" in b for b in P.check_run(r, "burst", SEEDS[0]))


def test_table_refuses_without_seeds_or_while_sealed(monkeypatch):
    with pytest.raises(ValueError, match="seeds"):
        P.table({**P.P8192, "seeds": ()})
    monkeypatch.setattr(P, "sealed", lambda: ["20261122-000000_burst_l1pl350cl_r2.2_b8192_s1002"])
    with pytest.raises(withheld.WithheldError):
        P.table()


def test_cold_side_is_reg19s_at_reg19s_lambda():
    from analysis.load import RUNS, find_runs
    runs = find_runs(f"*_{REG19['burst']}") if RUNS.exists() else []
    if not runs or not withheld.lifted("[RESULT] REG-19"):
        pytest.skip("REG-19's seed-15 burst run is not on this machine")
    b = runs[0]
    mine, theirs = P.cold_side(b, {**P.P8192, "lam": gate.REG19["lam"]}), gate.reg19_cold_side(b)
    assert json.dumps(mine, default=str) == json.dumps(theirs, default=str)
    assert P.cold_side(b)["c"] == theirs["c"]                     # lambda enters only the comparison


@pytest.mark.parametrize("c,lp,holds", [(2.1, 2.5, True),    # below lambda 2.2 and below lambda_p
                                        (2.1, 2.0, False),   # below lambda but not below lambda_p: min() decides
                                        (2.2, 2.5, False),   # equal to lambda: the test is strict
                                        (2.19, 2.19, False)])
def test_cold_side_rule_at_its_boundary(monkeypatch, c, lp, holds):
    import pandas as pd
    monkeypatch.setattr(gate, "burst_end", lambda b: 1080.0)
    monkeypatch.setattr(gate, "completion_rate", lambda b, a, z: c)
    monkeypatch.setattr(gate, "planned_session_rate", lambda b, a, z: lp)
    monkeypatch.setattr(gate, "verdict", lambda b: {"end": {"q": 5.0}})
    b = {"metrics": pd.DataFrame({"t": [1700.0, 1800.0], "vllm:num_requests_waiting": [10.0, 10.0]})}
    r = P.cold_side(b)
    assert r["holds"] is holds and all(s["below"] is holds for s in r["spans"])
    assert r["q_ratio"] == 0.5


R19 = {**P.P8192, "lam": 2.0, "check_settings": False, "floor": 1.789,
       "names": {"control": "stage2_control_l1pl350cl_r2.0_w3000_s{}", "burst": "burst_l1pl350cl_r2.0_w3000_s{}"}}
KEYS = ("outcome", "control_verdict", "burst_verdict", "label", "cold_side", "drain_time_s", "control_waiting",
        "after_empty", "exposure", "c1", "owed", "baseline_pair", "emptyings_s")


def _reg19_runs_here() -> bool:
    from analysis.load import RUNS, find_runs
    return RUNS.exists() and withheld.lifted("[RESULT] REG-19") and bool(find_runs("*_burst_l1pl350cl_r2.0_w3000_s15"))


def _same(a, b):
    """Field-by-field equality; numpy arrays compared whole (json.dumps would summarise long ones, r2 N-1)."""
    import numpy as np
    import pandas as pd
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, (np.ndarray, pd.Series, pd.DataFrame)):
        return np.array_equal(np.asarray(a, dtype=object), np.asarray(b, dtype=object))
    if isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _lam_dicts():
    return [v for v in vars(gate).values() if isinstance(v, dict) and "lam" in v]


@pytest.mark.parametrize("seed", (15, 16, 17, 18, 19))
def test_pair_is_reg19s_pair_at_reg19s_settings(monkeypatch, seed):
    """Rule F r1 P-1 and r2 N-1. (1) With REG-19's names and lambda, `pair` gives `gate.reg19_pair`'s outcome and
    readouts on each of REG-19's seeds. (2) Every gate dict with a "lam" key set to 1.0, below every seed's c (1.74 to
    1.76): a reused function that read a module lambda instead of `p` would fail the cold side and change the outcome;
    `pair` must not change. (3) at lambda 1.70 the cold side must fail where REG-19's own
    pair passes: the lambda in `p` is the one the cold side judges (`pair` cannot run at 1.70 itself: `reg10_gate`
    refuses a run whose rate is not the spec's lambda, so the test checks that `pair` hands `cold_side` its own `p`,
    and that `cold_side` at 1.70 fails on the same burst run)."""
    if not _reg19_runs_here():
        pytest.skip("REG-19's runs are not on this machine")
    mine, theirs = P.pair(seed, R19), gate.reg19_pair(seed)
    for k in KEYS:
        assert _same(mine[k], theirs[k]), k
    assert theirs["outcome"] == "pass_clear"
    for d in _lam_dicts():
        monkeypatch.setitem(d, "lam", 1.0)
    again = P.pair(seed, R19)
    for k in KEYS:
        assert _same(again[k], mine[k]), k
    monkeypatch.undo()
    seen, orig = [], P.cold_side
    monkeypatch.setattr(P, "cold_side", lambda b, p=P.P8192: (seen.append(p["lam"]), orig(b, p))[1])
    P.pair(seed, R19)
    assert seen == [R19["lam"]]                   # pair hands the cold side its own p
    monkeypatch.undo()
    from analysis.load import find_runs
    b = gate.reg12_only(find_runs(f"*_burst_l1pl350cl_r2.0_w3000_s{seed}"), f"s{seed}")
    assert P.cold_side(b, {**R19, "lam": 1.70})["holds"] is False      # a lambda below c fails it
    assert gate.reg19_cold_side(b)["holds"] is True


def test_no_reused_function_binds_lambda_at_definition():
    """r2 N-1: no function in gate or p8192 has a default that is a float equal to a registered lambda or a dict
    carrying one (a default bound at definition time escapes the sentinel), except p8192's own `p=P8192`."""
    import inspect
    lams = {2.0, 2.2, 2.4}
    for mod in (gate, P):
        for name, f in vars(mod).items():
            if not inspect.isfunction(f) or f.__module__ != mod.__name__:
                continue
            for prm in inspect.signature(f).parameters.values():
                d = prm.default
                if d is inspect.Parameter.empty:
                    continue
                if mod is P and d is P.P8192:
                    continue
                assert not (isinstance(d, float) and d in lams and "lam" in prm.name), (name, prm.name)
                assert not (isinstance(d, dict) and "lam" in d), (name, prm.name)


def test_pair_refuses_a_run_check_fails(monkeypatch):
    """r2 (P-7): a `check_run` fault makes `pair` raise; nothing is counted."""
    if not _reg19_runs_here():
        pytest.skip("REG-19's runs are not on this machine")
    monkeypatch.setattr(P, "check_run", lambda r, kind, seed, p=None: ["pool: 5850 blocks"])
    with pytest.raises(ValueError, match="pool"):
        P.pair(15, R19)


@pytest.mark.parametrize("seed", SEEDS)
def test_seed_rule_peak_is_the_exposure_peak(seed):
    """Rule F r1, P-6: the rule's schedule (REG-19's template at lambda 2.2) is the control config's schedule."""
    assert P.peak(seed)["peak"] == pytest.approx(P.exposed(seed)["peak"], abs=1e-12)


@pytest.mark.parametrize("field,value,msg", [("seed", 1, "seed"), ("rate", 2.0, "rate"),
                                             ("watch", 1500, "3000 s"), ("duration", 2580, "3000 s")])
def test_check_run_flags_each_config_field(monkeypatch, field, value, msg):
    from analysis import t4
    good = {"override_blocks": 6041, "pool_tokens": 96656, "non_default_args": {"max_num_batched_tokens": 8192}}
    monkeypatch.setattr(t4, "server_readouts", lambda d: good)
    for kind in ("control", "burst"):
        cfg = _resolved(P.config_name(kind, SEEDS[0]))
        assert P.check_run({"cfg": cfg, "dir": Path("x")}, kind, SEEDS[0]) == []     # P-7: a clean control passes
        if field == "seed":
            cfg["seed"] = value
        elif field == "rate":
            cfg["workload"]["rate"] = value
        elif field == "watch":
            cfg["verdict"]["watch_s"] = value
        else:
            cfg["workload"]["duration_s"] = value
        assert any(msg in b for b in P.check_run({"cfg": cfg, "dir": Path("x")}, kind, SEEDS[0]))
    monkeypatch.setattr(t4, "server_readouts", lambda d: {**good, "pool_tokens": 93600})
    assert any("pool" in b for b in P.check_run({"cfg": _resolved(P.config_name("burst", SEEDS[0])), "dir": Path("x")},
                                                "burst", SEEDS[0]))
