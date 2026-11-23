"""analysis/t2.py (T2, the cache-off burst control): configs, seal, labels, outcome and the refusal; no run folder is read."""
import fnmatch
import json
import math
from pathlib import Path

import pytest

from analysis import t2 as T
from analysis import withheld

REPO = Path(__file__).resolve().parent.parent
REG19 = {"burst": "burst_l1pl350cl_r2.0_w3000_s{}", "control": "stage2_control_l1pl350cl_r2.0_w3000_s{}"}


def _resolved(path: str) -> dict:
    from harness.config import with_defaults
    from harness.run import expand, load_config
    out = expand(with_defaults(load_config(REPO / path)))
    assert len(out) == 1
    return json.loads(json.dumps(out[0]))


@pytest.mark.parametrize("seed", T.T2["seeds"])
@pytest.mark.parametrize("kind", ["burst", "control"])
def test_configs_are_reg19s_with_caching_off_at_matched_load(kind, seed):
    new = _resolved(f"configs/{T.config_name(kind, seed)}.yaml")
    old = _resolved(f"configs/{REG19[kind].format(seed)}.yaml")
    assert new["name"] == T.config_name(kind, seed) and new["seed"] == seed
    assert new["server"]["args"].pop("--no-enable-prefix-caching") is True
    assert new["server"]["args"]["--enable-prefix-caching"] is None
    new["server"]["args"]["--enable-prefix-caching"] = old["server"]["args"]["--enable-prefix-caching"]
    assert new["workload"]["rate"] == 0.8 and new["workload"]["duration_s"] == 2580 and new["verdict"]["watch_s"] == 1500
    for c in (new, old):
        c.pop("name"), c["workload"].pop("rate"), c["workload"].pop("duration_s"), c["verdict"].pop("watch_s")
        c.pop("source_config", None)
    assert new == old
    if kind == "burst":
        assert new["trigger"] == {"kind": "burst", "at_s": 900, "duration_s": 180, "rate": 2.4, "prompt_tokens": 2944}
    else:
        assert not new.get("trigger")


def test_windows_fit_the_run():
    c = _resolved(f"configs/{T.config_name('burst', 15)}.yaml")
    w = T.windows(c)
    assert w["B"] == (300.0, 900.0) and w["E"] == (1980.0, 2580.0)
    assert w["E1"] == (1980.0, 2280.0) and w["E2"] == (2280.0, 2580.0)
    assert c["verdict"]["recovered_q_excess"] == T.T2["q_excess"]            # rule F r1, T2-8e
    r19 = _resolved(f"configs/{REG19['burst'].format(15)}.yaml")
    assert T.windows(r19)["E"] == (3480.0, 4080.0)                          # validate reads REG-19 on its own E


def test_seal_covers_the_six_and_nothing_else():
    rows = [r for r in withheld.SEALS if r[0] == "T2"]
    assert len(rows) == 1
    for sd in T.T2["seeds"]:
        for kind in ("burst", "control"):
            n = f"20261121-000000_{T.config_name(kind, sd)}"
            assert [r[0] for r in withheld.SEALS if fnmatch.fnmatchcase(n, r[1])] == ["T2"]
    for sd in (15, 16, 17, 18, 19):
        for fam in REG19.values():
            assert not any(r[0] == "T2" for r in withheld.SEALS if fnmatch.fnmatchcase("x_" + fam.format(sd), r[1]))
    assert not fnmatch.fnmatchcase("x_" + T.config_name("burst", 18), rows[0][1])


@pytest.mark.skipif(not (REPO / "docs/drafts/t2_draft2.md").exists(), reason="the registration drafts are not in this tree")
def test_only_its_result_lifts_it(tmp_path):
    d = tmp_path / "decisions.md"
    body = (REPO / "docs/drafts/t2_draft2.md").read_text()      # the registration's body names the seal string
    assert T.T2["seal"] in body
    d.write_text("## 2026-11-21 · [REG] T2 (cache-off burst control): …\n" + body)
    assert not withheld.lifted(T.T2["seal"], d)                 # rule F r1, T2-8a: headings only
    d.write_text(d.read_text() + "## 2026-11-22 · [RESULT] T2 (cache-off burst control): x\n")
    assert withheld.lifted(T.T2["seal"], d)


def test_table_refuses_while_sealed(tmp_path, monkeypatch):
    import analysis.load as L
    (tmp_path / f"20261121-000000_{T.config_name('burst', 15)}").mkdir()
    monkeypatch.setattr(L, "RUNS", tmp_path)
    monkeypatch.setattr(withheld, "lifted", lambda *a, **k: False)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    with pytest.raises(withheld.WithheldError):
        T.table()


def test_no_environment_access():
    src = (REPO / "analysis" / "t2.py").read_text()
    assert "os.environ" not in src and "putenv" not in src and "import os" not in src


B = {"q": 0.1, "c": 0.80, "p": 0.80}
E_OK = {"q": 0.3, "c": 0.79, "p": 0.80}


def _w(e, e1=None, e2=None, b=B):
    return {"B": b, "E": e, "E1": e1 or e, "E2": e2 or e}


def test_twin_labels():
    assert T.label_twin(_w(E_OK)) == "drained"
    assert T.label_twin(_w({"q": 5.0, "c": 0.79, "p": 0.8})) == "twin_not_drained"
    assert T.label_twin(_w(E_OK, b={"q": 3.0, "c": 0.8, "p": 0.8})) == "bad_baseline"
    assert T.label_twin(_w({"q": math.nan, "c": 0.8, "p": 0.8})) == "unclear"


def test_burst_labels_judge_against_the_twin():
    nan = math.nan
    assert T.label_burst(_w(E_OK), 0.80, 300.0, 900.0) == "drained"
    # rule F r1, T2-2: p follows the slowed server (c/p = 0.97), the twin's c shows the shortfall
    slow = {"q": 30.0, "c": 0.70, "p": 0.72}
    assert T.label_burst(_w(slow, {"q": 30.0}, {"q": 31.0}), 0.80, nan, 900.0) == "slow"
    assert T.label_burst(_w(slow, {"q": 40.0}, {"q": 20.0}), 0.80, nan, 900.0) == "draining"     # T2-6
    assert T.label_burst(_w(slow), 0.80, 200.0, 900.0) == "slow"           # rule F r2, T2-R2-1: emptied once, still slow
    rel = {"q": 9.0, "c": 0.79, "p": 0.8}
    assert T.label_burst(_w(rel, {"q": 9.0}, {"q": 14.0}), 0.80, 200.0, 900.0) == "relapse"     # T2-4
    assert T.label_burst(_w(slow, {"q": math.nan}, {"q": 31.0}), 0.80, nan, 900.0) == "unclear"  # T2-R2-2
    assert T.label_burst(_w(rel, {"q": 20.0}, {"q": 9.0}), 0.80, nan, 900.0) == "draining"     # T2-R2-5, at full rate
    assert T.label_burst(_w({"q": 30.0, "c": 0.79, "p": 0.8}), 0.80, nan, 900.0) == "unclear"   # line, no shortfall, never emptied
    assert T.label_burst(_w({"q": 0.2, "c": 0.60, "p": 0.8}), 0.80, nan, 900.0) == "unclear"    # shortfall, no line
    assert T.label_burst(_w(E_OK, b={"q": 0.1, "c": 0.70, "p": 0.8}), 0.80, 300.0, 900.0) == "bad_baseline"
    assert T.label_burst(_w(E_OK), nan, 300.0, 900.0) == "unclear"


def test_outcome():
    ok = {"twin": "drained"}
    d = lambda *b: T.outcome({15: {**ok, "burst": b[0]}, 16: {**ok, "burst": b[1]}, 17: {**ok, "burst": b[2]}})
    o = d("drained", "drained", "drained")
    assert o["decision"] == "drains" and o["slow_upper"] == pytest.approx(1 - 0.05 ** (1 / 3))
    assert d("drained", "slow", "drained")["decision"] == "slow_found"
    for other in ("unclear", "draining", "relapse"):
        assert d("drained", other, "drained")["decision"] == "mixed"
    o = T.outcome({15: {"twin": "twin_not_drained", "burst": "slow"}, 16: {**ok, "burst": None},
                   17: {**ok, "burst": "drained"}})
    assert o["decision"] == "too_few" and o["n"] == 1 and set(o["not_counted"]) == {15, 16}
    assert d("bad_baseline", "drained", "drained")["n"] == 2


def test_earliest_uses_valid_runs_only():
    import inspect
    from analysis import load
    assert inspect.signature(load.find_runs).parameters["valid_only"].default is True   # rule F r1, T2-8b
    assert "find_runs(pattern)" in inspect.getsource(T._earliest)


def test_validate_pass_rule(monkeypatch):
    """Rule F r2, T2-R2-8: `validate`'s pass condition on synthetic rows, no run read."""
    labs = iter([("slow", "drained")] * 4 + [("unclear", "drained")])
    monkeypatch.setattr(T, "_earliest", lambda pat: ({"dir": Path(pat)}, []))
    monkeypatch.setattr(T, "pair_row", lambda b, w: dict(zip(("burst_label", "twin_label"), next(labs))))
    v = T.validate()
    assert v["k_slow"] == 4 and v["k_twin_drained"] == 5 and v["passes"]
    labs = iter([("slow", "drained")] * 3 + [("drained", "drained")] * 2)
    assert not T.validate()["passes"]


def test_prediction_sums_to_one():
    assert sum(T.PREDICTED["outcome"].values()) == pytest.approx(1.0)
    assert sum(T.PREDICTED["per_burst"].values()) == pytest.approx(1.0)


@pytest.mark.skipif(not (REPO / "tools/launcher.py").exists(), reason="the GPU queue launcher and its plans are not in this tree")
def test_plan_is_the_registered_order():
    plan = json.loads((REPO / "docs/probe/t2_ops/plan.json").read_text())
    assert [i["key"] for i in plan["items"]] == [k.replace("_", "-") for k in T.T2["order"]]
    import importlib.util
    spec = importlib.util.spec_from_file_location("launcher", REPO / "tools/launcher.py")
    import sys
    L = sys.modules.setdefault("launcher", importlib.util.module_from_spec(spec))
    if not hasattr(L, "load_plan"):
        spec.loader.exec_module(L)
    assert L.load_plan(REPO / "docs/probe/t2_ops/plan.json", REPO)["id"] == "T2"    # the launcher's own plan check
    for i in plan["items"]:
        kind, sd = i["key"].split("-")
        assert i["run_name"] == T.config_name(kind, int(sd)) and i["config"] == f"configs/{i['run_name']}.yaml"
        assert _resolved(i["config"])["name"] == i["run_name"]
    assert plan["stop"]["max_attempts"] == 3 and "docs/probe/t4_ops/state/queue.lock" in plan["locks"]
    import re
    assert re.match(plan["approve_pat"], "## 2026-11-21 · [DECISION] (author) T2 starts: x")          # rule F r1, T2-8c
    assert not re.match(plan["approve_pat"], "## 2026-11-21 · [DECISION] T2 starts: x")
