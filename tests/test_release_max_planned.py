"""AIPerf's default release (`workload.sessions.release: max_planned`, AIPerf arm draft 2): harness.loadgen.release_mode,
session_chain's max(planned time, previous end) release, the run-time refusals, `release_chain`'s max_planned branch,
and the proof that every existing config resolves, releases and is checked exactly as before.

"Before" is T1's launch commit a108eb9 (harness/ there is identical to main at 68eafb7, this branch's base): its
harness/loadgen.py and harness/checks.py are loaded from git and run on the same inputs as today's, byte for byte."""
import asyncio
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import harness.loadgen as lg
from harness import checks, workload
from harness.config import load_config, with_defaults
from harness.run import expand
from tests import test_cap_pause as CP
from tests.test_e2e_fake import run_fake
from tests.test_session_cap import WL

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
BEFORE = "a108eb9"     # T1's launch commit
NEW = {f"{k}_l1pl350dr_r2.0_w3000_s{s}.yaml" for s in range(15, 20) for k in ("burst", "stage2_control")}
BURST = CP.BURST


def _old(name: str):
    if subprocess.run(["git", "cat-file", "-e", f"{BEFORE}^{{commit}}"], cwd=REPO, capture_output=True).returncode:
        pytest.skip(f"needs the study's git history (T1's launch commit {BEFORE}), which a fresh checkout of the "
                    "public release does not have")
    src = subprocess.run(["git", "show", f"{BEFORE}:harness/{name}.py"], cwd=REPO, check=True, capture_output=True,
                         text=True).stdout
    assert "max_planned" not in src
    spec = importlib.util.spec_from_loader(f"{name}_before", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src, f"{BEFORE}:harness/{name}.py", "exec"), mod.__dict__)
    return mod


@pytest.fixture(scope="module")
def old_lg():
    return _old("loadgen")


@pytest.fixture(scope="module")
def old_checks():
    return _old("checks")


def _old_mode(cfg: dict) -> str:
    """The release every existing config got before the change: loadgen.main's and checks.run_checks' expression."""
    return "closed" if bool((cfg["workload"].get("sessions") or {}).get("closed_loop")) else "open"


# ---------------------------------------------------------------- release_mode

def _cfg(**sessions):
    return {"workload": {"sessions": sessions}}


def test_release_mode_without_the_key_is_todays_behaviour():
    assert lg.release_mode({"workload": {}}) == "open"
    assert lg.release_mode({"workload": {"sessions": None}}) == "open"
    assert lg.release_mode(_cfg(turns=8)) == "open"
    assert lg.release_mode(_cfg(closed_loop=False)) == "open"
    assert lg.release_mode(_cfg(closed_loop=True)) == "closed"
    assert lg.release_mode(_cfg(closed_loop=True, cap=3, cap_mode="pause")) == "closed"
    assert lg.release_mode(_cfg(closed_loop=True, release=None)) == "closed"      # null = absent
    assert lg.release_mode(_cfg(closed_loop=True, release="max_planned")) == "max_planned"


@pytest.mark.parametrize("sessions", [
    {"closed_loop": True, "release": "closed"}, {"closed_loop": True, "release": "open"},
    {"closed_loop": True, "release": "Max_planned"}, {"closed_loop": True, "release": "max_planned "},
    {"closed_loop": True, "release": True}, {"closed_loop": True, "release": 1},
    {"closed_loop": True, "release": ["max_planned"]},
    {"release": "max_planned"}, {"closed_loop": False, "release": "max_planned"},
    {"closed_loop": 1, "release": "max_planned"},
    {"closed_loop": True, "release": "max_planned", "cap": 22},
    {"closed_loop": True, "release": "max_planned", "cap": 22, "cap_mode": "pause"},
])
def test_release_mode_refusals(sessions):
    with pytest.raises(ValueError):
        lg.release_mode({"workload": {"sessions": sessions}})


def test_session_chain_refuses_a_mode_it_does_not_know_or_a_cap_with_max_planned():
    rows = [SimpleNamespace(request_id=0, session_id=0, turn=0, t_planned=0.0, prompt_len=1, output_len=1)]
    for mode, cap in (("open", None), ("max_planned", lg.SessionCap(1))):
        with pytest.raises(ValueError):
            asyncio.run(lg.session_chain(rows, None, None, "", "", None, 0.0, None, {}, cap, mode))


@pytest.mark.parametrize("sessions", [{"release": "max_planned", "closed_loop": False},
                                      {"release": "max_planned", "cap": 2}, {"release": "maxplanned"}])
def test_a_bad_release_is_refused_before_the_server_starts(tmp_path, sessions):
    with pytest.raises(subprocess.CalledProcessError):
        run_fake(tmp_path, workload={**WL, "sessions": {**WL["sessions"], **sessions}})
    assert not any((tmp_path / "runs").glob("*")) if (tmp_path / "runs").exists() else True


# ---------------------------------------------------------------- every existing config is unchanged

def _resolved_all():
    """Every config in configs/ as run.py resolves it (defaults beneath, sweeps expanded). defaults.yaml is the layer
    merged beneath the others, not a run config (it has no workload); it is checked to set no `release`."""
    assert "release" not in json.dumps(yaml.safe_load((CONFIGS / "defaults.yaml").read_text()))
    out = {}
    for p in sorted(CONFIGS.glob("*.yaml")):
        if p.name != "defaults.yaml":
            out[p.name] = [c for c in expand(with_defaults(load_config(p)))]
    return out


def test_every_existing_config_resolves_to_its_old_release_mode():
    """Every config in configs/ but the arm's ten: no `release` key anywhere in its resolved form, and release_mode
    equals the expression loadgen and checks used before. The arm's ten: max_planned, and their resolved config differs
    from REG-19's parent only in `name` and `workload.sessions.release`."""
    from analysis.d1 import _diff_keys
    allc = _resolved_all()
    assert NEW <= set(allc), NEW - set(allc)
    old = {k: v for k, v in allc.items() if k not in NEW}
    assert len(old) >= 280
    modes = {"open": 0, "closed": 0}
    for name, cfgs in old.items():
        for c in cfgs:
            assert "release" not in (c["workload"].get("sessions") or {}), name
            assert lg.release_mode(c) == _old_mode(c), name
            modes[_old_mode(c)] += 1
    assert modes["open"] > 0 and modes["closed"] > 0
    for name in NEW:
        (c,) = allc[name]
        assert lg.release_mode(c) == "max_planned"
        (parent,) = allc[yaml.safe_load((CONFIGS / name).read_text())["extends"]]
        assert _diff_keys(c, parent) == ["name", "workload.sessions.release"], name
        assert c["workload"]["sessions"]["closed_loop"] is True and c["workload"]["sessions"].get("cap") is None


def test_the_release_key_leaves_the_schedule_unchanged():
    allc = _resolved_all()
    c = allc["burst_l1pl350dr_r2.0_w3000_s15.yaml"][0]
    p = allc["burst_l1pl350cl_r2.0_w3000_s15.yaml"][0]
    a, pa = workload.build_schedule(c["workload"], c["seed"], c.get("trigger"))
    b, pb = workload.build_schedule(p["workload"], p["seed"], p.get("trigger"))
    pd.testing.assert_frame_equal(a, b)
    assert all((x == y).all() for x, y in zip(pa.tokens, pb.tokens))


# ---------------------------------------------------------------- loadgen on a virtual clock, old and new

def _fake_send(mod, dur):
    async def send(session, url, model, row, prompt, gw, t0, out, inflight, t_released=None, extra=None):
        rec = mod.new_rec(row, t_released, **(extra or {}))
        inflight[rec["request_id"]] = rec
        try:
            rec["t_sent"] = mod.time.monotonic() - t0
            await asyncio.sleep(dur(row))
            rec["t_first_token"] = rec["t_end"] = mod.time.monotonic() - t0
            rec["prompt_tokens"], rec["output_tokens"], rec["status"] = int(row.prompt_len), int(row.output_len), "ok"
        finally:
            if rec["status"] != "unfinished":
                out.write(json.dumps(rec) + "\n")
                out.flush()
                inflight.pop(rec["request_id"], None)
        return rec
    return send


def run_virtual(monkeypatch, tmp_path, cfg, mod=lg, sched=None, dur=lambda row: 0.3 + 0.05 * (int(row.request_id) % 7)):
    """`mod.main` (today's harness.loadgen or the one at BEFORE) on test_cap_pause's virtual clock."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    if sched is None:
        sched, prompts = workload.build_schedule(cfg["workload"], cfg["seed"], cfg.get("trigger"))
    else:
        prompts = pd.DataFrame({"session_id": range(int(sched.session_id.max()) + 1),
                                "tokens": [[1]] * (int(sched.session_id.max()) + 1)})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    sched.to_parquet(tmp_path / "schedule.parquet", index=False)
    prompts.to_parquet(tmp_path / "prompts.parquet", index=False)
    (tmp_path / "env.json").write_text("{}")
    loop = CP.VirtualLoop()
    monkeypatch.setattr(mod, "time", SimpleNamespace(monotonic=loop.time))
    monkeypatch.setattr(mod, "send", _fake_send(mod, dur))
    monkeypatch.setattr(mod, "prompt_tokens", lambda *a: [1])
    t0 = loop.time() + 1.0
    try:
        loop.run_until_complete(mod.main(tmp_path, t0))
    finally:
        loop.close()
    return (tmp_path / "requests.jsonl").read_text()


DUR = lambda row: 0.3 + 0.05 * (int(row.request_id) % 7)     # test_cap_pause's default turn length
SLOW = lambda row: 2.5                                          # long enough that turns are cut at a 0.5 s drain
CASES = {   # (sessions over WL's, other config keys, turn length)
    "open": ({"closed_loop": False}, {}, DUR),
    "open_burst": ({"closed_loop": False}, {"trigger": BURST}, DUR),
    "closed": ({}, {}, DUR),
    "closed_burst_cutoff": ({}, {"trigger": BURST, "drain_s": 0.5}, SLOW),
    "cap_admission": ({"cap": 2, "cap_from_s": 10}, {"trigger": BURST}, DUR),
    "cap_pause": ({"cap": 2, "cap_from_s": 10, "cap_mode": "pause"}, {"trigger": BURST}, DUR),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_old_and_new_loadgen_write_the_same_records(monkeypatch, tmp_path, old_lg, case):
    sessions, over, dur = CASES[case]
    cfg = CP.resolved(sessions, **over)
    assert lg.release_mode(cfg) in ("open", "closed")
    new = run_virtual(monkeypatch, tmp_path / "new", cfg, dur=dur)
    old = run_virtual(monkeypatch, tmp_path / "old", cfg, mod=old_lg, dur=dur)
    assert len(new.splitlines()) > 30 and new == old


def _synthetic(gap=4.0, turns=5, sessions=4):
    rows = []
    for sid in range(sessions):
        rows += [dict(t_planned=0.5 * sid + j * gap, session_id=sid, turn=j, prompt_len=1, burst=False)
                 for j in range(turns)]
    df = pd.DataFrame(rows).sort_values("t_planned", kind="stable").reset_index(drop=True)
    df.insert(0, "request_id", range(len(df)))
    df["output_len"] = 1
    return df


SLOW_FAST = lambda row: 5.0 if (int(row.turn) + int(row.session_id)) % 2 == 0 else 1.0  # slower and faster than gap


def _recs(text):
    q = pd.DataFrame([json.loads(l) for l in text.splitlines()]).sort_values(["session_id", "turn"])
    g = q.groupby("session_id")
    return q.assign(prev_end=g.t_end.shift(), prev_planned=g.t_planned.shift())


def test_max_planned_releases_each_turn_at_max_of_planned_and_previous_end(monkeypatch, tmp_path):
    cfg = CP.resolved({"release": "max_planned"}, drain_s=1000)
    q = _recs(run_virtual(monkeypatch, tmp_path, cfg, sched=_synthetic(), dur=SLOW_FAST))
    assert len(q) == len(_synthetic()) and (q.status == "ok").all()
    t0 = q[q.turn == 0]
    assert (t0.t_released == t0.t_planned).all()                                   # turn 0: at its planned time
    later = q[q.turn > 0]
    want = later[["t_planned", "prev_end"]].max(axis=1)
    assert (later.t_released == want).all()                                       # exactly, per turn
    assert ((later.t_sent - later.t_released).abs() < 1e-9).all()
    assert (later.t_sent >= later.prev_end - 1e-9).all()                          # never while the previous runs
    late, early = later[later.prev_end > later.t_planned], later[later.prev_end < later.t_planned]
    assert len(late) >= 3 and len(early) >= 3                                     # both branches occurred
    assert (late.t_released == late.prev_end).all() and (early.t_released == early.t_planned).all()


def test_closed_and_open_release_rules_on_the_same_schedule(monkeypatch, tmp_path):
    """The same synthetic schedule without the key: closed releases at previous end + planned gap, open at the
    planned time, and both differ from max_planned exactly where the rule says they must."""
    closed = _recs(run_virtual(monkeypatch, tmp_path / "c", CP.resolved({}, drain_s=1000), sched=_synthetic(),
                               dur=SLOW_FAST))
    lc = closed[closed.turn > 0]
    assert ((lc.t_released - (lc.prev_end + lc.t_planned - lc.prev_planned)).abs() < 1e-9).all()
    opened = pd.DataFrame([json.loads(l) for l in run_virtual(
        monkeypatch, tmp_path / "o", CP.resolved({"closed_loop": False}, drain_s=1000), sched=_synthetic(),
        dur=SLOW_FAST).splitlines()])
    assert opened.t_released.isna().all() and ((opened.t_sent - opened.t_planned).abs() < 1e-9).all()
    mp = _recs(run_virtual(monkeypatch, tmp_path / "m", CP.resolved({"release": "max_planned"}, drain_s=1000),
                           sched=_synthetic(), dur=SLOW_FAST))
    m = mp.set_index("request_id").t_released
    c = closed.set_index("request_id").t_released
    later = closed[closed.turn > 0].request_id
    assert (m[later] <= c[later] + 1e-9).all() and (m[later] < c[later] - 1e-9).any()  # never later than closed


# ---------------------------------------------------------------- release_chain

def _chain(d, mod=checks):
    return next(c for c in mod.run_checks(d)["checks"] if c["name"] == "release_chain")


def _write(src, dst, recs):
    shutil.copytree(src, dst)
    (dst / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return dst


@pytest.mark.parametrize("case", ["closed_burst_cutoff", "cap_pause", "open", "closed"])
def test_checks_of_a_run_without_the_key_are_byte_for_byte_as_before(monkeypatch, tmp_path, old_checks, case):
    sessions, over, dur = CASES[case]
    d = tmp_path / "run"
    run_virtual(monkeypatch, d, CP.resolved(sessions, **over), dur=dur)
    new, old = checks.run_checks(d), old_checks.run_checks(d)
    assert json.dumps(new, default=str) == json.dumps(old, default=str)
    if case != "open":
        rc = next(c for c in new["checks"] if c["name"] == "release_chain")
        assert "release" not in rc["detail"]
    if case == "closed_burst_cutoff":
        rc = next(c for c in new["checks"] if c["name"] == "release_chain")
        assert rc["detail"]["unsent_later_turns"] >= 1                           # the blocked branch was exercised


def test_a_max_planned_run_cut_at_the_deadline_passes_release_chain(monkeypatch, tmp_path):
    d = tmp_path / "run"
    run_virtual(monkeypatch, d, CP.resolved({"release": "max_planned"}, trigger=BURST, drain_s=0.5), dur=SLOW)
    rc = _chain(d)
    assert rc["passed"] and rc["detail"]["release"] == "max_planned"
    assert rc["detail"]["unsent_later_turns"] >= 1 and rc["detail"]["unexplained"] == 0


def _two_sessions():
    """s0: turns at 0 and 10 s; s1: turns at 0.5 and 20 s. Deadline 20 + drain 2 = 22 s."""
    df = pd.DataFrame([dict(request_id=0, t_planned=0.0, session_id=0, turn=0),
                       dict(request_id=1, t_planned=0.5, session_id=1, turn=0),
                       dict(request_id=2, t_planned=10.0, session_id=0, turn=1),
                       dict(request_id=3, t_planned=20.0, session_id=1, turn=1)])
    return df.assign(prompt_len=1, output_len=1, burst=False)


def test_release_chain_fails_a_stalled_max_planned_chain_the_closed_rule_would_excuse(monkeypatch, tmp_path,
                                                                                   old_checks):
    """Failure control of the branch. s1's turn 0 lasts 5 s (ends 5.5): under max_planned its turn 1 is due at
    max(20, 5.5) = 20 <= 22 - 1, so a missing send is unexplained; the closed rule's 5.5 + 19.5 = 25 > 21 would call it
    blocked. Made unsent by hand, the new check fails on it and the check before the change passes it."""
    d = tmp_path / "run"
    text = run_virtual(monkeypatch, d, CP.resolved({"release": "max_planned"}, drain_s=2.0), sched=_two_sessions(),
                       dur=lambda row: 5.0 if (int(row.session_id), int(row.turn)) == (1, 0) else 1.0)
    recs = {r["request_id"]: r for r in (json.loads(l) for l in text.splitlines())}
    assert recs[3]["t_released"] == 20.0 and recs[1]["t_end"] == pytest.approx(5.5)
    assert _chain(d)["passed"]
    stalled = [dict(r) for r in recs.values()]
    for r in stalled:
        if r["request_id"] == 3:
            r.update(t_sent=None, t_first_token=None, t_end=None, t_released=None, status="unfinished",
                     prompt_tokens=None, output_tokens=None)
    d2 = _write(d, tmp_path / "stalled", stalled)
    rc = _chain(d2)
    assert not rc["passed"] and rc["detail"]["examples"] == [3] and rc["detail"]["unexplained"] == 1
    assert _chain(d2, old_checks)["passed"]                                       # the old rule would excuse it


def test_release_chain_blocks_a_turn_whose_previous_never_ended_or_ended_too_late(monkeypatch, tmp_path):
    """s1's turn 0 runs past the deadline (never ends): its turn 1 is blocked, not unexplained."""
    d = tmp_path / "run"
    run_virtual(monkeypatch, d, CP.resolved({"release": "max_planned"}, drain_s=2.0), sched=_two_sessions(),
                dur=lambda row: 100.0 if (int(row.session_id), int(row.turn)) == (1, 0) else 1.0)
    rc = _chain(d)
    assert rc["passed"] and rc["detail"]["unsent_later_turns"] == 1 and rc["detail"]["unexplained"] == 0


def test_release_chain_blocks_a_turn_whose_previous_ended_in_the_last_second(monkeypatch, tmp_path):
    """s1's turn 0 ends at 21.5 s, inside the 1 s slack before the 22 s deadline: its turn 1 is due at
    max(20, 21.5) = 21.5 > 21, so left unsent (made so by hand) it is blocked by its own session. A due that ignored
    the previous end (20 <= 21) would call it unexplained."""
    d = tmp_path / "run"
    text = run_virtual(monkeypatch, d, CP.resolved({"release": "max_planned"}, drain_s=2.0), sched=_two_sessions(),
                       dur=lambda row: 21.0 if (int(row.session_id), int(row.turn)) == (1, 0) else 1.0)
    recs = [json.loads(l) for l in text.splitlines()]
    assert next(r for r in recs if r["request_id"] == 1)["t_end"] == pytest.approx(21.5)
    for r in recs:
        if r["request_id"] == 3:
            r.update(t_sent=None, t_first_token=None, t_end=None, t_released=None, status="unfinished",
                     prompt_tokens=None, output_tokens=None)
    rc = _chain(_write(d, tmp_path / "late", recs))
    assert rc["passed"] and rc["detail"]["unsent_later_turns"] == 1 and rc["detail"]["unexplained"] == 0


# ---------------------------------------------------------------- end to end against fake_vllm

def test_max_planned_run_against_the_fake_server(tmp_path):
    from analysis import aiperf_arm
    from analysis.load import load_run
    run_dir = run_fake(tmp_path, workload={**WL, "sessions": {**WL["sessions"], "release": "max_planned"}},
                       trigger=BURST, drain_s=30)
    chk = json.loads((run_dir / "checks.json").read_text())
    assert chk["valid"], chk["failed"]
    rc = next(c for c in chk["checks"] if c["name"] == "release_chain")
    assert rc["detail"]["release"] == "max_planned"
    sched = pd.read_parquet(run_dir / "schedule.parquet")
    q = pd.DataFrame([json.loads(l) for l in (run_dir / "requests.jsonl").read_text().splitlines()])  # exact floats
    q = q[~q.request_id.isin(set(sched.request_id[sched.burst]))].sort_values(["session_id", "turn"])
    assert len(q) == int((~sched.burst).sum()) and (q.status == "ok").all()
    q = q.assign(prev_end=q.groupby("session_id").t_end.shift())
    later = q[q.turn > 0]
    assert len(later) > 10
    assert (later.t_released == later[["t_planned", "prev_end"]].max(axis=1)).all()
    assert (later.t_sent >= later.prev_end).all()
    audit = aiperf_arm.release_audit(load_run(run_dir))
    assert audit["mismatch"] == 0 and audit["before_prev_end"] == 0 and audit["later_sent"] == len(later)
