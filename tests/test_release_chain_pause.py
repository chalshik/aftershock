"""harness/checks.py `release_chain` in pause mode (`workload.sessions.cap_mode: pause`, T1 draft): a later turn left
unsent because its session was still paused at the drain deadline (or resumed too late) is blocked by its own session.
The branch runs only when the records carry `t_paused`, so every other run's check output is unchanged; that is checked
against the check as it was at b1b778c (the commit before the change), byte for byte."""
import importlib.util
import json
import shutil
import subprocess
from pathlib import Path

import pytest

import harness.loadgen as lg
from harness import checks
from tests.test_e2e_fake import run_fake
from tests.test_session_cap import WL

REPO = Path(__file__).resolve().parent.parent
BEFORE = "b1b778c"   # harness/checks.py before the pause branch
HAS_BEFORE = subprocess.run(["git", "cat-file", "-e", f"{BEFORE}^{{commit}}"], cwd=REPO, capture_output=True).returncode == 0
needs_before = pytest.mark.skipif(not HAS_BEFORE, reason=f"commit {BEFORE} is not in this tree's history")


def _old_checks(tmp_path):
    src = subprocess.run(["git", "show", f"{BEFORE}:harness/checks.py"], cwd=REPO, check=True, capture_output=True,
                         text=True).stdout
    assert "t_paused" not in src
    p = tmp_path / "checks_before.py"
    p.write_text(src)
    spec = importlib.util.spec_from_file_location("checks_before", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def pause_run(tmp_path_factory):
    """cap 1 from 8 s, ~16 sessions of 6 turns in service before it, drain 3 s: many sessions are still paused at
    the deadline."""
    tmp = tmp_path_factory.mktemp("pause")
    return run_fake(tmp, workload={**WL, "rate": 12.0, "sessions": {**WL["sessions"], "turns": 6, "cap": 1,
                                                                     "cap_from_s": 8, "cap_mode": "pause"}},
                    drain_s=3)


def _records(run_dir):
    return [json.loads(l) for l in (run_dir / "requests.jsonl").read_text().splitlines() if l.strip()]


def _copy(run_dir, tmp_path, recs):
    d = tmp_path / run_dir.name
    shutil.copytree(run_dir, d)
    (d / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return d


def _chain(d, mod=checks):
    return next(c for c in mod.run_checks(d)["checks"] if c["name"] == "release_chain")


def test_a_pause_run_ending_with_sessions_paused_at_the_deadline_passes(pause_run):
    chk = json.loads((pause_run / "checks.json").read_text())
    assert chk["valid"], chk["failed"]
    rc = next(c for c in chk["checks"] if c["name"] == "release_chain")
    assert rc["passed"] and rc["detail"]["unexplained"] == 0
    assert rc["detail"]["unsent_paused_at_deadline"] >= 1                  # the case the branch is for occurred
    recs = _records(pause_run)
    stuck = [r for r in recs if r.get("t_paused") is not None and r["t_resumed"] is None]
    assert stuck and all(r["t_sent"] is None for r in stuck)


@needs_before
def test_without_the_branch_the_same_run_would_fail(pause_run, tmp_path):
    """The pause fields stripped: the paused-at-the-deadline turns are unexplained, by the old check and the new."""
    recs = [{k: v for k, v in r.items() if k not in lg.PAUSE_FIELDS} for r in _records(pause_run)]
    d = _copy(pause_run, tmp_path, recs)
    rc = _chain(d)
    assert not rc["passed"] and rc["detail"]["unexplained"] >= 1
    assert json.dumps(_chain(d, _old_checks(tmp_path)), default=str) == json.dumps(rc, default=str)


def test_an_unexplained_unsent_turn_still_fails(pause_run, tmp_path):
    recs = _records(pause_run)
    # a paused turn whose session got its slot back early, yet the turn was never sent
    i = next(k for k, r in enumerate(recs) if r.get("t_paused") is not None and r["t_resumed"] is None)
    early = [dict(r) for r in recs]
    early[i]["t_resumed"] = early[i]["t_paused"] + 0.01
    rc = _chain(_copy(pause_run, tmp_path / "a", early))
    assert not rc["passed"] and rc["detail"]["examples"] == [recs[i]["request_id"]]
    # a turn of a session that never paused, its previous turn ended in time, never sent
    by = {(r["session_id"], r["turn"]): r for r in recs}
    j = next(k for k, r in enumerate(recs) if r["turn"] > 0 and r["status"] == "ok" and r.get("t_paused") is None
             and by[(r["session_id"], r["turn"] - 1)]["t_end"] is not None)
    unsent = [dict(r) for r in recs]
    unsent[j].update(t_sent=None, t_first_token=None, t_end=None, status="unfinished")
    rc = _chain(_copy(pause_run, tmp_path / "b", unsent))
    assert not rc["passed"] and recs[j]["request_id"] in rc["detail"]["examples"]


@needs_before
def test_a_run_without_the_pause_columns_checks_byte_for_byte_as_before(tmp_path):
    """An admission-mode capped run with turns left unsent at the deadline: the whole check output equals the one the
    check gave before the change (the pause-stripped copy above is a second such run)."""
    (tmp_path / "adm").mkdir()
    run_dir = run_fake(tmp_path / "adm", workload={**WL, "rate": 12.0,
                                                    "sessions": {**WL["sessions"], "turns": 6, "cap": 1,
                                                                 "cap_from_s": 8}}, drain_s=3)
    assert not any(k in r for r in _records(run_dir) for k in lg.PAUSE_FIELDS)
    new, old = checks.run_checks(run_dir), _old_checks(tmp_path).run_checks(run_dir)
    assert json.dumps(new, default=str) == json.dumps(old, default=str)
    rc = next(c for c in new["checks"] if c["name"] == "release_chain")
    assert set(rc["detail"]) == {"unsent_later_turns", "unexplained", "examples"}


def test_t1_reads_the_fake_pause_run_as_no_fault(pause_run):
    """analysis/t1.py's pause invariant on loadgen's own records against fake_vllm."""
    from analysis import t1
    from analysis.load import load_run
    run = load_run(pause_run)
    sl = t1.slots_paused(run, 1, 8.0)
    assert sl["fields"] and sl["pauses"] >= 1 and sl["handovers"] >= 1
    assert not t1.pause_fault(sl), sl


def test_a_deadline_between_the_resume_and_the_send_reads_as_resumed_late(monkeypatch, tmp_path):
    """T1 draft 3, review R6: session 0 pauses at 1 s (two hold a slot, cap 1), is handed its slot back at 4 s when
    session 1 ends, and its next turn is due at 1 + 50 = 51 s, past the deadline (50 + 0.5 s). Cut off in the release
    sleep, its placeholder must carry the resume, and `release_chain` must count it as resumed late, not paused."""
    import pandas as pd
    from tests import test_cap_pause as CP
    sched = pd.DataFrame([dict(request_id=0, t_planned=0.0, session_id=0, turn=0),
                          dict(request_id=1, t_planned=0.5, session_id=1, turn=0),
                          dict(request_id=2, t_planned=2.0, session_id=1, turn=1),
                          dict(request_id=3, t_planned=50.0, session_id=0, turn=1)])
    sched = sched.assign(prompt_len=1, output_len=1, burst=False)
    cfg = CP.resolved({"cap": 1, "cap_from_s": 1.0, "cap_mode": "pause"}, drain_s=0.5)
    text, _, _ = CP.run_virtual(monkeypatch, tmp_path, cfg, sched=sched, dur=lambda row: 1.0)
    recs = {r["request_id"]: r for r in (json.loads(l) for l in text.splitlines())}
    cut = recs[3]
    assert cut["status"] == "unfinished" and cut["t_sent"] is None
    assert cut["t_paused"] == pytest.approx(1.0) and cut["t_resumed"] == pytest.approx(4.0)
    assert cut["pause_s"] == pytest.approx(3.0) and cut["pause_s_cum"] == pytest.approx(3.0)
    assert cut["slots_at_resume"] == 1 and cut["resume_queue_at_resume"] == 0 and "done" not in cut
    (tmp_path / "env.json").write_text("{}")
    rc = _chain(tmp_path)
    assert rc["detail"]["unsent_resumed_late"] == 1 and rc["detail"]["unsent_paused_at_deadline"] == 0
    assert rc["passed"] and rc["detail"]["unexplained"] == 0


def test_a_session_still_paused_at_the_deadline_keeps_a_null_resume(monkeypatch, tmp_path):
    """The other branch of the same check: session 1 never ends before the deadline, so session 0 is never resumed."""
    import pandas as pd
    from tests import test_cap_pause as CP
    sched = pd.DataFrame([dict(request_id=0, t_planned=0.0, session_id=0, turn=0),
                          dict(request_id=1, t_planned=0.5, session_id=1, turn=0),
                          dict(request_id=2, t_planned=3.0, session_id=0, turn=1)])
    sched = sched.assign(prompt_len=1, output_len=1, burst=False)
    cfg = CP.resolved({"cap": 1, "cap_from_s": 1.0, "cap_mode": "pause"}, drain_s=2.0)
    text, _, _ = CP.run_virtual(monkeypatch, tmp_path, cfg, sched=sched,
                                dur=lambda row: 100.0 if int(row.session_id) == 1 else 1.0)
    recs = {r["request_id"]: r for r in (json.loads(l) for l in text.splitlines())}
    assert recs[2]["t_paused"] == pytest.approx(1.0) and recs[2]["t_resumed"] is None and recs[2]["pause_s"] is None
    (tmp_path / "env.json").write_text("{}")
    rc = _chain(tmp_path)
    assert rc["detail"]["unsent_paused_at_deadline"] == 1 and rc["detail"]["unsent_resumed_late"] == 0
    assert rc["detail"]["unexplained"] == 0
