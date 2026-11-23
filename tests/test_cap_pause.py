"""Pause mode of the session cap (`workload.sessions.cap_mode: pause`, T1 draft: a pause arm beside D1r's admission-only
arm): harness.loadgen.SessionCap's resume queue, session_chain's pause at a turn boundary, the startup refusals, and an
end-to-end run against fake_vllm.

Most chain tests drive harness.loadgen.main on a virtual clock (VirtualLoop: the event loop's time jumps to the next
timer instead of waiting), with `send` replaced by a fake whose turn durations are fixed. The run is then exactly
reproducible, so two configs can be compared byte for byte, and minutes of schedule take milliseconds."""
import asyncio
import io
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

import harness.loadgen as lg
from harness import workload
from harness.config import _merge, load_config, with_defaults
from tests.test_e2e_fake import run_fake
from tests.test_session_cap import WL

REPO = Path(__file__).resolve().parent.parent
BURST = {"kind": "burst", "at_s": 8, "duration_s": 2, "rate": 3.0, "prompt_tokens": 64}


# ---------------------------------------------------------------- a virtual clock

class VirtualLoop(asyncio.SelectorEventLoop):
    """time() is virtual: when no I/O is ready, a wait of `timeout` advances it by `timeout` at once."""

    def __init__(self):
        super().__init__()
        self.vt = 1000.0
        real, loop = self._selector, self

        class Sel:
            def __getattr__(self, k):
                return getattr(real, k)

            def select(self, timeout=None):
                ev = real.select(0)
                if not ev:
                    if timeout is None:
                        raise RuntimeError("virtual loop: nothing scheduled, would block forever")
                    loop.vt += max(0.0, timeout)
                return ev

        self._selector = Sel()

    def time(self):
        return self.vt


def fake_send(dur):
    """send() as harness.loadgen writes it, minus HTTP: the turn lasts dur(row) seconds."""
    async def send(session, url, model, row, prompt, gw, t0, out, inflight, t_released=None, extra=None):
        rec = lg.new_rec(row, t_released, **(extra or {}))
        inflight[rec["request_id"]] = rec
        try:
            rec["t_sent"] = lg.time.monotonic() - t0
            await asyncio.sleep(dur(row))
            rec["t_first_token"] = rec["t_end"] = lg.time.monotonic() - t0
            rec["prompt_tokens"], rec["output_tokens"], rec["status"] = int(row.prompt_len), int(row.output_len), "ok"
        finally:
            if rec["status"] != "unfinished":
                out.write(json.dumps(rec) + "\n")
                out.flush()
                inflight.pop(rec["request_id"], None)
        return rec
    return send


class SpyCap(lg.SessionCap):
    """Logs (virtual time, used, resume queue, admission queue, what) after every change."""
    log: list

    def _log(self, what):
        self.log.append((asyncio.get_running_loop().time(), self.used, len(self.resumers), len(self.waiters), what))

    async def acquire(self, now=None):
        await super().acquire(now)
        self._log("acquire")

    def release(self):
        super().release()
        self._log("release")

    def pause(self):
        fut = super().pause()
        self._log("pause")
        return fut


def resolved(sessions=None, **over):
    """first_real.yaml (as run_fake extends it) with WL's workload and `sessions` merged into it."""
    cfg = with_defaults(load_config(REPO / "configs" / "first_real.yaml"))
    return _merge(cfg, {"workload": {**WL, "sessions": {**WL["sessions"], **(sessions or {})}}, **over})


def run_virtual(monkeypatch, tmp_path, cfg, sched=None, dur=lambda row: 0.3 + 0.05 * (int(row.request_id) % 7)):
    """harness.loadgen.main on a virtual clock. Returns (requests.jsonl text, SpyCap log, t0)."""
    tmp_path.mkdir(parents=True, exist_ok=True)
    if sched is None:
        sched, prompts = workload.build_schedule(cfg["workload"], cfg["seed"], cfg.get("trigger"))
    else:
        prompts = pd.DataFrame({"session_id": range(int(sched.session_id.max()) + 1),
                                "tokens": [[1]] * (int(sched.session_id.max()) + 1)})
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    sched.to_parquet(tmp_path / "schedule.parquet", index=False)
    prompts.to_parquet(tmp_path / "prompts.parquet", index=False)
    loop, logs = VirtualLoop(), []
    monkeypatch.setattr(lg, "time", SimpleNamespace(monotonic=loop.time))
    monkeypatch.setattr(lg, "send", fake_send(dur))
    monkeypatch.setattr(lg, "prompt_tokens", lambda *a: [1])
    orig = lg.session_cap

    def spy(c):
        cap = orig(c)
        if cap is not None:
            cap.__class__, cap.log = SpyCap, logs
        return cap

    monkeypatch.setattr(lg, "session_cap", spy)
    t0 = loop.time() + 1.0
    try:
        loop.run_until_complete(lg.main(tmp_path, t0))
    finally:
        loop.close()
    return (tmp_path / "requests.jsonl").read_text(), logs, t0


# ---------------------------------------------------------------- 1. admission (or absent) is today's cap

def test_admission_mode_is_byte_for_byte_the_cap_without_the_key(monkeypatch, tmp_path):
    base = {"cap": 2, "cap_from_s": 10}
    a_text, a_log, _ = run_virtual(monkeypatch, tmp_path / "a", resolved(base, trigger=BURST))
    b_text, b_log, _ = run_virtual(monkeypatch, tmp_path / "b", resolved({**base, "cap_mode": "admission"},
                                                                         trigger=BURST))
    assert a_text == b_text and a_log == b_log
    recs = [json.loads(l) for l in a_text.splitlines()]
    assert len(recs) > 50 and all(r["status"] == "ok" for r in recs)
    assert sum(1 for r in recs if r.get("t_admitted") is not None and r["t_admitted"] - r["t_arrived"] > 0.05) >= 1
    assert not any(k in r for r in recs for k in (*lg.PAUSE_FIELDS, "pause_s", "pause_s_cum"))  # no new fields
    assert max(used for _, used, *_ in a_log) > 2                     # the overhang before the switch was there


def test_session_cap_reads_the_mode():
    s = {"closed_loop": True, "cap": 3}
    assert lg.session_cap({"workload": {"sessions": s}}).mode == "admission"
    assert lg.session_cap({"workload": {"sessions": {**s, "cap_mode": None}}}).mode == "admission"
    assert lg.session_cap({"workload": {"sessions": {**s, "cap_mode": "admission"}}}).mode == "admission"
    assert lg.session_cap({"workload": {"sessions": {**s, "cap_mode": "pause", "cap_from_s": 5}}}).mode == "pause"


# ---------------------------------------------------------------- 2. the schedule does not depend on the mode

def test_schedule_is_identical_across_modes():
    frames = []
    for mode in (None, "admission", "pause"):
        s = {"cap": 2, "cap_from_s": 10, **({"cap_mode": mode} if mode else {})}
        cfg = resolved(s, trigger=BURST)
        sched, prompts = workload.build_schedule(cfg["workload"], cfg["seed"], cfg["trigger"])
        frames.append((sched, prompts))
    for sched, prompts in frames[1:]:
        pd.testing.assert_frame_equal(sched, frames[0][0])
        assert all((a == b).all() for a, b in zip(prompts.tokens, frames[0][1].tokens))


# ---------------------------------------------------------------- 3. pause at a turn boundary, resume first

CAP, FROM = 2, 10.0
TURN_S, GAP_S = 1.0, 1.5


def _synthetic():
    """Sessions 0-5 start at 0-5 s (all before the switch: an overhang of 4 over cap 2), 6 turns each; sessions 6 and 7
    start at 11 and 12 s, after it, with 3 turns. Planned gap GAP_S between turns; every turn lasts TURN_S."""
    rows = []
    for sid, (start, turns) in enumerate([(float(i), 6) for i in range(6)] + [(11.0, 3), (12.0, 3)]):
        rows += [dict(t_planned=start + j * GAP_S, session_id=sid, turn=j, prompt_len=1, burst=False)
                 for j in range(turns)]
    df = pd.DataFrame(rows).sort_values("t_planned", kind="stable").reset_index(drop=True)
    df.insert(0, "request_id", range(len(df)))
    df["output_len"] = 1
    return df


def _run_synthetic(monkeypatch, tmp_path, mode="pause"):
    cfg = resolved({"cap": CAP, "cap_from_s": FROM, "cap_mode": mode}, drain_s=1000)
    text, log, t0 = run_virtual(monkeypatch, tmp_path, cfg, sched=_synthetic(), dur=lambda row: TURN_S)
    q = pd.DataFrame([json.loads(l) for l in text.splitlines()]).sort_values(["session_id", "turn"])
    return q, [(t - t0, *rest) for t, *rest in log]


def test_overhang_pauses_at_turn_boundaries_after_the_switch_only(monkeypatch, tmp_path):
    q, log = _run_synthetic(monkeypatch, tmp_path)
    paused = q[q.t_paused.notna()]
    assert len(paused) == 4                                           # 6 in service at the switch, cap 2
    assert (paused.t_paused >= FROM).all() and (paused.turn > 0).all()
    assert set(paused.session_id) <= set(range(6))                    # only the overhang pauses
    # each pause is at the end of the session's previous turn, and with more than the cap holding slots
    prev_end = q.groupby("session_id").t_end.shift()
    assert ((paused.t_paused - prev_end[paused.index]).abs() < 1e-9).all()
    assert (paused.slots_at_pause >= CAP).all()                       # used after giving the slot up: still >= cap
    # in service never goes above the cap again once it has reached it after the switch
    after = [(t, used) for t, used, *_ in log if t >= FROM]
    first = next(i for i, (_, used) in enumerate(after) if used <= CAP)
    assert all(used <= CAP for _, used in after[first:])
    assert after[first][0] <= FROM + TURN_S + GAP_S + TURN_S + 1e-9   # within one turn cycle of the switch
    # the mode changes nothing before the switch: no pause events, and the same records as admission up to it
    assert not any(what == "pause" for t, *_, what in log if t < FROM)


def test_resume_queue_is_served_first_and_in_pause_order(monkeypatch, tmp_path):
    q, _ = _run_synthetic(monkeypatch, tmp_path)
    paused = q[q.t_paused.notna()].sort_values(["t_paused", "resume_queue_at_pause"])  # ties: two pauses at one time
    assert paused.resume_queue_at_pause.tolist() == [1, 2, 3, 4]
    assert paused.t_resumed.notna().all()
    assert paused.t_resumed.is_monotonic_increasing                   # FIFO by pause time
    assert paused.resume_queue_at_resume.tolist() == [3, 2, 1, 0]
    new = q[(q.turn == 0) & (q.t_arrived >= FROM)]
    assert len(new) == 2
    for p in paused.itertuples():
        for n in new.itertuples():
            if n.t_admitted > p.t_paused:                             # a new session admitted after this pause...
                assert p.t_resumed <= n.t_admitted                    # ...only once the paused one had its slot back
    assert (paused.resume_queue_at_pause >= 1).all() and (paused.resume_queue_at_resume >= 0).all()
    assert (paused.slots_at_resume <= CAP).all()


def test_resumed_turn_is_released_at_the_later_of_its_closed_loop_time_and_the_resume(monkeypatch, tmp_path):
    q, _ = _run_synthetic(monkeypatch, tmp_path)
    g = q.groupby("session_id")
    q = q.assign(due=g.t_end.shift() + q.t_planned - g.t_planned.shift())
    later = q[q.turn > 0]
    assert ((later.t_released - later.due) > -1e-9).all()
    p = later[later.t_paused.notna()]
    assert ((p.t_released - p.t_resumed) > -1e-9).all()
    assert ((p.t_released - p[["due", "t_resumed"]].max(axis=1)).abs() < 1e-9).all()
    assert ((p.pause_s - (p.t_resumed - p.t_paused)).abs() < 1e-9).all()
    unpaused = later[later.t_paused.isna()]
    assert ((unpaused.t_released - unpaused.due).abs() < 1e-9).all() and (unpaused.pause_s == 0).all()
    assert (later.t_sent >= later.t_released - 1e-9).all()
    # pause_s_cum adds up the session's pauses, turn by turn
    assert ((q.groupby("session_id").pause_s.cumsum() - q.pause_s_cum).abs() < 1e-9).all()


def test_every_owed_turn_is_released_and_nothing_is_lost(monkeypatch, tmp_path):
    q, log = _run_synthetic(monkeypatch, tmp_path)
    sched = _synthetic()
    assert len(q) == len(sched) and (q.status == "ok").all()
    assert sorted(q.request_id) == sorted(sched.request_id)
    assert (q.prompt_tokens_planned.to_numpy() == sched.sort_values(["session_id", "turn"]).prompt_len.to_numpy()).all()
    assert log[-1][1:4] == (0, 0, 0)                                  # every slot given back, both queues empty


def test_admission_arm_on_the_same_schedule_never_pauses(monkeypatch, tmp_path):
    q, log = _run_synthetic(monkeypatch, tmp_path, mode="admission")
    assert "t_paused" not in q and not any(what == "pause" for *_, what in log)
    assert len(q) == len(_synthetic()) and (q.status == "ok").all()


# ---------------------------------------------------------------- SessionCap's resume queue alone

def test_resume_queue_before_admission_queue_and_fifo():
    async def go():
        cap, got = lg.SessionCap(1, 10.0, "pause"), []
        for i in range(3):
            await cap.acquire(float(i))                    # 3 in service before the switch
        assert cap.should_pause(10.0) and not cap.should_pause(9.9)
        a = cap.pause()                                    # 3 -> 2
        b = cap.pause()                                    # 2 -> 1
        assert cap.used == 1 and not cap.should_pause(11.0)

        async def new():
            await cap.acquire(12.0)
            got.append("new")

        async def res(name, fut):
            got.append((name, await cap.resume(fut)))

        tasks = [asyncio.create_task(new()), asyncio.create_task(res("a", a)), asyncio.create_task(res("b", b))]
        await asyncio.sleep(0)
        assert got == []                                   # nothing freed yet: nobody moves
        for _ in range(3):
            cap.release()
            await asyncio.sleep(0)
        await asyncio.gather(*tasks)
        cap.release()
        return got, cap.used, len(cap.resumers), len(cap.waiters)

    got, used, r, w = asyncio.run(go())
    assert got == [("a", (1, 1)), ("b", (1, 0)), "new"] and (used, r, w) == (0, 0, 0)


def test_cancelled_resume_leaves_the_queue_and_holds_no_slot():
    async def go():
        cap = lg.SessionCap(1, 0.0, "pause")
        cap.used = 2                                       # an overhang of one
        fut = cap.pause()
        t = asyncio.create_task(cap.resume(fut))
        await asyncio.sleep(0)
        t.cancel()
        await asyncio.gather(t, return_exceptions=True)
        cap.release()
        return cap.used, len(cap.resumers)

    assert asyncio.run(go()) == (0, 0)


def test_admission_mode_never_pauses():
    cap = lg.SessionCap(1, 0.0)
    cap.used = 5
    assert not cap.should_pause(100.0)
    with pytest.raises(AssertionError):
        asyncio.run(_pause(cap))


async def _pause(cap):
    cap.pause()


def test_a_chain_cancelled_while_paused_frees_nothing_and_keeps_the_pause(monkeypatch):
    """The drain deadline while paused: the next turn stays an unfinished placeholder that shows the pause."""
    monkeypatch.setattr(lg, "prompt_tokens", lambda *a: [1])
    from tests.test_session_cap import _fake_send, _rows
    monkeypatch.setattr(lg, "send", _fake_send([], dur=0.02))

    async def go():
        import time
        cap, inflight = lg.SessionCap(1, 0.0, "pause"), {}
        cap.used = 1                                       # another session holds the one slot
        cap.from_s = 1e9                                   # admit session 0 at once, as before a switch...
        t0 = time.monotonic()
        a = asyncio.create_task(lg.session_chain(_rows(0, n=3, gap=0.01), None, None, "", "", None, t0,
                                                 io.StringIO(), inflight, cap))
        await asyncio.sleep(0)
        cap.from_s = 0.0                                   # ...then the switch passes while its first turn runs
        await asyncio.sleep(0.2)
        assert not a.done() and len(cap.resumers) == 1 and cap.used == 1
        a.cancel()
        await asyncio.gather(a, return_exceptions=True)
        return cap, inflight

    cap, inflight = asyncio.run(go())
    assert cap.used == 1 and len(cap.resumers) == 0        # only the other session's slot is left
    (rec,) = [r for r in inflight.values() if r["turn"] == 1]
    assert rec["t_paused"] is not None and rec["t_resumed"] is None and rec["pause_s"] is None
    assert rec["slots_at_pause"] == 1 and rec["resume_queue_at_pause"] == 1 and rec["t_sent"] is None


# ---------------------------------------------------------------- 4. bad values are refused before the server starts

@pytest.mark.parametrize("bad", ["Pause", "", "stop", "admission ", 1, True, ["pause"]])
def test_cap_mode_must_be_admission_or_pause(bad):
    with pytest.raises(ValueError):
        lg.session_cap({"workload": {"sessions": {"closed_loop": True, "cap": 3, "cap_mode": bad}}})
    with pytest.raises(ValueError):
        lg.SessionCap(3, 0.0, bad)


@pytest.mark.parametrize("mode", ["pause", "admission"])
def test_cap_mode_without_a_cap_is_refused(mode):
    with pytest.raises(ValueError):
        lg.session_cap({"workload": {"sessions": {"closed_loop": True, "cap_mode": mode}}})


@pytest.mark.parametrize("sessions", [{"cap": 2, "cap_mode": "halt"}, {"cap_mode": "pause"}])
def test_bad_cap_mode_is_refused_before_the_server_starts(tmp_path, sessions):
    with pytest.raises(subprocess.CalledProcessError):
        run_fake(tmp_path, workload={**WL, "sessions": {**WL["sessions"], **sessions}})
    assert not any((tmp_path / "runs").glob("*")) if (tmp_path / "runs").exists() else True


# ---------------------------------------------------------------- 5. end to end against fake_vllm

def test_paused_run_against_the_fake_server_is_valid_and_releases_every_owed_turn(tmp_path):
    run_dir = run_fake(tmp_path, workload={**WL, "sessions": {**WL["sessions"], "cap": 3, "cap_from_s": 14,
                                                               "cap_mode": "pause"}},
                       trigger=BURST, drain_s=60)
    chk = json.loads((run_dir / "checks.json").read_text())
    assert chk["valid"], chk["failed"]
    q = pd.read_json(run_dir / "requests.jsonl", lines=True)
    sched = pd.read_parquet(run_dir / "schedule.parquet")
    assert len(q) == len(sched) and (q.status == "ok").all()          # every owed turn released and served
    sess = q[~q.request_id.isin(set(sched.request_id[sched.burst]))]
    paused = sess[sess.t_paused.notna()]
    assert len(paused) >= 1 and (paused.t_paused >= 14).all() and paused.t_resumed.notna().all()
    assert ((sess.t_released - sess.t_planned) > -1e-6).all()
    # 2. the schedule written before the run is the one the admission (and the keyless) config draws
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    for mode in ("admission", None):
        c = _merge(cfg, {"workload": {"sessions": {"cap_mode": mode}}})
        s, _ = workload.build_schedule(c["workload"], c["seed"], c.get("trigger"))
        pd.testing.assert_frame_equal(s, sched)
