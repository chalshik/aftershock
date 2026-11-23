"""Load generator. Open loop (default): sends every request at its planned time, never waits for earlier ones.
Closed-loop turns (`workload.sessions.closed_loop: true`, REG-11 draft): session starts and burst requests stay open
loop; a session's later turn is released only after its previous turn has ended, plus the schedule's own gap between
the two turns. `t_planned` stays the schedule's time; `t_released` records when the turn became due (None in open loop).
Session cap (`workload.sessions.cap`, closed loop only; step D, author 2026-11-12): at most `cap` sessions are in service
at once (from turn 0 sent to the last turn's end). A session that arrives at a full cap waits, first come first served;
its turn 0 records `t_arrived` and `t_admitted`, and its `t_released` is the admission time. Burst requests pass the cap.
Cap mode (`workload.sessions.cap_mode`, T1 draft): "admission" (or absent) is the cap above, unchanged. "pause" adds a
resume queue: from `cap_from_s` on, a session whose turn ends while more than `cap` sessions hold a slot gives its slot
up until a slot is handed back to it (see SessionCap). Its later turns stay owed; only their release times move.

Usage (started by run.py): python -m harness.loadgen <run_dir> <t0_monotonic>
Writes one line per request to requests.jsonl. No timeouts, no retries.
"""
import asyncio
import json
import math
import sys
import time
from collections import deque
from pathlib import Path

import aiohttp
import pandas as pd
import yaml

from harness.workload import prompt_tokens


class Gateway:
    """Optional concurrency cap in front of the server. Off in Stage 1 (config gateway.enabled)."""

    def __init__(self, cfg: dict):
        self.sem = asyncio.Semaphore(cfg["max_concurrency"]) if cfg["enabled"] else None

    async def __aenter__(self):
        if self.sem:
            await self.sem.acquire()

    async def __aexit__(self, *exc):
        if self.sem:
            self.sem.release()


class SessionCap:
    """At most `n` sessions in service at once, held at the load generator (the gateway stays off). Waiters are served
    strictly first come first served from an explicit queue; a freed slot is handed straight to the oldest waiter.

    `from_s` (D1r, author 2026-11-17; default 0, today's behaviour): a session that arrives before `from_s` (seconds
    since t0) is admitted at once and counted, so `used` may exceed `n`. From `from_s` on, a session is admitted only
    while fewer than `n` are in service. Sessions above the cap are never stopped: in service falls to `n` only as they
    end, and a freed slot is handed over only once `used` is at or below `n` (the overhang drains, it is not cut).

    `mode` (`workload.sessions.cap_mode`, T1 draft; default "admission", everything above, unchanged). With "pause"
    the overhang is not left to drain: from `from_s` on, a session whose turn ends while `used` > `n` gives its slot up
    (`pause`, called by session_chain at a turn boundary only) and joins a resume queue, first come first served by
    pause time. It is not ended. A freed slot goes first to the head of the resume queue and only when that is empty
    to the head of the admission queue (`waiters`). Before `from_s` nothing pauses, so both modes are the same there.
    A pause happens only with `used` > `n`, so it never hands a slot over itself; `used` falls to `n` within one turn
    of each overhang session, and a resumed session is handed a slot exactly as an admitted one is."""

    MODES = ("admission", "pause")

    def __init__(self, n: int, from_s: float = 0.0, mode: str = "admission"):
        if not isinstance(n, int) or isinstance(n, bool) or n < 1:
            raise ValueError(f"workload.sessions.cap must be a positive integer, got {n!r}")
        if isinstance(from_s, bool) or not isinstance(from_s, (int, float)) or not math.isfinite(from_s) or from_s < 0:
            raise ValueError(f"workload.sessions.cap_from_s must be a finite number >= 0, got {from_s!r}")
        if not isinstance(mode, str) or mode not in self.MODES:
            raise ValueError(f"workload.sessions.cap_mode must be one of {self.MODES}, got {mode!r}")
        self.n, self.from_s, self.used, self.waiters = n, float(from_s), 0, deque()
        self.mode, self.resumers = mode, deque()  # resumers: paused sessions, empty in admission mode

    async def acquire(self, now: float | None = None):
        """`now`: seconds since t0 at the session's arrival. None (or `from_s` 0) means the cap applies."""
        if now is not None and now < self.from_s:
            self.used += 1  # before the switch: admitted at once, counted toward the overhang
            return
        if self.used < self.n and not self.waiters and not self.resumers:
            self.used += 1
            return
        fut = asyncio.get_running_loop().create_future()
        self.waiters.append(fut)
        try:
            await fut  # resolved by release(): the slot passes over without `used` going down
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()  # handed a slot in the same tick as the cancel: give it on
            elif fut in self.waiters:
                self.waiters.remove(fut)
            raise

    def release(self):
        """A slot frees. The future of the session it is handed to gets (`used`, resume queue length) at the handover;
        acquire() ignores it, session_chain records it for a resume."""
        if self.used <= self.n:  # above the cap (an overhang from before `from_s`) the slot is not handed over
            for queue in (self.resumers, self.waiters):  # paused sessions first (empty in admission mode)
                while queue:
                    fut = queue.popleft()
                    if not fut.done():
                        fut.set_result((self.used, len(self.resumers)))
                        return
        self.used -= 1

    def should_pause(self, now: float) -> bool:
        """At a turn boundary `now` seconds after t0: pause mode, the switch has passed, and more than `n` hold slots."""
        return self.mode == "pause" and now >= self.from_s and self.used > self.n

    def pause(self) -> asyncio.Future:
        """Give the caller's slot up and join the resume queue. Returns the future resume() waits on."""
        assert self.mode == "pause" and self.used > self.n, (self.mode, self.used, self.n)
        self.release()  # used > n: no handover, used falls by one
        fut = asyncio.get_running_loop().create_future()
        self.resumers.append(fut)
        return fut

    async def resume(self, fut: asyncio.Future):
        """Wait until a slot is handed back; returns (`used`, resume queue length) at the handover."""
        try:
            return await fut
        except asyncio.CancelledError:
            if fut.done() and not fut.cancelled():
                self.release()  # handed a slot in the same tick as the cancel: give it on
            elif fut in self.resumers:
                self.resumers.remove(fut)
            raise


def session_cap(cfg: dict) -> SessionCap | None:
    """The configured cap, or None. Read with .get (added after defaults.yaml was used; absent = no cap). Refused
    with open-loop turns: an open-loop turn has no session in service to hold a slot. `cap_mode` (T1 draft, read with
    .get; absent or null = "admission") must be "admission" or "pause", and is refused without a `cap`."""
    sessions = (cfg["workload"].get("sessions") or {})
    n, mode = sessions.get("cap"), sessions.get("cap_mode")
    if n is None:
        if mode is not None:
            raise ValueError(f"workload.sessions.cap_mode ({mode!r}) needs workload.sessions.cap")
        return None
    if not sessions.get("closed_loop"):
        raise ValueError("workload.sessions.cap needs closed-loop turns (workload.sessions.closed_loop: true)")
    return SessionCap(n, sessions.get("cap_from_s", 0.0),  # added 2026-11-17 (D1r), so read with .get; absent = 0
                      "admission" if mode is None else mode)


PAUSE_FIELDS = ("t_paused", "t_resumed", "slots_at_pause", "resume_queue_at_pause", "slots_at_resume",
                "resume_queue_at_resume")  # request fields of a pause-mode chain (session_chain)


def new_rec(row, t_released=None, **extra) -> dict:
    return dict(request_id=int(row.request_id), session_id=int(row.session_id), turn=int(row.turn),
               t_planned=float(row.t_planned), t_released=t_released, prompt_tokens_planned=int(row.prompt_len),
               output_tokens_planned=int(row.output_len), t_sent=None, t_connected=None, t_first_token=None, t_end=None,
               prompt_tokens=None, output_tokens=None, cached_tokens=None, status="unfinished", error=None, **extra)


async def send(session, url, model, row, prompt, gateway, t0, out, inflight, t_released=None, extra=None) -> dict:
    rec = new_rec(row, t_released, **(extra or {}))
    inflight[rec["request_id"]] = rec
    body = {"model": model, "prompt": prompt, "stream": True,
            "max_tokens": rec["output_tokens_planned"], "min_tokens": rec["output_tokens_planned"],
            "ignore_eos": True, "temperature": 0.0,
            "stream_options": {"include_usage": True}}
    try:
        async with gateway:
            rec["t_sent"] = time.monotonic() - t0
            # t_connected: when a new connection for this request was open (None if a pooled one was reused), so the
            # connect time inside TTFT is measured (decisions.md 2026-11-09, amendment review).
            async with session.post(url, json=body, trace_request_ctx={"rec": rec, "t0": t0}) as resp:
                if resp.status != 200:
                    rec["status"], rec["error"] = "http_error", f"{resp.status}: {(await resp.text())[:300]}"
                    return rec  # session_chain needs the record (REG-11 review, T1)
                async for raw in resp.content:
                    line = raw.decode().strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    # Any choice chunk counts: a token's text can decode to "" (partial UTF-8).
                    if rec["t_first_token"] is None and chunk.get("choices"):
                        rec["t_first_token"] = time.monotonic() - t0
                    if usage := chunk.get("usage"):
                        rec["prompt_tokens"] = usage.get("prompt_tokens")
                        rec["output_tokens"] = usage.get("completion_tokens")
                        rec["cached_tokens"] = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
            rec["t_end"] = time.monotonic() - t0
            rec["status"] = "ok"
    except asyncio.CancelledError:
        raise  # stays "unfinished"; written by main()
    except Exception as e:  # recorded, never retried
        rec["status"], rec["error"] = "client_error", f"{type(e).__name__}: {e}"[:300]
    finally:
        if rec["status"] != "unfinished":
            out.write(json.dumps(rec) + "\n")
            out.flush()
            inflight.pop(rec["request_id"], None)
    return rec


async def session_chain(rows, prompts, session, url, model, gateway, t0, out, inflight, cap=None):
    """Closed-loop turns of one session. Turn 0 is released at its planned time; turn k at the end of turn k-1 (or the
    time it failed) plus the planned gap t_planned[k] - t_planned[k-1]. Turns not yet sent stay in `inflight` as
    unfinished placeholders, so a turn never released before the drain deadline is still recorded (as a miss).
    With a session cap the session first waits for a slot (it is called at turn 0's planned time); turn 0 is released
    at admission, and the slot is held until the chain ends (last turn ended, failed, or cancelled at the deadline).
    In pause mode (SessionCap) the chain checks, after each turn but the last has ended, whether to pause; a paused
    chain waits in the resume queue without a slot, and its next turn is released at max(end of the turn + planned
    gap, resume time). Every record of a pause-mode chain carries the PAUSE_FIELDS (None / 0.0 when not paused): on
    the turn after a pause `t_paused`, `slots_at_pause` (`used` after the slot was given up) and
    `resume_queue_at_pause` (queue length with this session in it), and, once resumed, `t_resumed`, `slots_at_resume`
    and `resume_queue_at_resume` (both at the handover, this session holding its slot and out of the queue);
    `pause_s` (this turn's pause; None while still paused) and `pause_s_cum` (the session's pauses up to this turn).
    The next turn's unfinished placeholder gets the pause fields at the pause and the resume fields at the resume,
    before the release sleep, so a turn cut off at the deadline shows how far its session got."""
    pausing = cap is not None and cap.mode == "pause"
    blank = {**dict.fromkeys(PAUSE_FIELDS), "pause_s": 0.0, "pause_s_cum": 0.0} if pausing else {}
    for row in rows:
        inflight[int(row.request_id)] = new_rec(row, **blank)
    held, extra = False, None
    paused_cum, pend = 0.0, None  # pend: the pause before the next turn, if any
    try:
        release, prev = float(rows[0].t_planned), None
        if cap is not None:
            arrived = time.monotonic() - t0
            inflight[int(rows[0].request_id)]["t_arrived"] = arrived  # kept if the deadline comes first
            await cap.acquire(arrived)
            held = True
            admitted = time.monotonic() - t0
            release, extra = max(release, admitted), {"t_arrived": arrived, "t_admitted": admitted}
        for i, row in enumerate(rows):
            prompt = prompt_tokens(prompts, int(row.session_id), int(row.prompt_len))
            if prev is not None:
                if pend is not None:  # the turn's end was taken before the pause
                    done = pend.pop("done")
                else:
                    done = prev["t_end"] if prev["t_end"] is not None else time.monotonic() - t0
                release = done + float(row.t_planned) - prev["t_planned"]
                if pend is not None:
                    release = max(release, pend["t_resumed"])
            delay = t0 + release - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            if pausing:
                kw = {"extra": {**(extra if prev is None else {}), **blank, **(pend or {}), "pause_s_cum": paused_cum}}
            else:
                kw = {"extra": extra} if extra is not None and prev is None else {}  # turn 0 under a cap only
            prev = await send(session, url, model, row, prompt, gateway, t0, out, inflight, t_released=release, **kw)
            pend = None
            if pausing and i + 1 < len(rows):
                now = time.monotonic() - t0
                if cap.should_pause(now):
                    # a turn boundary by construction: send() has returned, so nothing of this session is in flight
                    assert prev["status"] != "unfinished" and int(row.request_id) not in inflight, "pause mid-turn"
                    done = prev["t_end"] if prev["t_end"] is not None else now
                    held = False
                    fut = cap.pause()
                    pend = {"t_paused": now, "slots_at_pause": cap.used, "resume_queue_at_pause": len(cap.resumers)}
                    # kept on the next turn's placeholder if the deadline comes first
                    inflight[int(rows[i + 1].request_id)].update(pend, pause_s=None, pause_s_cum=paused_cum)
                    slots, queue = await cap.resume(fut)
                    held = True
                    resumed = time.monotonic() - t0
                    pend.update(t_resumed=resumed, slots_at_resume=slots, resume_queue_at_resume=queue,
                                pause_s=resumed - now, done=done)
                    paused_cum += resumed - now
                    # the resume goes onto the placeholder too, before the release sleep: a deadline between the
                    # resume and the send then leaves a record that shows the resume (T1 draft 3, review R6)
                    inflight[int(rows[i + 1].request_id)].update(
                        {k: v for k, v in pend.items() if k != "done"}, pause_s_cum=paused_cum)
    except asyncio.CancelledError:
        raise  # the drain deadline: unsent turns stay unfinished placeholders
    except Exception as e:  # a chain that dies must fail no_errors, not look like a slow server (REG-11 review, T2)
        for row in rows:
            rec = inflight.get(int(row.request_id))
            if rec is not None and rec["t_sent"] is None:
                rec["status"], rec["error"] = "client_error", f"session_chain: {type(e).__name__}: {e}"[:300]
                out.write(json.dumps(rec) + "\n")
                out.flush()  # one line at a time, so a reader never sees a buffer cut mid-line (A14)
                inflight.pop(int(row.request_id), None)
    finally:
        if held:
            cap.release()


async def _connected(session, ctx, params):
    req = ctx.trace_request_ctx or {}
    if "rec" in req:
        req["rec"]["t_connected"] = time.monotonic() - req["t0"]


def make_trace() -> aiohttp.TraceConfig:
    tc = aiohttp.TraceConfig()
    tc.on_connection_create_end.append(_connected)
    return tc


def make_connector(cfg: dict) -> aiohttp.TCPConnector:
    """Unlimited pool: the client must not cap concurrency. With `loadgen_force_close` (added after defaults.yaml, so
    read with .get; absent = the earlier behaviour) every request opens its own connection and none is reused. Reusing
    an idle one raced vLLM's 5 s keep-alive close against aiohttp's 15 s pool timeout, and failed 4 requests in 25,450
    at send time (decisions.md 2026-11-09). No retry either way (rule 2)."""
    return aiohttp.TCPConnector(limit=0, force_close=bool(cfg.get("loadgen_force_close")))


async def main(run_dir: Path, t0: float):
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    sched = pd.read_parquet(run_dir / "schedule.parquet")
    prompts = pd.read_parquet(run_dir / "prompts.parquet")
    srv = cfg["server"]
    url = f"http://{srv['host']}:{srv['port']}/v1/completions"
    gateway = Gateway(cfg["gateway"])
    cap = session_cap(cfg)
    deadline = t0 + (float(sched.t_planned.max()) if len(sched) else 0.0) + cfg["drain_s"]
    inflight, tasks = {}, []
    closed = bool((cfg["workload"].get("sessions") or {}).get("closed_loop"))  # absent = open loop (older configs)
    chains = {}
    if closed:
        for row in sched[~sched.burst].sort_values(["session_id", "turn"]).itertuples(index=False):
            chains.setdefault(int(row.session_id), []).append(row)

    connector = make_connector(cfg)
    timeout = aiohttp.ClientTimeout(total=None, sock_read=None, sock_connect=None)
    with open(run_dir / "requests.jsonl", "a") as out:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout, trace_configs=[make_trace()]) as session:
            for row in sched.itertuples(index=False):
                if closed and not row.burst:
                    if int(row.turn) == 0:  # the session starts open loop; its chain releases the later turns
                        delay = t0 + row.t_planned - time.monotonic()
                        if delay > 0:
                            await asyncio.sleep(delay)
                        tasks.append(asyncio.create_task(session_chain(
                            chains[int(row.session_id)], prompts, session, url, cfg["model"], gateway, t0, out,
                            inflight, cap)))
                    continue
                prompt = prompt_tokens(prompts, int(row.session_id), int(row.prompt_len))  # built before the wait
                delay = t0 + row.t_planned - time.monotonic()
                if delay > 0:
                    await asyncio.sleep(delay)
                tasks.append(asyncio.create_task(
                    send(session, url, cfg["model"], row, prompt, gateway, t0, out, inflight)))
            pending = [t for t in tasks if not t.done()]
            if pending:
                await asyncio.wait(pending, timeout=max(0.0, deadline - time.monotonic()))
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
        for rec in inflight.values():  # still running at the end: counted as misses
            out.write(json.dumps(rec) + "\n")
            out.flush()  # one line at a time (A14)


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), float(sys.argv[2])))
