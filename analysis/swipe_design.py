"""The sustained wipe's interval (REG-25 draft 1, docs/drafts/swipe_reg_draft1.md, "Wipe interval: the rule, fixed
before the estimate"; rule committed in b188fc2 before this code was written). Never writes into runs/.

A turn's cached prefix can survive a wipe series with calls every E s only if no call falls between the previous
turn's end and its own send, L = t_sent(k) - t_end(k - 1); the true exposure also holds the queue wait, so this is
the shortest. With calls at a phase independent of the turns, that chance is max(0, 1 - L/E). The hit-rate bound is

    H(E) = sum over turns sent in [a, b) of cached_tokens x max(0, 1 - L/E)  /  sum of their prompt_tokens

on a no-trigger control, where nothing is wiped. A wipe only adds recomputed prompt tokens to the denominator.
The rule: E = the largest of SD["every_s"] with H(E) <= SD["limit"] in every control; else None (told to the author).

Correction, 2026-11-14 (decisions.md [CORR], REG-25 rule F round 1, R5): H is not an upper bound. (1) Readmissions of
preempted requests are not in the exported hit/query counters (docs/open-questions.md S8), so a preempt wipe adds
nothing to the check's denominator. (2) In preempt mode turn ends are tied to the calls, not at an independent phase;
ends 0-0.6 s after a call give 0.27-0.33 at E = 2.5 s (PROBE(docs/drafts/swipe_r1/phase_lock.txt)). H is the
expected hit rate at a uniform phase with no queue wait; queue wait lowers the real rate. The numbers computed stand.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from analysis.load import RUNS, load_run
from analysis.wipe_gate import WG

SD = {
    "controls": WG["controls"],     # REG-19's five λ = 2.0 controls at L1 (also rule G's)
    "a": 900.0, "b": 1200.0,        # the span the wipes would cover (at_s 900, span_s 300)
    "every_s": (5.0, 2.5),          # the mean turn gap; about one turn's service time (BELIEF)
    "limit": 0.05,                  # half of wipe_max_hit 0.10
}


def hit_bound(reqs: pd.DataFrame, every_s: float, a: float, b: float) -> dict:
    r = reqs[reqs.status == "ok"].sort_values(["session_id", "turn"])
    prev_end = r.groupby("session_id").t_end.shift(1)
    prev_turn = r.groupby("session_id").turn.shift(1)
    inwin = (r.t_sent >= a) & (r.t_sent < b)
    gap = (r.t_sent - prev_end).where(prev_turn == r.turn - 1)       # a missing previous turn: no carried prefix
    if (gap[inwin].dropna() < 0).any():
        raise ValueError("a turn sent before its previous turn ended: not a closed-loop run")
    survive = np.clip(1.0 - gap / every_s, 0.0, 1.0).fillna(0.0)
    den = float(r.prompt_tokens[inwin].sum())
    num = float((r.cached_tokens * survive)[inwin].sum())
    return {"H": num / den if den else float("nan"), "turns": int(inwin.sum()),
            "cached_share": float(r.cached_tokens[inwin].sum()) / den if den else float("nan")}


def choose(rows: list[dict]) -> float | None:
    """The largest E whose bound is within the limit in every control (NaN fails)."""
    for e in sorted(SD["every_s"], reverse=True):
        hs = [x["H"] for x in rows if x["every_s"] == e]
        if hs and all(np.isfinite(h) and h <= SD["limit"] for h in hs):
            return e
    return None


def table() -> dict:
    rows = []
    for name in SD["controls"]:
        run = load_run(RUNS / name)
        if not (run.get("checks") and run["checks"].get("valid")):
            raise ValueError(f"{name}: not valid")
        if (run["cfg"].get("trigger") or {}).get("kind"):
            raise ValueError(f"{name}: not a no-trigger control")
        for e in SD["every_s"]:
            rows.append({"run": name, "every_s": e, **hit_bound(run["requests"], e, SD["a"], SD["b"])})
    return {"rows": rows, "E": choose(rows), "limit": SD["limit"]}


QUEUE = {"gauge": "vllm:num_requests_waiting", "window_s": 10.0}   # REG-25 draft 3, rule F round 1 O1


def queue_after_last_call(run: dict) -> dict:
    """REG-25's readout (reported, never judged): the mean of the waiting-queue gauge over the scrapes in
    [t_end of the last successful call, + QUEUE["window_s"]), against 1.5 Q* (`wipe_gate`). The gauge also counts
    preempted requests waiting to be admitted again (docs/decisions.md, the `preempted_*` logging run)."""
    trig = run["trigger"]
    ok = trig[trig.success] if len(trig) else trig
    if not len(ok):
        raise ValueError("no successful wipe call")
    t = float(ok.iloc[-1].t_end)
    m = run["metrics"]
    w = m[(m.t >= t) & (m.t < t + QUEUE["window_s"])] if len(m) and QUEUE["gauge"] in m else m.iloc[0:0]
    q = float(w[QUEUE["gauge"]].mean()) if len(w) else float("nan")
    return {"t_last_end": t, "queue": q, "scrapes": int(len(w))}


SPAN = {"min_span_c": 1.5}   # REG-25 rule F round 2 S1: the floor 1.681 less ≈ 2σ of a 300 s count (DERIVED there)


def span_readout(run: dict) -> dict:
    """REG-25's span readouts (reported; `span_c` sets how row 1 is read, rule F round 2 S1-S2):
      span_c     = `gate.completion_rate` over [first successful call's t_start, last successful call's t_end);
      owed_rise  = `gate.owed_backlog` at the last call's end minus at the first call's start: the backlog G2 predicts
                   (server waiting and running, plus closed-loop turns not yet released), unlike the waiting gauge."""
    from analysis.gate import completion_rate, owed_backlog
    trig = run["trigger"]
    ok = trig[trig.success] if len(trig) else trig
    if not len(ok):
        raise ValueError("no successful wipe call")
    a, b = float(ok.iloc[0].t_start), float(ok.iloc[-1].t_end)
    c = completion_rate(run, a, b)
    return {"a": a, "b": b, "span_c": c, "reprefill_reading": bool(c < SPAN["min_span_c"]),
            "owed_rise": owed_backlog(run, b) - owed_backlog(run, a)}
