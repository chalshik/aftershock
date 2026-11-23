"""Condition 4's order clause, judged literally, on stuck runs made after its registration only (draft until
registered). Never writes into runs/.

PROTOCOL §3 condition 4 ends "and the traces follow the plan's order". The author ruled that this clause is judged
literally, on new runs only, by a test registered before them, on new burst runs and on one new cache-wipe run
(decisions.md 2026-11-14 [DECISION] (author) entries on condition 4's order clause and on the order test; and the
author's answers to rule F round 1's R3, R4 and R9, recorded with the registration). The plan
(docs/experiment-plan.md, "Watch it happen"), with the trigger at t = 0:

  1. hit rate drops; 2. recomputed prefill per second rises; 3. running batch N grows;
  4. KV memory fills; cached prefixes are evicted sooner (shorter idle time before eviction);
  5. hit rate stays low.  Preemptions stay low throughout.

The origin is the trigger's start: a burst's `trigger.at_s`; a wipe's first successful call in trigger.jsonl
(`t_start`), with the watch from the last successful call's `t_end` (one call for a single wipe; amended 2026-11-14
for a sustained wipe, before any run was judged), as `load.verdict` takes them. The preemption exclusion covers
the whole span, from the first call's start to the last call's end + wipe_preempt_pad_s.
Onsets use REG-22's rules (`c2.reuse_in`, `c2.session_uncached_rate`, `c2.gauge`, `c2.C2`): the first point of a step_s
grid from the origin at which the trailing roll_s window meets its level:
  reuse of later turns <= baseline - stuck_hit_drop (at least min_turns turns)   item 1
  session uncached prompt tokens/s >= u_factor x baseline                       item 2
  mean N >= n_factor x baseline N                                               item 3
  mean KV >= min_kv, condition 4's own KV level (not REG-22's kv_onset; R9)     item 4
An onset is resolved only if every grid window from the origin up to it could be read (no gauge gap above
gauge_max_gap_s, at least min_turns turns for reuse); otherwise it is `unresolved` and the run is `undefined` (R7).
Items 1-4 hold unless a signal the plan puts later crosses more than tol_s before one it puts earlier (every pair,
not only neighbours, so tolerances do not add up). tol_s = roll_s + step_s is the onset rule's resolution for steps: a
trailing roll_s mean of a step crosses its level within roll_s of the step, if at all, on a step_s grid, so two
signals that step at one instant have onsets less than tol_s apart, in either order. For ramps an onset is when the
mean reaches the level, which is the literal reading. The test can contradict the plan's order; it cannot tell it
from steps less than tol_s apart.
Item 4, "evicted sooner": the mean idle time before eviction (histogram _sum / _count deltas) is shorter in the end
window than in the baseline; judged when both windows sampled at least min_evictions evictions, otherwise reported as
not measurable, which does not make the run undefined (R4).
Item 5 and "then stay there" (rule F round 2, S1 (a)): over the end window, reuse <= baseline - stuck_hit_drop
(`stays_low`), mean N >= n_factor x baseline N and mean KV >= min_kv (`stays_high`), condition 4's own levels.
Preemptions: the counter's increase over [origin, watch end), leaving out a wipe's own attempt from its t_start to
wipe_preempt_pad_s after its t_end (R3), <= max_preempt_frac x the requests that ended ok in that window.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np

from analysis import c2
from analysis.k1 import later_turns
from analysis.load import RUNS, load_run, verdict

ORDER = {
    "registered_after": "20261114-102959",   # UTC run-id stamp YYYYMMDD-HHMM59 of the registration's minute (R8)
    "max_preempt_frac": 0.01,                # preemptions per request ended ok, origin to watch end (BELIEF of "low")
    "wipe_preempt_pad_s": 10.0,              # a wipe's own preemptions: its t_start to this long after its t_end
    "min_evictions": 20,                     # sampled evictions each window needs for "evicted sooner" to be judged
    "tol_s": 70.0,                           # c2.C2 roll_s + step_s: the onset rule's resolution (docstring)
    "plan": ("reuse", "u", "n", "kv"),       # docs/experiment-plan.md, "Watch it happen", items 1 to 4
}
PREEMPT = "vllm:num_preemptions_total"
IDLE = c2.HIST["idle"]


def windows(run: dict) -> dict:
    """Origin, trigger end, baseline, watch end, end window; for a wipe also its attempt (t_start, t_end)."""
    t, v = run["cfg"]["trigger"], run["cfg"]["verdict"]
    if t["kind"] == "burst":
        at = float(t["at_s"]); end = at + float(t["duration_s"]); attempt = None
    elif t["kind"] == "cache_wipe":
        trig = run["trigger"]
        ok = trig[trig.success] if len(trig) else trig
        if not len(ok):
            raise ValueError("a cache wipe without a successful attempt in trigger.jsonl: no origin")
        at, end = float(ok.iloc[0].t_start), float(ok.iloc[-1].t_end)   # a sustained wipe: first call to last (amendment)
        attempt = (at, end)
    else:
        raise ValueError(f"trigger kind {t['kind']!r}: the order test is registered for burst and cache_wipe only")
    return {"at": at, "end": end, "attempt": attempt, "base": (at - v["baseline_s"], at),
            "watch_end": end + v["watch_s"], "end_window": (end + v["watch_s"] - v["end_s"], end + v["watch_s"])}


def is_new(name: str) -> bool:
    """A run made after the registration: its id's UTC stamp (harness/run.py:136) is later than registered_after."""
    return bool(re.fullmatch(r"\d{8}-\d{6}", name[:15])) and name[:15] > ORDER["registered_after"]


def _values(run: dict, turns, a: float, b: float) -> dict:
    return {"reuse": c2.reuse_in(turns, a, b, c2.C2["min_turns"]), "u": c2.session_uncached_rate(run, a, b),
            "n": c2.gauge(run, c2.N, a, b), "kv": c2.gauge(run, c2.KV, a, b)}


def _met(k: str, x: float, base: dict, drop: float) -> bool:
    C = c2.C2
    return {"reuse": lambda: x <= base["reuse"] - drop, "u": lambda: x >= C["u_factor"] * base["u"],
            "n": lambda: x >= C["n_factor"] * base["n"], "kv": lambda: x >= C["min_kv"]}[k]()   # NaN gives False


def onsets(run: dict, base: dict, w: dict) -> dict:
    """First grid time from the origin at which each signal meets its level (NaN if never in the watch); `unresolved`
    names the signals with an unreadable window before their onset; `pre` those already at their level over the
    roll_s before the origin, whose onset says nothing."""
    C, drop, turns = c2.C2, float(run["cfg"]["verdict"]["stuck_hit_drop"]), later_turns(run)
    v0 = _values(run, turns, w["at"] - C["roll_s"], w["at"])
    pre = sorted(k for k in ORDER["plan"] if _met(k, v0[k], base, drop))
    out, gap = {k: math.nan for k in ORDER["plan"]}, set()
    for t in np.arange(w["at"] + C["step_s"], w["watch_end"] + 1e-9, C["step_s"]):
        vals = _values(run, turns, t - C["roll_s"], t)
        for k in ORDER["plan"]:
            if not math.isnan(out[k]):
                continue
            if math.isnan(vals[k]):
                gap.add(k)
            elif _met(k, vals[k], base, drop):
                out[k] = float(t)
        if not any(math.isnan(x) for x in out.values()):
            break
    return {**out, "pre": pre, "unresolved": sorted(gap)}


def _counter_at(m, col: str, x: float) -> float:
    s = m[m.t <= x][col].dropna()
    return float(s.iloc[-1]) if len(s) else math.nan


def preemptions(run: dict, w: dict) -> dict:
    """Counter increase over [origin, watch end), less a wipe's own attempt (R3); NaN if the counter is missing or
    went down (a restart, flagged, not hidden). `ended_ok` counts the requests that ended ok in the window."""
    a, b = w["at"], w["watch_end"]
    q, m = run["requests"], run["metrics"]
    done = int(((q.status == "ok") & (q.t_end >= a) & (q.t_end < b)).sum())
    nan = {"count": math.nan, "ended_ok": done, "excluded": math.nan}
    if not len(m) or PREEMPT not in m:
        return nan
    s = m[(m.t >= a) & (m.t < b)][PREEMPT].dropna()
    if len(s) < 2 or (s.diff().dropna() < 0).any():
        return nan
    total, excl = float(s.iloc[-1] - s.iloc[0]), 0.0
    if w["attempt"] is not None:
        ws, we = w["attempt"][0], w["attempt"][1] + ORDER["wipe_preempt_pad_s"]
        lo = _counter_at(m, PREEMPT, ws) if ws > a else float(s.iloc[0])
        hi = _counter_at(m, PREEMPT, min(we, float(m[m.t < b].t.max())))
        excl = hi - lo if not (math.isnan(hi) or math.isnan(lo)) else math.nan   # O3: missing is not "none"
        if math.isnan(excl):
            return {**nan, "excluded": excl}
        excl = max(0.0, excl)
    return {"count": total - excl, "ended_ok": done, "excluded": excl}


def _hist_sum_count(path, name: str, a: float, b: float) -> tuple[float, float]:
    """Increase of a histogram's _sum and _count over [a, b), summed over label sets, from the raw metrics.jsonl;
    NaN, NaN if missing or if either went down (a restart)."""
    first = last = None
    down = False
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if "values" not in s or s["t"] < a:
                continue
            if s["t"] >= b:
                break
            vals = s["values"]
            cur = (sum(v for k, v in vals.items() if k == name + "_sum" or k.startswith(name + "_sum{")),
                   sum(v for k, v in vals.items() if k == name + "_count" or k.startswith(name + "_count{")))
            if first is None:
                first = cur
            elif cur[0] < last[0] or cur[1] < last[1]:
                down = True
            last = cur
    if first is None or down:
        return math.nan, math.nan
    return last[0] - first[0], last[1] - first[1]


def evictions(run: dict, w: dict) -> dict:
    """Mean sampled idle time before eviction in the baseline and the end window; `sooner` is True/False when both
    windows hold at least min_evictions sampled evictions, else None (not measurable)."""
    path = run["dir"] / "metrics.jsonl"
    out = {}
    for label, (a, b) in (("base", w["base"]), ("end", w["end_window"])):
        s, n = _hist_sum_count(path, IDLE, a, b) if path.exists() else (math.nan, math.nan)
        out[label] = {"n": n, "mean_s": s / n if (not math.isnan(n)) and n > 0 else math.nan}
    ok = all((not math.isnan(out[k]["n"])) and out[k]["n"] >= ORDER["min_evictions"] for k in ("base", "end"))
    return {**out, "sooner": bool(out["end"]["mean_s"] < out["base"]["mean_s"]) if ok else None}


def judge(run: dict) -> dict:
    """One run's order clause: 'follows' (items 1-5, evictions sooner where measurable, low preemptions),
    'does_not_follow' (`failed` names the items), 'undefined' (an onset never reached or unresolved, a signal at its
    level before the origin, or a value missing). Refuses a run made before the registration."""
    name = run["dir"].name
    if not is_new(name):
        raise ValueError(f"{name}: made before the order test's registration ({ORDER['registered_after']}); not judged")
    w = windows(run)
    base = c2.baseline(run, w["base"])
    on = onsets(run, base, w)
    drop = float(run["cfg"]["verdict"]["stuck_hit_drop"])
    stay = c2.reuse_in(later_turns(run), *w["end_window"], c2.C2["min_turns"])
    n_end, kv_end = c2.gauge(run, c2.N, *w["end_window"]), c2.gauge(run, c2.KV, *w["end_window"])
    pe, ev = preemptions(run, w), evictions(run, w)
    out = {"run": name, "onsets": on, "end_reuse": stay, "end_n": n_end, "end_kv": kv_end, "preempt": pe, "evictions": ev}
    times = [on[k] for k in ORDER["plan"]]
    missing = any(math.isnan(x) for x in (*times, stay, n_end, kv_end, pe["count"], base["reuse"], base["u"], base["n"]))
    if on["pre"] or on["unresolved"] or missing or pe["ended_ok"] == 0:
        return {**out, "label": "undefined", "stays_low": None, "failed": None}
    P = ORDER["plan"]
    failed = [f"{x}>{y}" for i, x in enumerate(P) for y in P[i + 1:] if on[x] > on[y] + ORDER["tol_s"]]
    if ev["sooner"] is False:
        failed.append("evictions")
    stays = stay <= base["reuse"] - drop
    if not stays:
        failed.append("stays_low")
    if n_end < c2.C2["n_factor"] * base["n"] or kv_end < c2.C2["min_kv"]:
        failed.append("stays_high")
    if pe["count"] > ORDER["max_preempt_frac"] * pe["ended_ok"]:
        failed.append("preemptions")
    u_only = bool(failed) and all(f.startswith("u>") for f in failed)   # O1: only u crossing late
    return {**out, "label": "does_not_follow" if failed else "follows", "stays_low": stays, "failed": failed,
            "u_only": u_only}   # every failure is u crossing late: possibly held first tokens (R6), reported as such


def table(names: list[str]) -> list[dict]:
    """The order clause on the named runs, loaded from runs/ (load_run applies the seal guard). Only a valid run whose
    own verdict is `stuck` is judged; an invalid run, or one with another verdict, is listed as such (R1)."""
    out = []
    for n in names:
        r = load_run(RUNS / n)
        if not (r.get("checks") and r["checks"].get("valid")):
            out.append({"run": n, "label": "invalid", "failed": None})
            continue
        v = verdict(r)["verdict"]
        out.append(judge(r) if v == "stuck" else {"run": n, "label": f"not_judged ({v})", "failed": None})
    return out
