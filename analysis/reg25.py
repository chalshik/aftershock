"""REG-25's registered computation, run once (decisions.md 2026-11-14 [REG] REG-25: a sustained cache wipe at L1,
λ = 2.0 (seed 44), with its same-seed control; approved the same day).

It gathers the registration's own readouts, each from the function the registration names, and adds only two things:
  - `a21`, the readout the registration lists as "reported, never judged" (A21; its O5 and round 2's O2), which had no
    function: the rise of `vllm:num_preemptions_total` over [first call's t_start, last call's t_end + the order test's
    `wipe_preempt_pad_s`), against Σ `vllm:num_requests_running` at the last scrape before each call, and the rise
    over the 10 s before the first call (the natural rate);
  - `outcome`, the registration's table "What each outcome decides", as a lookup.

    AFTERSHOCK_UNSEAL=REG-25 python -m analysis.reg25 table

Readouts, all reported: each run's `load.verdict`; the wipe run's `swipe_design.span_readout` (`span_c` sets how row 1
is read, at `swipe_design.SPAN["min_span_c"]`) and `swipe_design.queue_after_last_call`; `order.table`'s line for the
wipe run. Statistics (rule H): one seed per arm, labelled as such.
"""
from __future__ import annotations

import json
import math
import sys

REG25 = {
    "control": "20261115-141357_stage2_control_l1pl350cl_r2.0_s44",   # decisions.md 2026-11-15 [OPS] REG-25: the queue ended
    "wipe": "20261115-193119_swipe_l1pl350cl_r2.0_s44",
    "q_300": 67.0,          # the [REG]'s feasibility, Q_300 (decisions.md 2026-11-14 [RESULT], the sustained-wipe gate)
    "q_15": 45.3,           # 1.5 Q*, same source
    "natural_s": 10.0,      # the [REG]: "the rise over the 10 s before the first call is given as the natural rate"
}
PREEMPT = "vllm:num_preemptions_total"
RUNNING = "vllm:num_requests_running"


def _last_at_or_before(m, col: str, t: float) -> float:
    s = m[m.t <= t][col].dropna() if col in m else m.iloc[0:0]
    return float(s.iloc[-1]) if len(s) else math.nan


def a21(run: dict) -> dict:
    """A21 (reported, never judged): does a preempt reset count in the preemption counter? NaN where the counter or
    the gauge is missing, or the counter went down in the window (a restart, flagged)."""
    from analysis.order import ORDER
    trig = run["trigger"]
    ok = trig[trig.success] if len(trig) else trig
    if not len(ok):
        raise ValueError("no successful wipe call")
    m = run["metrics"]
    a, b = float(ok.iloc[0].t_start), float(ok.iloc[-1].t_end) + ORDER["wipe_preempt_pad_s"]
    nan = {"a": a, "b": b, "rise": math.nan, "running_sum": math.nan, "calls": int(len(ok)),
           "natural_rise": math.nan, "counter_down": None}
    if not len(m) or PREEMPT not in m:
        return nan
    s = m[(m.t >= a - REG25["natural_s"]) & (m.t < b)][PREEMPT].dropna()
    down = bool((s.diff().dropna() < 0).any())
    lo, hi = _last_at_or_before(m, PREEMPT, a), _last_at_or_before(m, PREEMPT, b)
    pre = _last_at_or_before(m, PREEMPT, a - REG25["natural_s"])
    running = [_last_at_or_before(m, RUNNING, float(t)) for t in ok.t_start]
    rsum = float(sum(running)) if running and not any(math.isnan(x) for x in running) else math.nan
    if down:
        return {**nan, "counter_down": True}
    return {"a": a, "b": b, "rise": hi - lo, "running_sum": rsum, "calls": int(len(ok)),
            "natural_rise": lo - pre, "counter_down": False}


COURSE = {"bin_s": 30.0}   # exploratory, added after the [REG]'s numbers were seen; it judges nothing


def preempt_course(run: dict) -> list[dict]:
    """Exploratory (not registered; added 2026-11-15 after the table was computed, and it changes no label): the rise
    of the preemption counter in bins of COURSE["bin_s"] from the order test's pad end (last call's t_end +
    `wipe_preempt_pad_s`) to the watch end. A wipe's preemptions are booked at each request's next output
    (open-questions A21), so ones booked after the pad bunch at its start; pressure preemptions spread over the watch."""
    from analysis.order import ORDER, windows
    w = windows(run)
    m = run["metrics"]
    t0, t1 = w["attempt"][1] + ORDER["wipe_preempt_pad_s"], w["watch_end"]
    out, a = [], t0
    while a < t1 - 1e-9:
        b = min(a + COURSE["bin_s"], t1)
        out.append({"a": a, "b": b, "rise": _last_at_or_before(m, PREEMPT, b) - _last_at_or_before(m, PREEMPT, a)})
        a = b
    return out


def outcome(wipe: str, control: str, span_c: float, owed_rise: float) -> dict:
    """The [REG]'s table "What each outcome decides", by the two runs' verdicts ('invalid' for an invalid run)."""
    from analysis.swipe_design import SPAN
    healthy = ("recovered", "slow_drain")
    if wipe == "invalid":
        return {"row": "invalid", "reading": None}
    if control in ("unclear", "bad_baseline") or wipe in ("unclear", "bad_baseline"):
        return {"row": "no_reading_listed", "reading": None}
    if wipe == "stuck" and control == "recovered":
        lab = "reprefill" if span_c < SPAN["min_span_c"] else "cache_loss"
        return {"row": "stuck_control_recovered", "reading": lab}
    if control in ("stuck", "slow_drain"):
        return {"row": "control_not_healthy", "reading": None}
    if wipe in healthy and control == "recovered":
        return {"row": "wipe_did_not_hold", "reading": "backlog_reached_1.5Q" if owed_rise >= REG25["q_15"]
                else "backlog_short_of_1.5Q"}
    raise ValueError(f"no row for wipe {wipe!r}, control {control!r}")


def table() -> dict:
    """Both runs, loaded through `load.load_run` (the seal guard applies: run with AFTERSHOCK_UNSEAL=REG-25)."""
    from analysis import load as L
    from analysis import order, swipe_design as SD
    out, verdicts = {}, {}
    for k in ("control", "wipe"):
        run = L.load_run(L.RUNS / REG25[k])
        valid = bool(run.get("checks") and run["checks"].get("valid"))
        v = L.verdict(run) if valid else {"verdict": "invalid"}
        verdicts[k] = v["verdict"]
        row = {"run": REG25[k], "valid": valid, "verdict": v}
        if k == "wipe" and valid:
            row.update(span=SD.span_readout(run), queue=SD.queue_after_last_call(run), a21=a21(run))
        out[k] = row
    span = out["wipe"].get("span", {})
    out["order"] = order.table([REG25["wipe"]])[0]
    out["outcome"] = outcome(verdicts["wipe"], verdicts["control"], span.get("span_c", math.nan),
                             span.get("owed_rise", math.nan))
    out["reference"] = {"q_300": REG25["q_300"], "q_15": REG25["q_15"], "min_span_c": SD.SPAN["min_span_c"]}
    return out


def _default(o):
    try:
        import numpy as np
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    return str(o)


if __name__ == "__main__":
    if sys.argv[1:] == ["table"]:
        print(json.dumps(table(), indent=1, default=_default))
    elif sys.argv[1:] == ["course"]:
        from analysis import load as L
        print(json.dumps(preempt_course(L.load_run(L.RUNS / REG25["wipe"])), indent=1, default=_default))
    else:
        sys.exit("usage: python -m analysis.reg25 table|course")
