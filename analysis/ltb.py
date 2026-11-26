"""Post-hoc load-test readouts (author 2026-11-26, the no-run route after the LTB review of the workshop paper;
decisions.md 2026-11-26 [DECISION] (author) "Post-hoc load-test readouts ..." defines each one before any number).

NOT pre-specified. Every run below had been seen and judged by its own registered test, which stays the result.
Nothing here judges a run: these are readouts reported beside the registered counts, for the cells of
`uniform.CELLS` (same runs, same `k1._runs` choice of the one valid run per seed, same windows `k1.windows`, a
control read at its burst partner's absolute windows). Windows: `base` [burst start - 600 s, burst start) and `end`
(the last `verdict.end_s` s of the watch).

Readouts, per run:
  latency   two origins, each with its own population:
            `scheduled`: requests (session and burst) PLANNED in the window, TTFT = t_first_token - t_planned. Under
                         closed or held release t_planned is the synthetic open-loop schedule, so this is how far
                         behind the session's schedule the first token came;
            `release`:   requests RELEASED in the window (t_released, or t_planned where the rule has no separate
                         release: open loop), TTFT = t_first_token - release time: what a generator that measures from
                         its own release reports.
            A request with no first token by the run's end (unfinished, never released, errored) has TTFT = inf, so
            it is a miss in every percentile (docs/definitions.md, percent fast: unfinished requests are misses).
            Percentiles p50, p95, p99 by nearest rank (`nearest_rank`), so an inf is reported, never interpolated.
            goodput: requests of the population that finished ok with TTFT <= slo.ttft_s (1.0 s), per second of the
            window; `fast_share` the same count over the population.
  send_lag  t_sent - (t_released, else t_planned) over requests sent with that due time in the span: the load
            generator's own lateness (the harness's `client_lag` check bounds its p99 at 0.05 s in every valid run).
            Spans: the measured span [warmup_s, duration_s), the burst [at, at + duration), and the two windows.
  backlog   `planned_backlog(t)`: session requests (burst excluded) planned before t (t_planned < t) and not finished
            by t (no t_end < t). It uses the schedule only, which is the same for every release rule on one seed, so
            it does not depend on the rule. Split exactly into `at_client` (planned, not yet sent by t: closed-loop
            turns not yet released, held turns, turns waiting at a session cap) and `at_server` (sent, not finished),
            because a request is never sent before its planned time. Reported beside the server's own gauges at the
            nearest scrape (waiting, running) and `burst_at_server` (burst requests sent and not finished), and beside
            `gate.owed_backlog` (the same with burst requests included). At the burst's end, at the end window's start
            and at its end.
  preempt   increase of vllm:num_preemptions_total over a span (interpolated between scrapes; None if the counter goes
            down), and the requests completed ok in it: whole run (first to last scrape), base, burst + watch, end.
Figure 3's accounting (`fig3_accounting`): the turns of `rewrite_figs.reuse_by_x`'s three arms in every `m.ages`
bucket, `held` and `neg_A` included, and for `neg_A` (a >= 0 but a negative allocation count A, counter timing) the
idle-age quantiles and the pooled reuse.

Usage: uv run --extra analysis python -m analysis.ltb   (prints JSON); `... ltb paper <summary JSON>` prints the papers'
latency table (`paper_table`).
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd

from analysis import k1, uniform
from analysis.gate import owed_backlog
from analysis.margin import is_burst_request

PCTS = (50, 95, 99)
PREEMPT = "vllm:num_preemptions_total"


def nearest_rank(values, p: float) -> float:
    """The p-th percentile by nearest rank (the ceil(p/100 * n)-th smallest); inf values allowed; NaN if empty."""
    v = np.sort(np.asarray(values, dtype=float))
    if not len(v):
        return math.nan
    return float(v[max(0, math.ceil(p / 100.0 * len(v)) - 1)])


def release_time(q: pd.DataFrame) -> pd.Series:
    """When the generator released each request. A rule that records releases (closed, held, cap: `t_released` set
    on some rows) keeps NaN for a turn never released, which then belongs to no release window. A rule that records
    none (open loop) releases at the planned time. Same due time as the harness's client_lag check where both exist."""
    if "t_released" in q and q.t_released.notna().any():
        return q.t_released.astype(float)
    return q.t_planned.astype(float)


def _lat(q: pd.DataFrame, origin: pd.Series, a: float, b: float, slo: float) -> dict:
    ttft = (q.t_first_token.astype(float) - origin).fillna(math.inf)
    fast = int(((q.status == "ok") & (ttft <= slo)).sum())
    out = {"n": int(len(q)), "no_first_token": int(q.t_first_token.isna().sum()),
           "goodput": fast / (b - a), "fast_share": fast / len(q) if len(q) else math.nan}
    out.update({f"p{p}": nearest_rank(ttft, p) for p in PCTS})
    return out


def latency(run: dict, a: float, b: float) -> dict:
    """Both origins over [a, b), each with its own population (module docstring)."""
    q, slo = run["requests"], float(run["cfg"]["slo"]["ttft_s"])
    planned = q[(q.t_planned >= a) & (q.t_planned < b)]
    rel = release_time(q)
    released = q[(rel >= a) & (rel < b)]
    return {"scheduled": _lat(planned, planned.t_planned.astype(float), a, b, slo),
            "release": _lat(released, rel[released.index], a, b, slo)}


def send_lag(run: dict, a: float, b: float) -> dict:
    q = run["requests"]
    due = release_time(q)
    s = q[q.t_sent.notna() & (due >= a) & (due < b)]
    lag = (s.t_sent.astype(float) - due[s.index]).to_numpy(float)
    if not len(lag):
        return {"n": 0}
    return {"n": int(len(lag)), "p50": nearest_rank(lag, 50), "p99": nearest_rank(lag, 99),
            "p999": nearest_rank(lag, 99.9), "max": float(lag.max()), "min": float(lag.min()),
            "over_0.05s": int((lag > 0.05).sum()), "over_1s": int((lag > 1.0).sum())}


def planned_backlog(run: dict, t: float) -> dict:
    """The rule-independent backlog at t and its exact split; the server's gauges at the nearest scrape."""
    q = run["requests"]
    burst = is_burst_request(run)
    s = q[~burst]
    pending = (s.t_planned < t) & ~(s.t_end < t)
    sent = s.t_sent < t
    bq = q[burst]
    out = {"t": float(t), "planned_backlog": int(pending.sum()), "at_client": int((pending & ~sent).sum()),
           "at_server": int((pending & sent).sum()),
           "burst_at_server": int(((bq.t_sent < t) & ~(bq.t_end < t)).sum()),
           "owed_backlog": owed_backlog(run, t)}
    m = run["metrics"]
    if len(m) and "vllm:num_requests_waiting" in m:
        i = int((m.t - t).abs().idxmin())
        out["server"] = {"t_scrape": float(m.t[i]), "waiting": float(m["vllm:num_requests_waiting"][i]),
                         "running": float(m["vllm:num_requests_running"][i])}
    return out


def backlog_mean(run: dict, a: float, b: float, step: float = 10.0) -> float:
    """Mean of planned_backlog over a grid of [a, b)."""
    q = run["requests"]
    s = q[~is_burst_request(run)]
    tp, te = s.t_planned.to_numpy(float), s.t_end.to_numpy(float)
    grid = np.arange(a, b, step)
    return float(np.mean([((tp < t) & ~(te < t)).sum() for t in grid])) if len(grid) else math.nan


def preemptions(run: dict, a: float, b: float) -> dict:
    m = run["metrics"]
    q = run["requests"]
    done = int(((q.status == "ok") & (q.t_end >= a) & (q.t_end < b)).sum())
    if not len(m) or PREEMPT not in m:
        return {"preemptions": None, "completed": done, "reason": "no counter"}
    c = m[["t", PREEMPT]].dropna()
    if np.any(np.diff(c[PREEMPT].to_numpy(float)) < 0):
        return {"preemptions": None, "completed": done, "reason": "counter went down"}
    t, v = c.t.to_numpy(float), c[PREEMPT].to_numpy(float)
    n = float(np.interp(b, t, v) - np.interp(a, t, v))
    return {"preemptions": n, "completed": done, "per_completion": n / done if done else math.nan}


def one(run: dict, win: dict, burst_span: tuple[float, float]) -> dict:
    sp = {"measured": (float(run["cfg"]["workload"]["warmup_s"]), float(run["cfg"]["workload"]["duration_s"])),
          "base": win["base"], "burst": burst_span, "end": win["end"]}
    m = run["metrics"]
    whole = (float(m.t.min()), float(m.t.max())) if len(m) else (math.nan, math.nan)
    e = burst_span[1]
    return {"run": run["dir"].name,
            "latency": {w: latency(run, *sp[w]) for w in ("base", "end")},
            "send_lag": {w: send_lag(run, *sp[w]) for w in sp},
            "backlog": {"burst_end": planned_backlog(run, e), "end_start": planned_backlog(run, win["end"][0]),
                        "end_end": planned_backlog(run, win["end"][1]),
                        "end_mean": backlog_mean(run, *win["end"])},
            "preempt": {"whole": preemptions(run, *whole), "base": preemptions(run, *win["base"]),
                        "burst_watch": preemptions(run, burst_span[0], win["end"][1]),
                        "end": preemptions(run, *win["end"])}}


def cell(name: str, burst_pat: str, control_pat: str | None, seeds: tuple[int, ...]) -> dict:
    rows = []
    for sd in seeds:
        b, nb = k1._runs(burst_pat.format(s=sd))
        if b is None:
            rows.append({"seed": sd, "burst": None, "invalid_burst_runs": nb})
            continue
        win, t = k1.windows(b["cfg"]), b["cfg"]["trigger"]
        bs = (float(t["at_s"]), float(t["at_s"] + t["duration_s"]))
        row = {"seed": sd, "burst": one(b, win, bs)}
        if control_pat:
            c, _ = k1._runs(control_pat.format(s=sd))
            row["control"] = one(c, win, bs) if c is not None else None
        rows.append(row)
    return {"cell": name, "rows": rows}


def table() -> list[dict]:
    return [cell(*c) for c in uniform.CELLS]


def fig3_accounting() -> dict:
    """Every turn of Figure 3's three arms by m.ages bucket, and what the neg_A turns are."""
    from analysis import b1, m
    from analysis.load import find_runs
    arms = {"closed": m.M["arms"]["closed"], "open": m.M["arms"]["open"], "cap40": b1.ARMS["cold"]}
    out = {}
    for arm, pat in arms.items():
        qs = [q for q in (m.ages(r) for r in find_runs(pat)) if q is not None]
        q = pd.concat(qs)
        neg = q[q.bucket == "neg_A"]
        den = float(neg.prev_prompt.sum())
        out[arm] = {"runs": len(qs), "total": int(len(q)),
                    "buckets": {bk: int((q.bucket == bk).sum()) for bk in m.BUCKETS},
                    "neg_A": {"n": int(len(neg)),
                              "a_s": {f"p{p}": nearest_rank(neg.a, p) for p in (0, 50, 90, 100)} if len(neg) else {},
                              "reuse": float(neg.r_num.sum()) / den if den > 0 else math.nan,
                              "per_run": {r: int(n) for r, n in neg.groupby("run").size().items()}}}
    return out


# ---------------------------------------------------------------- SIGMETRICS review (author 2026-11-26, scope extension)
# E1: does an open-loop follow-up turn overlap its predecessor only after the burst, or in the healthy state too?
# E3: how many sessions were in service in the open-loop arm at its peak? Defined in decisions.md 2026-11-26
# [DECISION] (author) "Two more post-hoc readouts ..." before computed. Post hoc; nothing judged.
OVERLAP = {"span_s": 300.0,
           "cells": ("open loop, λ 2.0", "open loop, λ 2.5", "open loop, λ 2.73", "open loop, later stack")}


def overlap_profile(run: dict, at: float, burst_s: float, watch_s: float) -> list[dict]:
    """Follow-up turns (b1.turns: ok later turns of a session, previous turn ok, burst excluded) by the span holding
    their first token: the baseline [at - 600, at), the burst [at, at + burst_s), then 300 s spans to the watch's end.
    `held` is m's bucket (m.bucket: a = t_first_token - previous turn's end < 0, the prefix still in use). Per span:
    turns, held turns and their share, cached tokens of held turns over all cached tokens, and pooled reuse
    (min(cached, previous prompt) / previous prompt, token-weighted, as k1.reuse) of held and of other turns."""
    from analysis import b1, m
    q = b1.turns(run).copy()
    a = (q.t_first_token - q.prev_end).to_numpy(float)
    q["held"] = m.bucket(a, np.full(len(a), np.nan)) == "held"
    e = at + burst_s
    edges = [(at - 600.0, at, "base"), (at, e, "burst")]
    edges += [(x, x + OVERLAP["span_s"], f"+{x - e:g}") for x in np.arange(e, e + watch_s, OVERLAP["span_s"])]
    out = []
    for lo, hi, name in edges:
        s = q[(q.t_first_token >= lo) & (q.t_first_token < hi)]
        h, o = s[s.held], s[~s.held]
        reuse = lambda d: float(d.r_num.sum() / d.prev_prompt.sum()) if len(d) and d.prev_prompt.sum() > 0 else math.nan
        cached = float(s.cached_tokens.sum())
        out.append({"span": name, "a": float(lo), "b": float(hi), "turns": int(len(s)), "held": int(len(h)),
                    "held_share": len(h) / len(s) if len(s) else math.nan,
                    "held_cached_share": float(h.cached_tokens.sum()) / cached if cached > 0 else math.nan,
                    "reuse_held": reuse(h), "reuse_other": reuse(o)})
    return out


def overlap_table() -> list[dict]:
    """overlap_profile for every burst run and control of the open-loop cells of uniform.CELLS; a control is read on
    its burst partner's spans."""
    from analysis import d1
    out = []
    for name, bp, cp, seeds in uniform.CELLS:
        if name not in OVERLAP["cells"]:
            continue
        rows = []
        for sd in seeds:
            b, _ = k1._runs(bp.format(s=sd))
            if b is None:
                continue
            t = b["cfg"]["trigger"]
            at, bs, w = float(t["at_s"]), float(t["duration_s"]), float(b["cfg"]["verdict"]["watch_s"])
            row = {"seed": sd, "burst": {"run": b["dir"].name, "spans": overlap_profile(b, at, bs, w),
                                         "in_service_peak": d1.peak(b, at)}}
            if cp:
                c, _ = k1._runs(cp.format(s=sd))
                if c is not None:
                    row["control"] = {"run": c["dir"].name, "spans": overlap_profile(c, at, bs, w),
                                      "in_service_peak": d1.peak(c, at)}
            rows.append(row)
        out.append({"cell": name, "rows": rows})
    return out


def _rng(vals: list) -> list | None:
    v = [x for x in vals if x is not None and not (isinstance(x, float) and math.isnan(x))]
    return [min(v), max(v)] if v else None


def summary(res: dict) -> dict:
    """Ranges (min, max over a cell's seeds) of the readouts the papers quote, from `table()`'s output (the saved
    probe JSON): per cell and role (burst, control). Presentation of the same numbers; nothing new is computed."""
    out = {}
    for c in res["cells"]:
        roles = {}
        for who in ("burst", "control"):
            rs = [r[who] for r in c["rows"] if r.get(who)]
            if not rs:
                continue
            get = lambda f: _rng([f(r) for r in rs])
            lat = lambda w, o, k: get(lambda r: r["latency"][w][o][k])   # w, o, k bound at each call
            roles[who] = {
                "n": len(rs),
                "ttft_release": {w: {k: lat(w, "release", k) for k in ("p50", "p95", "p99", "goodput")}
                                 for w in ("base", "end")},
                "ttft_scheduled": {w: {k: lat(w, "scheduled", k) for k in ("p50", "p95", "p99", "goodput")}
                                   for w in ("base", "end")},
                "end_no_first_token_share": get(lambda r: r["latency"]["end"]["scheduled"]["no_first_token"]
                                                / r["latency"]["end"]["scheduled"]["n"]),
                "send_lag_p99_max": max(max(r["send_lag"][w].get("p99", 0) for w in r["send_lag"]) for r in rs),
                "send_lag_max": max(max(r["send_lag"][w].get("max", 0) for w in r["send_lag"]) for r in rs),
                "send_lag_over_1s": sum(r["send_lag"]["measured"].get("over_1s", 0) for r in rs),
                "backlog": {t: {k: get(lambda r: r["backlog"][t][k])
                                for k in ("planned_backlog", "at_client", "at_server", "burst_at_server")}
                            for t in ("burst_end", "end_end")},
                "server_end": {k: get(lambda r: r["backlog"]["end_end"].get("server", {}).get(k))
                               for k in ("waiting", "running")},
                "backlog_end_mean": get(lambda r: r["backlog"]["end_mean"]),
                "preempt": {w: get(lambda r, w=w: r["preempt"][w]["preemptions"]) for w in rs[0]["preempt"]},
                "preempt_end_per_completion": get(lambda r: r["preempt"]["end"].get("per_completion")),
                "preempt_whole_per_completion": get(lambda r: r["preempt"]["whole"].get("per_completion")),
            }
        out[c["cell"]] = roles
    return out


# The papers' latency table (author 2026-11-26, the no-run route): which cells and roles it shows, in order, with the
# label printed. Presentation of `summary`'s ranges only; nothing is computed here.
PAPER_ROWS = (
    ("Closed loop", "closed loop, λ 2.0 (main runs)", "burst"),
    ("Closed loop, controls", "closed loop, λ 2.0 (main runs)", "control"),
    ("Closed loop, 4500 s window", "closed loop, long window", "burst"),
    ("Closed loop, 4500 s window, controls", "closed loop, long window", "control"),
    ("Held", "held, λ 2.0", "burst"),
    ("Held, controls", "held, λ 2.0", "control"),
    ("Open loop", "open loop, λ 2.0", "burst"),
    ("Open loop, λ = 2.73", "open loop, λ 2.73", "burst"),
    ("Open loop, λ = 2.73, controls", "open loop, λ 2.73", "control"),
    ("Cap at the control's peak", "closed loop, cap at the control's peak", "burst"),
    ("Cap of 40", "closed loop, cap of 40", "burst"),
)


def fmt_num(x: float) -> str:
    """Two decimals below 1, one below 10, none from 10; inf as such."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "—"
    if math.isinf(x):
        return "∞"
    if x == 0:
        return "0"
    return f"{x:.2f}" if x < 1 else f"{x:.1f}" if x < 10 else f"{x:.0f}"


def fmt_rng(r: list | None) -> str:
    """A [min, max] range as 'a–b', or one value when both print alike."""
    if not r:
        return "—"
    a, b = fmt_num(r[0]), fmt_num(r[1])
    return a if a == b else f"{a}–{b}"


def paper_table(summ: dict) -> list[str]:
    """Markdown rows of the papers' latency table from `summary()`'s output: release-origin TTFT (s) in the baseline
    (p50 / p99) and the end window (p50 / p95 / p99), and the end window's goodput (req/s), ranges over seeds."""
    rows = ["| Arm (burst runs unless stated) | Runs | Baseline p50 / p99 | End window p50 / p95 / p99 | "
            "End goodput |", "|---|---|---|---|---|"]
    for label, c, who in PAPER_ROWS:
        r = summ[c][who]
        b, e = r["ttft_release"]["base"], r["ttft_release"]["end"]
        rows.append(f"| {label} | {r['n']} | {fmt_rng(b['p50'])} / {fmt_rng(b['p99'])} | {fmt_rng(e['p50'])} / "
                    f"{fmt_rng(e['p95'])} / {fmt_rng(e['p99'])} | {fmt_rng(e['goodput'])} |")
    return rows


def main(argv: list[str]) -> int:
    what = argv[1] if len(argv) > 1 else "all"
    if what == "paper":   # python -m analysis.ltb paper <saved summary JSON>
        print("\n".join(paper_table(json.loads(open(argv[2]).read()))))
        return 0
    if what == "summary":   # python -m analysis.ltb summary <saved table JSON>
        print(json.dumps(summary(json.loads(open(argv[2]).read())), indent=1, default=str))
        return 0
    res = {}
    if what in ("all", "table"):
        res["cells"] = table()
    if what == "overlap":
        res["overlap"] = overlap_table()
    if what in ("all", "fig3"):
        res["fig3"] = fig3_accounting()
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
