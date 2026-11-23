"""T2 (registered: the [REG] T2 (cache-off burst control) entry in docs/decisions.md): with caching off, does REG-19's
burst leave a slow state at a matched load?

Three seeds (15, 16, 17), each a burst run and its no-burst twin on the same arrivals: REG-19's configs with caching
off, at D2's matched load lambda = 0.80 req/s (decisions.md 2026-11-14 [DECISION] (author) D2 is matched on the open-loop
mu_off = 1.0), 2580 s (900 s before the burst, its 180 s, the protocol's 1500 s watch). The cache-on verdict cannot
judge these runs: the prefix-cache query counter stays at 0, so the hit rate is NaN (decisions.md 2026-11-14
[DECISION] (author) C2's caching-off control …). T2 judges the server's line and its completions against the twin's.
Burst requests are one-off turn-0 requests (harness/workload.py:6), excluded from c and p (`margin.is_burst_request`).

Windows (`windows`): the baseline B = the 600 s before the burst; the end window E = the watch's last `end_s` s;
E's halves E1, E2. Per window: q = mean `vllm:num_requests_waiting` (`load._level`), c = session completions ok with
t_end in the window per s, p = session requests planned in the window per s. With closed-loop turns p follows the
server (a later turn is planned from the previous turn's end), so c is judged against the twin's c (rule F r1, T2-2).

Per pair (`label_pair`; names distinct from REG-19's `recovered`/`stuck`, rule F r1 T2-8d):
- a run's baseline is healthy iff q_B <= `base_q_max` and c_B >= `keep_frac` x p_B; else `bad_baseline`;
- the twin is `drained` iff q_E <= q_B + `q_excess` and c_E >= `keep_frac` x p_E; else `twin_not_drained`;
- the burst run, with lined = q_E > q_B + `q_excess`, short = c_E < `keep_frac` x the twin's c_E,
  dt = `gate.drain_time` (s after the burst's end):
  `drained` (not lined, not short); then, for a lined run with both halves finite (else `unclear`): `draining`
  (q_E2 < q_E1: still falling, at any rate); `slow` (short, q_E2 >= q_E1; whether the line emptied once after the
  burst is a flag, `emptied_before_E`, not a label; rule F r2 T2-R2-1); `relapse` (not short, the line emptied
  before E and came back; T2-4); `unclear` otherwise.
Outcome (`outcome`): a pair counts when both baselines are healthy and the twin is `drained`. `too_few` (n < 2) ->
`slow_found` (any counted burst `slow`) -> `drains` (every counted burst `drained`) -> `mixed`.
`validate` (rule F r1, T2-1): the same labels on REG-19's five burst-twin pairs (cache on, lambda = 2.0, seen and
judged stuck there by the hit rate), with a pass condition registered before it is computed.
Readouts, never judged: the drain time, the session backlog owed at the burst's end (`gate.shock_size`), sessions
started and not finished by the end (burst vs twin), q per 300 s span after the burst, the pool's peak in the burst,
completions in the first 300 s after it, percent fast at 1.0 s and the median TTFT per window, burst minus twin
(rule F r1, T2-3: a latency-only slow state is not judged).
"""
from __future__ import annotations

import math

import numpy as np

from analysis import withheld

T2 = {
    "seeds": (15, 16, 17),
    "lam": 0.8,
    "duration_s": 2580.0,
    "base_q_max": 1.0,     # a healthy baseline: at most one request waiting on average
    "q_excess": 1.0,       # REG-19's `verdict.recovered_q_excess` (checked against each run's config in `table`)
    "keep_frac": 0.95,     # REG-8's sat_frac / the capacity rule's min_throughput_frac
    "min_pairs": 2,
    "order": ("control_15", "burst_15", "burst_16", "control_16", "control_17", "burst_17"),
    "seal": "[RESULT] T2 (cache-off burst control)",
    "validate_pass": {"slow_min": 4, "twin_drained_min": 4},   # of REG-19's 5 pairs (rule F r1, T2-1)
    "reg19": {"burst": "burst_l1pl350cl_r2.0_w3000_s{}", "control": "stage2_control_l1pl350cl_r2.0_w3000_s{}",
              "seeds": (15, 16, 17, 18, 19)},
}
PREDICTED = {  # BELIEF, registered (ledger row 41); outcome by hand in the [REG], 0.97 validity per run assumed
    "per_burst": {"drained": 0.93, "unclear": 0.03, "draining": 0.02, "relapse": 0.005, "slow": 0.015},
    "per_twin": {"drained": 0.95},
    "outcome": {"drains": 0.79, "mixed": 0.135, "slow_found": 0.045, "too_few": 0.03},   # rule F r2, T2-R2-6
}


def config_name(kind: str, seed: int) -> str:
    return f"t2_{kind}_l1pl350cl_off_r0.8_s{seed}"


def windows(cfg: dict) -> dict:
    """B, E, E1, E2 from a burst run's config (trigger start, duration, verdict.end_s); a twin is read on its burst
    run's windows (`pair_row`), since the twin has no trigger."""
    at = float(cfg["trigger"]["at_s"])
    d, e = float(cfg["workload"]["duration_s"]), float(cfg["verdict"]["end_s"])
    return {"B": (at - 600.0, at), "E": (d - e, d), "E1": (d - e, d - e / 2), "E2": (d - e / 2, d)}


def window(run: dict, a: float, b: float) -> dict:
    """q, c and p over [a, b) (module docstring), plus percent fast at the configs' 1.0 s, median TTFT, KV, running."""
    from analysis.load import _level
    from analysis.margin import is_burst_request
    lv = _level(run, a, b)
    q = run["requests"]
    s = q[~is_burst_request(run)]
    c = float(((s.status == "ok") & (s.t_end >= a) & (s.t_end < b)).sum() / (b - a))
    p = float(((s.t_planned >= a) & (s.t_planned < b)).sum() / (b - a))
    planned = s[(s.t_planned >= a) & (s.t_planned < b)]
    ttft = float(planned.ttft.median()) if len(planned) and "ttft" in planned else math.nan
    return {"q": lv["q"], "c": c, "p": p, "pf": lv["pf"], "ttft_median": ttft, "kv": lv["kv"], "n": lv["n"]}


def _fin(*v) -> bool:
    return all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)


def healthy_baseline(b: dict, p: dict = T2) -> bool | None:
    if not _fin(b["q"], b["c"], b["p"]):
        return None
    return b["q"] <= p["base_q_max"] and b["c"] >= p["keep_frac"] * b["p"]


def label_twin(w: dict, p: dict = T2) -> str:
    hb = healthy_baseline(w["B"], p)
    if hb is None or not _fin(w["E"]["q"], w["E"]["c"], w["E"]["p"]):
        return "unclear"
    if not hb:
        return "bad_baseline"
    ok = w["E"]["q"] <= w["B"]["q"] + p["q_excess"] and w["E"]["c"] >= p["keep_frac"] * w["E"]["p"]
    return "drained" if ok else "twin_not_drained"


def label_burst(b: dict, twin_c_e: float, drain_s: float, e_start_after_burst: float, p: dict = T2) -> str:
    """b: the burst run's windows {"B", "E", "E1", "E2"}; twin_c_e: the twin's c over E; drain_s: `gate.drain_time`;
    e_start_after_burst: E's start minus the burst's end."""
    hb = healthy_baseline(b["B"], p)
    if hb is None or not _fin(b["E"]["q"], b["E"]["c"], twin_c_e):
        return "unclear"
    if not hb:
        return "bad_baseline"
    lined = b["E"]["q"] > b["B"]["q"] + p["q_excess"]
    short = b["E"]["c"] < p["keep_frac"] * twin_c_e
    if not lined and not short:
        return "drained"
    if not lined or not _fin(b["E1"]["q"], b["E2"]["q"]):      # rule F r2, T2-R2-2: a missing half is unclear
        return "unclear"
    if b["E2"]["q"] < b["E1"]["q"]:                             # T2-R2-5: a falling line, at any rate
        return "draining"
    if short:                                                   # T2-R2-1: tested before relapse
        return "slow"
    emptied_before_e = isinstance(drain_s, (int, float)) and math.isfinite(drain_s) and drain_s < e_start_after_burst
    return "relapse" if emptied_before_e else "unclear"


def outcome(pairs: dict, p: dict = T2) -> dict:
    """pairs: {seed: {"burst": label or None (no valid run), "twin": label or None}}."""
    counted = {sd: v for sd, v in pairs.items() if v.get("twin") == "drained"
               and v.get("burst") not in (None, "bad_baseline")}
    labels = [v["burst"] for v in counted.values()]
    n, k_slow = len(counted), labels.count("slow")
    out = {"n": n, "counted": sorted(counted), "burst_labels": {sd: v["burst"] for sd, v in counted.items()},
           "not_counted": {sd: v for sd, v in pairs.items() if sd not in counted},
           "k_drained": labels.count("drained"), "k_slow": k_slow,
           "slow_upper": 1 - 0.05 ** (1 / n) if n and k_slow == 0 else None}   # one-sided 95 % upper bound at 0 of n
    if n < p["min_pairs"]:
        out["decision"] = "too_few"
    elif k_slow >= 1:
        out["decision"] = "slow_found"
    elif out["k_drained"] == n:
        out["decision"] = "drains"
    else:
        out["decision"] = "mixed"
    return out


def run_windows(run: dict, w: dict) -> dict:
    withheld.guard(run["dir"].name)
    return {k: window(run, *v) for k, v in w.items()}


def unfinished_sessions(run: dict, at: float | None = None) -> int:
    """Sessions whose turn 0 was planned before `at` (default the run's end) and whose last turn had not ended ok by
    then."""
    from analysis.margin import is_burst_request
    q = run["requests"]
    s = q[~is_burst_request(run)]
    d = float(run["cfg"]["workload"]["duration_s"]) if at is None else float(at)
    turns = run["cfg"]["workload"]["sessions"]["turns"]
    started = set(s[(s.turn == 0) & (s.t_planned < d)].session_id)
    done = set(s[(s.turn == turns - 1) & (s.status == "ok") & (s.t_end <= d)].session_id)
    return len(started - done)


def _early_c(run: dict, before: float, win: tuple) -> float:
    """Rule F r2, T2-R2-4: completions per s in `win` from sessions whose turn 0 was planned before `before`."""
    from analysis.margin import is_burst_request
    q = run["requests"]
    s = q[~is_burst_request(run)]
    early = set(s[(s.turn == 0) & (s.t_planned < before)].session_id)
    a, b = win
    return float(((s.session_id.isin(early)) & (s.status == "ok") & (s.t_end >= a) & (s.t_end < b)).sum() / (b - a))


def pair_row(burst: dict, twin: dict) -> dict:
    from analysis import gate
    W = windows(burst["cfg"])
    bw, tw = run_windows(burst, W), run_windows(twin, W)
    end = gate.burst_end(burst)
    dt = gate.drain_time(burst)
    e0 = W["E"][0]
    row = {"burst": burst["dir"].name, "twin": twin["dir"].name, "windows": {"burst": bw, "twin": tw},
           "drain_time_s": dt, "twin_label": label_twin(tw),
           "burst_label": label_burst(bw, tw["E"]["c"], dt, e0 - end),
           "emptied_before_E": bool(math.isfinite(dt) and dt < e0 - end)}
    ro = {}
    for k, f in (("shock", lambda: gate.shock_size(burst)),
                 ("unfinished_sessions", lambda: {"burst": unfinished_sessions(burst), "twin": unfinished_sessions(twin)}),
                 ("unfinished_sessions_at_E_start", lambda: {"burst": unfinished_sessions(burst, e0),
                                                             "twin": unfinished_sessions(twin, e0)}),   # T2-R2-8
                 ("E_completions_from_sessions_before_burst_end", lambda: {"burst": _early_c(burst, end, W["E"]),
                                                                           "twin": _early_c(twin, end, W["E"])}),
                 ("q_spans_after_burst", lambda: [window(burst, a, a + 300.0)["q"]
                                                  for a in np.arange(end, float(burst["cfg"]["workload"]["duration_s"]) - 299.0, 300.0)]),
                 ("first_300s_after", lambda: window(burst, end, end + 300.0)),
                 ("kv_max_in_burst", lambda: float(burst["metrics"][(burst["metrics"].t >= burst["cfg"]["trigger"]["at_s"])
                                                   & (burst["metrics"].t < end)]["vllm:kv_cache_usage_perc"].max())),
                 ("latency_burst_minus_twin_E", lambda: {"pf": bw["E"]["pf"] - tw["E"]["pf"],
                                                         "ttft_median": bw["E"]["ttft_median"] - tw["E"]["ttft_median"]})):
        try:
            ro[k] = f()
        except Exception as ex:   # a readout never takes the result with it
            ro[k] = f"error: {type(ex).__name__}: {ex}"[:200]
    row["readouts"] = ro
    return row


def _earliest(pattern: str):
    """The earliest valid run of a name (`find_runs` returns valid runs only by default), and the later valid ones."""
    from analysis.load import find_runs
    rr = sorted(find_runs(pattern), key=lambda r: r["dir"].name)
    return (rr[0] if rr else None), [r["dir"].name for r in rr[1:]]


def validate() -> dict:
    """Rule F r1, T2-1: `pair_row` on REG-19's five pairs (seen; REG-19's seal is lifted). Passes iff at least
    `slow_min` bursts are `slow` and at least `twin_drained_min` twins `drained`."""
    rows = {}
    for sd in T2["reg19"]["seeds"]:
        b, _ = _earliest(f"*_{T2['reg19']['burst'].format(sd)}")
        w, _ = _earliest(f"*_{T2['reg19']['control'].format(sd)}")
        if b is None or w is None:
            rows[sd] = None
            continue
        rows[sd] = pair_row(b, w)
    lab = [r["burst_label"] for r in rows.values() if r]
    tw = [r["twin_label"] for r in rows.values() if r]
    vp = T2["validate_pass"]
    return {"rows": rows, "k_slow": lab.count("slow"), "k_twin_drained": tw.count("drained"),
            "passes": lab.count("slow") >= vp["slow_min"] and tw.count("drained") >= vp["twin_drained_min"]}


def sealed() -> list[str]:
    import analysis.load as L
    if not L.RUNS.exists():
        return []
    return sorted(d.name for d in L.RUNS.iterdir() if "_t2_" in d.name and withheld.sealed_by(d.name) == "T2")


def table() -> dict:
    """T2's registered computation, once after the queue: `AFTERSHOCK_UNSEAL=T2 uv run python -m analysis.t2 table`.
    Refuses while any T2 folder is sealed. Per run name the earliest valid run counts (later ones listed); its config
    must carry caching off, lambda 0.80, 2580 s, its seed, a trigger only in burst runs, and REG-19's q excess."""
    s = sealed()
    if s:
        raise withheld.WithheldError(f"{len(s)} T2 folder(s) sealed; set AFTERSHOCK_UNSEAL=T2")
    pairs, rows, later = {}, {}, {}
    for sd in T2["seeds"]:
        got = {}
        for kind in ("burst", "control"):
            r, later[config_name(kind, sd)] = _earliest(f"*_{config_name(kind, sd)}")
            if r is not None:
                cfg, args = r["cfg"], r["cfg"]["server"]["args"]
                if not (args.get("--no-enable-prefix-caching") is True and args.get("--enable-prefix-caching") is None
                        and cfg["workload"]["rate"] == T2["lam"] and cfg["workload"]["duration_s"] == T2["duration_s"]
                        and cfg["seed"] == sd and bool(cfg.get("trigger")) == (kind == "burst")
                        and cfg["verdict"]["recovered_q_excess"] == T2["q_excess"]):
                    raise ValueError(f"{r['dir'].name}: config is not T2's")
            got[kind] = r
        if got["burst"] is None or got["control"] is None:
            pairs[sd] = {"burst": None if got["burst"] is None else "no_twin",
                         "twin": None if got["control"] is None else "no_burst"}
            continue
        rows[sd] = pair_row(got["burst"], got["control"])
        pairs[sd] = {"burst": rows[sd]["burst_label"], "twin": rows[sd]["twin_label"]}
    return {"outcome": outcome(pairs), "pairs": pairs, "rows": rows, "later_valid_not_counted": later}


if __name__ == "__main__":
    import sys
    from pprint import pprint
    cmd = sys.argv[1:]
    if cmd == ["table"]:
        pprint(table(), width=120, sort_dicts=False)
    elif cmd == ["validate"]:
        pprint(validate(), width=120, sort_dicts=False)
    else:
        sys.exit("usage: python -m analysis.t2 table | validate")
