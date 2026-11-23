"""REG-31 (pending the author's approval; not registered): four readouts, 0 GPU h. Never judged, no verdict label; every
setting is in REG31. Each reuses a registered function; none writes into runs/. The tables refuse until decisions.md
holds REG31["heading"] (`withheld.lifted`), so nothing here is computed on real runs before the registration.

  calm_share   schedule only: on REG-19's control workload (`gate._burst_cfg(calm_cfg)`), at rate λ, for each seed
               the highest 120 s planned session rate over [240, duration) (`gate.schedule_peak_series`), and per level
               how many seeds reach it. `calm_table` over calm_lams x calm_durations_s x calm_levels.
  in_service_at_burst_end
               sessions in service at `gate.burst_end` (`d1.in_service_steps`: the value after the last event at or
               before that time, ends before starts at equal times), their peak over [0, burst end) (`d1.peak`'s
               `before`), and `gate.shock_size`'s `outstanding`. `in_service_table` over REG-19's five closed-loop burst
               runs and REG-20 arm A's five open-loop burst runs; per arm, k of 5 at the burst's end at or below pool_lo
               and k of 5 above pool_hi (the band pool / longest prompt to pool / mean prompt).
  slow_baseline
               per run of a pair, the session requests planned in base_window_s that are not fast (percent fast's
               rule in `load._level`: status ok and TTFT <= the config's SLO; unfinished counts as a miss), and how
               many of them sit at the edge, the last edge_s before the trigger's at_s (which must equal the window's
               end), by three clocks: t_planned, t_sent, and a first token (`t_first_token`) at or after at_s.
               `slow_table` over `paper_tables.PAIRS`, pooled over the ten burst runs per clock.
  batch_lead   the running gauge (`b3.running`, raw scrapes, whole run: K5's onsets are read over the whole run, and
               batch_level is a value, not a window) against K5's latency onset (`k5.event`'s onsets L): t_batch is
               the start of the unbroken stretch of scrapes above batch_level that runs into the L onset (NaN if the
               gauge is not above it at L), t_first the first scrape above it at or after K5's ref_start_s (a readout);
               lead = L - t_batch, and the lead over arrivals A - t_batch. `lead_table` over the 11 event runs and the 9
               alarm runs; REG-19's controls, where batch_level comes from (B3's gate: highest 22), are used nowhere.
"""
from __future__ import annotations

import math

import numpy as np

from analysis import b3, d1, gate, k5
from analysis import paper_tables as PT
from analysis.load import _level, find_runs
from analysis.margin import is_burst_request
from analysis.reg30 import REG30
from analysis.stats import cp_upper as _cp_upper_exact


def _binom_cdf_log(k: int, n: int, p: float) -> float:
    """P(X <= k), X ~ Bin(n, p), summed in log space so that n in the thousands does not overflow."""
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k + 1)]
    m = max(terms)
    return math.exp(m) * sum(math.exp(x - m) for x in terms)


def cp_upper(k: int, n: int, conf: float = 0.95) -> float:
    """The one-sided Clopper-Pearson upper bound of analysis.stats.cp_upper, the same bisection on the same CDF, for
    any n: stats' float sum overflows above n of about 1000 (REG-31's calm scan has n = 2000; [CORR] 2026-11-22).
    For n <= 200 it returns stats' value (tests/test_reg31.py checks the two agree)."""
    if n <= 200:
        return _cp_upper_exact(k, n, conf)
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if k == n:
        return 1.0
    alpha = 1 - conf
    lo, hi = k / n, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if _binom_cdf_log(k, n, mid) > alpha else (lo, mid)
    return hi

REG31 = {
    "heading": "[REG] REG-31",
    "conf": 0.95,
    # 1. calm schedules
    "calm_cfg": "stage2_control_l1pl350cl_r2.0_w3000_s15",   # REG-19's control config (gate.reg19_exposed, seed 15)
    "p0_cfg": "burst_l1pl350cl_r2.0_w3000_s15",              # REG-19's burst config: P0's planned duration
    "calm_lams": (1.8, 2.0, 2.2, 2.4),
    "calm_levels": (3.2, 3.4),
    "calm_long_s": 15000.0,                                  # the second duration; the first is P0's (p0_duration_s)
    "calm_seeds": range(200000, 202000),                     # no config uses them (tests/test_reg31.py)
    # 2. sessions in service at the burst's end
    "closed_burst": {sd: PT.PAIRS[("replication", sd)][1] for sd in (15, 16, 17, 18, 19)},   # REG-19
    "open_burst": {15: "20261115-064753_burst_l1pl350ol_r2.0_w3000_s15",                    # REG-20 arm A
                   16: "20261115-075808_burst_l1pl350ol_r2.0_w3000_s16",
                   17: "20261115-090827_burst_l1pl350ol_r2.0_w3000_s17",
                   18: "20261115-113416_burst_l1pl350ol_r2.0_w3000_s18",
                   19: "20261115-124444_burst_l1pl350ol_r2.0_w3000_s19"},
    # the band the paper prints (REG30["low_bound"], REG30["low_bounds_reported"][1]), rounded as printed
    "pool_lo": 24.4,                                         # pool / turn 8's prompt (longest): 96,656 / 3968
    "pool_hi": 31.5,                                         # pool / mean prompt: 96,656 / 3072
    "pool_band_label": "pool / longest prompt to pool / mean prompt",
    # 3. slow baseline requests
    "base_window_s": (300.0, 900.0),                         # = gate.REG19["baseline_pair"]
    "edge_s": 10.0,
    "edge_clock": "t_sent",                                  # the clock the consequence uses (rule F, R31c)
    # 4. running-batch lead
    "batch_level": 22,                                       # B3's gate: REG-19's controls' highest running batch
    "lead_s": 30.0,                                          # = k5.K5["min_lead_s"]
    "k5_events": ("20261109-102614_stage2_control_l1pl350cl_r2.4_s4",   # REG-23's six events (decisions.md
                  "20261109-120308_stage2_control_l1pl350cl_r2.4_s5",   # 2026-11-14 [RESULT] REG-23); a cross-check
                  "20261109-193526_sweep_l1pl350cl_pc_on_r2.6",         # of k5.candidates() + k5.event
                  "20261109-195349_sweep_l1pl350cl_pc_on_r2.8",
                  "20261109-201246_sweep_l1pl350cl_pc_on_r3",
                  "20261110-122001_stage2_control_l1pl350cl_r2.2_long_s14"),
    "k4_tipped": {133: "20261117-053015_k4_l1pl350cl_r2.2_s133", 137: "20261117-070951_k4_l1pl350cl_r2.2_s137",
                  175: "20261117-084905_k4_l1pl350cl_r2.2_s175", 196: "20261117-102749_k4_l1pl350cl_r2.2_s196"},
    "k4e_late": {914: "20261120-180402_k4e_l1pl350cl_r2.2_s914"},
    "reg27_control_seeds": (61, 68, 72, 79),
    "k4_calm": {102: "20261117-062210_k4_l1pl350cl_r2.2_s102", 103: "20261117-080145_k4_l1pl350cl_r2.2_s103",
                105: "20261117-094058_k4_l1pl350cl_r2.2_s105", 106: "20261117-111949_k4_l1pl350cl_r2.2_s106",
                107: "20261117-125335_k4_l1pl350cl_r2.2_s107"},
}


def _registered() -> None:
    from analysis import withheld
    if not withheld.lifted(REG31["heading"]):
        raise RuntimeError(f"REG-31 is not registered: decisions.md has no heading '{REG31['heading']}'")


def _one(rid: str) -> dict:
    found = find_runs(rid)
    if len(found) != 1:
        raise ValueError(f"{rid}: {len(found)} valid runs, REG-31 needs exactly one")
    return found[0]


# ---------------------------------------------------------------- 1. calm schedules

def p0_duration_s(name: str | None = None) -> float:
    """P0's planned duration from REG-19's burst config: trigger at_s + its duration_s + the verdict's watch_s."""
    c = gate._burst_cfg(name or REG31["p0_cfg"])
    return float(c["trigger"]["at_s"] + c["trigger"]["duration_s"] + c["verdict"]["watch_s"])


def calm_share(lam: float, seeds, duration_s: float, levels) -> dict:
    """Schedule only. Per level: n seeds, k whose highest 120 s planned session rate is >= level, k / n, and the
    one-sided Clopper-Pearson upper bound on it."""
    cfg = gate._burst_cfg(REG31["calm_cfg"])
    seeds = list(seeds)
    peaks = np.array([float(np.max(gate.schedule_peak_series(cfg, lam, sd, duration_s)[1])) for sd in seeds])
    n = len(seeds)
    out = {}
    for lv in levels:
        k = int((peaks >= lv).sum())
        out[lv] = {"n": n, "k": k, "share": k / n if n else math.nan,
                   "cp_upper": cp_upper(k, n, REG31["conf"]) if n else math.nan}
    return {"lam": lam, "duration_s": duration_s, "seeds": seeds, "levels": out}


def calm_table() -> dict:
    """`calm_share` over calm_lams x (P0's duration, calm_long_s); the seed list once, not per row."""
    _registered()
    durs = (p0_duration_s(), REG31["calm_long_s"])
    rows = [calm_share(lam, REG31["calm_seeds"], dur, REG31["calm_levels"]) for lam in REG31["calm_lams"] for dur in durs]
    seeds = rows[0]["seeds"]
    return {"seeds": seeds, "durations_s": durs, "rows": [{k: v for k, v in r.items() if k != "seeds"} for r in rows]}


# ---------------------------------------------------------------- 2. sessions in service at the burst's end

def in_service_at_burst_end(run: dict) -> dict:
    end = gate.burst_end(run)
    t, n = d1.in_service_steps(run)
    i = int(np.searchsorted(t, end, side="right")) - 1
    return {"run": run["dir"].name, "burst_end_s": end, "at_end": int(n[i]) if i >= 0 else 0,
            "peak_before_end": d1.peak(run, before=end)["before"],
            "outstanding": gate.shock_size(run)["outstanding"]}


def _k_of_n(k: int, n: int) -> dict:
    return {"k": k, "n": n, "share": k / n if n else math.nan, "cp_upper": cp_upper(k, n, REG31["conf"]) if n else math.nan}


def band_counts(rows: list[dict]) -> dict:
    """Over one arm's rows: k with `at_end` <= pool_lo and k with `at_end` > pool_hi, of n."""
    v = [r["at_end"] for r in rows]
    return {"at_or_below_lo": _k_of_n(sum(x <= REG31["pool_lo"] for x in v), len(v)),
            "above_hi": _k_of_n(sum(x > REG31["pool_hi"] for x in v), len(v))}


def in_service_table() -> dict:
    _registered()
    rows = {loop: [{"loop": loop, "seed": sd, **in_service_at_burst_end(_one(rid))} for sd, rid in ids.items()]
            for loop, ids in (("closed", REG31["closed_burst"]), ("open", REG31["open_burst"]))}
    return {"rows": rows["closed"] + rows["open"], "arms": {loop: band_counts(r) for loop, r in rows.items()},
            "band": {"label": REG31["pool_band_label"], "lo": REG31["pool_lo"], "hi": REG31["pool_hi"]}}


# ---------------------------------------------------------------- 3. slow baseline requests

CLOCKS = ("t_planned", "t_sent", "first_token_after")


def _not_fast(run: dict, a: float, b: float):
    """Requests planned in [a, b) and their 'not fast' mask, by `load._level`'s rule (status ok and TTFT <= the
    config's SLO is fast; unfinished or errored is not). Raises if the mask's complement does not give `_level`'s pf
    over the same window, so the rule cannot drift from percent fast's."""
    r, slo = run["requests"], run["cfg"]["slo"]["ttft_s"]
    w = r[(r.t_planned >= a) & (r.t_planned < b)]
    slow = ~((w.status == "ok") & (w.ttft <= slo))
    pf = _level(run, a, b)["pf"]
    if len(w) and not math.isclose(1.0 - float(slow.mean()), pf, abs_tol=1e-12):
        raise AssertionError(f"{run['dir'].name}: not-fast rule gives {1 - slow.mean()}, _level's pf {pf}")
    return w, slow


def _f(x) -> float | None:
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def _slow_one(run: dict, at: float) -> dict:
    """Edge counts by three clocks: t_planned and t_sent in [at - edge_s, at); and a first token (`t_first_token`,
    the column load.py measures TTFT from) at or after at. A request with no first token is counted apart."""
    a, b = REG31["base_window_s"]
    w, slow = _not_fast(run, a, b)
    burst = is_burst_request(run).loc[w.index]
    sess, s = w[~burst], w[slow & ~burst]
    lo = at - REG31["edge_s"]
    in_edge = lambda col: (s[col] >= lo) & (s[col] < at)
    edge = {"t_planned": int(in_edge("t_planned").sum()), "t_sent": int(in_edge("t_sent").sum()),
            "first_token_after": int((s.t_first_token >= at).sum())}
    planned_edge = int(((sess.t_planned >= lo) & (sess.t_planned < at)).sum())
    return {"run": run["dir"].name, "n_session": int(len(sess)), "slow": int(len(s)), "edge": edge,
            "slow_no_first_token": int(s.t_first_token.isna().sum()),
            "edge_share_of_planned": planned_edge / len(sess) if len(sess) else math.nan,   # the uniform-rate reference
            "edge_share_uniform": REG31["edge_s"] / (b - a),
            "at_s": at, "requests": [{"t_planned": float(x.t_planned), "t_sent": _f(x.t_sent),
                                      "t_first_token": _f(x.t_first_token), "turn": int(x.turn), "ttft": _f(x.ttft),
                                      "status": x.status}
                                     for x in s.sort_values("t_planned").itertuples()]}


def slow_baseline(control: dict, burst: dict) -> dict:
    """Per run: session requests planned in base_window_s that are not fast, the edge counts by each clock (the
    control is read at the burst config's at_s: one schedule), and each one's times, turn and TTFT. Raises unless
    the burst's at_s is the window's end."""
    at = float(burst["cfg"]["trigger"]["at_s"])
    if at != REG31["base_window_s"][1]:
        raise ValueError(f"{burst['dir'].name}: trigger at_s {at} is not the baseline window's end "
                         f"{REG31['base_window_s'][1]}")
    return {"control": _slow_one(control, at), "burst": _slow_one(burst, at)}


def pooled_edge(burst_rows: list[dict]) -> dict:
    """Over the burst runs, per clock: k slow requests at the edge, of n slow, with the one-sided CP upper bound;
    beside the edge's pooled share of planned session requests (the uniform-rate reference)."""
    n = sum(r["slow"] for r in burst_rows)
    out = {c: _k_of_n(sum(r["edge"][c] for r in burst_rows), n) for c in CLOCKS}
    out["edge_clock"] = REG31["edge_clock"]
    out["planned_reference"] = {
        "edge_share_of_planned": (sum(r["edge_share_of_planned"] * r["n_session"] for r in burst_rows)
                                  / sum(r["n_session"] for r in burst_rows)) if burst_rows else math.nan,
        "edge_share_uniform": REG31["edge_s"] / (REG31["base_window_s"][1] - REG31["base_window_s"][0])}
    return out


def slow_table() -> dict:
    _registered()
    rows = [{"experiment": e, "seed": s, **slow_baseline(_one(c), _one(b))} for (e, s), (c, b) in PT.PAIRS.items()]
    return {"rows": rows, "pooled_burst": pooled_edge([r["burst"] for r in rows])}


# ---------------------------------------------------------------- 4. running-batch lead

def batch_cross(run: dict, l_onset: float = math.nan) -> dict:
    """On the raw running gauge over the whole run (measured=False: K5's onsets are read over the whole run, and
    batch_level is a value, not a window):
      t_batch  the start of the unbroken stretch of scrapes above batch_level that holds at the last scrape at or
               before l_onset (every scrape from t_batch to L above it); NaN if the gauge is not above it at L;
      t_first  the first scrape at or after K5's ref_start_s above batch_level (a readout); `exceeds` if there is one."""
    t, n = b3.running(run, measured=False)   # whole run, not load.window: see the docstring
    above = n > REG31["batch_level"]
    after = t >= k5.K5["ref_start_s"]
    first = after & above
    t_batch = math.nan
    if not math.isnan(l_onset):
        i = int(np.searchsorted(t, l_onset, side="right")) - 1
        if i >= 0 and above[i]:
            below = np.flatnonzero(~above[:i + 1])
            t_batch = float(t[below[-1] + 1] if len(below) else t[0])
    return {"t_batch": t_batch, "t_first": float(t[first][0]) if first.any() else math.nan,
            "exceeds": bool(first.any()), "n_max_after_ref": float(n[after].max()) if after.any() else math.nan}


def batch_lead(run: dict) -> dict:
    e = k5.event(run)
    on = (e.get("onsets") or {}) if e["why"] is None else {}
    on_l, on_a = float(on.get("L", math.nan)), float(on.get("A", math.nan))
    bc = batch_cross(run, on_l)
    tb = bc["t_batch"]
    lead = on_l - tb if not (math.isnan(on_l) or math.isnan(tb)) else math.nan
    # k5's convention (lead_over_A): A - X, +inf when arrivals never had an onset
    over_a = math.nan if math.isnan(tb) else (math.inf if math.isnan(on_a) else on_a - tb)
    return {"run": run["dir"].name, **bc, "l_onset": on_l, "a_onset": on_a, "lead": lead, "lead_over_A": over_a,
            "why": e["why"]}


def lead_counts(ev_rows: list[dict]) -> dict:
    """n_with_lead: events with why None and both times; k of them with lead above lead_s."""
    ok = [x for x in ev_rows if x["why"] is None and not math.isnan(x["lead"])]
    return {"n_events": len(ev_rows), "n_with_lead": len(ok),
            "lead_above": _k_of_n(sum(x["lead"] > REG31["lead_s"] for x in ok), len(ok))}


def alarm_counts(al_rows: list[dict]) -> dict:
    """Warm alarm runs (no finite `gate.kv_onset`) and how many of them ever exceed batch_level (by t_first)."""
    warm = [x for x in al_rows if math.isnan(x["kv_onset"])]
    return {"n_alarm": len(al_rows), "n_tipped": len(al_rows) - len(warm),
            "warm_exceeding": _k_of_n(sum(x["exceeds"] for x in warm), len(warm))}


def lead_table() -> dict:
    _registered()
    k5_ev = [r for r in k5.candidates() if k5.event(r)["why"] is None]
    names = tuple(r["dir"].name for r in k5_ev)
    events = [("k5", r) for r in k5_ev]
    events += [("k4_surge_tipped", _one(rid)) for rid in REG31["k4_tipped"].values()]
    events += [("k4e_late", _one(rid)) for rid in REG31["k4e_late"].values()]
    alarms = [("reg27_control", _one(REG30["controls"][sd])) for sd in REG31["reg27_control_seeds"]]
    alarms += [("k4_calm", _one(rid)) for rid in REG31["k4_calm"].values()]
    ev_rows = [{"source": s, **batch_lead(r)} for s, r in events]
    al_rows = [{"source": s, "run": r["dir"].name, "kv_onset": gate.kv_onset(r), **batch_cross(r)} for s, r in alarms]
    return {"events": ev_rows, "alarms": al_rows, "leads": lead_counts(ev_rows), "alarm": alarm_counts(al_rows),
            "k5_events": names, "k5_events_are_reg23s": names == REG31["k5_events"],
            "params": {k_: REG31[k_] for k_ in ("batch_level", "lead_s")}}


if __name__ == "__main__":
    import json
    import sys
    fns = {"calm_table": calm_table, "in_service_table": in_service_table, "slow_table": slow_table,
           "lead_table": lead_table}
    if len(sys.argv) != 2 or sys.argv[1] not in fns:
        raise SystemExit("usage: python -m analysis.reg31 " + " | ".join(fns))
    print(json.dumps(fns[sys.argv[1]](), indent=1, default=str))
