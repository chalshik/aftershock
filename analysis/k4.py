"""K4 (docs/drafts/k4_draft1.md; not registered): the warm state ends at an arrival surge. This module holds draft 1's
design investigation (the author, 2026-11-17: "solve them based on rigour investigation"), computed on runs already on
record and on schedules alone. It judges nothing.

  natural_record   every valid no-trigger, cache-on, closed-loop L1 350 W run (`gate.SURGE["pattern"]`): its warm time
                   and tip (`gate.surge_run`), and per level the first 120 s planned-rate crossing in warm time and its
                   lag to the KV onset. Cache-off runs and triggered runs are left out by config (`natural`).
  crossings        per level: natural runs whose warm time crossed it, and how many of those tipped.
  surgeless        the tips with no 120 s peak at or above a level in their warm time, against the warm hours on record,
                   with a one-sided Poisson upper bound on the rate (`stats.poisson_rate_upper`).
  scan             schedule only: per seed, whether it qualifies for the surge arm or the calm arm at λ.
  design_scan      `scan` over K4["design_seeds"], a seed range kept apart from the range a registration would judge
                   (K4["judged_seeds"]), so λ can be chosen on it without seeing the pass condition's own count.
  design_power     exact P(outcome) for candidate designs under the claim and under its null (`outcome_rule`).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from analysis import gate
from analysis.stats import cp_upper, fisher_one_sided, poisson_rate_upper

K4 = {
    "levels": (3.0, 3.1, 3.2, 3.3, 3.4, 3.5),     # readout levels, req/s, 120 s planned session rate
    "surge_level": 3.4, "calm_level": 3.1,        # draft 1
    "surge_window": (1200.0, 1800.0),             # draft 1: the first crossing of surge_level falls here
    "pre_level": 3.2,                             # draft 1: no peak at or above this before the window
    "duration_s": 2700.0,                         # draft 1
    "base_cfg": "stage2_control_l1pl350cl_r2.2_s2",   # REG-14's control workload (L1, 8 turns, 5 s gaps)
    "lams": (2.0, 2.1, 2.2, 2.3, 2.4),
    "design_seeds": tuple(range(1000, 1300)),     # for choosing λ; never run, never judged
    "judged_seeds": tuple(range(100, 600)),       # the feasibility count a registration would judge; widened from
                                                  # 100-399 before any judged scan (decisions.md 2026-11-17, K4 points)
    "conf": 0.95,
    # Draft 2's design (decisions.md 2026-11-17 [DECISION] K4's four open points):
    "n": 5, "r_c": 2, "r_s": 2, "need_s": 4,
    # Prediction only (BELIEF; draft 2): per surge run and per calm run.
    "prediction": {"surge": {"tipped": 0.80, "warm": 0.12, "unclear": 0.08},
                   "calm": {"tipped": 0.05, "warm": 0.88, "unclear": 0.07}},
    # After rule F round 1 (BELIEF): a surge tip is counted only if it persists; "tipped" in the calm arm is any onset.
    "prediction_r1": {"surge": {"tipped": 0.75, "tipped_recovered": 0.05, "warm": 0.12, "unclear": 0.08},
                      "calm": {"tipped": 0.05, "warm": 0.88, "unclear": 0.07}},
}


def natural(run: dict) -> bool:
    """No trigger of any kind, prefix caching on, closed-loop turns."""
    cfg = run["cfg"]
    off = bool(((cfg.get("server") or {}).get("args") or {}).get("--no-enable-prefix-caching"))
    return cfg.get("trigger") is None and not off and gate.closed_loop(run)


def _series(run: dict, end: float) -> tuple[np.ndarray, np.ndarray]:
    from analysis.margin import is_burst_request
    q = run["requests"]
    s = q[~is_burst_request(run) & q.turn.notna()]
    grid = np.arange(gate.SURGE["start_s"], end, gate.SURGE["grid_s"])
    return grid, gate.rolling_rate(s.t_planned.to_numpy(float), grid, gate.SURGE["window_s"])


def natural_row(run: dict) -> dict:
    r = gate.surge_run(run)
    on = r["onset"]
    end = on if r["tipped"] else gate.SURGE["start_s"] + r["warm_s"]
    g, x = _series(run, end)
    cross = {}
    for lv in K4["levels"]:
        i = np.flatnonzero(x >= lv)
        t = float(g[i[0]]) if len(i) else None
        cross[lv] = {"t": t, "lag_to_onset": (on - t) if (t is not None and r["tipped"]) else None}
    return {"run": r["run"], "lam": r["lam"], "tipped": bool(r["tipped"]), "onset": on if r["tipped"] else None,
            "warm_s": r["warm_s"], "peak_warm": float(x.max()) if len(x) else math.nan, "cross": cross,
            "peak_tip": r["peak_tip"] if r["tipped"] else None}


def natural_record() -> list[dict]:
    from analysis.load import find_runs
    return [natural_row(r) for r in find_runs(gate.SURGE["pattern"]) if natural(r)]


def crossings(rows: list[dict]) -> dict:
    """Per level: natural runs whose warm time (to the onset, for a tipped run) reached it; how many of them tipped;
    and the lags from the first crossing to the onset."""
    out = {}
    for lv in K4["levels"]:
        hit = [r for r in rows if r["cross"][lv]["t"] is not None]
        lags = sorted(r["cross"][lv]["lag_to_onset"] for r in hit if r["tipped"])
        out[lv] = {"n_crossed": len(hit), "n_tipped": sum(r["tipped"] for r in hit), "lags_s": lags,
                   "runs_crossed_not_tipped": [r["run"] for r in hit if not r["tipped"]]}
    return out


def surgeless(rows: list[dict], level: float) -> dict:
    """Tips whose warm time never reached `level`, against the warm hours of all natural runs."""
    n = sum(r["tipped"] and r["cross"][level]["t"] is None for r in rows)
    hours = sum(r["warm_s"] for r in rows) / 3600.0
    return {"level": level, "n_tips_below": n, "warm_h": hours, "rate_per_h_upper": poisson_rate_upper(n, hours,
                                                                                                         K4["conf"])}


def classify(g: np.ndarray, x: np.ndarray) -> dict:
    """Draft 1's arms on one schedule's 120 s planned-rate series."""
    w0, w1 = K4["surge_window"]
    i = np.flatnonzero(x >= K4["surge_level"])
    t = float(g[i[0]]) if len(i) else None
    pre = bool((x[g < w0] >= K4["pre_level"]).any())
    surge = t is not None and w0 <= t <= w1 and not pre
    calm = bool(x.max() < K4["calm_level"]) if len(x) else False
    return {"surge": bool(surge), "calm": calm, "t_cross": t, "peak": float(x.max()) if len(x) else math.nan}


def scan(lam: float, seeds) -> dict:
    cfg = gate._burst_cfg(K4["base_cfg"])
    rows = []
    for sd in seeds:
        g, x = gate.schedule_peak_series(cfg, lam, sd, K4["duration_s"])
        rows.append({"seed": sd, **classify(g, x)})
    return {"lam": lam, "n": len(rows), "surge": [r["seed"] for r in rows if r["surge"]],
            "calm": [r["seed"] for r in rows if r["calm"]],
            "t_cross": [r["t_cross"] for r in rows if r["surge"]]}


def design_scan() -> dict:
    out = {}
    for lam in K4["lams"]:
        s = scan(lam, K4["design_seeds"])
        out[lam] = {"n": s["n"], "n_surge": len(s["surge"]), "n_calm": len(s["calm"]),
                    "first_surge": s["surge"][:5], "first_calm": s["calm"][:5]}
    return out


# ---------------------------------------------------------------- design error rates

def outcome_rule(n: int, surge: tuple[int, int], calm_tipped: int, r_c: int, r_s: int, need_s: int,
                 calm_warm: int = 0, need_cw: int = 0) -> str:
    """surge = (tipped, warm) of n surge runs; calm_tipped of n calm runs. refutes: calm_tipped >= r_c or surge warm >=
    r_s; supports: surge tipped >= need_s, calm_tipped == 0 and calm_warm >= need_cw (0 in draft 2's `design_power`;
    need_s after rule F round 1, `predicted`); mixed otherwise."""
    if calm_tipped >= r_c or surge[1] >= r_s:
        return "refutes"
    if surge[0] >= need_s and calm_tipped == 0 and calm_warm >= need_cw:
        return "supports"
    return "mixed"


def _dist(n: int, p: dict) -> dict:
    """Multinomial over n runs with per-run probabilities p (keys: tipped, warm, unclear): (tipped, warm) -> prob."""
    out = {}
    for c in itertools.product(p, repeat=n):
        k = (c.count("tipped"), c.count("warm"))
        out[k] = out.get(k, 0.0) + math.prod(p[x] for x in c)
    return out


def power(n: int, p_surge: dict, p_calm: dict, r_c: int, r_s: int, need_s: int, need_cw: int = 0) -> dict:
    ds, dc = _dist(n, p_surge), _dist(n, p_calm)
    res = {"supports": 0.0, "refutes": 0.0, "mixed": 0.0}
    for (ts, ws), ps in ds.items():
        for (tc, wc), pc in dc.items():
            res[outcome_rule(n, (ts, ws), tc, r_c, r_s, need_s, wc, need_cw)] += ps * pc
    return res


def design_power() -> dict:
    """Candidate designs (n per arm 4, 5, 6; r_c 1 or 2; need_s n - 1; r_s 2) under:
    - the claim, per surge run tipped p_s with warm (1 - p_s) * 0.6 and unclear the rest, per calm run tipped p_c;
    - its null, both arms tipping at one rate p (the surge does not matter; unclear 0.10 in both).
    Also each design's Fisher one-sided p at its weakest `supports` (need_s of n against 0 of n)."""
    designs = [(n, r_c) for n in (4, 5, 6) for r_c in (1, 2)]
    alt = [(p_s, p_c) for p_s in (0.5, 0.75, 0.9) for p_c in (0.02, 0.05, 0.12)]
    null = (0.05, 0.2, 0.4, 0.6)
    out = []
    for n, r_c in designs:
        need = n - 1
        row = {"n": n, "r_c": r_c, "need_s": need, "r_s": 2,
               "fisher_weakest_supports": fisher_one_sided(need, n, 0, n), "claim": {}, "null": {}}
        for p_s, p_c in alt:
            ps = {"tipped": p_s, "warm": (1 - p_s) * 0.6, "unclear": (1 - p_s) * 0.4}
            pc = {"tipped": p_c, "warm": (1 - p_c) * 0.93, "unclear": (1 - p_c) * 0.07}
            row["claim"][f"ps{p_s}_pc{p_c}"] = power(n, ps, pc, r_c, 2, need)
        for p in null:
            pp = {"tipped": p, "warm": 0.9 - p, "unclear": 0.10}
            row["null"][f"p{p}"] = power(n, pp, pp, r_c, 2, need)
        out.append(row)
    return {"designs": out}


def predicted() -> dict:
    """Draft 2's outcome distribution under K4["prediction"] (`power`; `supports` here is before the timing split)."""
    pr = K4["prediction"]
    for v in pr.values():
        if abs(sum(v.values()) - 1) > 1e-9:
            raise ValueError("a prediction does not sum to 1")
    return power(K4["n"], pr["surge"], pr["calm"], K4["r_c"], K4["r_s"], K4["need_s"])


def predicted_r1() -> dict:
    """The registered rule's outcome distribution (rule F round 1: persistent surge tips, calm warm >= need_s) under
    K4["prediction_r1"]; `supports` here is before the timing split."""
    pr = K4["prediction_r1"]
    for v in pr.values():
        if abs(sum(v.values()) - 1) > 1e-9:
            raise ValueError("a prediction does not sum to 1")
    return power(K4["n"], pr["surge"], pr["calm"], K4["r_c"], K4["r_s"], K4["need_s"], need_cw=K4["need_s"])


# ---------------------------------------------------------------- the registration (draft 2, author 2026-11-17)

REG = {
    "lam": 2.2,
    "delta_s": 212.0,                 # the timing band: largest record lag at 3.4 (92.1 s, `crossings`) + 120 s
    "tip_from_s": 600.0,              # an onset counts as a tip from here (before: `unclear`)
    "base": (300.0, 600.0),           # baseline window; ends at tip_from_s (rule F round 1, item 1; was [300, 900))
    "end": (2100.0, 2700.0),          # end window
    "most": 4,                        # of 5 valid per arm (too_few below)
}
CONFIGS_DIR = None  # set in tests; configs/ otherwise


def _configs():
    from pathlib import Path
    return CONFIGS_DIR or Path(__file__).resolve().parent.parent / "configs"


def config_name(seed: int) -> str:
    return f"k4_l1pl350cl_r2.2_s{seed}"


def run_pattern(seed: int) -> str:
    return f"*_{config_name(seed)}"


def pick(seeds=None) -> dict:
    """Draft 2's arms: the lowest K4["n"] qualifying seeds of each arm among `seeds` (default K4["judged_seeds"]) at
    λ = REG["lam"], in seed order, and the run order: alternating surge, calm (so host drift does not fall on one arm)."""
    s = scan(REG["lam"], K4["judged_seeds"] if seeds is None else seeds)
    surge, calm = s["surge"][:K4["n"]], s["calm"][:K4["n"]]
    t = dict(zip(s["surge"], s["t_cross"]))
    order = [x for pair in itertools.zip_longest(surge, calm) for x in pair if x is not None]
    return {"lam": REG["lam"], "n_scanned": s["n"], "n_surge_qualifying": len(s["surge"]),
            "n_calm_qualifying": len(s["calm"]), "surge": surge, "calm": calm,
            "t_cross": {sd: t[sd] for sd in surge}, "order": order}


def feasibility() -> dict:
    """Rule G, the pass condition in draft 2 (fixed before this was run): at least K4["n"] seeds per arm."""
    pk = pick()
    return {**pk, "passes": bool(len(pk["surge"]) >= K4["n"] and len(pk["calm"]) >= K4["n"])}


def config_text(seed: int, arm: str) -> str:
    if arm not in ("surge", "calm"):
        raise ValueError(arm)
    return (f"# K4 (docs/decisions.md [REG] K4 (the warm state ends at a surge)): REG-14's control workload at\n"
            f"# λ = {REG['lam']:g}, seed {seed} ({arm} arm, picked by `k4.pick` from the schedule alone), {K4['duration_s']:g} s, no trigger.\n"
            f"# Written by `python -m analysis.k4 configs`. Runs only after `[DECISION] ... K4 starts`.\n"
            f"extends: {K4['base_cfg']}.yaml\n"
            f"name: {config_name(seed)}\n"
            f"seed: {seed}\n"
            f"workload:\n"
            f"  duration_s: {K4['duration_s']:g}\n")


def resolved(seed: int, arm: str) -> dict:
    import yaml
    from harness.config import _merge, load_config, with_defaults
    from harness.run import expand
    raw = yaml.safe_load(config_text(seed, arm))
    parent = raw.pop("extends")
    return expand(with_defaults(_merge(load_config(_configs() / parent), raw)))[0]


def write_configs(pk: dict, configs_dir=None) -> list:
    d = configs_dir or _configs()
    items = [(sd, "surge") for sd in pk["surge"]] + [(sd, "calm") for sd in pk["calm"]]
    paths = [d / f"{config_name(sd)}.yaml" for sd, _ in items]
    if any(p.exists() for p in paths):
        raise FileExistsError([str(p) for p in paths if p.exists()])
    for p, (sd, arm) in zip(paths, items):
        p.write_text(config_text(sd, arm))
    return paths


REGISTERED = {"surge": [133, 137, 175, 196, 287], "calm": [102, 103, 105, 106, 107],
              "t_cross": {133: 1775.0, 137: 1365.0, 175: 1685.0, 196: 1685.0, 287: 1760.0}}   # `feasibility`, 92acfbd
CALM_TIPS = ("tipped", "tipped_recovered", "tipped_unclear", "tipped_early")   # any onset counts against the calm arm
MARGINAL_S = 30.0   # a surge seed with fewer seconds than this at or above surge_level is marginal (rule F r1, item 8)


def label(run: dict, t_cross: float | None) -> dict:
    """One run (rule F round 1 applied). The onset is read first, whatever the baseline (item 1):
    - an onset (`gate.kv_onset`) before tip_from_s: `tipped_early`;
    - an onset in [tip_from_s, duration): by the end window against the baseline (k1's rule, `k1.cold_backlogged`):
      `tipped` if not recovered (the cold state persisted, item 2), `tipped_recovered` if recovered, `tipped_unclear`
      if that is undefined (NaN);
    - no onset: `bad_baseline` if the baseline's percent fast is below min_baseline_pf or NaN; `warm` if the end window
      is recovered; `unclear` otherwise.
    Baseline [300, 600), so it ends before tip_from_s; end [2100, 2700). For a surge-arm `tipped` run, `on_time`: the
    onset in [t_cross, t_cross + delta_s]."""
    from analysis.k1 import cold_backlogged
    v = run["cfg"]["verdict"]
    st = cold_backlogged(run, {"base": REG["base"], "end": REG["end"]}, v)
    on = gate.kv_onset(run)
    dur = float(run["cfg"]["workload"]["duration_s"])
    pf = st["base"]["pf"]
    rec = st["recovered"]
    if np.isfinite(on) and on < REG["tip_from_s"]:
        lab = "tipped_early"
    elif np.isfinite(on) and on < dur:
        lab = "tipped" if rec is False else "tipped_recovered" if rec is True else "tipped_unclear"
    elif pf is None or (isinstance(pf, float) and math.isnan(pf)) or pf < v["min_baseline_pf"]:
        lab = "bad_baseline"
    elif rec is True:
        lab = "warm"
    else:
        lab = "unclear"
    on_time = None
    if t_cross is not None and lab == "tipped":
        on_time = bool(t_cross <= on <= t_cross + REG["delta_s"])
    return {"label": lab, "onset": float(on) if np.isfinite(on) else None, "on_time": on_time, "state": st}


def outcome(surge: list[dict], calm: list[dict], marginal: set | None = None) -> dict:
    """The rule (rule F round 1 applied), over each arm's runs with a usable baseline (`bad_baseline` is left out of n,
    item 3), in this order:
    too_few (either arm under REG["most"]) -> refutes (calm runs with any onset >= r_c, or surge runs `warm` >= r_s) ->
    supports / supports_untimed (surge `tipped` >= need_s, no calm onset, and calm `warm` >= need_s; untimed if fewer
    than need_s surge tips are on_time) -> mixed. Rows carry "seed" when `marginal` (the marginal surge seeds) is given;
    `refutes_source` and `surge_warm_marginal_only` feed the decision text (item 8). Readouts: k of n per arm, one-sided
    95 % Clopper-Pearson upper bounds, Fisher's one-sided p (surge tipped against calm with an onset)."""
    surge = [r for r in surge if r["label"] != "bad_baseline"]
    calm = [r for r in calm if r["label"] != "bad_baseline"]
    ns, nc = len(surge), len(calm)
    ts = sum(r["label"] == "tipped" for r in surge)
    ws = sum(r["label"] == "warm" for r in surge)
    tc = sum(r["label"] in CALM_TIPS for r in calm)
    wc = sum(r["label"] == "warm" for r in calm)
    on = sum(bool(r["on_time"]) for r in surge)
    need = K4["need_s"]
    src = None
    if ns < REG["most"] or nc < REG["most"]:
        dec = "too_few"
    elif tc >= K4["r_c"] or ws >= K4["r_s"]:
        dec = "refutes"
        src = "both" if (tc >= K4["r_c"] and ws >= K4["r_s"]) else "calm" if tc >= K4["r_c"] else "surge"
    elif ts >= need and tc == 0 and wc >= need:
        dec = "supports" if on >= need else "supports_untimed"
    else:
        dec = "mixed"
    warm_seeds = [r.get("seed") for r in surge if r["label"] == "warm"]
    marg_only = (bool(warm_seeds) and marginal is not None and all(sd in marginal for sd in warm_seeds))
    return {"decision": dec, "refutes_source": src, "surge_warm_marginal_only": marg_only if src == "surge" else None,
            "n_surge": ns, "n_calm": nc, "surge_tipped": ts, "surge_warm": ws, "surge_on_time": on,
            "calm_tipped": tc, "calm_warm": wc,
            "cp_upper_calm_tipped": cp_upper(tc, nc, K4["conf"]) if nc else math.nan,
            "cp_upper_surge_not_tipped": cp_upper(ns - ts, ns, K4["conf"]) if ns else math.nan,
            "fisher_p": fisher_one_sided(ts, ns, tc, nc) if ns and nc else math.nan}


def _planned_rate(tp: np.ndarray, a: float, b: float) -> float:
    return float(((tp >= a) & (tp < b)).sum() / (b - a)) if b > a else math.nan


def schedule_load(pk: dict | None = None) -> dict:
    """Schedule only (rule F round 1, item 4): per registered seed, the planned session rate over the whole run
    [240, duration), over [t_c - 720, t_c - 120) and over [t_c - 120, t_c + 212), and the seconds at or above the surge
    level (5 s grid). A calm seed is read at the spans of the surge seed it is paired with in the run order (pair i =
    surge[i], calm[i]). Also each surge seed's `marginal` flag (fewer than MARGINAL_S seconds at or above 3.4)."""
    from harness.workload import build_schedule
    pk = pk or REGISTERED
    cfg = gate._burst_cfg(K4["base_cfg"])
    dur = K4["duration_s"]
    out = {}
    for i, (ss, cs) in enumerate(zip(pk["surge"], pk["calm"])):
        tc = float(pk["t_cross"][ss])
        for sd, arm in ((ss, "surge"), (cs, "calm")):
            s, _ = build_schedule({**cfg["workload"], "rate": REG["lam"], "duration_s": dur, "steps": None}, sd, None)
            tp = s.t_planned.to_numpy(float)
            g, x = gate.schedule_peak_series(cfg, REG["lam"], sd, dur)
            above = float((x >= K4["surge_level"]).sum() * gate.SURGE["grid_s"])
            out[sd] = {"arm": arm, "pair": i, "t_c": tc, "rate_run": _planned_rate(tp, gate.SURGE["start_s"], dur),
                       "rate_before": _planned_rate(tp, tc - 720.0, tc - 120.0),
                       "rate_around": _planned_rate(tp, tc - 120.0, tc + REG["delta_s"]),
                       "peak": float(x.max()), "s_at_or_above": above,
                       "marginal": bool(arm == "surge" and above < MARGINAL_S)}
    return out


def readouts(run: dict, t_cross: float | None) -> dict:
    """Reported, never judged: the planned and the sent 120 s peaks over [240, duration); the 120 s new-session
    (turn 0) planned rate at t_cross (surge arm); the planned session rate and planned prompt tokens per second over
    [240, duration); and `gate.surge_run`'s peak_tip (the highest 120 s planned rate in the last 180 s of warm time
    before a tip; item 7)."""
    from analysis.margin import is_burst_request
    q = run["requests"]
    q = q[~is_burst_request(run)]
    dur = float(run["cfg"]["workload"]["duration_s"])
    g = np.arange(gate.SURGE["start_s"], dur, gate.SURGE["grid_s"])
    w = gate.SURGE["window_s"]
    planned = gate.rolling_rate(q.t_planned.to_numpy(float), g, w)
    sent = gate.rolling_rate(q.t_sent.dropna().to_numpy(float), g, w)
    t0 = q[q.turn == 0].t_planned.to_numpy(float)
    m = (q.t_planned >= gate.SURGE["start_s"]) & (q.t_planned < dur)
    span = dur - gate.SURGE["start_s"]
    mx = lambda a: float(a.max()) if len(a) else math.nan
    try:
        pt = gate.surge_run(run)["peak_tip"]
    except Exception as e:  # reported only; a readout never stops the table
        pt = f"error: {type(e).__name__}"
    return {"planned_peak": mx(planned), "sent_peak": mx(sent),
            "new_session_rate_at_cross": (None if t_cross is None else
                                          float(gate.rolling_rate(t0, np.array([t_cross]), w)[0])),
            "planned_rate": float(m.sum() / span) if span > 0 else math.nan,
            "prompt_tokens_per_s": float(q.prompt_tokens_planned[m].sum() / span) if span > 0 else math.nan,
            "peak_tip": pt}


def _sealed(seeds) -> list:
    from analysis import withheld
    from analysis.load import RUNS
    return [p.name for sd in seeds for p in sorted(RUNS.glob(run_pattern(sd))) if withheld.sealed_by(p.name)]


def table() -> dict:
    """K4's registered computation, once after the ten runs: `AFTERSHOCK_UNSEAL=K4 python -m analysis.k4 table`. The
    seeds and crossings are REGISTERED's, and `pick()` must give them again (rule F round 1, item 5). Per seed the counted
    run is the earliest valid one (later valid ones listed); its resolved config must equal `resolved(seed, arm)`.
    Refuses while any K4 folder is sealed."""
    from analysis import d1, withheld
    from analysis.load import find_runs
    pk = REGISTERED
    again = pick()
    if again["surge"] != pk["surge"] or again["calm"] != pk["calm"] or \
            any(again["t_cross"][sd] != pk["t_cross"][sd] for sd in pk["surge"]):
        raise AssertionError(f"pick() no longer gives the registered seeds: {again}")
    seeds = pk["surge"] + pk["calm"]
    sealed = _sealed(seeds)
    if sealed:
        raise withheld.WithheldError(f"{len(sealed)} K4 folder(s) sealed; the registered computation sets "
                                     f"AFTERSHOCK_UNSEAL=K4")
    load = schedule_load(pk)
    rows = {"surge": [], "calm": []}
    not_run = []
    for arm in ("surge", "calm"):
        for sd in pk[arm]:
            runs = sorted(find_runs(run_pattern(sd)), key=lambda r: r["dir"].name)
            if not runs:
                not_run.append(sd)
                continue
            r = runs[0]
            got = {k: v for k, v in r["cfg"].items() if k not in ("run_id", "source_config")}
            diff = d1._diff_keys(got, resolved(sd, arm))
            if diff:
                raise ValueError(f"{r['dir'].name}: config differs from K4's at {diff}")
            tc = pk["t_cross"].get(sd) if arm == "surge" else None
            lab = label(r, tc)
            rows[arm].append({"seed": sd, "run": r["dir"].name, "later_valid": [x["dir"].name for x in runs[1:]],
                              "t_cross": tc, **lab, "readouts": readouts(r, tc), "schedule": load[sd]})
    marginal = {sd for sd in pk["surge"] if load[sd]["marginal"]}
    return {"pick": pk, "rows": rows, "not_run": not_run, "marginal_surge_seeds": sorted(marginal),
            "outcome": outcome(rows["surge"], rows["calm"], marginal)}


def investigation() -> dict:
    """Everything above in one call: `python -m analysis.k4 investigation`."""
    rows = natural_record()
    return {"natural": rows, "crossings": crossings(rows),
            "surgeless": {lv: surgeless(rows, lv) for lv in (3.1, 3.2)},
            "design_scan": design_scan(), "design_power": design_power(), "predicted": predicted(),
            "cp_upper_0_of_n": {n: cp_upper(0, n, K4["conf"]) for n in (4, 5, 6)}}


if __name__ == "__main__":
    import json
    import sys

    dump = lambda x: print(json.dumps(x, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    cmd = sys.argv[1] if len(sys.argv) == 2 else None
    if cmd == "investigation":
        dump(investigation())
    elif cmd == "feasibility":
        dump(feasibility())
    elif cmd == "configs":
        print("\n".join(str(p) for p in write_configs(pick())))
    elif cmd == "table":
        dump(table())
    elif cmd == "schedule_load":
        dump(schedule_load())
    elif cmd == "predicted":
        dump({"predicted_r1": predicted_r1()})
    else:
        raise SystemExit("usage: python -m analysis.k4 investigation | feasibility | configs | table | schedule_load")
