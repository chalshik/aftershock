"""REG-30 (Z1 of the 2026-11-21 panel, paper repo `reviews/panel_20261121/colleague.md:200-204`): when does later-turn
reuse come back during the drop of the five long-watch burst runs, and how many sessions are in service then? A
retrospective reading of runs whose summaries are reported (REG-24's seed 43, REG-27's seeds 61, 68, 72, 79). Never
writes into runs/, and refuses to compute until the [REG] REG-30 heading is in docs/decisions.md.

Question (rule F round 1, B4). Not loop against owed backlog: both readings need the population link. The reading
asks whether reuse returns as the sessions in service fall to about what the pool holds (the population link,
Huang and Liu's N* = C / L-bar; 96,656 / 3968 = 24.4 to 96,656 / 3072 = 31.5 sessions), or stays low for a while
after the population is already below that (hysteresis, as Lu et al. show in simulation).

Per run, all times absolute (s), phases from `confirm.phases` (the drop is [watch end, drop end)):
  base     `k1.reuse` (Σ min(cached, previous prompt) / Σ previous prompt, ok later turns by first token) over the
           baseline (300, 900) s. NaN gives `bad_base`.
  level    base - the config's `verdict.recovered_hit_drop` (K1's and REG-16's `recovered` rule, 0.06)
  onset    (rule F r1 B3) the first t on the bin_s grid from the drop's start with `k1.reuse` pooled over
           [t, t + win_s) and over [t + win_s, t + 2 win_s) both at or above level (both windows inside the drop)
  onset60  readout and stability check: the first 60 s bin from which hold_bins consecutive 60 s bins reach level
  drained  `confirm.drained_at`: the total line back at its level at the trigger (REG-24's drain, K3's value)
  lead     drained - onset;  n_onset = `k1.sessions_in_service` over [onset, onset + bin_s)
  t_low    (rule F r1 B1, B2) the start of the unbroken run of bins ending just before the onset's bin in which every
           bin's sessions in service are at most low_bound (24.4 = pool / turn 8's prompt, the lowest N*); NaN if
           the bin before the onset is above it
Label, in this order: `bad_base`; `no_onset`; `no_drain` (the line never drained in the drop); `held_low` when
onset - t_low >= lead_min_s and `k1.reuse` pooled over [t_low, onset) is below level (the population was at or below
the lowest N* for that long while reuse stayed low: hysteresis); `before_drain` when lead >= lead_min_s and n_onset
is in in_service (the colleague's prediction); `early_other` when lead >= lead_min_s and n_onset is outside it;
`near_drain` when |lead| < lead_min_s; `after_drain` when lead <= -lead_min_s.
Stability (B3): the label is recomputed with onset60 in place of onset; when the two differ the run is
`hold_sensitive` (its two labels are reported).
Outcome over the runs: `population_link` when before_drain >= k_support; `hysteresis` when held_low >= k_support;
`drain_bound` when near_drain >= k_support; `early_population` when early_other >= k_support; `mixed` otherwise;
`too_few` when fewer than min_runs runs load. A pinned run that does not load is listed and k_support stays 4, so
with four runs every one must agree. k of n with one-sided 95 % Clopper-Pearson bounds (rule H); seed 43, which
chose the design, is also left out in a second count.
Readouts, never judged: the sessions-in-service series of the drop; the label with low_bound at 20, 31.5 and 40;
the owed backlog, waiting and KV at the onset; reuse and sessions in service in the drain's bin; the same-seed
control's sessions in service at the onset and its median over the drop (rule F r1, item 7: the warm state at λ = 1).

Z2 (`z2`, a readout, not registered and never judged). REG-21's r of REG-19's five stuck runs against a second
reference: E1's four saturated closed-loop cache-off runs (λ_sat 2.5, rates 2.5-4.0; decisions.md 2026-11-14 [RESULT]
E1) in place of REG-8's open-loop ones. Not registered because its value follows from numbers already reported (REG-21
batch 2's r and E1's U per run), so no threshold can be set blind to it (rule A5).
"""
from __future__ import annotations

import math

import numpy as np

from analysis import confirm, k1, withheld
from analysis import gate as G
from analysis.load import _level, find_runs
from analysis.stats import cp_upper

REG30 = {
    "heading": "[REG] REG-30",
    "runs": {43: "20261116-182704_confirm_l1pl350cl_r2.0_s43", 61: "20261118-015033_confirm_l1pl350cl_r2.0_s61",
             68: "20261119-104834_confirm_l1pl350cl_r2.0_s68", 72: "20261119-195054_confirm_l1pl350cl_r2.0_s72",
             79: "20261120-045201_confirm_l1pl350cl_r2.0_s79"},
    "controls": {43: "20261116-135154_stage2_control_l1pl350cl_r2.0_long_s43",
                 61: "20261117-211937_stage2_control_l1pl350cl_r2.0_long_s61",
                 68: "20261118-062104_stage2_control_l1pl350cl_r2.0_long_s68",
                 72: "20261119-151949_stage2_control_l1pl350cl_r2.0_long_s72",
                 79: "20261120-002152_stage2_control_l1pl350cl_r2.0_long_s79"},
    "design_seed": 43,
    "bin_s": 60.0,                  # this registration's grid (k1's sessions-in-service grid is 10 s)
    "win_s": 300.0,                 # pooled onset windows: REG-15's 300 s watch window (confirm.REG24["timeline_s"])
    "hold_bins": 5,                 # onset60: 5 x 60 s = the same 300 s
    "lead_min_s": 300.0,            # colleague.md:202, "at least 300 s before the line empties"
    "in_service": (20.0, 40.0),     # colleague.md:202, "20 to 40 sessions are in service"
    "low_bound": 96656 / 3968,      # 24.4: the lowest N* (pool / turn 8's prompt, PROTOCOL L1), rule F r1 B1
    "low_bounds_reported": (20.0, 96656 / 3072, 40.0),
    "k_support": 4,                 # colleague.md:202, "in at least 4 of 5 runs"
    "min_runs": 3,
}
LABELS = ("bad_base", "no_onset", "no_drain", "held_low", "before_drain", "early_other", "near_drain", "after_drain",
          "hold_sensitive")


def onset_pooled(reuse_of, a: float, b: float, level: float, step: float, win: float) -> float:
    """First t in [a, b - 2 win] on the step grid with reuse pooled over [t, t + win) and [t + win, t + 2 win) both
    at or above level; NaN if none. `reuse_of(x, y)` gives the pooled reuse over [x, y)."""
    for t in np.arange(a, b - 2 * win + 1e-9, step):
        r1, r2 = reuse_of(t, t + win), reuse_of(t + win, t + 2 * win)
        if not (math.isnan(r1) or math.isnan(r2)) and r1 >= level and r2 >= level:
            return float(t)
    return math.nan


def onset_of(series: list[tuple[float, float]], level: float, hold: int) -> float:
    """The start time of the first bin from which `hold` consecutive bins all reach `level`; NaN if none."""
    ok = [(not math.isnan(r)) and r >= level for _, r in series]
    for i in range(len(series) - hold + 1):
        if all(ok[i:i + hold]):
            return float(series[i][0])
    return math.nan


def low_run_start(n_series: list[tuple[float, float]], onset: float, bound: float) -> float:
    """The start of the unbroken run of bins ending just before the bin that starts at `onset` in which every bin's
    sessions in service are at most `bound` (a NaN bin breaks it); NaN if the bin before the onset is above it."""
    before = [(t, n) for t, n in n_series if t < onset]
    start = math.nan
    for t, n in reversed(before):
        if math.isnan(n) or n > bound:
            break
        start = t
    return float(start)


def label(lead: float, n_onset: float, onset: float, drained: float, t_low: float = math.nan,
          reuse_low: float = math.nan, level: float = math.nan, p: dict = REG30) -> str:
    if math.isnan(level):
        return "bad_base"
    if math.isnan(onset):
        return "no_onset"
    if math.isnan(drained):
        return "no_drain"
    if (not math.isnan(t_low) and onset - t_low >= p["lead_min_s"] and not math.isnan(reuse_low)
            and reuse_low < level):
        return "held_low"
    if lead >= p["lead_min_s"]:
        lo, hi = p["in_service"]
        return "before_drain" if lo <= n_onset <= hi else "early_other"
    return "after_drain" if lead <= -p["lead_min_s"] else "near_drain"


def _label_at(run, onset, t_d, n_series, level, bound, p):
    if math.isnan(onset) or math.isnan(level):
        return label(math.nan, math.nan, onset, t_d, level=level, p=p), {}
    lead = t_d - onset if not math.isnan(t_d) else math.nan
    n_on = k1.sessions_in_service(run, onset, onset + p["bin_s"])
    t_low = low_run_start(n_series, onset, bound)
    r_low = k1.reuse(run, t_low, onset) if not math.isnan(t_low) else math.nan
    return (label(lead, n_on, onset, t_d, t_low, r_low, level, p),
            {"lead": lead, "n_onset": n_on, "t_low": t_low, "reuse_low": r_low})


def rewarm(run: dict, p: dict = REG30, control: dict | None = None) -> dict:
    """One run's onset, drain, lead, sessions in service and label, with its stability check and readouts."""
    ph = confirm.phases(run["cfg"])
    we, de = ph["drop"]
    base = k1.reuse(run, *ph["baseline"])
    level = base - float(run["cfg"]["verdict"]["recovered_hit_drop"]) if not math.isnan(base) else math.nan
    edges = np.arange(we, de, p["bin_s"])
    n_series = [(float(t), k1.sessions_in_service(run, t, min(t + p["bin_s"], de))) for t in edges]
    owed0 = G.owed_backlog(run, float(run["cfg"]["trigger"]["at_s"]))
    t_d = confirm.drained_at(run, we, de, owed0)
    if math.isnan(level):
        t_on = t_60 = math.nan
        series = []
    else:
        t_on = onset_pooled(lambda x, y: k1.reuse(run, x, y), we, de, level, p["bin_s"], p["win_s"])
        series = [(float(t), k1.reuse(run, t, min(t + p["bin_s"], de))) for t in edges]
        t_60 = onset_of(series, level, p["hold_bins"])
    lab, d = _label_at(run, t_on, t_d, n_series, level, p["low_bound"], p)
    lab60, d60 = _label_at(run, t_60, t_d, n_series, level, p["low_bound"], p)
    rel = lambda t: t - we if not math.isnan(t) else math.nan
    out = {"run": run["dir"].name, "base_reuse": base, "level": level, "onset_s": rel(t_on), "onset60_s": rel(t_60),
           "drained_s": rel(t_d), "lead_s": d.get("lead", math.nan), "n_onset": d.get("n_onset", math.nan),
           "t_low_s": rel(d.get("t_low", math.nan)), "reuse_low": d.get("reuse_low", math.nan),
           "label_pooled": lab, "label_60": lab60, "label": lab if lab == lab60 else "hold_sensitive"}
    try:   # readouts never take the judged values with them
        rd = {"owed_at_trigger": owed0, "n_series": [(t - we, n) for t, n in n_series],
              "reuse_60": [(t - we, r) for t, r in series],
              "labels_at_low_bound": {round(b, 1): _label_at(run, t_on, t_d, n_series, level, b, p)[0]
                                      for b in p["low_bounds_reported"]}}
        if not math.isnan(t_on):
            lv = _level(run, t_on, t_on + p["bin_s"])
            rd.update(owed_at_onset=G.owed_backlog(run, t_on), waiting_at_onset=lv["q"], kv_at_onset=lv["kv"])
            if control is not None:
                cn = [k1.sessions_in_service(control, t, min(t + p["bin_s"], de)) for t in edges]
                rd.update(control_n_at_onset=k1.sessions_in_service(control, t_on, t_on + p["bin_s"]),
                          control_n_median_drop=float(np.nanmedian(cn)) if len(cn) else math.nan)
        if not math.isnan(t_d):
            rd.update(reuse_at_drain=k1.reuse(run, t_d, t_d + p["bin_s"]),
                      n_at_drain=k1.sessions_in_service(run, t_d, t_d + p["bin_s"]))
        out["readouts"] = rd
    except Exception as e:
        out["readouts"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


def judge(rows: list[dict], p: dict = REG30) -> dict:
    """The outcome over the runs that loaded, with k of n and one-sided 95 % bounds for the decisive labels."""
    n = len(rows)
    counts = {lab: sum(r["label"] == lab for r in rows) for lab in LABELS}
    if n < p["min_runs"]:
        return {"outcome": "too_few", "n": n, "counts": counts}
    k = p["k_support"]
    out = ("population_link" if counts["before_drain"] >= k else "hysteresis" if counts["held_low"] >= k
           else "drain_bound" if counts["near_drain"] >= k else "early_population" if counts["early_other"] >= k
           else "mixed")
    bounds = {lab: {"k": counts[lab], "n": n, "lower": 1 - cp_upper(n - counts[lab], n),
                    "upper": cp_upper(counts[lab], n)}
              for lab in ("before_drain", "held_low", "near_drain", "early_other")}
    four = [r for r in rows if r["seed"] != p["design_seed"]]
    return {"outcome": out, "n": n, "counts": counts, "bounds": bounds,
            "without_design_seed": {lab: sum(r["label"] == lab for r in four) for lab in LABELS} | {"n": len(four)}}


def table() -> dict:
    """REG-30's table, once, after its [REG] heading is in docs/decisions.md. A pinned run that does not load (or is
    not valid) is listed under `missing`; the outcome is then judged on the rest with k_support unchanged."""
    if not withheld.lifted(REG30["heading"]):
        raise RuntimeError("REG-30 is computed only after its [REG] heading is in docs/decisions.md")
    rows, missing = [], []
    for seed, rid in REG30["runs"].items():
        found = find_runs(rid)
        if len(found) != 1:
            missing.append(rid)
            continue
        ctl = find_runs(REG30["controls"][seed])
        rows.append(rewarm(found[0], control=ctl[0] if len(ctl) == 1 else None) | {"seed": seed})
    return {"rows": rows, "missing": missing, "judged": judge(rows)}


Z2 = {"ref_runs": ("20261114-025146_sweep_l1pl350cl_pc_off_r2.5", "20261114-031247_sweep_l1pl350cl_pc_off_r3",
                   "20261114-033348_sweep_l1pl350cl_pc_off_r3.5", "20261114-035448_sweep_l1pl350cl_pc_off_r4")}


def z2() -> dict:
    """Z2's readout: `k3w.reference` on E1's saturated closed-loop runs, and `k3w.row` of REG-19's five burst runs
    against it and against REG-8's open-loop reference, side by side."""
    from analysis import k3w
    found = [find_runs(i) for i in Z2["ref_runs"]]
    if any(len(f) != 1 for f in found):
        raise ValueError(f"Z2 needs E1's four saturated runs, each valid: {Z2['ref_runs']}")
    ref_cl, ref_ol = k3w.reference([f[0] for f in found]), k3w.reference()
    rows = []
    for r in k3w.batch2_runs():
        a, b = k3w.row(r, ref_ol, with_readouts=False), k3w.row(r, ref_cl, with_readouts=False)
        rows.append({"run": a["run"], "U": a["U"], "r_open": a["r"], "r_closed": b["r"], "why_not": a["why_not"]})
    rc = np.array([x["r_closed"] for x in rows], float)
    ro = np.array([x["r_open"] for x in rows], float)
    return {"U_off_open": ref_ol["mean"], "U_off_closed": ref_cl["mean"], "closed_runs": ref_cl["runs"], "rows": rows,
            "median_r_open": float(np.median(ro)), "median_r_closed": float(np.median(rc)),
            "range_r_closed": (float(rc.min()), float(rc.max())),
            "max_abs_dev_closed": float(np.abs(rc - 1).max()), "max_abs_dev_open": float(np.abs(ro - 1).max())}


if __name__ == "__main__":
    import json
    import sys
    fns = {"reg30": table, "z2": z2}
    print(json.dumps(fns[sys.argv[1] if len(sys.argv) > 1 else "reg30"](), indent=1, default=str))
