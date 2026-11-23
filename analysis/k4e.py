"""K4e (docs/drafts/k4e_draft1.md): K4's edge. K4 (decisions.md 2026-11-17 [RESULT] K4 …) tipped 4 of 5 schedules with a
120 s planned peak >= 3.4 req/s and 0 of 5 with none >= 3.1, at λ = 2.2; the band 3.1-3.4 is untested, and its one warm
surge run had the lowest surge peak (3.433). K4e runs no-burst schedules whose highest 120 s planned peak lies inside
the band, in two strata, to locate where a surge starts to end the warm state.

Everything per run is K4's: REG-14's control workload at λ = 2.2, 2700 s, no trigger (`k4.K4["base_cfg"]`), K4's
labels (`k4.label`) and readouts (`k4.readouts`). Only the schedules differ.

The pick (`classify`, `pick`), fixed in this file before any schedule in its seed range was scanned: in seed order over
K4E["seeds"] (no config uses them), a schedule is in stratum `lo` or `hi` when, on its 120 s planned session-rate series
over [240, 2700) (`gate.schedule_peak_series`, K4's grid):
  - no value >= 3.1 before 1200 s;
  - its highest value is reached in [1200, 1800] s (K4's surge window), so a tip has time to persist into the end
    window [2100, 2700);
  - that highest value is in [3.1, 3.25) (`lo`) or [3.25, 3.4) (`hi`).
The first K4E["n"] seeds of each stratum are taken. If either stratum has fewer, the range K4E["seeds_more"] is scanned
next, in order, by the same rule (rule G: fixed before the scan).
"""
from __future__ import annotations

import itertools
import math

import numpy as np

from analysis import gate
from analysis import k4 as K

K4E = {
    "lam": 2.2,
    "lo": (3.1, 3.25), "hi": (3.25, 3.4),       # strata of the highest 120 s planned peak, req/s
    "pre_level": 3.1,                           # nothing at or above this before window[0]
    "window": (1200.0, 1800.0),                 # K4's surge window: the time of the highest peak
    "seeds": tuple(range(600, 1000)),           # never run, never scanned at λ = 2.2 (K4: 100-599 judged,
                                                # 1000-1299 design)
    "seeds_more": tuple(range(1300, 2000)),     # only if a stratum is short in `seeds`
    "n": 5,                                     # per stratum
}


def classify(g: np.ndarray, x: np.ndarray, p: dict = K4E) -> dict:
    """The stratum of one schedule's 120 s planned-rate series (None if neither), with the time of its highest value
    and the first crossing of pre_level."""
    if not len(x):
        return {"stratum": None, "peak": math.nan, "t_peak": None, "t_cross": None}
    i = int(np.argmax(x))
    peak, t_peak = float(x[i]), float(g[i])
    c = np.flatnonzero(x >= p["pre_level"])
    t_cross = float(g[c[0]]) if len(c) else None
    w0, w1 = p["window"]
    ok = t_cross is not None and t_cross >= w0 and w0 <= t_peak <= w1
    st = None
    if ok:
        for name in ("lo", "hi"):
            a, b = p[name]
            if a <= peak < b:
                st = name
    return {"stratum": st, "peak": peak, "t_peak": t_peak, "t_cross": t_cross}


def scan(seeds, p: dict = K4E) -> list[dict]:
    cfg = gate._burst_cfg(K.K4["base_cfg"])
    out = []
    for sd in seeds:
        g, x = gate.schedule_peak_series(cfg, p["lam"], sd, K.K4["duration_s"])
        out.append({"seed": sd, **classify(g, x, p)})
    return out


def pick(p: dict = K4E) -> dict:
    """The rule above: the first n seeds of each stratum in `seeds`, then in `seeds_more` only if a stratum is short;
    the run order alternates lo, hi (so host drift does not fall on one stratum)."""
    rows = scan(p["seeds"], p)
    take = lambda rr, s: [r for r in rr if r["stratum"] == s][:p["n"]]
    lo, hi = take(rows, "lo"), take(rows, "hi")
    scanned_more = False
    if len(lo) < p["n"] or len(hi) < p["n"]:
        more = scan(p["seeds_more"], p)
        rows, scanned_more = rows + more, True
        lo, hi = take(rows, "lo"), take(rows, "hi")
    order = [r["seed"] for pair in itertools.zip_longest(lo, hi) for r in pair if r is not None]
    return {"n_scanned": len(rows), "scanned_more": scanned_more,
            "n_lo": sum(r["stratum"] == "lo" for r in rows), "n_hi": sum(r["stratum"] == "hi" for r in rows),
            "lo": [r["seed"] for r in lo], "hi": [r["seed"] for r in hi],
            "rows": {r["seed"]: r for r in lo + hi}, "order": order,
            "passes": len(lo) >= p["n"] and len(hi) >= p["n"]}


# ---------------------------------------------------------------- the registration (draft 3: rule F rounds 1 and 2 applied)
#
# Rule F r1 (docs/probe/k4e_ruleF/review_r1.md) showed that five per stratum, with strata set on the 5 s grid and
# confounded with sessions per 120 s, cannot place an edge inside [3.1, 3.4); the question is therefore the band as a
# whole (stated assumption for pooling the two strata, rule H: the band is judged as one cell; the strata are readouts),
# tips are K4's (persistent, `tipped`) inside K4's timing window, and one K4 surge seed is rerun as a positive control.
# Rule F r2 (docs/probe/k4e_ruleF/review_r2.md): an onset before the window is `surgeless` whatever its time (N1,
# N2); persistent tips outside the window block `band_warm` (N1); K4's calm seed with the highest peak is rerun as
# a calm control (N3).

# `pick()` run once from the rule committed at b2f7510 (PROBE(docs/probe/k4e_draft/pick_b2f7510_20261117T2033Z.txt)):
# 400 seeds scanned, 19 `lo` and 7 `hi`; the first five of each. t_cross: the first 120 s value >= 3.1 (grid).
REGISTERED = {
    "lo": [630, 650, 663, 669, 694], "hi": [609, 711, 772, 813, 914],
    "t_cross": {630: 1660.0, 650: 1640.0, 663: 1245.0, 669: 1310.0, 694: 1560.0,
                609: 1285.0, 711: 1495.0, 772: 1405.0, 813: 1510.0, 914: 1645.0},
    "peak": {630: 3.1083333333333334, 650: 3.1416666666666666, 663: 3.225, 669: 3.1666666666666665, 694: 3.1,
             609: 3.275, 711: 3.3666666666666667, 772: 3.283333333333333, 813: 3.3916666666666666,
             914: 3.3666666666666667},
    # the positive control: K4's surge seed 137 (crossing of 3.4 at 1365 s, tipped 68 s later in K4,
    # RUN(20261117-070951_k4_l1pl350cl_r2.2_s137, k4.table)), rerun under a K4e name, judged against K4's crossing
    "positive": 137, "positive_t_cross": 1365.0,
    # the calm control: K4's calm seed with the highest 120 s planned peak, 103 (3.017 at 535 s; warm in K4,
    # RUN(20261117-080145_k4_l1pl350cl_r2.2_s103, k4.table)), rerun under a K4e name; it passes if it stays `warm`
    "calm": 103,
    "order": [630, 609, 650, 711, 663, 137, 772, 669, 813, 103, 694, 914],
}
WINDOW = {"lead_s": 60.0, "delta_s": 212.0}   # a tip counts with onset in [t_cross - 60, t_cross + 212]: K4's 212 s
                                              # band, opened 60 s early because the 120 s rate's arrivals sit in its
                                              # last 60 s (rule F r1 B1)
OUTCOME = {
    "most": 8,       # counted runs in the band (of 10), else too_few
    "warm_max": 1,   # band_warm: at most this many tips
    "tips_min": 4,   # band_tips: at least this many tips
}
CLASSES = ("tip", "warm", "recovered", "late", "surgeless", "bad_baseline", "unclear")
COUNTED = ("tip", "warm", "recovered")   # n; `recovered` (an onset in the window that did not stay cold) is not a tip


def config_name(seed: int) -> str:
    return f"k4e_l1pl350cl_r2.2_s{seed}"


def run_pattern(seed: int) -> str:
    return f"*_{config_name(seed)}"


def _stratum(seed: int) -> str:
    return "lo" if seed in REGISTERED["lo"] else "hi" if seed in REGISTERED["hi"] else \
        "positive" if seed == REGISTERED["positive"] else "calm" if seed == REGISTERED["calm"] else None


def config_text(seed: int, stratum: str) -> str:
    if stratum not in ("lo", "hi", "positive", "calm") or _stratum(seed) != stratum:
        raise ValueError((seed, stratum))
    what = (f"{stratum} stratum, highest 120 s planned peak {REGISTERED['peak'][seed]:.3f}, picked by `k4e.pick`\n"
            f"# from the schedule alone" if stratum in ("lo", "hi") else
            "the positive control: K4's surge seed 137 rerun under a K4e name" if stratum == "positive" else
            "the calm control: K4's calm seed 103 rerun under a K4e name")
    return (f"# K4e (docs/decisions.md [REG] K4e (K4's edge)): K4's workload (REG-14's control) at λ = {K4E['lam']:g},\n"
            f"# seed {seed} ({what}), {K.K4['duration_s']:g} s, no trigger. Runs only after `[DECISION] ... K4e starts`.\n"
            f"extends: {K.K4['base_cfg']}.yaml\n"
            f"name: {config_name(seed)}\n"
            f"seed: {seed}\n"
            f"workload:\n"
            f"  duration_s: {K.K4['duration_s']:g}\n")


def _all_seeds() -> list[tuple[int, str]]:
    return [(sd, "lo") for sd in REGISTERED["lo"]] + [(sd, "hi") for sd in REGISTERED["hi"]] + \
        [(REGISTERED["positive"], "positive"), (REGISTERED["calm"], "calm")]


def write_configs(configs_dir=None) -> list:
    d = configs_dir or K._configs()
    items = _all_seeds()
    paths = [d / f"{config_name(sd)}.yaml" for sd, _ in items]
    if any(q.exists() for q in paths):
        raise FileExistsError([str(q) for q in paths if q.exists()])
    for q, (sd, st) in zip(paths, items):
        q.write_text(config_text(sd, st))
    return paths


def resolved(seed: int, stratum: str) -> dict:
    import yaml
    from harness.config import _merge, load_config, with_defaults
    from harness.run import expand
    raw = yaml.safe_load(config_text(seed, stratum))
    parent = raw.pop("extends")
    return expand(with_defaults(_merge(load_config(K._configs() / parent), raw)))[0]


def counted(lab: dict, t_cross: float, w: dict = WINDOW) -> str:
    """The class of one run from K4's label (`k4.label`). An onset before t_cross - lead (or `tipped_early`) is
    `surgeless`: by the pick no 120 s value >= 3.1 precedes t_cross, so it is an onset with no surge, K4's refuting
    case (rule F r2 N1, N2: no 1200 s floor, so seed 663's window opens at 1185 s). An onset in
    [t_cross - lead, t_cross + delta]: `tip` if it stayed cold (`tipped`), `recovered` if it recovered, `unclear` if
    undefined; after the window, `late`. No onset: `warm`, `bad_baseline`, `unclear`. Counted (n): tip, warm,
    recovered."""
    l, on = lab["label"], lab["onset"]
    if l == "tipped_early":
        return "surgeless"
    if l in ("tipped", "tipped_recovered", "tipped_unclear"):
        if on < t_cross - w["lead_s"]:
            return "surgeless"
        if on > t_cross + w["delta_s"]:
            return "late"
        return {"tipped": "tip", "tipped_recovered": "recovered", "tipped_unclear": "unclear"}[l]
    return l if l in ("warm", "bad_baseline", "unclear") else "unclear"


def _k_of_n(rr: list[dict]) -> dict:
    from analysis.stats import cp_upper
    k = sum(r["class"] == "tip" for r in rr)
    n = sum(r["class"] in COUNTED for r in rr)
    return {"k": k, "n": n, "upper": cp_upper(k, n) if n else math.nan,
            "lower": 1 - cp_upper(n - k, n) if n else math.nan,
            "not_counted": [(r.get("seed"), r["class"]) for r in rr if r["class"] not in COUNTED]}


def outcome(rows: dict[str, list[dict]], positive: dict | None, calm: dict | None = None,
            p: dict = OUTCOME) -> dict:
    """rows: {"lo": [...], "hi": [...]}, each row with "class" and "label" (K4's); positive, calm: the controls' rows
    (or None). The band (both strata as one cell, the stated assumption; its bound is on the mean tip probability over
    these ten schedules): too_few (n < most) -> band_warm (in-window tips plus persistent `tipped` runs outside the
    window, `surgeless` or `late`, <= warm_max: rule F r2 N1) -> band_tips (in-window tips >= tips_min) -> mixed.
    `positive_ok`: 137 is a `tip`; `calm_ok`: 103 is `warm`; each True, False or None (not run), with its class.
    With either False or None the band's outcome is not read as a contrast with the host at K4's time, and goes to the
    author as such. Readouts: k of n per stratum, the tips' peaks, `surgeless` runs, not-counted runs."""
    rr = rows["lo"] + rows["hi"]
    band = _k_of_n(rr)
    k, n = band["k"], band["n"]
    outside = [r.get("seed") for r in rr if r["class"] in ("surgeless", "late") and r.get("label") == "tipped"]
    if n < p["most"]:
        d = "too_few"
    elif k + len(outside) <= p["warm_max"]:
        d = "band_warm"
    elif k >= p["tips_min"]:
        d = "band_tips"
    else:
        d = "mixed"
    return {"decision": d, "band": band, "lo": _k_of_n(rows["lo"]), "hi": _k_of_n(rows["hi"]),
            "positive_ok": None if positive is None else positive["class"] == "tip",
            "positive_class": None if positive is None else positive["class"],
            "calm_ok": None if calm is None else calm["class"] == "warm",
            "calm_class": None if calm is None else calm["class"],
            "recovered": [r.get("seed") for r in rr if r["class"] == "recovered"],
            "tipped_outside_window": outside,
            "tip_peaks": sorted(r.get("peak") for r in rr if r["class"] == "tip"),
            "surgeless": [r.get("seed") for r in rr if r["class"] == "surgeless"]}


def predicted(p_lo: float = 0.10, p_hi: float = 0.35, n: int = 5, p: dict = OUTCOME) -> dict:
    """DERIVED from per-run tip probabilities (BELIEF), every run counted (so too_few is 0 here): exact P(outcome)."""
    from math import comb
    b = lambda k, q: comb(n, k) * q ** k * (1 - q) ** (n - k)
    out = {"band_warm": 0.0, "mixed": 0.0, "band_tips": 0.0}
    for kl in range(n + 1):
        for kh in range(n + 1):
            k, pr = kl + kh, b(kl, p_lo) * b(kh, p_hi)
            out["band_warm" if k <= p["warm_max"] else "band_tips" if k >= p["tips_min"] else "mixed"] += pr
    return out


def exact_peak(seed: int) -> dict:
    """Schedule only (rule F r1, advisory): the highest count of planned session requests in any 120 s window (exact,
    not on the 5 s grid) / 120, the new sessions (turn 0) in the 120 s grid window at the grid peak, the seconds at or
    above 3.1, 3.25 and 3.4 on the grid, the planned rate around t_cross ([t_c - 120, t_c + 212)) and over the end
    window [2100, 2700)."""
    from harness.workload import build_schedule
    cfg = gate._burst_cfg(K.K4["base_cfg"])
    dur, w = K.K4["duration_s"], gate.SURGE["window_s"]
    s, _ = build_schedule({**cfg["workload"], "rate": K4E["lam"], "duration_s": dur, "steps": None}, seed, None)
    tp = np.sort(s.t_planned.to_numpy(float))
    tp = tp[tp >= gate.SURGE["start_s"] - w]
    j, best = 0, 0
    for i in range(len(tp)):                       # windows (tp[i] - 120, tp[i]]
        while tp[j] <= tp[i] - w:
            j += 1
        best = max(best, i - j + 1)
    g, x = gate.schedule_peak_series(cfg, K4E["lam"], seed, dur)
    t_pk = float(g[int(np.argmax(x))])
    t0 = s[s.turn == 0].t_planned.to_numpy(float)
    tc = REGISTERED["t_cross"].get(seed, REGISTERED["positive_t_cross"] if seed == REGISTERED["positive"] else None)
    # the calm control has no crossing: rate_around is NaN
    return {"exact_peak": best / w, "grid_peak": float(x.max()), "t_peak": t_pk,
            "new_sessions_at_peak": int(((t0 > t_pk - w) & (t0 <= t_pk)).sum()),
            "s_at_or_above": {lv: float((x >= lv).sum() * gate.SURGE["grid_s"]) for lv in (3.1, 3.25, 3.4)},
            "rate_around": K._planned_rate(tp, tc - 120.0, tc + K.REG["delta_s"]) if tc is not None else math.nan,
            "rate_end_window": K._planned_rate(tp, *K.REG["end"])}


def _sealed() -> list:
    from analysis import withheld
    from analysis.load import RUNS
    return [q.name for sd, _ in _all_seeds() for q in sorted(RUNS.glob(run_pattern(sd))) if withheld.sealed_by(q.name)]


def _readout(fn, *a):
    try:
        return fn(*a)
    except Exception as e:   # noqa: BLE001 — reported, never decisive
        return {"error": f"{type(e).__name__}: {e}"}


def table() -> dict:
    """K4e's registered computation, once after the twelve runs: `AFTERSHOCK_UNSEAL=K4e uv run python -m analysis.k4e
    table`. `pick()` must give REGISTERED again (seeds, t_cross and peaks); it refuses while any K4e folder is sealed.
    Per seed the counted run is the earliest valid one (later valid ones listed); its config must equal
    `resolved(seed, stratum)`. Labels, classes and the outcome first; readouts after, each guarded. Closed-loop turn
    times are nominal: the planned rate holds only while the server keeps up, so the sent 120 s peak (`k4.readouts`)
    is reported beside the planned one."""
    from analysis import d1, withheld
    from analysis.load import find_runs
    pk = REGISTERED
    again = pick()
    if again["lo"] != pk["lo"] or again["hi"] != pk["hi"] or any(
            again["rows"][sd]["t_cross"] != pk["t_cross"][sd] or again["rows"][sd]["peak"] != pk["peak"][sd]
            for sd in pk["lo"] + pk["hi"]) or pk["positive_t_cross"] != K.REGISTERED["t_cross"][pk["positive"]] \
            or pk["calm"] not in K.REGISTERED["calm"]:
        raise AssertionError(f"pick() no longer gives the registered seeds: {again['lo']}, {again['hi']}")
    sealed = _sealed()
    if sealed:
        raise withheld.WithheldError(f"{len(sealed)} K4e folder(s) sealed; set AFTERSHOCK_UNSEAL=K4e")
    rows, runs_by_seed, not_run, ctl = {"lo": [], "hi": []}, {}, [], {"positive": None, "calm": None}
    for sd, st in _all_seeds():
        rr = sorted(find_runs(run_pattern(sd)), key=lambda r: r["dir"].name)
        if not rr:
            not_run.append(sd)
            continue
        r = rr[0]
        got = {k: v for k, v in r["cfg"].items() if k not in ("run_id", "source_config")}
        diff = d1._diff_keys(got, resolved(sd, st))
        if diff:
            raise ValueError(f"{r['dir'].name}: config differs from K4e's at {diff}")
        tc = pk["positive_t_cross"] if st == "positive" else None if st == "calm" else pk["t_cross"][sd]
        lab = K.label(r, tc)
        row = {"seed": sd, "stratum": st, "run": r["dir"].name, "later_valid": [x["dir"].name for x in rr[1:]],
               "t_cross": tc, "peak": pk["peak"].get(sd), **lab,
               "class": lab["label"] if st == "calm" else counted(lab, tc)}
        runs_by_seed[sd] = r
        if st in ctl:
            ctl[st] = row
        else:
            rows[st].append(row)
    out = outcome(rows, ctl["positive"], ctl["calm"])
    for row in rows["lo"] + rows["hi"] + [c for c in ctl.values() if c]:
        sd = row["seed"]
        row["readouts"] = _readout(K.readouts, runs_by_seed[sd], row["t_cross"])
        row["schedule"] = _readout(exact_peak, sd)
        row["onset_minus_cross"] = (row["onset"] - row["t_cross"]) if None not in (row["onset"], row["t_cross"]) \
            else None
    return {"pick": pk, "rows": rows, "positive": ctl["positive"], "calm": ctl["calm"], "not_run": not_run,
            "outcome": out}


if __name__ == "__main__":
    import sys
    print({"pick": pick, "table": table, "predicted": predicted}[sys.argv[1]]())
