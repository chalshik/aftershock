"""Reading PW (0 GPU h; retrospective, on runs already seen): REG-19's frozen whole-window per-seed rule
(`gate.reg19_seed_outcome`) applied to the ten burst seeds run at λ = 2.0 after that rule was written: REG-19's five
(15-19, 3000 s watch) and the five long-watch seeds of REG-24 (43) and REG-27 (61, 68, 72, 79; 4500 s watch, then a
drop to λ = 1 and a return). Draft 2: rule F round 1 applied (docs/drafts/pw_draft2.md).

Not blind: every run's registered outcome is known. The long-watch seeds were screened by REG-24's seed rule and seed
43 fixed the design REG-27 repeated. The count is a retrospective count beside P0's registered result, never a test
of P0. The only decision at stake is whether the papers carry a sentence with it (rule A1).

The view (fixed before any PW number): a long-watch pair is judged as REG-19's rule sees a REG-19 pair, cut at
T_CUT = 4080 s (burst end 1080 s + REG-19's 3000 s watch): cfg verdict.watch_s = 3000, workload.duration_s = 4080;
metrics and GPU samples at t >= 4080 dropped; requests planned at or after 4080 dropped. Requests planned before the
cut keep their records, so a turn first served after 4080 is served under continuing load in a long run and in a
draining tail in a REG-19 run. That difference is bounded, not corrected (`VARIANTS`): every request planned in the
verdict's two windows [2880, 4080) without a first token before the cut ("late") is set all fast, then all missed,
in the burst run and the control independently; a seed whose label changes under any of the four settings is
`cut_sensitive` and is not counted as a pass. Before any long-watch row, `reg19_check` computes REG-19's five seeds
both ways (this view and `gate.reg19_pair`) and stops on any difference beyond `TOL`.

Pooling (rule H): the ten seeds are not exchangeable (screened schedules, a design chosen on seed 43, a later kernel,
arrivals past 4080 s). Each group's k of n is reported; the pooled k of n and its bound are reported only as a bound
under an exchangeability assumption known not to hold, never as a bound on P0. REG-18's matrix rule (`most` = 3, set
for n = 5) is not applied to the pool.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from analysis import gate as G

T_CUT = 4080.0
GROUPS = {"reg19": (15, 16, 17, 18, 19), "long": (43, 61, 68, 72, 79)}
LONG = {"burst": "*_confirm_l1pl350cl_r2.0_s{seed}", "control": "*_stage2_control_l1pl350cl_r2.0_long_s{seed}"}
REG_HEADING = "[REG] PW (REG-19's whole-window rule on the ten post-rule seeds)"
CONF = 0.95
# resolved-config keys a long-watch run may differ in from REG-19's run of the same kind (checked, `check_pair`)
ALLOWED_DIFF = {"name", "run_id", "seed", "source_config", "verdict.watch_s", "workload.duration_s", "workload.steps"}
TEMPLATE_SEED = 15
# REG-19-check tolerances: labels, booleans and counts exact; rates and shares within these (the cut drops the tail
# scrapes `reg19_pair` reads at b = 4080 and in `kv_onset`)
TOL = {"rate_rel": 0.01, "share_abs": 0.01, "time_abs": 1.0, "peak_abs": 1e-9}
LATE_WINDOW = (T_CUT - 1200.0, T_CUT)     # the verdict's end window [3480, 4080) and its previous [2880, 3480)
VARIANTS = (None, ("fast", "fast"), ("fast", "miss"), ("miss", "fast"), ("miss", "miss"))   # (burst, control)
# Prior, set before computing (BELIEF): 10 of 10 counted passes 0.80; one or more cut_sensitive long seeds 0.12; a
# long seed failing the rule outright 0.05; the REG-19 check stopping PW 0.03.
PREDICTED = {"10_of_10": 0.80, "cut_sensitive": 0.12, "fails": 0.05, "check_stops": 0.03}


def _flat(d: dict, p: str = "") -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, dict):
            out.update(_flat(v, p + k + "."))
        else:
            out[p + k] = v
    return out


def check_pair(control: dict, burst: dict, template: dict | None = None) -> None:
    """Refuses a pair the view does not fit: configs other than REG-19's beyond ALLOWED_DIFF, a rate step before
    the cut, unequal durations, a pooled connection, a release rule other than closed loop, or a session cap."""
    from analysis.load import find_runs
    if control["cfg"]["workload"]["duration_s"] != burst["cfg"]["workload"]["duration_s"]:
        raise ValueError(f"{burst['dir'].name}: control and burst durations differ")
    for kind, r in (("control", control), ("burst", burst)):
        name = r["dir"].name
        c = r["cfg"]
        if c.get("loadgen_force_close") is not True:
            raise ValueError(f"{name}: not one connection per request")
        s = c["workload"].get("sessions") or {}
        if not G.closed_loop(r) or s.get("release", "closed") not in ("closed", None) or s.get("cap"):
            raise ValueError(f"{name}: not closed-loop turns without a cap")
        for st in c["workload"].get("steps") or []:
            if float(st["at_s"]) < T_CUT:
                raise ValueError(f"{name}: a rate step at {st['at_s']} s, before the cut")
        t = template[kind] if template else G.reg12_only(find_runs(G.reg19_spec(TEMPLATE_SEED)[kind]), kind)["cfg"]
        a, b = _flat(t), _flat(c)
        diff = {k for k in set(a) | set(b) if a.get(k) != b.get(k)} - ALLOWED_DIFF
        if diff:
            raise ValueError(f"{name}: config differs from REG-19's in {sorted(diff)}")


def truncate(run: dict, t_cut: float = T_CUT) -> dict:
    """The run as REG-19's rule sees a REG-19 run that ended its arrivals at t_cut (module docstring)."""
    out = dict(run)
    cfg = copy.deepcopy(run["cfg"])
    start = G.burst_end(run) if (cfg.get("trigger") or {}).get("kind") == "burst" else float(cfg["verdict"]["control_t_s"])
    if start + G.REG19["watch_s"] != t_cut:
        raise ValueError(f"{run['dir'].name}: watch start {start} + {G.REG19['watch_s']} is not {t_cut}")
    cfg["verdict"]["watch_s"] = float(G.REG19["watch_s"])
    cfg["workload"]["duration_s"] = float(t_cut)
    out["cfg"] = cfg
    out["metrics"] = run["metrics"][run["metrics"].t < t_cut].reset_index(drop=True)
    if len(run["gpu"]):
        out["gpu"] = run["gpu"][run["gpu"].t < t_cut].reset_index(drop=True)
    out["requests"] = run["requests"][run["requests"].t_planned < t_cut].reset_index(drop=True)
    return out


def late_mask(run: dict) -> np.ndarray:
    """Requests planned in LATE_WINDOW with no first token before the cut: their outcome was decided after it."""
    q = run["requests"]
    a, b = LATE_WINDOW
    return ((q.t_planned >= a) & (q.t_planned < b) & ~(q.t_first_token < T_CUT)).to_numpy()


def set_late(run: dict, how: str | None) -> dict:
    """`how` None: as recorded; 'fast': late requests ok with TTFT 0; 'miss': late requests unfinished."""
    if how is None:
        return run
    q = run["requests"].copy()
    m = late_mask({**run, "requests": q})
    if how == "fast":
        q.loc[m, "status"], q.loc[m, "ttft"] = "ok", 0.0
    elif how == "miss":
        q.loc[m, "status"], q.loc[m, "ttft"] = "unfinished", math.nan
    else:
        raise ValueError(how)
    return {**run, "requests": q}


def exposure(control: dict, t_cut: float = T_CUT) -> dict:
    """`gate.reg19_exposed` from the control's own planned session requests before t_cut (checked against the
    schedule on REG-19's seeds by `reg19_check`)."""
    q = control["requests"]
    s = q[~G.is_burst_request(control) & q.turn.notna() & (q.t_planned < t_cut)]
    grid = np.arange(G.SURGE["start_s"], t_cut, G.SURGE["grid_s"])
    x = G.rolling_rate(s.t_planned.to_numpy(float), grid, G.SURGE["window_s"])
    i = np.flatnonzero(x >= G.REG19["surge_tip"])
    return {"peak": float(x.max()), "t_peak": float(grid[x.argmax()]), "exposed": bool(len(i)),
            "t_first": float(grid[i[0]]) if len(i) else math.nan}


def _inputs(c: dict, b: dict, ex: dict) -> dict:
    """REG-19's per-seed inputs on a (truncated) pair, computed as `gate.reg19_pair` computes them, and the label."""
    res = G.reg8_gate_verdict(c, b)
    end = G.burst_end(b)
    cw = G.control_waiting(c, [(end + x, end + y) for x, y in G.reg19_spans()])
    sc = G.surge_run(c)
    cold = G.reg19_cold_side(b)
    em = G.reg19_emptyings(b)
    drain = em[0] if em else math.nan
    after = G.reg19_after_empty(b, drain)
    outcome = G.reg19_seed_outcome(res["control_verdict"], res["burst_verdict"], sc, cw, cold["holds"],
                                   bool(np.isfinite(drain)), ex, after["rewarmed"])
    return {"control_verdict": res["control_verdict"], "burst_verdict": res["burst_verdict"], "outcome": outcome,
            "c": cold["c"], "lam_p": cold["lam_p"], "cold_side": cold["holds"],
            "shares": [w["share_waiting"] for w in cw], "emptyings_s": em, "rewarmed": after["rewarmed"],
            "tipped": bool(sc["tipped"]), "exposed": ex["exposed"], "peak": ex["peak"]}


def _pair_runs(seed: int) -> tuple[dict, dict]:
    from analysis.load import find_runs
    spec = G.reg19_spec(seed) if seed in GROUPS["reg19"] else {k: v.format(seed=seed) for k, v in LONG.items()}
    runs = [G.reg12_only(find_runs(spec[k]), spec[k]) for k in ("control", "burst")]
    for r in runs:
        if r["cfg"]["seed"] != seed or float(r["cfg"]["workload"]["rate"]) != G.REG19["lam"]:
            raise ValueError(f"{r['dir'].name}: not seed {seed} at λ = {G.REG19['lam']}")
    return runs[0], runs[1]


def seed_row(seed: int, control: dict | None = None, burst: dict | None = None,
             template: dict | None = None) -> dict:
    """The as-recorded row and the four late-outcome variants; `cut_sensitive` if their labels differ; `counted`
    True only for a pass under every variant. Checks the pair first, also for runs passed in."""
    if control is None or burst is None:
        control, burst = _pair_runs(seed)
    check_pair(control, burst, template)
    c0, b0 = truncate(control), truncate(burst)
    ex = exposure(control)
    rows = {}
    for v in VARIANTS:
        bc = (set_late(b0, v[0]), set_late(c0, v[1])) if v else (b0, c0)
        rows["recorded" if v is None else f"burst_{v[0]}_control_{v[1]}"] = _inputs(bc[1], bc[0], ex)
    labels = {r["outcome"] for r in rows.values()}
    rec = rows["recorded"]
    return {"seed": seed, "group": "reg19" if seed in GROUPS["reg19"] else "long",
            "runs": (control["dir"].name, burst["dir"].name), **rec,
            "late": {"burst": int(late_mask(b0).sum()), "control": int(late_mask(c0).sum())},
            "variant_outcomes": {k: r["outcome"] for k, r in rows.items()},
            "cut_sensitive": len(labels) > 1,
            "counted": len(labels) == 1 and rec["outcome"] in G.PASS19}


def compare(pw_row: dict, reg19: dict) -> list[str]:
    """Differences between this view's inputs and `gate.reg19_pair`'s on a REG-19 seed, beyond TOL."""
    out = []
    eq = {"control_verdict": reg19["control_verdict"], "burst_verdict": reg19["burst_verdict"],
          "outcome": reg19["outcome"], "cold_side": reg19["cold_side"]["holds"],
          "rewarmed": reg19["after_empty"]["rewarmed"], "tipped": bool(reg19["surge_control"]["tipped"]),
          "exposed": reg19["exposure"]["exposed"]}
    out += [f"{k}: {pw_row[k]!r} != {v!r}" for k, v in eq.items() if pw_row[k] != v]
    for k in ("c", "lam_p"):
        x, y = pw_row[k], reg19["cold_side"][k]
        if not (np.isfinite(x) and np.isfinite(y) and abs(x - y) <= TOL["rate_rel"] * abs(y)):
            out.append(f"{k}: {x} vs {y}")
    ys = [w["share_waiting"] for w in reg19["control_waiting"]]
    if len(ys) != len(pw_row["shares"]) or any(abs(a - b) > TOL["share_abs"] for a, b in zip(pw_row["shares"], ys)):
        out.append(f"shares: {pw_row['shares']} vs {ys}")
    e1, e2 = pw_row["emptyings_s"], reg19["emptyings_s"]
    if len(e1) != len(e2) or any(abs(a - b) > TOL["time_abs"] for a, b in zip(e1, e2)):
        out.append(f"emptyings: {e1} vs {e2}")
    if abs(pw_row["peak"] - reg19["exposure"]["peak"]) > TOL["peak_abs"]:
        out.append(f"peak: {pw_row['peak']} vs {reg19['exposure']['peak']}")
    return out


def reg19_check() -> dict:
    """REG-19's five seeds, this view against `gate.reg19_pair`; raises on any difference (PW stops)."""
    rows, diffs = [], {}
    for s in GROUPS["reg19"]:
        r = seed_row(s)
        d = compare(r, G.reg19_pair(s))
        if d:
            diffs[s] = d
        rows.append(r)
    if diffs:
        raise RuntimeError(f"PW stops: the view differs from REG-19's own computation: {diffs}")
    return {"rows": rows}


def counts(rows: list[dict]) -> dict:
    """k counted passes of n, with one-sided 95 % Clopper-Pearson bounds, and the labels."""
    from analysis.stats import cp_upper
    n, k = len(rows), sum(r["counted"] for r in rows)
    return {"k": k, "n": n, "cp_lower": 1.0 - cp_upper(n - k, n, CONF) if n else math.nan,
            "cp_upper": cp_upper(k, n, CONF) if n else math.nan,
            "outcomes": {r["seed"]: r["outcome"] for r in rows},
            "cut_sensitive": [r["seed"] for r in rows if r["cut_sensitive"]]}


def readouts(seed: int) -> dict:
    """Decides nothing: from a long-watch pair's full data, emptyings in [4050, 4380) and the control's KV onset if
    it falls in [4080, 4140)."""
    from analysis import confirm
    control, burst = _pair_runs(seed)
    on = G.kv_onset({**control, "metrics": control["metrics"][control["metrics"].t < 4140.0]})
    return {"emptyings_4050_4380": confirm.emptyings(burst, 4050.0, 4380.0),
            "control_onset_4080_4140": on if np.isfinite(on) and on >= T_CUT else None}


def _registered() -> bool:
    from analysis import withheld
    return withheld.lifted(REG_HEADING)


def table() -> dict:
    """The registered computation, once: refuses until decisions.md holds the [REG] PW heading; the REG-19 check runs
    and must pass before any long-watch row is computed."""
    if not _registered():
        raise RuntimeError(f"PW is not registered: decisions.md has no heading '{REG_HEADING}'")
    r19 = reg19_check()["rows"]
    long_rows = [seed_row(s) for s in GROUPS["long"]]
    return {"groups": {"reg19": counts(r19), "long": counts(long_rows)},
            "pooled_exchangeable_assumed": counts(r19 + long_rows),
            "readouts": {s: readouts(s) for s in GROUPS["long"]},
            "rows": r19 + long_rows}


if __name__ == "__main__":
    import json
    print(json.dumps(table(), indent=1, default=str))
