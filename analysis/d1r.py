"""D1r (docs/drafts/d1r_draft1.md; [REG] D1r (C2, cap as a rescue) when registered): REG-19's burst runs at λ = 2.0
on seeds 15–19 with D1's per-seed session cap switched on at 1680 s (`workload.sessions.cap_from_s`, harness/loadgen.py,
author 2026-11-17), 600 s after the burst's end, once the stuck state has formed. Before the switch every session is
admitted at once; after it a session is admitted only while fewer than `cap` are in service, and the sessions above the
cap finish their turns (the overhang drains, it is not cut). Each run's config is D1's capped config of the seed plus
`cap_from_s` (and its name), so it shares its schedule row for row with its REG-19 run and its D1 run.

Per run (`row`):
  established   the precondition, judged on [1380, 1680) against the baseline by REG-20's readout (`k1.cold_backlogged`
                on those windows): the run is cold and backlogged before the cap acts. A run that is not is
                `not_established`: reported, and left out of n.
  label         `d1.label` on its end window [3480, 4080) against [300, 900), D1's thresholds, unchanged. A run that
                `harness_fault` flags (the switched cap did not do what loadgen says) counts in n as unclear (D1's DM2).
Outcome (`outcome`): D1's rule (`d1.outcome`, most = 3 of 5, REG-19's 5 of 5 as the uncapped twin), with D1's
`supports` named `rescues`: too_few -> refutes (stuck >= 3) -> rescues (no stuck, cut >= 3) -> admission_only
(server_only >= 3) -> mixed.
Feasibility (rule G; pass condition written in draft 1 before this was computed): `first_order`.
Not separated: C2's loop from the owed backlog against the pool (REG-24's alternative 5); see the draft.
"""
from __future__ import annotations

import itertools
import math
import re
from pathlib import Path

import numpy as np
import yaml

from analysis import d1

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"

D1R = {
    "seeds": (15, 16, 17, 18, 19),   # REG-19's and D1's seeds, in that order (author 2026-11-17: the draft's choice)
    "from_s": 1680.0,                # the switch: the start of REG-19's W, 600 s after the burst's end (author 2026-11-17)
    "established": (1380.0, 1680.0), # the 300 s before the switch, judged against the baseline (k1's readout)
    "most": 3,                       # of 5, D1's (d1.D1["most"])
    "max_idle_share": 0.05,          # harness_fault after the switch: D1's d1.D1["max_idle_share"]
    "max_wait_before_s": 1.0,        # harness_fault: a session arriving before the switch waits longer than this
    # Feasibility (rule G), written in docs/drafts/d1r_draft1.md before any computation:
    "feasible_by": 3480.0,           # the cap's queue empty, or emptying within drain_horizon_s, by the end window's start
    "feasible_k": 3,                 # in at least 3 of 5 seeds
    "x_cold": d1.D1["x_cold"],       # req/s until in service reaches the cap (REG-19's cold rate over W)
    "x_after": 2.65,                 # req/s after: the lowest post-exit cold-span completion rate of D1's cut seeds
                                     # (decisions.md 2026-11-16 [RESULT] D1: 2.65-3.01)
    "held_turns": 4.0,               # turns still owed by each of the `cap` sessions left in service when phase A ends
                                     # (half of 8; BELIEF, a first-order term)
    "rate_ref": d1.D1["lam"],        # req/s, the planned load the queue's arrivals carry
    # Prediction only (BELIEF; docs/drafts/d1r_draft1.md "Prediction"): P(label) per valid established seed.
    "prediction": {"recovered": 0.30, "recovered_cap_draining": 0.35, "server_only": 0.10, "stuck": 0.08,
                   "rewarming": 0.07, "partial": 0.03, "unclear": 0.07},
    "p_not_established": 0.03,
}


def config_name(seed: int) -> str:
    return f"burst_l1pl350cl_r2.0_w3000_s{seed}_rescue"


def run_pattern(seed: int) -> str:
    """The rescue runs' folders; the `_rescue` suffix keeps REG-19's, D1's, D1b's and REG-20's patterns off them."""
    return f"*_{config_name(seed)}"


def d1_cap(seed: int) -> int:
    """D1's cap for the seed, read from D1's committed capped config (written by `d1.write_configs` from
    `d1.cap_values`; [RESULT] D1 caps: 29, 25, 29, 26, 25)."""
    raw = yaml.safe_load((CONFIGS / f"{d1.config_name(seed)}.yaml").read_text())
    return int(raw["workload"]["sessions"]["cap"])


def config_text(seed: int) -> str:
    if seed not in D1R["seeds"]:
        raise ValueError(f"seed {seed} is not one of D1r's")
    fs = D1R["from_s"]
    return (f"# D1r (docs/decisions.md [REG] D1r (C2, cap as a rescue)): D1's capped run at seed {seed} with the cap\n"
            f"# switched on at {fs:g} s (cap_from_s), the only change; REG-19's run of this seed differs only in the cap.\n"
            f"# Written by `python -m analysis.d1r configs`. Runs only after `[DECISION] ... D1r starts`.\n"
            f"extends: {d1.config_name(seed)}.yaml\n"
            f"name: {config_name(seed)}\n"
            f"workload:\n"
            f"  sessions:\n"
            f"    cap_from_s: {fs:g}\n")


def resolved(seed: int) -> dict:
    from harness.config import _merge, load_config, with_defaults
    from harness.run import expand
    raw = yaml.safe_load(config_text(seed))
    parent = raw.pop("extends")
    return expand(with_defaults(_merge(load_config(CONFIGS / parent), raw)))[0]


def write_configs(configs_dir: Path | None = None) -> list[Path]:
    """Writes the five rescue configs as new files; refuses if any exists (a used config is never edited)."""
    d = configs_dir or CONFIGS
    paths = [d / f"{config_name(sd)}.yaml" for sd in D1R["seeds"]]
    if any(p.exists() for p in paths):
        raise FileExistsError([str(p) for p in paths if p.exists()])
    for p, sd in zip(paths, D1R["seeds"]):
        p.write_text(config_text(sd))
    return paths


# ---------------------------------------------------------------- the switched cap's invariant

def _held(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per session (no burst row): when it took a slot (admission, else first send) and when it gave it back (its last
    reply; inf if any turn never ended), as `d1.slots` reads them."""
    from analysis.margin import is_burst_request
    q = run["requests"]
    q = q[~is_burst_request(run)]
    g = q.groupby("session_id")
    t0 = q[q.turn == 0].set_index("session_id")
    held_from = (t0.t_admitted if "t_admitted" in t0 else t0.t_sent).fillna(t0.t_sent)
    done = g.t_end.max().where(g.t_end.apply(lambda s: s.notna().all()), np.inf)
    start = held_from.reindex(done.index)
    keep = start.notna()
    return start[keep].to_numpy(float), done[keep].to_numpy(float)


def in_service(run: dict, t: np.ndarray) -> np.ndarray:
    s, d = _held(run)
    t = np.asarray(t, float)
    return (np.searchsorted(np.sort(s), t, side="right") - np.searchsorted(np.sort(d), t, side="right")).astype(float)


def slots_switched(run: dict, cap: int, from_s: float) -> dict:
    """The switched cap's invariant, read from the records (harness/loadgen.py SessionCap with `from_s`):
    - before the switch no session waits (`max_wait_before_s`, the longest admission delay of a session arriving
      before `from_s`);
    - every admission at or after the switch found fewer than `cap` sessions in service (`late_admissions_over`, the
      count that did not);
    - reported: the share of grid seconds after the switch with a session waiting at the cap while fewer than `cap` were
      in service (`idle_share`), and the first time in service fell to the cap (`t_at_cap`)."""
    q = run["requests"]
    from analysis.margin import is_burst_request
    s0 = q[(q.turn == 0) & ~is_burst_request(run)]
    arr = s0.t_arrived.to_numpy(float) if "t_arrived" in s0 else np.full(len(s0), np.nan)
    adm = s0.t_admitted.to_numpy(float) if "t_admitted" in s0 else np.full(len(s0), np.nan)
    before = (arr < from_s) & np.isfinite(adm)
    wait_before = float(np.nanmax(adm[before] - arr[before])) if before.any() else 0.0
    s, d = _held(run)
    late = np.sort(adm[np.isfinite(adm) & (adm >= from_s)])
    # sessions in service just before each late admission (the admitted one excluded: its own start is at t)
    in_before = (np.searchsorted(np.sort(s), late, side="left") - np.searchsorted(np.sort(d), late, side="right"))
    over = int((in_before >= cap).sum())
    t_end = float(run["cfg"]["workload"]["duration_s"])
    grid = np.arange(from_s, t_end, d1.D1["grid_s"])
    held = in_service(run, grid)
    waiting = d1.cap_queue(run, grid)
    busy = waiting > 0
    at_cap = grid[held <= cap]
    return {"cap": cap, "from_s": from_s, "max_wait_before_s": wait_before, "late_admissions": int(len(late)),
            "late_admissions_over": over,
            "idle_share": float(((held < cap) & busy).sum() / busy.sum()) if busy.any() else 0.0,
            "t_at_cap": float(at_cap[0]) if len(at_cap) else None,
            "max_in_service_after": float(held.max()) if len(held) else math.nan}


def harness_fault(sl: dict) -> bool:
    """The switched cap did not do what harness/loadgen.py says: a session waited before the switch, an admission after
    it found the cap full, or sessions waited after it while a slot stood free. A flag, not a label: the run counts in
    n as unclear (D1's DM2)."""
    return (sl["max_wait_before_s"] > D1R["max_wait_before_s"] or sl["late_admissions_over"] > 0
            or sl["idle_share"] > D1R["max_idle_share"])


# ---------------------------------------------------------------- one run

def established(run: dict) -> dict:
    """REG-20's readout (`k1.cold_backlogged`) on [1380, 1680) against the run's own baseline window."""
    from analysis.k1 import cold_backlogged, windows
    w = windows(run["cfg"])
    return cold_backlogged(run, {"base": w["base"], "end": D1R["established"]})


def rewarm(run: dict, base_reuse: float) -> dict:
    """Reported, never judged: from the switch, in 300 s steps, the first span whose reuse of later turns (at least
    c2's min turns) is back within recovered_hit_drop of the baseline reuse; the completion rate from there to the
    run's end, against λ; and the cap's queue's peak after the switch and the first time it is 0 after that peak."""
    from analysis.c2 import C2, reuse_in
    from analysis.gate import completion_rate
    v = run["cfg"]["verdict"]
    t_end = float(run["cfg"]["workload"]["duration_s"])
    turns = d1.later_turns(run)
    t_rw = None
    a = D1R["from_s"]
    while a + d1.D1["span_s"] <= t_end and not math.isnan(base_reuse):
        r = reuse_in(turns, a, a + d1.D1["span_s"], C2["min_turns"])
        if not math.isnan(r) and r >= base_reuse - v["recovered_hit_drop"]:
            t_rw = a
            break
        a += d1.D1["span_s"]
    grid = np.arange(D1R["from_s"], t_end, d1.D1["grid_s"])
    cq = d1.cap_queue(run, grid)
    peak_i = int(np.nanargmax(cq)) if len(cq) and not np.isnan(cq).all() else None
    empty = None
    if peak_i is not None:
        z = np.nonzero(cq[peak_i:] == 0)[0]
        empty = float(grid[peak_i + z[0]]) if len(z) else None
    return {"t_rewarm": t_rw, "completion_after": None if t_rw is None else completion_rate(run, t_rw, t_end),
            "lam": d1.D1["lam"], "cap_queue_peak": None if peak_i is None else float(cq[peak_i]),
            "t_cap_queue_peak": None if peak_i is None else float(grid[peak_i]), "t_cap_queue_empty": empty}


def rescue_run(seed: int) -> tuple[dict | None, list[str]]:
    """The seed's counted rescue run (the earliest valid one, as `d1.capped_run`) and any later valid ones. Its resolved
    config must equal `resolved(seed)` (run id and source path aside)."""
    from analysis.load import find_runs
    runs = sorted(find_runs(run_pattern(seed)), key=lambda r: r["dir"].name)
    if not runs:
        return None, []
    r = runs[0]
    got = {k: v for k, v in r["cfg"].items() if k not in ("run_id", "source_config")}
    diff = d1._diff_keys(got, resolved(seed))
    if diff:
        raise ValueError(f"{r['dir'].name}: config differs from D1r's at {diff}")
    return r, [x["dir"].name for x in runs[1:]]


def row(seed: int) -> dict | None:
    from analysis.c2 import eviction_readout, last_burst_end, levels
    from analysis.load import RUNS, load_run, verdict
    run, later = rescue_run(seed)
    if run is None:
        return None
    un = load_run(RUNS / d1.D1["uncapped"][seed])
    cap = d1_cap(seed)
    est = established(run)
    lab = d1.label(run)
    sl = slots_switched(run, cap, D1R["from_s"])
    fault = harness_fault(sl)
    sp, sp_un = d1.spans(run), d1.spans(un)
    if est["cold_backlogged"] is not True:
        label = "not_established"
    else:
        label = "unclear" if fault else lab["label"]
    return {"seed": seed, "cap": cap, "run": run["dir"].name, "later_valid": later, "uncapped": un["dir"].name,
            "label": label, "established": est, "harness_fault": fault, "label_values": lab,
            "verdict": verdict(run)["verdict"], "slots": sl, "rewarm": rewarm(run, lab["reuse"]["base"]),
            "first_order": d1.first_order(cap), "spans": sp, "uncapped_spans": sp_un, "levels": levels(run),
            "eviction": eviction_readout(run), "exit": last_burst_end(run)}


# ---------------------------------------------------------------- outcome, prediction, feasibility

def outcome(labels: list[str]) -> dict:
    """D1's rule on the established seeds' labels (`not_established` left out of n and counted apart), with `supports`
    named `rescues`. McNemar's p and the bounds are D1's readouts against REG-19's 5 of 5."""
    kept = [lab for lab in labels if lab != "not_established"]
    out = d1.outcome(kept, n_seeds=len(labels))
    out["decision"] = {"supports": "rescues"}.get(out["decision"], out["decision"])
    out["n_not_established"] = len(labels) - len(kept)
    return out


def predicted(n: int = 5) -> dict:
    """Exact over n established seeds drawn independently from D1R["prediction"] (not_established left aside)."""
    p = D1R["prediction"]
    if abs(sum(p.values()) - 1) > 1e-9:
        raise ValueError("the prediction does not sum to 1")
    dist: dict[str, float] = {}
    for combo in itertools.product(p, repeat=n):
        pr = math.prod(p[c] for c in combo)
        dec = outcome(list(combo))["decision"]
        dist[dec] = dist.get(dec, 0.0) + pr
    return dict(sorted(dist.items()))


def empty_time(work_a: float, cap: int, arrivals: np.ndarray, from_s: float, t_end: float, horizon: float) -> dict:
    """The first-order estimate (prediction only; BELIEF as a model). Phase A: the server at x_cold serves the admitted
    sessions' remaining turns (`work_a`) until in service is down to `cap` sessions, each still owing `held_turns`; it
    serves nothing from the cap's queue. Phase B, from `t_a`: the capped server at x_after serves the queue's turns.
    The queue's work arrives at the planned times of the turns of the sessions arriving after the switch (`arrivals`)
    and is a fluid queue on a 1 s grid: backlog <- max(0, backlog + arrived - served). Judged (the pass condition in
    docs/drafts/d1r_draft1.md): at `feasible_by` the backlog is 0, or the queue is falling (x_after above the queue's
    planned arrival rate) and the backlog would empty within `horizon` (the rescue config's verdict.drain_horizon_s).
    Reported: the backlog's peak and the first time after it that the backlog is 0 (inf if never on the grid)."""
    x_c, x_b, by = D1R["x_cold"], D1R["x_after"], D1R["feasible_by"]
    t_a = from_s + max(0.0, work_a - cap * D1R["held_turns"]) / x_c
    arrivals = np.sort(np.asarray(arrivals, float))
    last = max(t_end, float(arrivals.max()) if len(arrivals) else t_end)
    edges = np.arange(from_s, last + horizon + 2.0, 1.0)
    arrived = np.diff(np.searchsorted(arrivals, edges, side="left"))       # turns planned in [edges[i], edges[i+1])
    serve = np.where(edges[:-1] >= t_a, x_b, 0.0)
    backlog, b = np.empty(len(arrived)), 0.0
    for i in range(len(arrived)):
        b = max(0.0, b + arrived[i] - serve[i])
        backlog[i] = b
    t = edges[1:]
    i_by = int(np.searchsorted(t, by, side="left"))
    b_by = float(backlog[i_by])
    span = (last - from_s)
    rate_after = len(arrivals) / span if span > 0 else 0.0
    if b_by == 0.0:
        meets = True
    elif x_b > rate_after:
        meets = b_by / (x_b - rate_after) <= horizon
    else:
        meets = False
    pk = int(np.argmax(backlog))
    z = np.nonzero(backlog[pk:] == 0.0)[0]
    return {"t_a": t_a, "backlog_at_by": b_by, "backlog_peak": float(backlog[pk]), "t_peak": float(t[pk]),
            "t_empty": float(t[pk + z[0]]) if len(z) else math.inf, "rate_after": rate_after, "meets": bool(meets)}


def first_order() -> dict:
    """Rule G (the pass condition in docs/drafts/d1r_draft1.md, written before this was computed): per seed, from REG-19's
    uncapped run (identical to the rescue run up to the switch): the admitted sessions' turns not ended by the switch
    (`work_a`), and the planned times of the turns of sessions arriving after it (the rescue's queue); `empty_time`."""
    from analysis.load import RUNS, load_run
    from analysis.margin import is_burst_request
    fs = D1R["from_s"]
    rows = []
    for sd in D1R["seeds"]:
        un = load_run(RUNS / d1.D1["uncapped"][sd])
        q = un["requests"][~is_burst_request(un)]
        first = q[q.turn == 0].set_index("session_id").t_planned
        sess_in = first[first < fs].index
        mine = q[q.session_id.isin(sess_in)]
        work_a = float((~(mine.t_end < fs)).sum())
        later = q[q.session_id.isin(first[first >= fs].index)]
        cap = d1_cap(sd)
        e = empty_time(work_a, cap, later.t_planned.to_numpy(float), fs, float(un["cfg"]["workload"]["duration_s"]),
                       float(resolved(sd)["verdict"]["drain_horizon_s"]))
        rows.append({"seed": sd, "uncapped": un["dir"].name, "cap": cap, "work_a": work_a,
                     "sessions_after": int(later.session_id.nunique()), "turns_after": int(len(later)), **e})
    k = sum(r["meets"] for r in rows)
    return {"rows": rows, "k_meets": k, "passes": bool(k >= D1R["feasible_k"]),
            "inputs": {k_: D1R[k_] for k_ in ("x_cold", "x_after", "held_turns", "feasible_by", "feasible_k")}}


def _sealed() -> list[str]:
    from analysis import withheld
    from analysis.load import RUNS
    return [p.name for sd in D1R["seeds"] for p in sorted(RUNS.glob(run_pattern(sd))) if withheld.sealed_by(p.name)]


def table() -> dict:
    """D1r's registered computation, run once after the five rescue runs: `AFTERSHOCK_UNSEAL=D1r python -m analysis.d1r
    table`. Refuses while any rescue folder is sealed, and if D1 is unsealed (D1's labels are read through its lifted
    seal). Reported beside the decision: `not_run`, `harness_faults`, D1's label per seed, `decision_underlying` (on
    each established row's `label_values["label"]`), and REG-20's drift rerun (`d1.drift_readout`)."""
    from analysis import withheld
    sealed = _sealed()
    if sealed:
        raise withheld.WithheldError(f"{len(sealed)} rescue folder(s) sealed by D1r; the registered computation sets "
                                     f"AFTERSHOCK_UNSEAL=D1r")
    if "D1" in withheld.unsealed_ids():
        raise withheld.WithheldError("D1r's computation does not unseal D1")
    rows, not_run = [], []
    for sd in D1R["seeds"]:
        r = row(sd)
        (not_run if r is None else rows).append(sd if r is None else r)
    labels_d1 = {}
    for sd in D1R["seeds"]:
        r1 = d1.row(sd)
        if r1 is not None:
            labels_d1[sd] = r1["label"]
    under = [r["label_values"]["label"] if r["label"] != "not_established" else "not_established" for r in rows]
    return {"rows": rows, "not_run": not_run, "outcome": outcome([r["label"] for r in rows]),
            "labels_d1": labels_d1, "harness_faults": [r["seed"] for r in rows if r["harness_fault"]],
            "decision_underlying": outcome(under)["decision"], "drift": d1.drift_readout()}


if __name__ == "__main__":
    import json
    import sys

    cmd = sys.argv[1] if len(sys.argv) == 2 else None
    if cmd == "configs":
        print("\n".join(str(p) for p in write_configs()))
    elif cmd == "predicted":
        print(json.dumps({"predicted": predicted(), "first_order_kv": {sd: d1.first_order(d1_cap(sd))
                                                                       for sd in D1R["seeds"]}}, indent=1))
    elif cmd == "feasibility":
        print(json.dumps(first_order(), indent=1, default=str))
    elif cmd == "table":
        print(json.dumps(table(), indent=1, default=str))
    else:
        raise SystemExit("usage: python -m analysis.d1r configs | predicted | feasibility | table")
