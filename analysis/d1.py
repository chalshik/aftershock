"""C2, the session-cap arm: D1 (draft 3; not registered: docs/drafts/d1_draft3.md). Step D of docs/experiment-plan.md;
the cap's rule is decisions.md 2026-11-12 [DECISION] (design choices, item 2), the cap itself decisions.md 2026-11-12
[DECISION] (session cap). Never writes into runs/.

The cap's value for a seed is the highest number of sessions in service at once in that seed's REG-19 control
(`cap.max_sessions_in_service`; in service: first turn sent to last reply, burst requests excluded). The capped arm is
REG-19's burst config with `workload.sessions.cap` set to that value, the only change (`config_text`). Burst requests
are not session chains and pass the cap (harness/loadgen.py), so the trigger is the one REG-19's burst runs had.

Each capped run is read against its own pre-burst baseline [300, 900) s, in the end window [3480, 4080) s of its 3000 s
watch (`k1.windows`), with the verdict's registered thresholds (configs/stage2_base.yaml, stage2_cap_base.yaml):
  server side: reuse of later turns (`c2.reuse_in`, at least c2.C2["min_turns"] turns; not the hit counter, whose
               served mix shifts with first turns, REG-20 review T1) and mean waiting (`load._level` q);
  client side: the queue of sessions waiting at the cap (`cap_queue`), its mean against its own baseline mean plus
               recovered_q_excess, and, when above, its slope against drain_horizon_s (as `load.verdict` reads a line).
               Under the cap the server's own line is held near 0 by construction, so a stuck state's backlog is the
               cap's queue: a cold reuse is `stuck_*` only with that queue backlogged (rule F d1 r1, DG1), else `partial`;
  the pool:    mean KV in the end window (`c2.gauge`) against condition 4's min_kv (0.80), a readout only (`kv_class`:
               full, free, or None if missing). The gauge counts only blocks held by running requests (open question S7),
               and cached free blocks are evicted about least recently released first (S9); under caps below ~34 the
               gauge cannot reach 0.80 (`first_order`), so the class would follow the cap's value, not the mechanism.
               One `stuck` label, the gauge reported beside it (author, decisions.md 2026-11-14, D1 goes behind D2;
               docs/drafts/d1_r3/discrimination.md).
Labels (`label`), in this order: unclear (a missing baseline pf), bad_baseline, unclear (a missing input),
recovered, recovered_cap_draining, server_only, unclear (no climb test), rewarming, partial, unclear, stuck, unclear.
The matrix (`outcome`): too_few, refutes, supports, admission_only, mixed.
A run whose records show more sessions holding a slot than the cap, or a session waiting at the cap while a slot is free
in more than max_idle_share of the waiting grid seconds (`slots`, `harness_fault`), is flagged: it counts in n as
unclear, and `row` keeps the flag and the label its records would otherwise give (design review d1 r2, DM2). Taking such
a run out of n would mark a run invalid beyond the automatic checks, which is the author's (docs/rules.md, C); `outcome`
refuses a harness_fault label, so no path of the registered computation does it. `table` lists the flagged seeds and
gives `decision_underlying`, the decision on the underlying labels (a readout, never judged).
Readouts, never judged: `load.verdict` of the capped run and of REG-19's uncapped one (percent fast counts turns never
sent as misses, and the cap's wait is not in latency), per 300 s span after the burst's end (`spans`, preemptions among
them), the separation readout (`load_reading`) and, for a cut seed, whether the cold state formed after the burst's own
requests had left (`onset_class`) and the first 60 s window from the exit back near the baseline (`clear_time`), the
waits (`cap.cap_wait`, before the burst apart), the longest stretch with a session waiting and a slot free (`slots`),
the cap's queue at the burst's end and at the exit, the planned session starts over the end window and its halves
(`planned_end_rates`), the rewarming or partial seeds whose cap queue ends above its baseline ("possibly stuck",
`possibly_stuck`), condition 4's levels (`c2.levels`), the eviction readout (`c2.eviction_readout`), the shock
(`gate.shock_size`), the baseline pair (`gate.reg19_baseline_pair`), and REG-20's drift rerun once REG-20's own
[RESULT] has lifted its seal (`drift_readout`).
"""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from analysis import cap as capa
from analysis.c2 import C2, KV, N, gauge, last_burst_end, levels, reuse_in, turn_spacing
from analysis.k1 import later_turns, mcnemar_one_sided, sessions_in_service, windows
from analysis.load import _level
from analysis.margin import is_burst_request
from analysis.stats import cp_upper

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"

D1 = {
    "seeds": (15, 16, 17, 18, 19),            # REG-19's seeds; run in this order (a rule fixed before any capped run)
    # REG-19's counted runs (decisions.md 2026-11-13 [RESULT] REG-19, its per-seed table). Seed 18's lost attempt-2
    # control (20261113-033842_...) is outside every count and is never read here.
    "controls": {15: "20261112-050920_stage2_control_l1pl350cl_r2.0_w3000_s15",
                 16: "20261112-073516_stage2_control_l1pl350cl_r2.0_w3000_s16",
                 17: "20261112-103423_stage2_control_l1pl350cl_r2.0_w3000_s17",
                 18: "20261113-165516_stage2_control_l1pl350cl_r2.0_w3000_s18",
                 19: "20261113-192110_stage2_control_l1pl350cl_r2.0_w3000_s19"},
    "uncapped": {15: "20261112-061951_burst_l1pl350cl_r2.0_w3000_s15",
                 16: "20261112-084533_burst_l1pl350cl_r2.0_w3000_s16",
                 17: "20261112-135538_burst_l1pl350cl_r2.0_w3000_s17",
                 18: "20261113-180511_burst_l1pl350cl_r2.0_w3000_s18",
                 19: "20261113-203108_burst_l1pl350cl_r2.0_w3000_s19"},
    "lam": 2.0, "duration_s": 4080.0, "watch_s": 3000.0,
    "most": 3,                                # REG-18's and REG-19's "most seeds" (3 of 5)
    "conf": 0.95,                             # one-sided Clopper-Pearson (rule H)
    "reg19_k_n": (5, 5),                      # REG-19's k_clear of n on the same seeds (decisions.md:16527); readout only
    "span_s": 300.0, "n_spans": 10,           # readouts: 300 s spans from the burst's end to the watch's end
    "clear_s": 60.0,                          # readout only (rule F d1 r2, DJ-O5): `clear_time`'s window, from the exit
    "grid_s": 1.0,                            # the cap queue is read on this grid (the recorder's scrape interval)
    "max_idle_share": 0.05,                   # harness_fault above this share of waiting grid seconds with a slot free
                                              # (a hand-over takes milliseconds; a slot that leaks keeps one free for good)
                                              # NEW in D1 (no earlier entry); the author's choice, fixed before any run
    "floor_ol": 1.681,                        # req/s, REG-8's open-loop cache-off floor (k2.E0["floor_8"]); readout only
    "warm_cl": 2.4,                           # req/s, REG-13's closed-loop warm capacity (k2.E0["warm_cl_8"]); readout only
    # REG-20's drift rerun of seed 15 on the new harness (sealed by REG-20): read only once REG-20's [RESULT] has
    # lifted its seal; D1's own command never unseals it (rule F d1 r1, DG-O2). Readout only.
    "drift_pattern": "*_burst_l1pl350cl_r2.0_w3000_s15_drift",
    # First-order model of the cold state under a cap (`first_order`; prediction only, BELIEF as a model):
    "x_cold": 1.75,                           # req/s, REG-19's cold completion rate over W, 1.740-1.758 (FILE decisions.md:16533)
    "gap_s": 5.0,                             # mean think time, configs/gap5_base.yaml (exponential)
    "kv_per_running": 0.951 / 30.0,           # KV share per running request in the cold state: REG-22 batch 2's end windows
                                              # (KV 0.950-0.952 at N 29.80-30.29, FILE decisions.md:16752-16756)
    # Prediction only (BELIEF, docs/drafts/d1_draft3.md "Prediction"): P(label) per valid seed, seeds independent.
    # `predicted_matrix` counts the harness_fault belief as unclear (a flagged run stays in n, DM2).
    "prediction": {"recovered": 0.60, "recovered_cap_draining": 0.16, "server_only": 0.03, "rewarming": 0.04,
                   "partial": 0.03, "stuck": 0.04, "bad_baseline": 0.04,
                   "unclear": 0.05, "harness_fault": 0.01},
}
LABELS = ("recovered", "recovered_cap_draining", "server_only", "rewarming", "partial", "stuck", "bad_baseline",
          "unclear")
CUT = ("recovered", "recovered_cap_draining")       # the server warm and the cap's queue back to (or draining to) its base
STUCK = ("stuck",)                                  # cold reuse and the cap's queue backlogged, not draining
ONSET = ("never_cold", "burst_only", "formed_then_cleared", "undefined")


def config_name(seed: int) -> str:
    return f"burst_l1pl350cl_r2.0_w3000_s{seed}_cap"


def run_pattern(seed: int) -> str:
    """The capped runs' folders. It matches none of the patterns REG-19, REG-20, REG-21, REG-22 or E1 select or seal
    (tests/test_d1.py), since the name ends in `_cap`."""
    return f"*_{config_name(seed)}"


# ---------------------------------------------------------------- the cap's value (registered computation, once)

def _check_control(r: dict, seed: int) -> None:
    c, name = r["cfg"], r["dir"].name
    if not (r["checks"] and r["checks"]["valid"]):
        raise ValueError(f"{name}: not valid")
    wl = c["workload"]
    ses = wl.get("sessions") or {}
    bad = {"seed": c["seed"] != seed, "name": c["name"] != f"stage2_control_l1pl350cl_r2.0_w3000_s{seed}",
           "trigger": c.get("trigger") is not None, "closed_loop": ses.get("closed_loop") is not True,
           "cap": ses.get("cap") is not None, "rate": float(wl["rate"]) != D1["lam"],
           "duration": float(wl["duration_s"]) != D1["duration_s"]}
    if any(bad.values()):
        raise ValueError(f"{name}: not REG-19's seed-{seed} control: {[k for k, v in bad.items() if v]}")


def control_run(seed: int) -> dict:
    from analysis.load import RUNS, load_run
    r = load_run(RUNS / D1["controls"][seed])
    _check_control(r, seed)
    return r


def in_service_steps(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """The in-service count after each event, events in time order with ends before starts at equal times (the same
    sweep as `cap.max_sessions_in_service`)."""
    start, done = capa._intervals(run)
    t = np.concatenate([start, done])
    step = np.concatenate([np.ones(len(start)), -np.ones(len(done))])
    order = np.lexsort((step, t))
    return t[order], np.cumsum(step[order])


def peak(run: dict, before: float) -> dict:
    """The peak (= `cap.max_sessions_in_service`), the first time it is reached, and the peak over [0, before)."""
    t, n = in_service_steps(run)
    if not len(t):
        return {"value": 0, "t_first": math.nan, "before": 0}
    i = int(np.argmax(n))
    pre = n[t < before]
    return {"value": int(n[i]), "t_first": float(t[i]), "before": int(pre.max()) if len(pre) else 0}


def cap_value(control: dict) -> int:
    v = capa.max_sessions_in_service(control)
    if v < 1:
        raise ValueError(f"{control['dir'].name}: no session in service, no cap")
    return v


def first_order(cap: int) -> dict:
    """Prediction only (BELIEF as a model): a cold server saturated at x_cold under a binding cap holds x_cold * gap_s
    sessions in think time (Little's law on that stage), so at most cap - x_cold * gap_s requests are in flight; at the
    cold state's KV per running request that bounds the KV gauge, which counts only blocks held by running requests
    (open question S7). It does not bound eviction: cached free blocks are evicted about least recently released first
    (S9). Returns the bound and the caps at which it reaches 0.80 / 0.95."""
    think = D1["x_cold"] * D1["gap_s"]
    kv = max(0.0, cap - think) * D1["kv_per_running"]
    return {"think": think, "in_flight_max": max(0.0, cap - think), "kv_max": kv,
            "cap_kv_080": think + C2["min_kv"] / D1["kv_per_running"], "cap_kv_095": think + 0.95 / D1["kv_per_running"]}


def cap_values() -> dict:
    """D1's registered computation on REG-19's five controls, run once after D1's approval: per seed the cap, when the
    control first reached it, its peak before the burst's start (900 s), its mean sessions in service over the baseline,
    and the first-order KV bound at that cap (a prediction readout). The last key, `caps`, feeds `write_configs`."""
    at = float(_burst_cfg(D1["seeds"][0])["trigger"]["at_s"])
    rows = []
    for sd in D1["seeds"]:
        r = control_run(sd)
        p = peak(r, at)
        if p["value"] != cap_value(r):
            raise AssertionError("peak and cap.max_sessions_in_service disagree")
        rows.append({"seed": sd, "control": r["dir"].name, "cap": p["value"], "t_peak": p["t_first"],
                     "peak_before_burst": p["before"], "mean_in_service_baseline": sessions_in_service(r, at - 600, at),
                     "first_order": first_order(p["value"])})
    return {"rows": rows, "caps": {r["seed"]: r["cap"] for r in rows}}


# ---------------------------------------------------------------- the five capped configs

def _burst_cfg(seed: int) -> dict:
    from harness.config import load_config, with_defaults
    from harness.run import expand
    return expand(with_defaults(load_config(CONFIGS / f"burst_l1pl350cl_r2.0_w3000_s{seed}.yaml")))[0]


def config_text(seed: int, cap: int) -> str:
    if seed not in D1["seeds"]:
        raise ValueError(f"seed {seed} is not one of REG-19's")
    if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
        raise ValueError(f"cap must be a positive integer, got {cap!r}")
    return (f"# D1 (docs/decisions.md [REG] D1 (C2, session cap)): REG-19's burst run at seed {seed} with the session\n"
            f"# cap on, the only change. cap = the highest number of sessions in service at once in seed {seed}'s\n"
            f"# REG-19 control {D1['controls'][seed]}\n"
            f"# (d1.cap_values, computed once after D1's approval; [RESULT] D1 caps).\n"
            f"# Written by `python -m analysis.d1 configs <caps file>`. Runs only after `[DECISION] ... D1 starts`.\n"
            f"extends: burst_l1pl350cl_r2.0_w3000_s{seed}.yaml\n"
            f"name: {config_name(seed)}\n"
            f"workload:\n"
            f"  sessions:\n"
            f"    cap: {cap}\n")


def resolved(seed: int, cap: int) -> dict:
    """The resolved config `config_text(seed, cap)` gives, without writing it (its parent is resolved from configs/)."""
    from harness.config import _merge, load_config, with_defaults
    from harness.run import expand
    raw = yaml.safe_load(config_text(seed, cap))
    parent = raw.pop("extends")
    return expand(with_defaults(_merge(load_config(CONFIGS / parent), raw)))[0]


def config_chain(seed: int) -> list[str]:
    """The repo files a capped run's config resolves from (harness/config.py: `extends` chains, then
    configs/defaults.yaml beneath): the capped config, every config in its chain, and defaults.yaml. The launch commit's
    tree check restricts `git diff --name-status <registered commit> <launch commit>` to these files and harness/,
    pyproject.toml, uv.lock and .python-version (draft 3, "The tree D1 runs"); it must print exactly the five capped
    configs, each `A` (rule F d1 r2, DJ-O6). The chain is printed by `python -m analysis.d1 chain` at the registered
    commit."""
    out, name = [f"configs/{config_name(seed)}.yaml"], f"burst_l1pl350cl_r2.0_w3000_s{seed}.yaml"
    while name:
        out.append(f"configs/{name}")
        name = (yaml.safe_load((CONFIGS / name).read_text()) or {}).get("extends")
    return out + ["configs/defaults.yaml"]


def write_configs(caps: dict, configs_dir: Path | None = None) -> list[Path]:
    """Writes the five capped configs as new files; refuses if any exists (a used config is never edited)."""
    d = configs_dir or CONFIGS
    caps = {int(k): v for k, v in caps.items()}
    if sorted(caps) != sorted(D1["seeds"]):
        raise ValueError(f"need a cap for each of seeds {D1['seeds']}, got {sorted(caps)}")
    paths = [d / f"{config_name(sd)}.yaml" for sd in D1["seeds"]]
    if any(p.exists() for p in paths):
        raise FileExistsError([str(p) for p in paths if p.exists()])
    texts = [config_text(sd, caps[sd]) for sd in D1["seeds"]]
    for p, txt in zip(paths, texts):
        p.write_text(txt)
    return paths


def caps_from_file(path: Path) -> dict:
    """The `caps` of the last JSON line of a saved `cap_values` output."""
    lines = [ln for ln in Path(path).read_text().splitlines() if ln.strip().startswith("{")]
    return {int(k): int(v) for k, v in json.loads(lines[-1])["caps"].items()}


def inputs() -> dict:
    """Design inputs from the configs alone (no run): per seed, whether the capped config's planned schedule equals
    REG-19's burst config's row for row (the cap is not read by the schedule) and its session rows the control's; the
    resolved keys in which the capped config differs from REG-19's burst config (with a placeholder cap)."""
    from harness.workload import build_schedule
    from harness.config import load_config, with_defaults
    from harness.run import expand
    out = []
    for sd in D1["seeds"]:
        b, k = _burst_cfg(sd), resolved(sd, 1)
        c = expand(with_defaults(load_config(CONFIGS / f"stage2_control_l1pl350cl_r2.0_w3000_s{sd}.yaml")))[0]
        sb, _ = build_schedule(b["workload"], b["seed"], b["trigger"])
        sk, _ = build_schedule(k["workload"], k["seed"], k["trigger"])
        sc, _ = build_schedule(c["workload"], c["seed"], c.get("trigger"))
        s = sk[~sk.burst].reset_index(drop=True)
        out.append({"seed": sd, "capped_equals_burst": bool(sb.equals(sk)),
                    "sessions_equal_control": bool(len(s) == len(sc) and (s.t_planned.values == sc.t_planned.values).all()
                                                   and (s.prompt_len.values == sc.prompt_len.values).all()),
                    "n_session": int(len(s)), "n_burst": int(sk.burst.sum()), "differs": _diff_keys(b, k)})
    return {"seeds": out}


def _flat(d: dict, p: str = "") -> dict:
    out = {}
    for key, v in d.items():
        out.update(_flat(v, f"{p}{key}.")) if isinstance(v, dict) else out.__setitem__(f"{p}{key}", v)
    return out


def _diff_keys(a: dict, b: dict) -> list[str]:
    fa, fb = _flat(a), _flat(b)
    return sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))


# ---------------------------------------------------------------- readouts of one capped run

def _cap_times(run: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """The sessions that reached the cap (turn 0, no burst row, with t_arrived): their arrival and admission times,
    each sorted; a session never admitted has admission inf. None if the run has no `t_arrived` (no cap)."""
    q = run["requests"]
    if "t_arrived" not in q:
        return None
    s = q[(q.turn == 0) & q.t_arrived.notna() & ~is_burst_request(run)]
    adm = s.t_admitted.to_numpy(float) if "t_admitted" in s else np.full(len(s), np.nan)
    return np.sort(s.t_arrived.to_numpy(float)), np.sort(np.where(np.isnan(adm), np.inf, adm))


def cap_queue(run: dict, t: np.ndarray) -> np.ndarray:
    """Sessions waiting at the cap at each time t: turn 0 (no burst row) with t_arrived <= t < t_admitted; a session
    never admitted waits to the end of the record. NaN everywhere if the run has no `t_arrived` (no cap)."""
    t = np.asarray(t, float)
    ct = _cap_times(run)
    if ct is None:
        return np.full(len(t), np.nan)
    arr, adm = ct
    return (np.searchsorted(arr, t, side="right") - np.searchsorted(adm, t, side="right")).astype(float)


def cap_queue_at(run: dict, burst_end_t: float, exit_t: float) -> dict:
    """Reported, never judged (rule F d1 r2, DJ-O4): the cap's queue at the burst's end and at the exit
    (`c2.last_burst_end`; NaN if unknown). `spans`' first span gives its mean over the span, not these values."""
    val = lambda t: math.nan if math.isnan(t) else float(cap_queue(run, np.array([t]))[0])
    return {"burst_end": val(burst_end_t), "exit": val(exit_t)}


def cap_queue_level(run: dict, a: float, b: float) -> dict:
    """Mean and least-squares slope (sessions/s) of the cap's queue on a grid over [a, b)."""
    g = np.arange(a, b, D1["grid_s"])
    x = cap_queue(run, g)
    if not len(g) or np.isnan(x).any():
        return {"mean": math.nan, "slope": math.nan}
    return {"mean": float(x.mean()), "slope": float(np.polyfit(g, x, 1)[0]) if len(g) >= 2 else math.nan}


def _backlogged(cq: dict, v: dict) -> bool | None:
    """The cap's queue in the end window is backlogged: its mean above its baseline mean + recovered_q_excess and not
    draining (not: slope < 0 and the queue would empty within drain_horizon_s). None if that needs a NaN slope.
    The two thresholds are the verdict's, set for the server's line in requests; here they are applied to a queue of
    sessions (up to 8 turns each), a new use fixed before any capped run (rule F d1 r1, DG11; the author's choice)."""
    c_end, c_base, sl = cq["end"]["mean"], cq["base"]["mean"], cq["end"]["slope"]
    if c_end <= c_base + v["recovered_q_excess"]:
        return False
    if math.isnan(sl):
        return None
    return not (sl < 0 and c_end / -sl <= v["drain_horizon_s"])


def label(run: dict) -> dict:
    """D1's label for one capped run (module docstring), with the values it used."""
    v, w = run["cfg"]["verdict"], windows(run["cfg"])
    (b0, b1), (e0, e1) = w["base"], w["end"]
    prev = (e0 - float(v["end_s"]), e0)
    turns = later_turns(run)
    ru = {"base": reuse_in(turns, b0, b1, C2["min_turns"]), "prev": reuse_in(turns, *prev, C2["min_turns"]),
          "end": reuse_in(turns, e0, e1, C2["min_turns"])}
    lb, le = _level(run, b0, b1), _level(run, e0, e1)
    cq = {"base": cap_queue_level(run, b0, b1), "end": cap_queue_level(run, e0, e1)}
    kv_end = gauge(run, KV, e0, e1)
    nan = lambda *x: any(isinstance(y, float) and math.isnan(y) for y in x)
    vals = {"reuse": ru, "q": {"base": lb["q"], "end": le["q"]}, "pf": {"base": lb["pf"], "end": le["pf"]},
            "hit": {"base": lb["hit"], "end": le["hit"]}, "cap_queue": cq, "kv_end": kv_end, "windows": w,
            # readout, never judged (author 2026-11-14): condition 4's KV class in the end window; not part of the label
            "kv_class": None if nan(kv_end) else ("full" if kv_end >= C2["min_kv"] else "free"),
            "cap_backlogged": None,
            # reported, never judged (rule F d1 r2, DJ-O3; design review d1 r2, DM3): the cap's queue ends above its
            # baseline mean + recovered_q_excess, draining or not; `possibly_stuck` reads it for rewarming and partial
            "cap_above_base": (None if nan(cq["base"]["mean"], cq["end"]["mean"])
                               else bool(cq["end"]["mean"] > cq["base"]["mean"] + v["recovered_q_excess"]))}
    if nan(lb["pf"]):
        lab = "unclear"
    elif lb["pf"] < v["min_baseline_pf"]:
        lab = "bad_baseline"
    elif nan(ru["base"], ru["end"], lb["q"], le["q"], cq["base"]["mean"], cq["end"]["mean"]):
        lab = "unclear"
    elif ru["end"] >= ru["base"] - v["recovered_hit_drop"] and le["q"] <= lb["q"] + v["recovered_q_excess"]:
        c_end, c_base, sl = cq["end"]["mean"], cq["base"]["mean"], cq["end"]["slope"]
        if c_end <= c_base + v["recovered_q_excess"]:
            lab = "recovered"
        elif nan(sl):
            lab = "unclear"
        elif sl < 0 and c_end / -sl <= v["drain_horizon_s"]:
            lab = "recovered_cap_draining"
        else:
            lab = "server_only"
    elif nan(ru["prev"]):
        lab = "unclear"            # a climb cannot be ruled out
    elif ru["end"] - ru["prev"] >= v["hit_climb"]:   # hit_climb was calibrated on the hit counter, not on reuse (DG11)
        lab = "rewarming"
        vals["cap_backlogged"] = _backlogged(cq, v)  # reported only (rule F d1 r2, DJ-O3): a rewarming seed may be stuck
    elif ru["end"] <= ru["base"] - v["stuck_hit_drop"]:
        bl = vals["cap_backlogged"] = _backlogged(cq, v)
        if bl is None:
            lab = "unclear"        # above its baseline, but whether it drains is unknown
        elif not bl:
            lab = "partial"        # no line at the server (by construction) or at the cap: a lower reuse, carried
        else:
            lab = "stuck"          # the KV gauge is a readout only (`kv_class`), never part of the label
    else:
        lab = "unclear"
    return {"label": lab, **vals}


def server_rate(run: dict, a: float, b: float) -> float:
    """Requests (session and burst) sent to the server per second in [a, b), by t_sent: the load the server was offered."""
    q = run["requests"]
    return float(((q.t_sent >= a) & (q.t_sent < b)).sum() / (b - a))


PREEMPT = "vllm:num_preemptions_total"


def preemptions(run: dict, a: float, b: float) -> float:
    """Reported, never judged (rule F d1 r2, DJ-O4): the rise of vllm:num_preemptions_total over [a, b), from the first
    to the last scrape inside it, as `tables._preemptions` reads a run. NaN if the counter is missing, if its scrapes
    do not cover the window (`load._covered` with c2.C2["gauge_max_gap_s"], as `c2.gauge`), or if it goes down inside
    (a restart: flagged, not hidden)."""
    from analysis.load import _covered
    m = run["metrics"]
    if not len(m) or PREEMPT not in m:
        return math.nan
    g = m[["t", PREEMPT]].dropna()
    if not _covered(g.t.to_numpy(float), a, b, C2["gauge_max_gap_s"]):
        return math.nan
    w = g[(g.t >= a) & (g.t < b)][PREEMPT]
    if len(w) < 2 or (w.diff().dropna() < 0).any():
        return math.nan
    return float(w.iloc[-1] - w.iloc[0])


def spans(run: dict) -> list[dict]:
    """Reported, never judged: per 300 s span from the burst's end, reuse (min turns), the server-side arrival rate,
    completions, waiting, N and KV (`c2.gauge`), preemptions (`preemptions`), the cap's queue, sessions in service, turn
    spacing."""
    from analysis.gate import burst_end, completion_rate
    end, out = burst_end(run), []
    for i in range(D1["n_spans"]):
        a, b = end + i * D1["span_s"], end + (i + 1) * D1["span_s"]
        out.append({"span": (a, b), "reuse": reuse_in(later_turns(run), a, b, C2["min_turns"]),
                    "server_rate": server_rate(run, a, b), "completion_rate": completion_rate(run, a, b),
                    "waiting": gauge(run, "vllm:num_requests_waiting", a, b), "n": gauge(run, N, a, b),
                    "kv": gauge(run, KV, a, b), "preemptions": preemptions(run, a, b),
                    "cap_queue": cap_queue_level(run, a, b)["mean"],
                    "in_service": sessions_in_service(run, a, b), "turn_spacing": turn_spacing(run, a, b)})
    return out


def clear_time(run: dict, exit_t: float) -> float | None:
    """Reported, never judged (rule F d1 r2, DJ-O5): the start of the first clear_s window, in steps of clear_s from the
    exit (`c2.last_burst_end`), whose reuse of later turns (at least c2.C2["min_turns"] turns) is back within
    recovered_hit_drop of the run's own baseline reuse, up to the end of the watch; None if no window is, or if the exit
    or the baseline reuse is unknown. Finer than the 300 s spans: it shows where a seed falls against the cut that the
    spans put between burst_only and formed_then_cleared (span 1 starts at 1380 s)."""
    v, w = run["cfg"]["verdict"], windows(run["cfg"])
    turns = later_turns(run)
    base = reuse_in(turns, *w["base"], C2["min_turns"])
    if math.isnan(exit_t) or math.isnan(base):
        return None
    step, end = D1["clear_s"], w["end"][1]
    i = 0
    while exit_t + (i + 1) * step <= end:
        a = exit_t + i * step
        r = reuse_in(turns, a, a + step, C2["min_turns"])
        if not math.isnan(r) and r >= base - v["recovered_hit_drop"]:
            return float(a)
        i += 1
    return None


def planned_end_rates(cfg: dict) -> dict:
    """Reported, never judged (rule F d1 r2, DJ-O2): planned session starts per second over the end window and over its
    two halves, from the schedule alone (turn 0 of the non-burst rows of `harness.workload.build_schedule` on the run's
    config). A cold seed's cap queue shrinks only where sessions finish faster than these."""
    from harness.workload import build_schedule
    s, _ = build_schedule(cfg["workload"], cfg["seed"], cfg.get("trigger"))
    t0 = s[(s.turn == 0) & ~s.burst].t_planned.to_numpy(float)
    e0, e1 = windows(cfg)["end"]
    mid = 0.5 * (e0 + e1)
    rate = lambda a, b: float(((t0 >= a) & (t0 < b)).sum() / (b - a))
    return {"end": rate(e0, e1), "first_half": rate(e0, mid), "second_half": rate(mid, e1)}


def load_reading(capped_spans: list[dict], uncapped_spans: list[dict] | None, base_reuse: float, rules: dict,
                 exit_t: float = math.nan) -> dict:
    """Reported, never judged (the separation readout). Over the spans from the burst's end up to the first span whose
    reuse is back within recovered_hit_drop of the baseline: the cold spans (reuse <= baseline - stuck_hit_drop), and in
    each the server-side arrival rate against the open-loop floor and against the same seed's uncapped REG-19 burst run
    in the same span, and whether the span starts at or after `exit_t`, the end of the last burst request
    (`c2.last_burst_end`; None if either is unknown). `reading`: never_cold (no cold span before the re-warm),
    no_rewarm, in_window (the server was offered at least the open-loop floor in every cold span before the re-warm),
    below_floor, or undefined (NaN). With a binding cap and closed-loop turns the server is sent what it completes, so
    in_window is expected whenever a cold span exists (rule F d1 r1, DG10 (e)): it is no evidence against admission."""
    if math.isnan(base_reuse):
        return {"reading": "undefined", "rewarm_span": None, "cold": [], "cold_any": [], "nan_any": True,
                "last_warm": None}
    up = uncapped_spans or [None] * len(capped_spans)
    after = lambda s: (None if s.get("span") is None or math.isnan(exit_t) else bool(s["span"][0] >= exit_t))
    # Every cold span of the watch, not only those before the first re-warm (rule F d1 r2, DJ2): a cold state can
    # form after an early re-warm, and a single warm span can interrupt one. `onset_class` reads these.
    cold_any = [{"span": s.get("span"), "after_exit": after(s)} for s in capped_spans
                if not math.isnan(s["reuse"]) and s["reuse"] <= base_reuse - rules["stuck_hit_drop"]]
    nan_any = any(math.isnan(s["reuse"]) for s in capped_spans)
    last = capped_spans[-1]["reuse"] if capped_spans else math.nan
    last_warm = None if math.isnan(last) else bool(last >= base_reuse - rules["recovered_hit_drop"])
    extra = {"cold_any": cold_any, "nan_any": nan_any, "last_warm": last_warm}
    rew = next((i for i, s in enumerate(capped_spans)
                if not math.isnan(s["reuse"]) and s["reuse"] >= base_reuse - rules["recovered_hit_drop"]), None)
    upto = capped_spans if rew is None else capped_spans[:rew]
    if any(math.isnan(s["reuse"]) for s in upto):
        return {"reading": "undefined", "rewarm_span": rew, "cold": [], **extra}
    cold = [{"span": s.get("span"), "server_rate": s["server_rate"],
             "uncapped_server_rate": None if u is None else u["server_rate"],
             "below_floor": s["server_rate"] < D1["floor_ol"], "after_exit": after(s)}
            for s, u in zip(upto, up) if s["reuse"] <= base_reuse - rules["stuck_hit_drop"]]
    if rew is None:
        reading = "no_rewarm"
    elif not cold:
        reading = "never_cold"
    else:
        reading = "below_floor" if any(c["below_floor"] for c in cold) else "in_window"
    return {"reading": reading, "rewarm_span": rew, "cold": cold, **extra}


def onset_class(lr: dict) -> str:
    """Reported, never judged (rule F d1 r1, DG10; d1 r2, DJ2): for a seed, whether a cold state formed and cleared,
    over every span of the watch (`load_reading`'s `cold_any`), not only those before the first re-warm.
    never_cold: no cold span in the watch (the cap kept the cold state from forming); burst_only: cold only in spans
    that start before the last burst request ended (the burst's own requests still in the system; REG-22 judges no
    such span, analysis/c2.py `sustain`); formed_then_cleared: a cold span that starts at or after the exit, and the
    last span warm; undefined: no re-warm, a NaN reuse in any span, the exit unknown while a cold span exists, or a
    cold span after the exit with the last span not warm. Only formed_then_cleared bears on whether the cap breaks a
    stuck state once it has formed."""
    r = lr["reading"]
    if r not in ("never_cold", "in_window", "below_floor") or lr.get("nan_any"):
        return "undefined"
    flags = [c["after_exit"] for c in lr.get("cold_any", [])]
    if not flags:
        return "never_cold"
    if any(f is None for f in flags):
        return "undefined"
    if not any(flags):
        return "burst_only"
    return "formed_then_cleared" if lr.get("last_warm") else "undefined"


def _longest_idle(s: np.ndarray, d: np.ndarray, arr: np.ndarray, adm: np.ndarray, cap: int, t_end: float) -> float:
    """The longest stretch in [0, t_end) with a session waiting at the cap (arr <= t < adm) while fewer than `cap`
    sessions hold a slot (s <= t < d), exact: both counts are right-continuous steps, constant between events."""
    if t_end <= 0:
        return 0.0
    fin = [x[np.isfinite(x) & (x >= 0) & (x < t_end)] for x in (s, d, arr, adm)]
    ev = np.unique(np.concatenate([np.zeros(1)] + fin))
    ss, dd, aa, mm = (np.sort(x) for x in (s, d, arr, adm))
    held = np.searchsorted(ss, ev, side="right") - np.searchsorted(dd, ev, side="right")
    wait = np.searchsorted(aa, ev, side="right") - np.searchsorted(mm, ev, side="right")
    idle = (held < cap) & (wait > 0)
    dur = np.where(idle, np.append(ev[1:], t_end) - ev, 0.0)
    return float(np.bincount(np.cumsum(~idle), weights=dur).max())


def slots(run: dict, cap: int) -> dict:
    """The cap's invariant, read from the records: sessions holding a slot (from admission, else first send, to the
    last reply) never exceed the cap (`max_held`, exact); and, reported only, the share of grid seconds with a session
    waiting at the cap while a slot was free (`idle_share`; a hand-over takes milliseconds, so not exactly 0), and the
    longest such stretch, exact on the event sweep over the same [0, duration_s) (`max_idle_stretch_s`; rule F d1 r2,
    DJ-O1; never judged)."""
    q = run["requests"]
    q = q[~is_burst_request(run)]
    g = q.groupby("session_id")
    t0 = q[q.turn == 0].set_index("session_id")
    held_from = (t0.t_admitted if "t_admitted" in t0 else t0.t_sent).fillna(t0.t_sent)
    done = g.t_end.max().where(g.t_end.apply(lambda s: s.notna().all()), np.inf)
    start = held_from.reindex(done.index)
    keep = start.notna()
    s, d = start[keep].to_numpy(float), done[keep].to_numpy(float)
    t = np.concatenate([s, d])
    step = np.concatenate([np.ones(len(s)), -np.ones(len(d))])
    order = np.lexsort((step, t))
    max_held = int(np.cumsum(step[order]).max()) if len(t) else 0
    t_end = float(run["cfg"]["workload"]["duration_s"])
    grid = np.arange(0.0, t_end, D1["grid_s"])
    held = np.searchsorted(np.sort(s), grid, side="right") - np.searchsorted(np.sort(d), grid, side="right")
    waiting = cap_queue(run, grid)
    busy = waiting > 0
    ct = _cap_times(run)
    stretch = 0.0 if ct is None else _longest_idle(s, d, ct[0], ct[1], cap, t_end)
    return {"cap": cap, "max_held": max_held, "holds": max_held <= cap,
            "idle_share": float(((held < cap) & busy).sum() / busy.sum()) if busy.any() else 0.0,
            "max_idle_stretch_s": stretch}


def harness_fault(sl: dict) -> bool:
    """The cap did not do what harness/loadgen.py says: more sessions held a slot than the cap, or sessions waited at
    the cap while a slot stood free (a leaked slot). Read from the records; `slots` output. A flag, not a label: the
    run counts in n as unclear (`row`; design review d1 r2, DM2)."""
    return (not sl["holds"]) or sl["idle_share"] > D1["max_idle_share"]


def possibly_stuck(r: dict) -> bool:
    """Reported, never judged (rule F d1 r2, DJ-O3; design review d1 r2, R16): a seed labelled rewarming or partial whose
    cap queue's end mean lies above its baseline mean + recovered_q_excess (`label`'s `cap_above_base`), draining or
    not. Such seeds are neither cut nor stuck and are already counted by `cp_upper_not_cut`."""
    return r["label"] in ("rewarming", "partial") and r["label_values"]["cap_above_base"] is True


def waits(run: dict) -> dict:
    """`cap.cap_wait` over all sessions, and apart for the sessions that reached the cap before the burst's start."""
    q = run["requests"]
    at = float(run["cfg"]["trigger"]["at_s"])
    pre = {**run, "requests": q[~((q.turn == 0) & q.t_arrived.notna() & (q.t_arrived >= at))]} if "t_arrived" in q else run
    return {"all": capa.cap_wait(run), "before_burst": capa.cap_wait(pre)}


# ---------------------------------------------------------------- the result

def _check_capped(r: dict, seed: int, cap: int) -> None:
    name = r["dir"].name
    if not (r["checks"] and r["checks"]["valid"]):
        raise ValueError(f"{name}: not valid")
    got = {k: v for k, v in r["cfg"].items() if k not in ("run_id", "source_config")}
    diff = _diff_keys(got, resolved(seed, cap))
    if diff:
        raise ValueError(f"{name}: config differs from D1's at {diff}")


def capped_run(seed: int, cap: int) -> tuple[dict | None, list[str]]:
    """The seed's counted capped run and the names of any later valid ones. A rerun follows only an invalid run, so two
    valid runs of one seed mean something outside the registration happened; as in E1 (`e1.first_valid`), the earliest
    counts (run ids start with their launch time) and the later ones are listed (rule F d1 r1, DG-O4). Invalid runs are
    skipped (`load.find_runs`, valid only). (None, []) if the seed has no valid run."""
    from analysis.load import find_runs
    runs = sorted(find_runs(run_pattern(seed)), key=lambda r: r["dir"].name)
    if not runs:
        return None, []
    _check_capped(runs[0], seed, cap)
    return runs[0], [r["dir"].name for r in runs[1:]]


def row(seed: int) -> dict | None:
    """One seed: the cap from its control, the counted capped run, its label and readouts, and REG-19's uncapped
    burst run's spans for the separation readout. None if the seed has no valid capped run. A run that `harness_fault`
    flags is labelled unclear and stays in n; `harness_fault` is True and `label_values["label"]` keeps the label its
    records would otherwise give (design review d1 r2, DM2)."""
    from analysis.gate import burst_end, reg19_baseline_pair, shock_size
    from analysis.c2 import eviction_readout
    from analysis.load import RUNS, load_run, verdict
    control = control_run(seed)
    cap = cap_value(control)
    run, later = capped_run(seed, cap)
    if run is None:
        return None
    un = load_run(RUNS / D1["uncapped"][seed])
    lab = label(run)
    sp, sp_un = spans(run), spans(un)
    sl = slots(run, cap)
    fault = harness_fault(sl)
    ex, ex_un = last_burst_end(run), last_burst_end(un)
    lr = load_reading(sp, sp_un, lab["reuse"]["base"], run["cfg"]["verdict"], ex)
    return {"seed": seed, "cap": cap, "run": run["dir"].name, "later_valid": later, "control": control["dir"].name,
            "uncapped": un["dir"].name,
            "label": "unclear" if fault else lab["label"], "harness_fault": fault, "label_values": lab,
            "verdict": verdict(run)["verdict"], "verdict_uncapped": verdict(un)["verdict"],
            "load_reading": lr, "onset": onset_class(lr),
            "spans": sp, "uncapped_spans": sp_un, "slots": sl, "waits": waits(run),
            "exit": ex, "exit_uncapped": ex_un,
            # reported, never judged (rule F d1 r2, DJ-O2, DJ-O4, DJ-O5; design review d1 r2, DM3)
            "planned_end": planned_end_rates(run["cfg"]), "cap_queue_at": cap_queue_at(run, burst_end(run), ex),
            "clear_60s": clear_time(run, ex), "clear_60s_uncapped": clear_time(un, ex_un),
            "levels": levels(run), "shock": shock_size(run), "shock_uncapped": shock_size(un),
            "baseline_pair": reg19_baseline_pair(control, run), "eviction": eviction_readout(run)}


def _cp_lower(k: int, n: int) -> float:
    """One-sided 95 % Clopper-Pearson lower bound, as 1 - cp_upper(n - k, n)."""
    return math.nan if not n else 1.0 - cp_upper(n - k, n, D1["conf"])


def outcome(labels: list[str], n_seeds: int | None = None) -> dict:
    """The registered decision on the labels of the valid capped runs (one per seed run); every one counts in n. A run
    that `harness_fault` flags comes in as unclear (`row`; design review d1 r2, DM2), and a label outside LABELS, the old
    harness_fault label among them, is refused (ValueError), so no path of the registered computation takes a valid
    capped run out of n. Rules, in this order: too_few (n < most) -> refutes (stuck >= most; claims.md counts any run
    stuck under the cap against C2, whatever the KV gauge, author 2026-11-14) -> supports (no stuck label at all and recovered + recovered_cap_draining >= most) ->
    admission_only (server_only >= most) -> mixed. Bounds (rule H, one-sided Clopper-Pearson), readouts: the upper bound
    on the share stuck under the cap, on k_stuck of n and on n - k_cut of n (every seed not cut counted as possibly
    stuck, rule F d1 r1 DG4); the upper and lower bounds on the share cut; `below_reg19`, whether the second stuck bound
    lies below REG-19's lower bound on the same seeds (a readout comparing two one-sided bounds of cells that share
    seeds, not a test); McNemar's exact one-sided p with the cut seeds as discordant pairs against REG-19's 5 of 5;
    `k_cut_server_side`, the count with server_only taken as cut (item 2's reading on the server side alone)."""
    bad = sorted({lab for lab in labels if lab not in LABELS})
    if bad:
        raise ValueError(f"labels outside LABELS: {bad}; a run flagged by harness_fault counts in n as unclear (DM2)")
    kept = list(labels)
    n = len(kept)
    k_cut = sum(lab in CUT for lab in kept)
    k_stuck, k_srv = kept.count("stuck"), kept.count("server_only")
    most = D1["most"]
    if n < most:
        dec = "too_few"
    elif k_stuck >= most:
        dec = "refutes"            # one stuck class: DJ1's split and cap_did_not_cut are folded in (author 2026-11-14)
    elif k_stuck == 0 and k_cut >= most:
        dec = "supports"
    elif k_srv >= most:
        dec = "admission_only"
    else:
        dec = "mixed"
    ub = (lambda k: cp_upper(k, n, D1["conf"])) if n else (lambda k: math.nan)
    reg19_lower = _cp_lower(*D1["reg19_k_n"])
    ub_not_cut = ub(n - k_cut)
    return {"decision": dec, "n": n, "n_seeds": len(labels) if n_seeds is None else n_seeds, "k_cut": k_cut, "k_stuck": k_stuck,
            "k_server_only": k_srv, "k_partial": kept.count("partial"),
            # readout, never judged: item 2 read on the server side alone (decisions.md 2026-11-12, item 2:
            # "judged on the server's hit rate and waiting line"), under which server_only would count as cut
            "k_cut_server_side": k_cut + k_srv,
            "cp_upper_stuck": ub(k_stuck), "cp_upper_not_cut": ub_not_cut,
            "cp_upper_cut": ub(k_cut), "cp_lower_cut": _cp_lower(k_cut, n),
            "reg19_lower": reg19_lower, "below_reg19": bool(ub_not_cut < reg19_lower) if n else None,
            "mcnemar_p": mcnemar_one_sided(k_cut, 0) if n else math.nan,
            "counts": {lab: kept.count(lab) for lab in sorted(set(kept))}}


def predicted_matrix(p: dict | None = None, n: int | None = None) -> dict:
    """Prediction only (reads no run): for n seeds whose labels are drawn independently from `p` (D1["prediction"],
    BELIEF), the probability of each matrix decision of `outcome`, of `supports` with `below_reg19`, E[k_cut],
    E[k_stuck] and P(k_cut = j); exact, over the multisets of n labels. The harness_fault belief is counted as unclear
    before `outcome` is called, as `row` labels a flagged run (design review d1 r2, DM2)."""
    from itertools import combinations_with_replacement
    p = dict(p or D1["prediction"])
    n = n or len(D1["seeds"])
    if abs(sum(p.values()) - 1.0) > 1e-9:
        raise ValueError(f"the label probabilities sum to {sum(p.values())}, not 1")
    if "harness_fault" in p:
        p["unclear"] = p.get("unclear", 0.0) + p.pop("harness_fault")
    dec, k_cut, out = Counter(), [0.0] * (n + 1), {"e_k_cut": 0.0, "e_k_stuck": 0.0, "p_supports_below_reg19": 0.0}
    for combo in combinations_with_replacement(sorted(p), n):
        c = Counter(combo)
        w = math.factorial(n) / math.prod(math.factorial(x) for x in c.values())
        w *= math.prod(p[lab] ** x for lab, x in c.items())
        o = outcome(list(combo))
        dec[o["decision"]] += w
        k_cut[o["k_cut"]] += w
        out["e_k_cut"] += w * o["k_cut"]
        out["e_k_stuck"] += w * o["k_stuck"]
        if o["decision"] == "supports" and o["below_reg19"]:
            out["p_supports_below_reg19"] += w
    return {"decision": dict(sorted(dec.items())), "p_k_cut": k_cut, **out}


def onset_split(rows: list[dict]) -> dict:
    """Reported, never judged (DG10): the cut seeds counted by `onset_class`."""
    cut = [r for r in rows if r["label"] in CUT]
    return {k: sum(r["onset"] == k for r in cut) for k in ONSET}


def drift_readout() -> dict:
    """Reported, never judged (DG-O2): REG-20's drift rerun of seed 15 (closed loop, REG-19's config renamed, on the
    new harness), only once REG-20's own [RESULT] has lifted its seal. D1's command sets AFTERSHOCK_UNSEAL=D1 only, so a
    folder still sealed by REG-20 is named and not read. The earliest valid run counts."""
    from analysis import withheld
    from analysis.k1 import cold_backlogged
    from analysis.load import RUNS, find_runs, verdict
    names = sorted(p.name for p in RUNS.glob(D1["drift_pattern"]))
    if not names:
        return {"status": "none"}
    by = sorted({withheld.sealed_by(x) for x in names} - {None})
    if by:
        return {"status": "sealed", "by": by, "folders": len(names)}
    runs = sorted(find_runs(D1["drift_pattern"]), key=lambda r: r["dir"].name)
    if not runs:
        return {"status": "no_valid_run", "folders": names}
    r = runs[0]
    return {"status": "read", "run": r["dir"].name, "later_valid": [x["dir"].name for x in runs[1:]],
            "verdict": verdict(r)["verdict"], "cold_backlogged": cold_backlogged(r)["cold_backlogged"]}


def _sealed_capped() -> list[str]:
    from analysis import withheld
    from analysis.load import RUNS
    return [p.name for sd in D1["seeds"] for p in sorted(RUNS.glob(run_pattern(sd))) if withheld.sealed_by(p.name)]


def table() -> dict:
    """D1's result: one row per seed with a valid capped run, in D1's order, then `outcome`. A seed with no valid run
    (attempts spent, or the stopping rule ended D1 first) is listed in `not_run`. Reported beside the decision, never
    judged: `harness_faults`, the seeds `harness_fault` flags (each counted in n as unclear), and `decision_underlying`,
    the decision `outcome` gives on every row's `label_values["label"]` (design review d1 r2, DM2); `possibly_stuck`, the
    seeds `possibly_stuck` names (DJ-O3, DM3); `stuck_kv_class`, each stuck seed's KV class. Refuses while any capped folder is
    sealed (run without AFTERSHOCK_UNSEAL=D1 before the [RESULT]): `find_runs` would skip it, and every seed would read
    as not run (the REG-20 pattern-bug lesson, decisions.md:4133)."""
    from analysis import withheld
    sealed = _sealed_capped()
    if sealed:
        raise withheld.WithheldError(f"{len(sealed)} capped folder(s) sealed by D1; the registered computation sets "
                                     f"AFTERSHOCK_UNSEAL=D1")
    rows, not_run = [], []
    for sd in D1["seeds"]:
        r = row(sd)
        if r is None:
            not_run.append(sd)
        else:
            rows.append(r)
    dec = outcome([r["label"] for r in rows], n_seeds=len(D1["seeds"]))
    under = outcome([r["label_values"]["label"] for r in rows], n_seeds=len(D1["seeds"]))["decision"]
    return {**dec, "not_run": not_run, "harness_faults": [r["seed"] for r in rows if r["harness_fault"]],
            "decision_underlying": under, "possibly_stuck": [r["seed"] for r in rows if possibly_stuck(r)],
            "onset_split": onset_split(rows),
            # readout, never judged (author 2026-11-14): condition 4's KV class of each stuck seed, underlying labels
            "stuck_kv_class": {r["seed"]: r["label_values"]["kv_class"] for r in rows
                               if r["label_values"]["label"] in STUCK},
            "later_valid": {r["seed"]: r["later_valid"] for r in rows if r["later_valid"]},
            "drift": drift_readout(), "rows": rows}


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1]
    if cmd == "caps":
        res = cap_values()
        for r in res["rows"]:
            print(json.dumps(r, default=str))
        print(json.dumps({"caps": res["caps"]}))
    elif cmd == "configs":
        print([str(p) for p in write_configs(caps_from_file(Path(sys.argv[2])))])
    elif cmd == "inputs":
        print(json.dumps(inputs(), default=str))
    elif cmd == "chain":
        print("\n".join(sorted({f for sd in D1["seeds"] for f in config_chain(sd)})))
    elif cmd == "prediction":
        print(json.dumps(predicted_matrix(), default=str))
    elif cmd == "d1":
        print(json.dumps(table(), default=str))
    else:
        raise SystemExit("usage: python -m analysis.d1 caps | configs <caps file> | inputs | chain | prediction | d1")
