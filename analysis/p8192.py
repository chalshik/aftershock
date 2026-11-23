"""P8192 (draft, to be registered: `docs/drafts/p8192_draft1.md`; author, decisions.md 2026-11-21 [DECISION] (author)
P0 at 8192 …): REG-19's burst test at vLLM's 8192-token step budget.

T4 found that at 8192 the closed-loop cold floor is 1.931 req/s and the warm capacity 2.4 (decisions.md 2026-11-21
[RESULT] T4 (step budget)), so lambda = 2.0 sits outside the window there. This test puts the load at lambda = 2.2:
1.139 x the 8192 floor and 0.917 of its warm capacity, the default cell's place in its window (1.118 x, 0.909).

Design: REG-19's configs and test, with four changes only:
- `--max-num-batched-tokens 8192`;
- T4's pool pin, `--num-gpu-blocks-override 6041`;
- lambda = 2.2;
- five new seeds chosen by `choose_seeds`, a rule on schedules only.
Everything else is REG-19's: closed-loop L1 at 350 W, the burst at 900 s for 180 s at 2.4 req/s of 2944-token
prompts, arrivals to 4080 s, the 3000 s watch, and `gate.reg19_seed_outcome` with its whole-window cold side, judged
against lambda = 2.2 (`cold_side`, REG-19's code with this lambda).

Seed rule (`choose_seeds`, fixed before any schedule is computed): the first `n_seeds` seeds from `seed_from` upward
that no config uses and whose control schedule at lambda = 2.2 to 4080 s has no 120 s planned session rate >=
`calm_peak` (3.1, K4's calm-arm bound). This is a selection on the workload, stated as "calm schedules".

Validity (beyond the harness checks): every counted run passes T4's server check for the 8192 arm (`t4.check_settings`
on `t4.server_readouts`): the 6041-block pool took effect and `max_num_batched_tokens` is 8192.

Judged: `gate.reg19_outcome` (REG-18's matrix rule) over the seeds' `pair` outcomes; k of n with the one-sided 95 %
Clopper-Pearson upper bound. REG-19's `outcome_18` readout (REG-17's per-span rule, which judges against REG-17's
lambda 2.0) is not computed here.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np

from analysis import gate, withheld

REPO = Path(__file__).resolve().parent.parent

P8192 = {**gate.REG19,
         "lam": 2.2,
         "budget": 8192,
         "pool_blocks": 6041,
         "seed_from": 1000, "n_seeds": 5,
         "calm_peak": 3.1,          # K4's calm-arm bound (decisions.md 2026-11-17 [RESULT] K4 ...)
         "template": "stage2_control_l1pl350cl_r2.0_w3000_s15",   # REG-19's control: its workload, at lambda 2.2
         "seeds": (1002, 1004, 1007, 1008, 1009),   # `choose_seeds` at 5849afe (PROBE(docs/probe/p8192/seeds_*.txt));
                                                     # checked by tests/test_p8192.py
         "names": {"control": "stage2_control_l1pl350cl_r2.2_b8192_s{}", "burst": "burst_l1pl350cl_r2.2_b8192_s{}"},
         "check_settings": True,    # T4's 8192 server check on every counted run (False only in the equivalence test)
         "floor": 1.931,            # T4's 8192 cold floor (RUN(T4's runs, t4.table)); a readout's yardstick only
         "seal": "[RESULT] P8192 (P0 at an 8192-token step budget)"}


def config_name(kind: str, seed: int, p: dict = P8192) -> str:
    """kind: 'burst' or 'control'."""
    return p["names"][kind].format(seed)


def spec(seed: int, p: dict = P8192) -> dict:
    return {**p, "seed": seed, "control": f"*_{config_name('control', seed, p)}",
            "burst": f"*_{config_name('burst', seed, p)}"}


def used_seeds() -> set[int]:
    """Every top-level `seed:` in configs/ (a child without one inherits its parent's, which is listed in the
    parent's file), plus base.yaml's default 0."""
    out = {0}
    for f in (REPO / "configs").glob("*.yaml"):
        for m in re.finditer(r"^seed:\s*(\d+)\s*$", f.read_text(), re.M):
            out.add(int(m.group(1)))
    return out


def peak(seed: int, p: dict = P8192) -> dict:
    """Schedule only: the control's highest 120 s planned session rate at lambda over [start_s, 4080 s), and when."""
    c = gate._burst_cfg(p["template"])
    g, x = gate.schedule_peak_series(c, p["lam"], seed, p["duration_s"])
    return {"seed": seed, "peak": float(x.max()), "t_peak": float(g[x.argmax()])}


def choose_seeds(p: dict = P8192, used: set[int] | None = None, peak_fn=None) -> dict:
    """The seed rule (module docstring): candidates from seed_from upward, skipping used seeds, taking the first
    n_seeds whose peak is below calm_peak. Returns the chosen seeds and every candidate's peak, in order."""
    used = used_seeds() if used is None else used
    peak_fn = peak_fn or (lambda s: peak(s, p))
    seen, chosen, s = [], [], p["seed_from"]
    while len(chosen) < p["n_seeds"]:
        if s not in used:
            r = peak_fn(s)
            r["calm"] = bool(r["peak"] < p["calm_peak"])
            seen.append(r)
            if r["calm"]:
                chosen.append(s)
        s += 1
        if s > p["seed_from"] + 10_000:
            raise RuntimeError("no calm seeds found")
    return {"seeds": tuple(chosen), "candidates": seen}


def exposed(seed: int, p: dict = P8192) -> dict:
    """`gate.reg19_exposed` on this test's control config: the highest 120 s planned rate, whether it reaches
    REG-17's surge_tip (3.2), and when it first does. With the seed rule no seed can be exposed (3.1 < 3.2)."""
    c = gate._burst_cfg(config_name("control", seed, p))
    g, x = gate.schedule_peak_series({"workload": {**c["workload"], "steps": None}}, float(c["workload"]["rate"]),
                                     int(c["seed"]), float(c["workload"]["duration_s"]))
    i = np.flatnonzero(x >= p["surge_tip"])
    return {"seed": seed, "peak": float(x.max()), "t_peak": float(g[x.argmax()]),
            "exposed": bool(len(i)), "t_first": float(g[i[0]]) if len(i) else math.nan}


def order(p: dict = P8192) -> tuple[int, ...]:
    """REG-19's rule: exposed seeds first, then the rest, ascending within each group."""
    ex = {sd: exposed(sd, p)["exposed"] for sd in p["seeds"]}
    return tuple(sorted(p["seeds"], key=lambda sd: (not ex[sd], sd)))


def cold_side(burst: dict, p: dict = P8192) -> dict:
    """`gate.reg19_cold_side` with this test's lambda (REG-19's judges against REG19['lam'] = 2.0): over
    W = [end + 600, end + 3000), the completion rate is below min(lambda, W's planned session rate)."""
    end, lam = gate.burst_end(burst), p["lam"]
    a, b = p["window"]
    c, lp = gate.completion_rate(burst, end + a, end + b), gate.planned_session_rate(burst, end + a, end + b)
    ok = np.isfinite(c) and np.isfinite(lp)
    spans = []
    for x, y in gate.reg19_spans():
        cs, ls = gate.completion_rate(burst, end + x, end + y), gate.planned_session_rate(burst, end + x, end + y)
        spans.append({"span": (x, y), "c": cs, "lam_p": ls,
                      "below": None if not (np.isfinite(cs) and np.isfinite(ls)) else bool(cs < min(lam, ls))})
    m = burst["metrics"]
    q = m["vllm:num_requests_waiting"][(m.t >= end + a) & (m.t < end + a + p["span_s"])]
    q_first = float(q.mean()) if len(q) else math.nan
    q_end = gate.verdict(burst)["end"]["q"]
    return {"c": c, "lam_p": lp, "holds": bool(c < min(lam, lp)) if ok else None, "spans": spans,
            "n_spans_below": sum(r["below"] is True for r in spans), "q_first": q_first, "q_end": q_end,
            "q_ratio": float(q_end / q_first) if np.isfinite(q_first) and q_first > 0 else math.nan}


def check_run(r: dict, kind: str, seed: int, p: dict = P8192) -> list[str]:
    """The config and the server settings of one counted run (module docstring, Validity)."""
    from analysis import t4
    cfg, bad = r["cfg"], []
    if cfg["seed"] != seed or float(cfg["workload"]["rate"]) != p["lam"]:
        bad.append(f"seed {cfg['seed']} / rate {cfg['workload']['rate']}")
    if (float(cfg["verdict"]["watch_s"]) != p["watch_s"] or float(cfg["workload"]["duration_s"]) != p["duration_s"]):
        bad.append("not a 3000 s watch with arrivals to 4080 s")
    if bool(cfg.get("trigger")) != (kind == "burst"):
        bad.append("trigger does not match the kind")
    return bad + (t4.check_settings("on_8192", t4.server_readouts(r["dir"])) if p["check_settings"] else [])


def pair(seed: int, p: dict = P8192) -> dict:
    """`gate.reg19_pair` for this test: exactly one valid control and burst run of the seed (`gate.reg12_only`), each
    passing `check_run`; `gate.reg10_gate` on this spec (lambda 2.2); this test's cold side; REG-19's drain time,
    control waiting over W's eight spans, surge readouts, after-empty labels, owed backlog and baseline pair; and
    `gate.reg19_seed_outcome`."""
    from analysis.load import RUNS, find_runs, load_run
    sp = spec(seed, p)
    for kind in ("control", "burst"):
        r = gate.reg12_only(find_runs(sp[kind]), sp[kind])
        bad = check_run(r, kind, seed, p)
        if bad:
            raise ValueError(f"{r['dir'].name}: {bad}")
    res = gate.reg10_gate(sp)
    control, burst = (load_run(RUNS / n) for n in res["runs"])
    end = gate.burst_end(burst)
    res["seed"] = seed
    res["control_waiting"] = gate.control_waiting(control, [(end + a, end + b) for a, b in gate.reg19_spans()])
    res["surge_control"], res["surge_burst"] = gate.surge_run(control), gate.surge_run(burst)
    res["cold_side"], res["drain_time_s"] = cold_side(burst, p), gate.reg19_drain_time(burst)
    res["drained"] = bool(np.isfinite(res["drain_time_s"]))
    res["emptyings_s"] = gate.reg19_emptyings(burst)
    res["after_empty"] = gate.reg19_after_empty(burst, res["drain_time_s"])
    res["end_window"] = gate.end_window(burst)
    a, b = p["window"]
    res["owed"] = {"window_start": gate.owed_backlog(burst, end + a), "window_end": gate.owed_backlog(burst, end + b)}
    res["baseline_pair"] = gate.reg19_baseline_pair(control, burst)
    res["exposure"] = exposed(seed, p)
    res["exposed"] = res["exposure"]["exposed"]
    res["c1"] = res["burst_verdict"] == "stuck" and res["control_verdict"] == "recovered"
    c, f = res["cold_side"]["c"], p["floor"]   # readout (rule F r1, P-3): where W's completions sit between floor and λ
    res["c_rel_floor"] = float((c - f) / (p["lam"] - f)) if np.isfinite(c) and p["lam"] != f else math.nan
    res["outcome"] = gate.reg19_seed_outcome(res["control_verdict"], res["burst_verdict"], res["surge_control"],
                                             res["control_waiting"], res["cold_side"]["holds"], res["drained"],
                                             res["exposure"], res["after_empty"]["rewarmed"])
    return res


def step(p: dict = P8192, pair_fn=None, has_fn=None) -> dict:
    """The seeds done in `order()`, never skipping a seed, and REG-18's futility flag (`reg18_continue`) as a readout.
    The queue runs all ten items: the detached launcher has no hook for a futility stop (rule F r1, P-4), so the stop
    is not applied, which changes the cost of a failing test, not its outcome."""
    from analysis.load import find_runs
    pair_fn = pair_fn or (lambda s: pair(s, p))
    has_fn = has_fn or (lambda s: tuple(bool(find_runs(spec(s, p)[k])) for k in ("control", "burst")))
    od, done, gap = order(p), [], None
    for sd in od:
        has = has_fn(sd)
        if gap is not None and any(has):
            raise ValueError(f"seed {sd} has a valid run but seed {gap} has no valid pair: P8192 never skips a seed")
        if gap is None and all(has):
            done.append(sd)
        elif gap is None:
            gap = sd
    rows = {}
    for sd in done:
        r = pair_fn(sd)
        rows[sd] = {"outcome": r["outcome"], "control_verdict": r["control_verdict"], "exposed": r["exposed"]}
    remaining = len(od) - len(done)
    return {"done": done, "continue": bool(remaining and gate.reg18_continue(rows, remaining))}


def sealed() -> list[str]:
    import analysis.load as L
    if not L.RUNS.exists():
        return []
    return sorted(d.name for d in L.RUNS.iterdir() if "_b8192_s" in d.name and withheld.sealed_by(d.name) == "P8192")


def table(p: dict = P8192) -> dict:
    """The registered computation, once after the queue's end entry:
    `AFTERSHOCK_UNSEAL=P8192 uv run python -m analysis.p8192 table`. Refuses while any of its folders is sealed or
    before the seeds are filled in."""
    from analysis.stats import cp_upper
    if len(p["seeds"]) != p["n_seeds"]:
        raise ValueError("P8192's seeds are not filled in")
    s = sealed()
    if s:
        raise withheld.WithheldError(f"{len(s)} P8192 folder(s) sealed; set AFTERSHOCK_UNSEAL=P8192")
    od, done = order(p), step(p)["done"]
    rows = [pair(sd, p) for sd in done]
    dec = gate.reg19_outcome({r["seed"]: {"outcome": r["outcome"], "control_verdict": r["control_verdict"],
                                          "exposed": r["exposed"]} for r in rows})
    counts = {}
    for r in rows:
        counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
    early = len(done) < len(od)
    keys = ("seed", "control_verdict", "burst_verdict", "label", "outcome", "c1", "exposed", "drain_time_s", "owed",
            "runs")
    return {**dec, "order": od, "seeds_run": done, "stopped_early": early,
            "cp_upper": math.nan if early or not dec["n"] else cp_upper(dec["k"], dec["n"], p["conf"]),
            "rows": [{key: r[key] for key in keys} | {"cold_side": r["cold_side"]["holds"], "c": r["cold_side"]["c"],
                                                       "lam_p": r["cold_side"]["lam_p"],
                                                       "q_ratio": r["cold_side"]["q_ratio"],
                                                       "end_q_slope": r["end_window"]["q_slope"],
                                                       "unreleased_end": r["end_window"]["unreleased"],
                                                       "rewarmed": r["after_empty"]["rewarmed"],
                                                       "emptyings_s": r["emptyings_s"],
                                                       "baseline_diff": r["baseline_pair"]["diff"],
                                                       "order_flag": r["baseline_pair"]["order_flag"],
                                                       "t_recold": r["after_empty"]["t_recold"],
                                                       "c_rel_floor": r["c_rel_floor"],
                                                       "control_tipped": bool(r["surge_control"]["tipped"])}
                     for r in rows],
            "k_c1": sum(r["c1"] for r in rows), "n_drains": sum(r["outcome"] == "drains" for r in rows),
            "outcome_counts": counts, "details": rows}


if __name__ == "__main__":
    import sys
    from pprint import pprint
    cmd = sys.argv[1:]
    if cmd == ["table"]:
        pprint(table(), width=120, sort_dicts=False)
    elif cmd == ["step"]:
        pprint(step(), width=120, sort_dicts=False)
    elif cmd == ["seeds"]:
        pprint(choose_seeds(), width=120, sort_dicts=False)
    else:
        sys.exit("usage: python -m analysis.p8192 table | step | seeds")
