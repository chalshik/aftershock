"""REG-28 (docs/drafts/reg28_draft3.md): two blind readings of REG-27's stuck burst runs, registered before REG-27's
[RESULT] and computed once after it (K3b's pattern: the code refuses until the heading exists and never sets
AFTERSHOCK_UNSEAL).

Window W per counted burst run: [burst end + 600 s, watch end), `confirm.phases`. A burst run counts when REG-27's
code labels it stuck through the watch, cold and never emptied (`STUCK_WATCH`: `confirms`, `no_return`,
`drop_inconclusive`), whatever its control (the control is reported).

Q1 (Z3, model F's session law). Observed: the share of completed session turns in W that end a session, times T
(8 x last turns ok / session turns ok, by t_end). Quasi-static reading (every admitted session finishes at the cold
rate, no growth of the session backlog): 1. Model F: the same share in W of model F's replay of the run's own schedule
(`modelf_sim.simulate`, REG-21's FIFO all-miss server, proportional cost), at the all-miss capacity mu_X at which the
replay reproduces the run's own session-turn completion rate in W (`k3.local_fit` over `mu_grid`). The replay carries
the burst's transient and the run's schedule into the prediction (rule F r1 B1: no long-run value on a ~1.6-lifetime
window), and mu is fitted, not set to the completion rate (r1 B2: model F serves 1.016-1.025 x mu). Judged on the mean
over counted runs of obs - pred (model F) and of obs - 1 (quasi-static), each against `tol`; per-run k of n reported.
What `law_holds` shows (rule F r2 N2): a FIFO all-miss replay at the fitted mu reproduces the session share; the
all-miss assumption itself is K3's token-work reading, not tested here. Sensitivity (r2 N1): the prediction is also
computed with every request a miss only from the burst's end, and with the run's measured residence after prefill
(`decode_s`, REG-21's variant); pred is the mean of the three, and a run whose three spread by more than half its
own gap between the two readings (1 - pred) is listed (`replay_sensitive`) and not counted: the replay cannot then
say which reading the run is nearer. The rule was set after the schedule-only probe on REG-27's four configs
(PROBE(docs/probe/reg28_draft/replay_sensitivity.txt): spreads 0.009-0.034, gaps 0.074-0.18), before any run was
read.

Q2 (Z5, preemption churn). An upper bound on the share of the stuck engine's prompt work that is re-prefill after
preemption: the preemption counter's increase in W (`order.preemptions`) x the largest L1 request (3968 + 128 tokens)
over the uncached prompt tokens of the ok requests whose first token falls in W (`k3w.work_rate` x |W|). An upper
bound can bound churn, never name it: above `churn_bounded` the label is `not_bounded` (rule F r1 B3).

Blindness (rule F r1 B4): `reg27.table` prints completion rates and preemption counts over the same W, so this
registration is committed before REG-27's computation first runs, not only before its heading.
"""
from __future__ import annotations

import math

import numpy as np

from analysis import k3, k3w, modelf, modelf_sim, withheld

REG28 = {
    "after": "[RESULT] REG-27",
    "offset_s": 600.0,                 # W starts 600 s after the burst's end (REG-19's and REG-21's offset)
    "turns": 8,
    "max_request_tokens": 3968 + 128,  # turn 8's prompt + max output (PROTOCOL L1)
    "tol": 0.03,                       # Q1: |mean deviation| inside this is "inside"
    "churn_bounded": 0.05,             # Q2: every run at or below this
    "min_runs": 3,
    "mu_ref": 1.681,                   # readout only: model F at docs/model.md M5's rho
    "mu_grid": (1.50, 2.10, 0.01),     # req/s: where mu_X is searched (k3.local_fit)
    "cost": (0.0, 1.0),                # proportional turn cost (docs/model.md M5)
    "sens_frac": 0.5,                  # a run counts only if the replays' spread <= this x its own gap 1 - pred
}
STUCK_WATCH = ("confirms", "no_return", "drop_inconclusive")
REG19_PATTERN = "*_burst_l1pl350cl_r2.0_w3000_s1[5-9]"   # retrospective readout (seen runs)


def window(run: dict) -> tuple[float, float]:
    """W for REG-24's design (`confirm.phases`)."""
    from analysis import confirm
    ph = confirm.phases(run["cfg"])
    return ph["burst_end"] + REG28["offset_s"], ph["watch_end"]


def window_reg19(run: dict) -> tuple[float, float]:
    """W for REG-19's 3000 s watch: [burst end + 600 s, burst end + watch_s)."""
    e = k3.burst_end(run)
    return e + REG28["offset_s"], e + float(run["cfg"]["verdict"]["watch_s"])


def predicted_share(lam: float, c: float, turns: int = REG28["turns"]) -> float:
    """Model F's T x (sessions completed / turns completed) at rho = lam / c; NaN at or below c (model F drains)."""
    if not (lam > c > 0):
        return math.nan
    t = modelf.l1(lam, c, turns)
    return float(t.sessions / t.requests)


def share_obs(run: dict, a: float, b: float, turns: int = REG28["turns"]) -> dict:
    """T x last turns / session turns, ended ok in [a, b), session turns only; and the session-turn rate."""
    w = k3.completed(run, a, b)
    n, last = len(w), int((w.turn == turns - 1).sum())
    return {"turns_ok": n, "last_ok": last, "X": n / (b - a), "obs": turns * last / n if n else math.nan}


def replay_share(cfg: dict, mu: float, win: tuple[float, float], sched=None, p: dict = REG28, **kw) -> dict:
    """Model F's replay of the schedule at all-miss capacity mu: share and session-turn rate in the same W. `kw`
    (all_miss_from, decode_s) go to `modelf_sim.simulate` for the sensitivity variants."""
    sim = modelf_sim.as_run(cfg, modelf_sim.simulate(cfg, mu, p["cost"], sched=sched, **kw))
    return share_obs(sim, *win, p["turns"])


def session_share(run: dict, a: float, b: float, turns: int = REG28["turns"], p: dict = REG28) -> dict:
    """Q1's observation over [a, b) against model F's replay at the fitted mu_X."""
    o = share_obs(run, a, b, turns)
    lam = k3.lam_w(run, a, b, turns)
    cfg = run["cfg"]
    sched = modelf_sim.schedule(cfg)
    lo, hi, st = p["mu_grid"]
    g = np.round(np.arange(lo, hi + st / 2, st), 6)
    reps = [replay_share(cfg, float(m), (a, b), sched, p) for m in g]
    fit = k3.local_fit(g, np.array([r["X"] for r in reps]), o["X"])
    pred = alt_end = alt_dec = math.nan
    res = residence(run, a, b)
    if fit["status"] == "ok":
        pred = replay_share(cfg, fit["mu"], (a, b), sched, p)["obs"]
        alt_end = replay_share(cfg, fit["mu"], (a, b), sched, p, all_miss_from=k3.burst_end(run))["obs"]
        if not math.isnan(res):
            alt_dec = replay_share(cfg, fit["mu"], (a, b), sched, p, decode_s=res)["obs"]
    vals = [x for x in (pred, alt_end, alt_dec) if not math.isnan(x)]
    spread = max(vals) - min(vals) if vals else math.nan
    mean_pred = sum(vals) / len(vals) if vals else math.nan
    sensitive = bool(vals) and spread > p["sens_frac"] * (1.0 - mean_pred)
    counted_pred = math.nan if sensitive else mean_pred
    return {**o, "lam_W": lam, "mu_X": fit["mu"], "mu_X_status": fit["status"], "mu_X_jitter": fit["jitter"],
            "mu_X_slope": fit["slope"], "pred_F_all": pred, "pred_F_from_burst_end": alt_end,
            "pred_F_residence": alt_dec, "residence_s": res, "replay_spread": spread, "replay_sensitive": sensitive,
            "pred_F": counted_pred, "dev_F": o["obs"] - counted_pred, "dev_Q": o["obs"] - 1.0,
            "gap_F_Q": 1.0 - mean_pred,
            "long_run_F_mu_ref": predicted_share(lam, p["mu_ref"], turns),
            "nearer": ("F" if abs(o["obs"] - pred) < abs(o["obs"] - 1.0) else "Q") if not math.isnan(pred) else None}


def residence(run: dict, a: float, b: float) -> float:
    """Mean time from first token to end of the ok session turns ending in [a, b) (REG-21's decode residence)."""
    w = k3.completed(run, a, b)
    d = (w.t_end - w.t_first_token).dropna()
    return float(d.mean()) if len(d) else math.nan


def churn(run: dict, a: float, b: float) -> dict:
    """Q2's bound over [a, b)."""
    from analysis import order
    p = order.preemptions(run, {"at": a, "watch_end": b, "attempt": None})
    u = k3w.work_rate(run, a, b)
    tok = u * (b - a) if not math.isnan(u) else math.nan
    cnt = p["count"]
    share = cnt * REG28["max_request_tokens"] / tok if tok and not (math.isnan(cnt) or math.isnan(tok)) else math.nan
    return {"preemptions": cnt, "uncached_tokens": tok, "share": share}


def _readout(fn, *a):
    try:
        return fn(*a)
    except Exception as e:   # noqa: BLE001 — readouts are never decisive
        return {"error": f"{type(e).__name__}: {e}"}


def row(run: dict, win: tuple[float, float]) -> dict:
    a, b = win
    nan1 = {"obs": math.nan, "dev_F": math.nan, "dev_Q": math.nan}
    q1 = _readout(session_share, run, a, b)
    q2 = _readout(churn, run, a, b)
    return {"run": run["dir"].name, "W": [a, b], "q1": q1 if "error" not in q1 else {**nan1, **q1},
            "q2": q2 if "error" not in q2 else {"share": math.nan, **q2}}


def judge_q1(rows: list[dict], p: dict = REG28) -> dict:
    ok = [r for r in rows if not math.isnan(r["q1"]["dev_F"]) and not math.isnan(r["q1"]["obs"])]
    if len(ok) < p["min_runs"]:
        return {"label": "too_few", "n": len(ok)}
    dF = sum(r["q1"]["dev_F"] for r in ok) / len(ok)
    dQ = sum(r["q1"]["dev_Q"] for r in ok) / len(ok)
    inF, inQ = abs(dF) <= p["tol"], abs(dQ) <= p["tol"]
    lab = ("not_separated" if inF and inQ else "law_holds" if inF else "quasi_static" if inQ else "neither")
    return {"label": lab, "n": len(ok), "mean_dev_F": dF, "mean_dev_Q": dQ,
            "side_F": "high" if dF > 0 else "low",
            "nearer_F": sum(r["q1"].get("nearer") == "F" for r in ok),
            "left_out": [r.get("run") for r in rows if r not in ok],
            "replay_sensitive": [r.get("run") for r in rows if r["q1"].get("replay_sensitive")]}


def judge_q2(rows: list[dict], p: dict = REG28) -> dict:
    s = [r["q2"]["share"] for r in rows]
    ok = [x for x in s if not math.isnan(x)]
    if len(ok) < p["min_runs"]:
        return {"label": "too_few", "n": len(ok), "shares": s}
    lab = "churn_bounded" if all(x <= p["churn_bounded"] for x in ok) else "not_bounded"
    return {"label": lab, "n": len(ok), "max": max(ok), "shares": s}


def table() -> dict:
    """REG-28's computation, once, after "[RESULT] REG-27" is in decisions.md: `python -m analysis.reg28 table`."""
    if not withheld.lifted(REG28["after"]):
        raise withheld.WithheldError("REG-28 reads REG-27's runs only after the heading " + REG28["after"])
    from analysis import confirm, reg27
    from analysis.load import RUNS, find_runs, load_run
    r27 = reg27.table()
    rows, left = [], {}
    for sd, r in r27["rows"].items():
        b = (r.get("table") or {}).get("burst")
        lab = r.get("burst")
        if not b or lab not in STUCK_WATCH:
            left[sd] = lab
            continue
        run = load_run(RUNS / b["run"])
        rows.append({"seed": sd, "burst_label": lab, "control": r.get("control"),
                     "same_build": (r.get("table") or {}).get("same_build"), **row(run, window(run))})
    retro = {"reg24_s43": _readout(lambda: (lambda t: row(load_run(RUNS / t["burst"]["run"]),
                                                          window(load_run(RUNS / t["burst"]["run"]))))(
                                                              confirm.table(43))),
             "reg19": [_readout(lambda x: row(x, window_reg19(x)), x) for x in find_runs(REG19_PATTERN)]}
    return {"rows": rows, "left_out": left, "q1": judge_q1(rows), "q2": judge_q2(rows),
            "retrospective": retro, "reg27_outcome": r27["outcome"], "params": REG28}


def predictions_replay(mus=(1.70, 1.75, 1.80)) -> dict:
    """Schedule only (no run is read): model F's replay share in W on each REG-27 seed's burst config at a few mu,
    beside the quasi-static 1 — the probe committed with the registration."""
    from analysis import confirm, reg27
    out = {}
    for sd in reg27.REG27["seeds"]:
        cfg = confirm.burst_cfg(sd, reg27.REG27["drop_s"])
        ph = confirm.phases(cfg)
        win = (ph["burst_end"] + REG28["offset_s"], ph["watch_end"])
        sched = modelf_sim.schedule(cfg)
        out[sd] = {str(m): replay_share(cfg, m, win, sched) for m in mus}
    return out


if __name__ == "__main__":
    import json
    import sys

    dump = lambda x: print(json.dumps(x, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    if len(sys.argv) == 2 and sys.argv[1] == "table":
        dump(table())
    elif len(sys.argv) == 2 and sys.argv[1] == "replay":
        dump(predictions_replay())
    elif len(sys.argv) == 2 and sys.argv[1] == "predictions":
        dump({str(l): {str(c): predicted_share(l, c) for c in (1.681, 1.72, 1.75, 1.78, 1.798)}
              for l in (1.9, 2.0, 2.07, 2.1)})
    else:
        raise SystemExit("usage: python -m analysis.reg28 predictions | replay | table")
