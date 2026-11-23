"""K3b (docs/drafts/k3b_reg_draft1.md; not registered): model F (Valse and Jannesari, arXiv 2611.08902 §3; BELIEF as
to the model) predicted before the data, on K4's runs that tip and stay cold. K3's first test written before its runs
were read: K4's ten runs are sealed until "[RESULT] K4 (the warm state ends at a surge)" and this module reads them
only after that heading exists. It never sets AFTERSHOCK_UNSEAL.

Runs: every K4 run (either arm) that `k4.table` labels `tipped` (an onset in [600, 2700) whose end window did not
recover). W = [onset + offset_s, duration); a run needs |W| >= min_w_s.

Gate per run (REG-21's judging conditions, `k3w`): in W, reuse of later turns <= allmiss_reuse_max (`k1.reuse`) and a
request waiting in >= busy_min of the scrapes (`k3w.busy_share`), and `k1.cold_backlogged` with K4's baseline and W
as the end window. A run that fails is listed with its reason, not judged.

Model F per run, at the run's own planned request rate in W (`k3.lam_w`) and the all-miss rate mu_off = 1.681 req/s
(REG-8's open-loop cache-off floor, whose mean prompt is the schedule's mean, 3072 tokens; so U_off / 3072 = mu_off
up to REG-8's spread). `modelf.l1` with a turn cost proportional to the prompt (docs/model.md M5):
  c         request completion rate over W (`gate.completion_rate`)         predicted Gamma x mu_off
  r         U over W / U_off (`k3w.work_rate`, `k3w.reference`)              predicted Gamma x served (1 at this cost)
  served    mean prompt of ok session turns by first token in W / 3072      predicted served_prompt
  sessions  sessions whose last turn ended ok in W per second / (mu_off/8)  predicted D / lambda_miss; REPORTED ONLY
"sessions" is the rate form of the backlog slope: slope = lambda_s - D. It is reported, never judged (rule F round 1,
item 2): the sessions that end in W started about one session lifetime earlier, many inside the surge, and that cohort
is not in lambda_W, so a right model reads high. The least-squares slope (`k3w.sessions_slope`) is reported too.
c counts requests of any status by t_end; served uses the planned prompt lengths, whose schedule mean is 3072.

Judged, per quantity in JUDGED, over the judged runs: the median deviation (c: c / c_pred - 1; the others: obs - pred)
within its tolerance (K3B["tol"]); undefined with fewer than min_runs judged runs. Served is judged only on runs where
pred + tol < 1 (its tolerance band excludes the untilted mix; rule F round 1, item 4). c and r follow mostly from the
gate and from served (a saturated all-miss engine gives r near 1; rule F round 1, item 3). Reported: k inside of n per
run with the one-sided 95 % Clopper-Pearson upper bound (rule H); every run that tipped, stayed cold and had lambda_W
<= mu_off, as a count against model F's Corollary 3 (item 5); the deviations against E1's closed-loop floor (mu_alt);
the values over W from onset + offset_alt_s; Valse and Jannesari's trap rule; and model F's long-run kbar.
Not tested here: the drain threshold (K4 has no drop in load) and the healthy-side condition (that is K4).
"""
from __future__ import annotations

import math

import numpy as np

from analysis import gate, k3, k3w, modelf, withheld
from analysis.k1 import cold_backlogged, reuse
from analysis.stats import cp_upper

K3B = {
    "mu_off": 1.681,            # req/s, REG-8's open-loop cache-off floor (REG21["mu_off"])
    "mu_alt": 1.798,            # req/s, E1's closed-loop cache-off floor; reported only
    "turns": 8,
    "mean_prompt": 3072.0,      # DERIVED: mean of modelf.l1_prompts(8) = 2176 + 256 x 3.5
    "offset_s": 300.0,          # W starts this long after the onset
    "offset_alt_s": 600.0,      # REG-21's offset; a second W, reported only
    "min_w_s": 300.0,
    "allmiss_reuse_max": 0.05, "busy_min": 0.99,    # REG-21's
    "tol": {"c": 0.05, "r": 0.05, "served": 0.015, "sessions": 0.07},
    "min_runs": 3,
    "conf": 0.95,
    "after": "[RESULT] K4 (the warm state ends at a surge)",
}
QUANTITIES = ("c", "r", "served", "sessions")
JUDGED = ("c", "r", "served")      # "sessions" is reported only (rule F round 1, item 2)
EPS = 1e-9


def predict(lam: float, mu: float | None = None) -> dict:
    """Model F's values at planned request rate lam and all-miss rate mu. Below or at mu the model drains."""
    mu = K3B["mu_off"] if mu is None else mu
    if not lam > mu:
        return {"lam": lam, "mu": mu, "rho": lam / mu, "drains": True}
    t = modelf.l1(lam, mu, K3B["turns"])
    return {"lam": lam, "mu": mu, "rho": t.rho, "drains": False, "c": t.requests * mu, "r": t.requests * t.served_prompt,
            "served": t.served_prompt, "sessions": t.sessions,
            "slope": t.backlog_growth * (lam - mu) / K3B["turns"]}


def predictions_grid(lams=(1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6)) -> dict:
    """The predictions as a table over lam, from the model alone (no run is read): the probe committed before K4's
    [RESULT]."""
    return {str(mu): {str(l): predict(l, mu) for l in lams} for mu in (K3B["mu_off"], K3B["mu_alt"])}


def window(onset: float, duration: float, offset: float | None = None) -> tuple[float, float]:
    return onset + (K3B["offset_s"] if offset is None else offset), float(duration)


def observed(run: dict, a: float, b: float, u_off: float, mu: float | None = None) -> dict:
    mu = K3B["mu_off"] if mu is None else mu
    q = k3.session_turns(run)
    ok = q[(q.status == "ok") & (q.t_first_token >= a) & (q.t_first_token < b)]
    last = q[(q.turn == K3B["turns"] - 1) & (q.status == "ok") & (q.t_end >= a) & (q.t_end < b)]
    u = k3w.work_rate(run, a, b)
    return {"lam_W": k3.lam_w(run, a, b, K3B["turns"]), "c": gate.completion_rate(run, a, b),
            "r": u / u_off if u_off and not math.isnan(u) else math.nan,
            "served": float(ok.prompt_tokens_planned.mean() / K3B["mean_prompt"]) if len(ok) else math.nan,
            "sessions": float(len(last) / (b - a) / (mu / K3B["turns"])),
            "slope": k3w.sessions_slope(run, a, b), "U": u}


def deviation(qty: str, obs: float, pred: float) -> float:
    if obs is None or pred is None or math.isnan(obs) or math.isnan(pred):
        return math.nan
    return obs / pred - 1 if qty == "c" else obs - pred


def _why(w: tuple, ru: float, busy: float, cb, pr: dict) -> str | None:
    why = []
    if w[1] - w[0] < K3B["min_w_s"]:
        why.append("W shorter than min_w_s")
    else:
        if math.isnan(ru):
            why.append("no later turn in W")
        elif ru > K3B["allmiss_reuse_max"]:
            why.append("not all-miss in W")
        if math.isnan(busy):
            why.append("no waiting gauge in W")
        elif busy < K3B["busy_min"]:
            why.append("idle in W")
        if cb is not True:
            why.append("not cold_backlogged in W" if cb is False else "cold_backlogged undefined in W")
        if pr.get("drains"):
            why.append("lam_W at or below mu_off: model F drains (counted against Corollary 3 if the gate holds)")
    return "; ".join(why) or None


def row(run: dict, onset: float, ref: dict) -> dict:
    """One tipped run: the gate, the judged values against model F, and the reported ones."""
    from analysis.k4 import REG
    dur = float(run["cfg"]["workload"]["duration_s"])
    a, b = window(onset, dur)
    out = {"run": run["dir"].name, "seed": run["cfg"].get("seed"), "onset": onset, "W": (a, b)}
    if b - a < K3B["min_w_s"]:
        out.update(why_not=_why((a, b), math.nan, math.nan, None, {}), judged=False)
        return out
    ru, busy = reuse(run, a, b), k3w.busy_share(run, a, b)
    cb = cold_backlogged(run, {"base": REG["base"], "end": (a, b)}, run["cfg"]["verdict"])["cold_backlogged"]
    obs = observed(run, a, b, ref["mean"])
    pr = predict(obs["lam_W"])
    why = _why((a, b), ru, busy, cb, pr)
    out.update(reuse_W=ru, busy_W=busy, cold_backlogged_W=cb, observed=obs, predicted=pr, why_not=why,
               judged=why is None)
    out["against_corollary3"] = bool(pr.get("drains")) and _why((a, b), ru, busy, cb, {}) is None  # stuck where F drains
    out["dev"] = {qn: (math.nan if pr.get("drains") else deviation(qn, obs[qn], pr[qn])) for qn in QUANTITIES}
    out["served_discriminates"] = (not pr.get("drains")) and pr["served"] + K3B["tol"]["served"] < 1 - EPS
    if not out["served_discriminates"]:
        out["dev"]["served"] = math.nan
    out["inside"] = {qn: (None if math.isnan(d) else bool(abs(d) <= K3B["tol"][qn] + EPS)) for qn, d in out["dev"].items()}
    try:                                   # reported only; never takes the judged values with it
        alt = predict(obs["lam_W"], K3B["mu_alt"])
        obs_alt = observed(run, a, b, ref["mean"], K3B["mu_alt"])
        obs_alt["r"] = obs["r"]
        rep = {"vs_E1_floor": {"predicted": alt, "observed": obs_alt,
                               "dev": {qn: (math.nan if alt.get("drains") else deviation(qn, obs_alt[qn], alt[qn]))
                                       for qn in QUANTITIES}},
               "vj_rule": k3w.vj_rule(run, obs["r"], a, b), "gauges": k3w._gauges(run, a, b, obs["U"]),
               "kbar_obs": k3.kbar_obs(run, a, b), "long_run": k3.long_run(obs["lam_W"])}
        a2, b2 = window(onset, dur, K3B["offset_alt_s"])
        if b2 - a2 >= K3B["min_w_s"]:
            o2 = observed(run, a2, b2, ref["mean"])
            p2 = predict(o2["lam_W"])
            rep["W_alt"] = {"W": (a2, b2), "observed": o2, "predicted": p2}
        out["reported"] = rep
    except Exception as e:
        out["reported"] = {"error": f"{type(e).__name__}: {e}"[:200]}
    return out


def judge(rows: list[dict]) -> dict:
    """Per quantity over the judged rows: median deviation, inside / outside / undefined, k inside of n, and the
    one-sided upper bound on the share outside."""
    js = [r for r in rows if r.get("judged")]
    out = {}
    for qn in QUANTITIES:          # "sessions" is computed the same way and marked reported
        d = [r["dev"][qn] for r in js if not math.isnan(r["dev"][qn])]
        n, k = len(d), sum(abs(x) <= K3B["tol"][qn] + EPS for x in d)
        med = float(np.median(d)) if d else math.nan
        lab = "undefined" if n < K3B["min_runs"] else ("inside" if abs(med) <= K3B["tol"][qn] + EPS else "outside")
        side = None if lab != "outside" else ("high" if med > 0 else "low")
        out[qn] = {"label": lab, "side": side, "median_dev": med, "n": n, "k_inside": k,
                   "cp_upper_outside": cp_upper(n - k, n, K3B["conf"]) if n else math.nan,
                   "judged": qn in JUDGED}
    out["n_judged"] = len(js)
    out["against_corollary3"] = [r["run"] for r in rows if r.get("against_corollary3")]
    out["left_out"] = {r["run"]: r["why_not"] for r in rows if not r.get("judged")}
    return out


def table() -> dict:
    """K3b's computation, once, after "[RESULT] K4 …" is in decisions.md: `python -m analysis.k3b table`."""
    if not withheld.lifted(K3B["after"]):
        raise withheld.WithheldError("K3b reads K4's runs only after the heading " + K3B["after"])
    from analysis import k4
    from analysis.load import RUNS, load_run
    k4t = k4.table()
    ref = k3w.reference()
    rows = []
    for arm in ("surge", "calm"):
        for r in k4t["rows"][arm]:
            if r["label"] != "tipped":
                continue
            rows.append({"arm": arm, **row(load_run(RUNS / r["run"]), float(r["onset"]), ref)})
    return {"rows": rows, "judge": judge(rows), "u_off": ref["mean"], "k4_outcome": k4t["outcome"]["decision"]}


if __name__ == "__main__":
    import json
    import sys

    dump = lambda x: print(json.dumps(x, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    cmd = sys.argv[1] if len(sys.argv) == 2 else None
    if cmd == "predictions":
        dump(predictions_grid())
    elif cmd == "table":
        dump(table())
    else:
        raise SystemExit("usage: python -m analysis.k3b predictions | table")
