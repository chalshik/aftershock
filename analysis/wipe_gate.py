"""Rule G feasibility gate for the sustained-wipe arm at L1, λ = 2.0 (decisions.md 2026-11-14 [DECISION] (author), the
wipe arm is a sustained wipe; pass condition registered there, in commit a066fb1, before this code was written).
Never writes into runs/.

G1: c_cold < λ < mu_warm_cl, and at least one valid no-trigger control at λ of >= min_control_s is `recovered`.
    c_cold = the highest span completion rate of `gate.reg17_cold_side` over the six stuck burst runs (conservative).
G2: Q_D = (λ - c_cold) x D >= margin x Q*, Q* = expected_kv_tokens / (mean session prompt + output tokens) of the burst
    config; D = the smallest of spans_s that passes, or None.
"""
from __future__ import annotations

import math

from analysis.gate import reg17_cold_side
from analysis.load import RUNS, load_run, verdict

WG = {
    "lam": 2.0,
    "mu_warm_cl": 2.4,         # REG-13's closed-loop warm capacity, one seed (docs/model.md M2, REG-13)
    "min_control_s": 2400,     # docs/process.md §4
    "margin": 1.5,             # registered with the gate (BELIEF)
    "spans_s": (180.0, 300.0, 600.0),
    "bursts": ("20261110-182920_burst_l1pl350cl_r2.0_s2", "20261112-061951_burst_l1pl350cl_r2.0_w3000_s15",
               "20261112-084533_burst_l1pl350cl_r2.0_w3000_s16", "20261112-135538_burst_l1pl350cl_r2.0_w3000_s17",
               "20261113-180511_burst_l1pl350cl_r2.0_w3000_s18", "20261113-203108_burst_l1pl350cl_r2.0_w3000_s19"),
    "controls": ("20261112-050920_stage2_control_l1pl350cl_r2.0_w3000_s15",
                 "20261112-073516_stage2_control_l1pl350cl_r2.0_w3000_s16",
                 "20261112-103423_stage2_control_l1pl350cl_r2.0_w3000_s17",
                 "20261113-165516_stage2_control_l1pl350cl_r2.0_w3000_s18",
                 "20261113-192110_stage2_control_l1pl350cl_r2.0_w3000_s19"),
}


def q_star(cfg: dict) -> float:
    """Session requests whose KV fills the pool: turn k's prompt is system + user + k x (user + reply) tokens."""
    s, w = cfg["workload"]["sessions"], cfg["workload"]
    mean_prompt = s["system_tokens"] + s["user_tokens"] + (s["turns"] - 1) / 2 * (s["user_tokens"] + s["reply_tokens"])
    return cfg["checks"]["expected_kv_tokens"] / (mean_prompt + w["output_tokens"])


def decide(c_cold: float, controls_recovered: int, qstar: float, lam: float = WG["lam"]) -> dict:
    g1 = bool(c_cold < lam < WG["mu_warm_cl"] and controls_recovered >= 1)
    rows = [{"D_s": d, "Q_D": (lam - c_cold) * d, "need": WG["margin"] * qstar} for d in WG["spans_s"]]
    ok = [r for r in rows if r["Q_D"] >= r["need"]]
    d = ok[0]["D_s"] if ok else None
    return {"g1": g1, "rows": rows, "D_s": d, "g2": d is not None, "passes": g1 and d is not None}


def gate_g() -> dict:
    spans = []
    for name in WG["bursts"]:
        r = load_run(RUNS / name)
        if not (r.get("checks") and r["checks"].get("valid")):
            raise ValueError(f"{name}: not valid")
        spans += [(name, s["span"], s["c"]) for s in reg17_cold_side(r)["spans"] if math.isfinite(s["c"])]
    if not spans:
        raise ValueError("no finite cold-span completion rate")
    c_cold = max(c for _, _, c in spans)
    rec = []
    for name in WG["controls"]:
        r = load_run(RUNS / name)
        long_enough = r["cfg"]["workload"]["duration_s"] >= WG["min_control_s"]
        valid = bool(r.get("checks") and r["checks"].get("valid"))
        rec.append({"run": name, "valid": valid, "long_enough": long_enough,
                    "verdict": verdict(r)["verdict"] if valid else None})
    n_rec = sum(1 for x in rec if x["valid"] and x["long_enough"] and x["verdict"] == "recovered")
    qs = q_star(load_run(RUNS / WG["bursts"][1])["cfg"])
    return {"c_cold": c_cold, "c_cold_at": max(spans, key=lambda x: x[2])[:2], "n_spans": len(spans),
            "controls": rec, "controls_recovered": n_rec, "q_star": qs, **decide(c_cold, n_rec, qs)}
