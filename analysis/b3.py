"""B3's feasibility gate (rule G; author, decisions.md 2026-11-17 [DECISION] (author) C1's order clause … C2 goes on with
B1 (0 GPU h) and then B3): before B3 (REG-19's burst at λ = 2.0 with the server's `--max-num-seqs` at 16) is drafted,
the highest running batch N in REG-19's five warm controls must be below 16, so that the flag does not bind in the
warm state the controls stand for.

The gauge is `len(scheduler.running)`, the list `--max-num-seqs` caps (PROBE(docs/probe/20261117_b3_source/
scheduler_running_cap_20261117T165548Z.txt)). The gate reads it at the 1 s scrapes in each valid control's measured
window, warm-up <= t < end of arrivals (`load.window`; rule F r1, item 1: the start, warm-up and drain are not the warm
state). N = 16 counts as binding: under the flag the cap is then full and the next arrival waits (rule F r1, T6). A
scrape samples the gauge, so a peak between scrapes is not seen: a pass says the flag was not reached at any scrape,
not that it never was. B3's own same-flag control, if B3 runs, is the test of the warm state under the flag.

Outcome (`gate_outcome`):
  undefined   not exactly one usable control for each seed 15-19 (rule F r1, item 3);
  pass        every control's highest N < 16: B3 may be drafted at 16;
  fail_redraft 16 <= highest N <= 22: a flag k in [highest + 1, 23] still keeps k x 4096 below the 96,656-token pool;
               a new draft at such k goes to the author first (rule F r1, item 4);
  infeasible  highest N >= 23: no flag both spares the warm state and bounds the pool below full.
Readouts, never judged: per control the mean N, the share of scrapes with N >= 16, the number of scrapes at the peak,
the 0.999 quantile, the time of the first peak, and the whole-frame peak (start and drain included).
"""
from __future__ import annotations

import math

import numpy as np

from analysis import withheld

B3_GATE = {
    "flag": 16,           # --max-num-seqs in B3 (author, 2026-11-17)
    "controls": "*_stage2_control_l1pl350cl_r2.0_w3000_s1[5-9]",   # REG-19's five no-burst controls, valid only
    "seeds": (15, 16, 17, 18, 19),
    "pool_tokens": 96656,  # KV pool (docs/model.md); a request holds at most 3968 + 128 = 4096 tokens
    "max_request_tokens": 4096,
    "q_hi": 0.999,        # readout only
}
GATE_HEADING = "[REG] B3's gate (rule G):"   # with its colon, so a draft or withdrawal heading does not match
N = "vllm:num_requests_running"


def running(run: dict, measured: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """(t, N) at every scrape with the gauge recorded; by default only in the measured window (`load.window`)."""
    from analysis.load import window
    m = run["metrics"]
    if N not in m:
        return np.array([]), np.array([])
    m = m[m[N].notna()]
    if measured:
        m = window(m, run["cfg"])
    return m.t.to_numpy(dtype=float), m[N].to_numpy(dtype=float)


def seed_of(name: str) -> int | None:
    tail = name.rsplit("_s", 1)
    return int(tail[1]) if len(tail) == 2 and tail[1].isdigit() else None


def control_row(run: dict, p: dict = B3_GATE) -> dict:
    name = run["dir"].name
    t, n = running(run)
    _, n_all = running(run, measured=False)
    if not len(n):
        return {"run": name, "seed": seed_of(name), "n_max": math.nan, "scrapes": 0}
    i = int(np.argmax(n))
    return {"run": name, "seed": seed_of(name), "scrapes": int(len(n)), "n_max": float(n[i]), "t_max": float(t[i]),
            "at_max": int((n == n[i]).sum()), "n_mean": float(n.mean()),
            "share_at_flag": float((n >= p["flag"]).mean()),
            "q_hi": float(np.quantile(n, p["q_hi"], method="inverted_cdf")),
            "n_max_whole_frame": float(n_all.max()) if len(n_all) else math.nan}


def k_max(p: dict = B3_GATE) -> int:
    """The largest flag k with k x max_request_tokens below the pool (23 at 96,656 and 4096)."""
    return (p["pool_tokens"] - 1) // p["max_request_tokens"]


def gate_outcome(rows: list[dict], p: dict = B3_GATE) -> dict:
    usable = [r for r in rows if r.get("scrapes", 0) > 0 and not math.isnan(r["n_max"])]
    seeds = sorted(r["seed"] for r in usable if r["seed"] is not None)
    highest = max((r["n_max"] for r in usable), default=math.nan)
    if seeds != sorted(p["seeds"]) or len(usable) != len(p["seeds"]):
        d = "undefined"
    elif highest < p["flag"]:
        d = "pass"
    elif highest < k_max(p):
        d = "fail_redraft"
    else:
        d = "infeasible"
    return {"decision": d, "n_usable": len(usable), "seeds": seeds, "highest": highest, "flag": p["flag"],
            "k_max": k_max(p)}


def gate() -> dict:
    """The registered computation, once: refuses until decisions.md holds `[REG] B3's gate`."""
    if not withheld.lifted(GATE_HEADING):
        raise RuntimeError(f"B3's gate is not registered: decisions.md has no heading '{GATE_HEADING}'")
    from analysis.load import find_runs
    rows = [control_row(r) for r in find_runs(B3_GATE["controls"])]
    return {"rows": rows, "outcome": gate_outcome(rows), "params": B3_GATE}


if __name__ == "__main__":
    import json
    import sys
    if sys.argv[1:] != ["gate"]:
        sys.exit("usage: python -m analysis.b3 gate")
    print(json.dumps(gate(), indent=1, default=str))
