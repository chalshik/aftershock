"""G2 (draft; to be registered after T4's [RESULT]): does vLLM's development mode change the cache-off engine's speed?

Every capacity sweep ran with the mode off; every Stage 2 trigger run and control with it on (stage2_base.yaml), so
each margin compares a burst run with the mode on against a floor measured with it off. Open-questions A6 answered from
the pinned source that no code path of a request that succeeds depends on the mode. G2 measures it: T4's 2048-token
cache-off sweep at two saturated rates, once with the mode on and once off, the four runs interleaved in one queue
(off, on at the first rate; on, off at the second), same seed, pool and stack. T4's same-rate runs (mode off, a day
earlier) are a readout, never judged.

Validity (rule F r1 G-3, G-4): the mode is read from the server, not the config. vLLM logs each route it serves at
startup; `/reset_prefix_cache` is a dev route, registered only with the mode on (open-questions A6), so its route line
must be in every mode-on server.log and in no mode-off one (`dev_routes_logged`; checked on every unsealed run on
record by `route_probe`). Every counted run also passes T4's server check (`t4.check_settings`, the 2048 arm).

Judged (`outcome`): per rate r = throughput_on / throughput_off - 1 (`load.summarize` throughput through
`gate.sweep_rows`, the rows every floor uses), in order:
- `dev_undefined`: a pair is missing, or any counted run keeps up (throughput >= sat_frac x planned, T4's 0.95), so
  r would measure the offered load (rule F r1 G-8);
- `dev_neutral`: |r| <= `band` (the colleague's 2 %) at both rates;
- `dev_moves`: |r| > band at both rates with the same sign;
- `dev_unclear`: otherwise (one rate beyond the band, or opposite signs; rule F r1 G-7).
Reported, never judged: mean KV per run, and r of the mode-on runs against T4's same-rate runs. One run per rate and
arm (rule H): r has no interval.
"""
from __future__ import annotations

import math
import re
from pathlib import Path

from analysis import t4, withheld

G2 = {
    "parents": {"on": "sweep_g2_l1pl350cl_pc_off_b2048_dev", "off": "sweep_g2_l1pl350cl_pc_off_b2048_nodev"},
    "readout_parent": t4.T4["parents"]["off_2048"],   # T4's mode-off runs at the same rates, a day earlier
    "rates": (3.0, 4.0),          # above E1's lam_sat 2.5 (decisions.md 2026-11-14 [RESULT] E1); saturation is checked
    "band": 0.02,                 # the colleague's +-2 % (colleague.md §5.3 G2)
    "sat_frac": t4.T4["sat_frac"],   # 0.95, REG-8's: a row is behind when throughput < 0.95 x planned
    "env_key": "VLLM_SERVER_DEV_MODE",
    "seal": "[RESULT] G2 (development-mode twin)",
}
_DEV_ROUTE = re.compile(r"Route: /reset_prefix_cache\b")


def outcome(tp_on: dict, tp_off: dict, behind: dict | None = None, p: dict = G2) -> dict:
    """tp_on, tp_off: {rate: throughput} from the counted runs; behind: {(arm, rate): bool}, whether each counted run
    fell behind its planned rate (None: not checked, as for the T4 readout)."""
    r = {x: (tp_on[x] / tp_off[x] - 1 if x in tp_on and x in tp_off and tp_off[x] > 0 else math.nan)
         for x in p["rates"]}
    v = list(r.values())
    if not all(math.isfinite(x) for x in v) or (behind is not None and not all(behind.values())):
        label = "dev_undefined"
    elif all(abs(x) <= p["band"] + 1e-12 for x in v):
        label = "dev_neutral"
    elif all(abs(x) > p["band"] + 1e-12 for x in v) and (all(x > 0 for x in v) or all(x < 0 for x in v)):
        label = "dev_moves"
    else:
        label = "dev_unclear"
    return {"ratio_minus_1": r, "decision": label}


def dev_routes_logged(run_dir) -> bool:
    """Whether the server served vLLM's dev routes: the `/reset_prefix_cache` route line in its server.log."""
    withheld.guard(Path(run_dir).name)
    log = Path(run_dir) / "server.log"
    return bool(_DEV_ROUTE.search(log.read_text(errors="replace"))) if log.exists() else False


def check_mode(run_dir, arm: str) -> list[str]:
    on = dev_routes_logged(run_dir)
    return [] if on == (arm == "on") else [f"dev routes {'absent' if arm == 'on' else 'served'} in the {arm} arm"]


def route_probe() -> dict:
    """Over every unsealed run on record (valid or not) with a config and a server.log: does the dev-route line match the config's mode?
    Run once before G2's [REG] (rule F r1 G-3); the counts go into it."""
    from analysis.load import RUNS
    from harness.config import load_config
    out = {"on_logged": 0, "on_missing": [], "off_logged": [], "off_clean": 0, "skipped_sealed": 0}
    for d in sorted(RUNS.iterdir()):
        cfgp, log = d / "config.yaml", d / "server.log"
        if not (cfgp.exists() and log.exists()):
            continue
        if withheld.sealed_by(d.name):
            out["skipped_sealed"] += 1
            continue
        cfg = load_config(cfgp)
        on = ((cfg.get("server") or {}).get("env") or {}).get(G2["env_key"]) == "1"
        got = dev_routes_logged(d)
        if on and got:
            out["on_logged"] += 1
        elif on:
            out["on_missing"].append(d.name)
        elif got:
            out["off_logged"].append(d.name)
        else:
            out["off_clean"] += 1
    return out


def sealed() -> list[str]:
    import analysis.load as L
    if not L.RUNS.exists():
        return []
    return sorted(d.name for d in L.RUNS.iterdir() if "_sweep_g2_" in d.name and withheld.sealed_by(d.name) == "G2")


def table() -> dict:
    """G2's computation, once after its queue and its [REG]: `AFTERSHOCK_UNSEAL=G2 uv run python -m analysis.g2 table`.
    Refuses while any G2 folder is sealed, before T4's result, or when a counted run fails `check_mode` or T4's
    server check. Per rate the earliest valid run counts."""
    from analysis import e1
    from analysis.gate import sweep_rows
    from analysis.load import RUNS
    if not withheld.lifted(t4.T4["seal"]):
        raise withheld.WithheldError("G2's readout reads T4's 2048 runs: T4's [RESULT] first")
    s = sealed()
    if s:
        raise withheld.WithheldError(f"{len(s)} G2 folder(s) sealed; set AFTERSHOCK_UNSEAL=G2")
    tp, kv, behind, problems, later = {}, {}, {}, {}, {}
    for arm, par in G2["parents"].items():
        rows = sweep_rows(par)
        rows = rows[rows.rate.isin(G2["rates"])]
        if not len(rows):
            raise RuntimeError(f"{par}: no valid run at {G2['rates']} can be read")
        keep, later[arm] = e1.first_valid(rows)
        for r in keep.to_dict("records"):
            x = float(r["rate"])
            tp.setdefault(arm, {})[x] = float(r["throughput"])
            kv.setdefault(arm, {})[x] = float(r["kv_mean"])
            behind[(arm, x)] = bool(r["throughput"] < G2["sat_frac"] * r["planned_rate"])
            bad = check_mode(RUNS / r["run"], arm) + t4.check_settings("off_2048", t4.server_readouts(RUNS / r["run"]))
            if bad:
                problems[r["run"]] = bad
    if problems:
        raise ValueError(f"counted runs that fail G2's checks: {problems}")
    res = {"outcome": outcome(tp.get("on", {}), tp.get("off", {}), behind), "throughput": tp, "kv": kv,
           "behind": {f"{a}@{x}": b for (a, x), b in behind.items()}, "later_valid_not_counted": later}
    try:   # a readout never takes the result with it
        t4rows, _ = e1.first_valid(sweep_rows(G2["readout_parent"]))
        t4tp = {float(r["rate"]): float(r["throughput"]) for r in t4rows.to_dict("records") if r["rate"] in G2["rates"]}
        res["readout_vs_t4"] = outcome(tp.get("on", {}), t4tp)["ratio_minus_1"]
    except withheld.WithheldError:
        raise
    except Exception as e:
        res["readout_vs_t4_error"] = f"{type(e).__name__}: {e}"[:300]
    res["decision"] = res["outcome"]["decision"]
    return res


if __name__ == "__main__":
    import sys
    from pprint import pprint
    if sys.argv[1:] == ["table"]:
        pprint(table(), width=120, sort_dicts=False)
    elif sys.argv[1:] == ["probe"]:
        pprint(route_probe(), width=120, sort_dicts=False)
    else:
        sys.exit("usage: python -m analysis.g2 table | probe")
