"""Post-hoc readout (author 2026-11-26, after an outside review of the workshop paper): the shared turn-release criterion
of REG-20 (`k1.cold_backlogged`: the follow-up hit rate at least `stuck_hit_drop` below baseline, the server queue more
than `recovered_q_excess` above it and KV usage at least the surge onset; `recovered`: hit rate within
`recovered_hit_drop` and queue within `recovered_q_excess` of baseline) applied alike to every run behind a burst-cell
count in the papers, controls included.

This is NOT a pre-specified test. It was chosen after every run below had been seen and judged by its own registered
test, which stays the result. It is reported beside those results as one ruler across arms, and every count it gives
is labelled post hoc. Nothing here changes `k1`; the criterion, thresholds and windows are each run's own config, and a
control is read at its burst run's absolute windows (`k1.windows`), as REG-20 does.

Usage: uv run --extra analysis python -m analysis.uniform   (prints JSON: one row per cell, one entry per seed)
"""
from __future__ import annotations

import json
import sys

from analysis import k1

# (cell, burst run pattern, arm-specific control pattern or None, seeds). Patterns are the registered ones: REG-19
# (k1.closed_pattern), REG-20 arms A and B (k1.open_pattern), TR arm H (tr, `_trh`) and arm D (aiperf_arm, `dr`),
# T1 arm U (t1, `_t1u`), arm O (armo, `_t1o`), D1 (`_cap`), D1b (`_cap40`), REG-24/27 (confirm, the long window).
# None: no control was run for that arm (REG-20 ran open-loop controls only after a cold burst run; T1 U, D1 and D1b
# compared against REG-19's runs, not a control of their own).
CELLS = (
    ("closed loop, λ 2.0 (main runs)", "*_burst_l1pl350cl_r2.0_w3000_s{s}", "*_stage2_control_l1pl350cl_r2.0_w3000_s{s}",
     (15, 16, 17, 18, 19)),
    ("open loop, λ 2.0", "*_burst_l1pl350ol_r2.0_w3000_s{s}", None, (15, 16, 17, 18, 19)),
    ("open loop, λ 2.5", "*_burst_l1pl350ol_r2.5_w3000_s{s}", None, (15, 16, 17)),
    ("open loop, λ 2.73", "*_burst_l1pl350ol_r2.73_w3000_s{s}_trh", "*_stage2_control_l1pl350ol_r2.73_w3000_s{s}_trh",
     (15, 16, 17, 18, 19)),
    ("held, λ 2.0", "*_burst_l1pl350dr_r2.0_w3000_s{s}", "*_stage2_control_l1pl350dr_r2.0_w3000_s{s}",
     (15, 16, 17, 18, 19)),
    ("closed loop, later stack", "*_burst_l1pl350cl_r2.0_w3000_s{s}_t1u", None, (15, 16, 17, 18, 19)),
    ("open loop, later stack", "*_burst_l1pl350ol_r2.0_w3000_s{s}_t1o", "*_stage2_control_l1pl350ol_r2.0_w3000_s{s}_t1o",
     (15, 16, 17, 18, 19)),
    ("closed loop, cap at the control's peak", "*_burst_l1pl350cl_r2.0_w3000_s{s}_cap", None, (15, 16, 17, 18, 19)),
    ("closed loop, cap of 40", "*_burst_l1pl350cl_r2.0_w3000_s{s}_cap40", None, (15, 16, 17)),
    ("closed loop, long window", "*_confirm_l1pl350cl_r2.0_s{s}", "*_stage2_control_l1pl350cl_r2.0_long_s{s}",
     (43, 61, 68, 72, 79)),
)


def outcome(cold: bool | None, recovered: bool | None) -> str:
    """cold / recovered / neither / unclear. The two columns exclude each other (the hit-rate and queue tests point
    opposite ways), so a run that meets neither is 'neither'; a NaN input anywhere is 'unclear'."""
    if cold is None or recovered is None:
        return "unclear"
    if cold and recovered:
        raise ValueError("cold and recovered at once: the criterion's columns should exclude each other")
    return "cold" if cold else "recovered" if recovered else "neither"


def _one(run: dict, win: dict, rules: dict) -> dict:
    s = k1.cold_backlogged(run, win, rules)
    return {"run": run["dir"].name, "outcome": outcome(s["cold_backlogged"], s["recovered"]), "parts": s["parts"],
            "reuse": s["reuse"], "q": {"base": s["base"]["q"], "end": s["end"]["q"]}, "kv_end": s["end"]["kv"]}


def cell(name: str, burst_pat: str, control_pat: str | None, seeds: tuple[int, ...]) -> dict:
    rows = []
    for sd in seeds:
        b, nb = k1._runs(burst_pat.format(s=sd))
        if b is None:
            rows.append({"seed": sd, "burst": None, "invalid_burst_runs": nb}); continue
        win, rules = k1.windows(b["cfg"]), b["cfg"]["verdict"]
        row = {"seed": sd, "burst": _one(b, win, rules), "invalid_burst_runs": nb}
        if control_pat:
            c, nc = k1._runs(control_pat.format(s=sd))
            row["control"] = _one(c, win, rules) if c is not None else None
            row["invalid_control_runs"] = nc
        rows.append(row)
    count = lambda who, o: sum(1 for r in rows if r.get(who) and r[who]["outcome"] == o)
    out = {"cell": name, "n": len(seeds), "rows": rows,
           "burst": {o: count("burst", o) for o in ("cold", "recovered", "neither", "unclear")}}
    if control_pat:
        out["control"] = {o: count("control", o) for o in ("cold", "recovered", "neither", "unclear")}
    return out


def table2_readout() -> list[dict]:
    return [cell(*c) for c in CELLS]


def main(argv: list[str]) -> int:
    print(json.dumps(table2_readout(), indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
