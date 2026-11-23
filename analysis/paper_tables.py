"""Tables the preprint prints that no registered table gives in one place. Readouts only, never judged; each reuses a
registered readout function on pinned, valid, unsealed runs. Never writes into runs/.

`baseline_pairs`: the run-order readout `gate.reg19_baseline_pair` (review T3 of REG-19: over [300, 900) s, before the
burst, each run's percent fast, hit, median TTFT, completion rate, busy SM clock and power, temperature, and the burst
run minus its control) for every burst-twin pair of the replication (REG-19) and of the long-watch runs (REG-24,
REG-27). The control always ran first.
"""
from __future__ import annotations

from analysis import gate as G
from analysis.load import find_runs
from analysis.reg30 import REG30

PAIRS = {   # (control, burst run), by experiment and seed
    ("replication", 15): ("20261112-050920_stage2_control_l1pl350cl_r2.0_w3000_s15", "20261112-061951_burst_l1pl350cl_r2.0_w3000_s15"),
    ("replication", 16): ("20261112-073516_stage2_control_l1pl350cl_r2.0_w3000_s16", "20261112-084533_burst_l1pl350cl_r2.0_w3000_s16"),
    ("replication", 17): ("20261112-103423_stage2_control_l1pl350cl_r2.0_w3000_s17", "20261112-135538_burst_l1pl350cl_r2.0_w3000_s17"),
    ("replication", 18): ("20261113-165516_stage2_control_l1pl350cl_r2.0_w3000_s18", "20261113-180511_burst_l1pl350cl_r2.0_w3000_s18"),
    ("replication", 19): ("20261113-192110_stage2_control_l1pl350cl_r2.0_w3000_s19", "20261113-203108_burst_l1pl350cl_r2.0_w3000_s19"),
    **{("long-watch", s): (REG30["controls"][s], REG30["runs"][s]) for s in REG30["runs"]},
}


def _one(rid: str) -> dict:
    found = find_runs(rid)
    if len(found) != 1:
        raise ValueError(f"{rid}: not one valid run")
    return found[0]


def baseline_pairs() -> list[dict]:
    """One row per pair: experiment, seed, the hours between the two starts, and `gate.reg19_baseline_pair`."""
    rows = []
    for (exp, seed), (c, b) in PAIRS.items():
        bp = G.reg19_baseline_pair(_one(c), _one(b))
        gap_h = (int(b[9:11]) * 3600 + int(b[11:13]) * 60 - int(c[9:11]) * 3600 - int(c[11:13]) * 60
                 + (int(b[6:8]) - int(c[6:8])) * 86400) / 3600
        rows.append({"experiment": exp, "seed": seed, "control": c, "burst": b, "gap_h": round(gap_h, 2), **bp})
    return rows


if __name__ == "__main__":
    import json
    import sys
    fns = {"baseline_pairs": baseline_pairs}
    print(json.dumps(fns[sys.argv[1]](), indent=1, default=str))
