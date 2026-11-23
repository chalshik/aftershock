"""G1 (draft; to be registered after T4's [RESULT]): a second workload seed for the capacity sweeps of setup L1.

The paper's margin (lambda = 2.0 over the cold floor, 1.11 to 1.19) and its warm capacity (2.4) rest on one seed per
sweep. G1 runs T4's 2048-token arm (same pinned pool, rates and 900 s per rate) with the workload seed 1 instead of 0,
on the same stack as T4, so the seed is the only difference from T4's 2048 arm.

Judged (`outcome`), with T4's functions and constants (analysis/t4.py), so nothing new is chosen here:
- each seed's cold floor (`t4.floor`) and the shared comparison (`t4.shared_floors`, a = seed 0, b = seed 1);
- `delta` = floor_s1 / floor_s0 - 1: `seed_same` if |delta| <= `band` (the colleague's 3 %), else `seed_up` /
  `seed_down`;
- each seed's warm capacity (`t4.warm_capacity`);
- the cell at lambda = 2.0 on each seed (`t4.label_cell`: 2.0 >= 1.05 x floor and 2.0 < warm capacity);
- `decision`, in order: `no_floor` (a floor or the shared comparison undefined), `no_warm` (a warm capacity
  undefined), `reference_moved` (the seed-0 cell fails: T4's own result), then, when the seed-1 cell fails,
  `cell_seed_bound` (the seeds differ beyond the band or in warm capacity, and `day_confounded` is clear) or
  `cell_edge` (otherwise: the cell sits at its edge on these runs); when both hold, `seed_stable` (`seed_same` and the
  same warm capacity) or `seed_varies`. An `above_grid` seed-1 floor is `seed_up` and fails its cell (as T4).
Flag `day_confounded` (registered with the rule, reported beside the decision): `day_drift` is G2's mode-off twins
(seed 0, run in G1's queue) against T4's same-rate 2048 runs (seed 0, a day earlier); see `day_confounded()`. It
blocks `cell_seed_bound`, and under `seed_varies` the paper says the spread includes day-to-day change.
Reported, never judged: the margin lambda / floor on each seed, against E1's 1.798 and REG-13's 2.4 (seen).
One run per rate and seed (rule H): delta has no interval.
"""
from __future__ import annotations

import math

from analysis import t4, withheld

G1 = {
    "parents": {"off_s0": t4.T4["parents"]["off_2048"], "on_s0": t4.T4["parents"]["on_2048"],
                "off_s1": "sweep_g1_l1pl350cl_pc_off_b2048_s1", "on_s1": "sweep_g1_l1pl350cl_pc_on_b2048_s1"},
    "band": 0.03,                 # the colleague's +-3 % (colleague.md §5.3 G1), = T4's band
    "seed": 1,                    # the workload seed (harness/run.py:145 -> workload.build_schedule's RNG)
    "twin_rates": (3.0, 4.0),     # G2's mode-off twins, the day-drift readout (G2["rates"])
    "seen": t4.T4["seen"],        # E1's 1.798, REG-13's 2.4
    "seal": "[RESULT] G1 (second-seed capacities)",
}


def label_seed(delta: float, band: float) -> str:
    if not math.isfinite(delta):
        return "undefined"
    return "seed_same" if abs(delta) <= band + 1e-12 else ("seed_up" if delta > 0 else "seed_down")


def day_confounded(drift: dict, delta: float, p: dict = G1) -> bool:
    """Rule F r1 G-10: the seed comparison is not separated from day-to-day change when a twin rate is missing, when
    |drift| > band at either rate, or, for a delta beyond the band, when the largest |drift| is at least half |delta|."""
    vals = [v for v in drift.values() if math.isfinite(v)]
    if len(vals) < len(p["twin_rates"]):
        return True
    worst = max(abs(v) for v in vals)
    if worst > p["band"] + 1e-12:
        return True
    return math.isfinite(delta) and abs(delta) > p["band"] + 1e-12 and worst >= abs(delta) / 2


def outcome(floors: dict, own: dict, warm: dict, drift: dict, p: dict = G1) -> dict:
    """floors: `t4.shared_floors(seed-0 rows, seed-1 rows, ...)`; own: {"s0": `t4.floor`, "s1": `t4.floor`} on each
    sweep's own rates; warm: {"s0": mu, "s1": mu} from `t4.warm_capacity`; drift: {rate: G2's mode-off twin / T4's
    same-rate run - 1}. The flag `day_confounded` is reported beside the decision and changes it only where stated
    (`cell_seed_bound` needs it clear)."""
    fa, fb = floors["a"]["mu_sat_off"], floors["b"]["mu_sat_off"]
    delta = fb / fa - 1 if math.isfinite(fa) and math.isfinite(fb) and fa > 0 else math.nan
    above = own["s1"].get("status") == "above_grid"            # rule F r1 G-5: as T4's floor_up
    lam = t4.T4["cell_lambda"]
    out = {"floor_s0": fa, "floor_s1": fb, "from": floors.get("from"), "delta": delta,
           "seed_label": "seed_up" if above else label_seed(delta, p["band"]),
           "warm_s0": warm["s0"], "warm_s1": warm["s1"],
           "cell_s0": t4.label_cell(own["s0"], warm["s0"]), "cell_s1": t4.label_cell(own["s1"], warm["s1"]),
           "margin": {k: lam / v["mu_sat_off"] if math.isfinite(v["mu_sat_off"]) and v["mu_sat_off"] > 0 else math.nan
                      for k, v in own.items()},
           "day_drift": drift, "day_confounded": day_confounded(drift, delta, p)}
    warm_ok = all(math.isfinite(w) for w in warm.values())
    same_warm = warm_ok and abs(warm["s0"] - warm["s1"]) < 1e-9
    if out["seed_label"] == "undefined" or ("cell_undefined" in (out["cell_s0"], out["cell_s1"]) and warm_ok):
        out["decision"] = "no_floor"
    elif not warm_ok:
        out["decision"] = "no_warm"
    elif out["cell_s0"] != "cell_holds":
        out["decision"] = "reference_moved"
    elif out["cell_s1"] != "cell_holds":
        # rule F r1 G-1: seed-dependent only when the seeds differ beyond the band (or in warm capacity) and the
        # difference is not day-to-day change; otherwise the cell sits at its edge on these runs
        moved = out["seed_label"] != "seed_same" or not same_warm
        out["decision"] = "cell_seed_bound" if moved and not out["day_confounded"] else "cell_edge"
    elif out["seed_label"] == "seed_same" and same_warm:
        out["decision"] = "seed_stable"
    else:
        out["decision"] = "seed_varies"
    return out


def schedule_hash(run_dir) -> str:
    """sha256 of the run's written schedule (`schedule.parquet`, written before the load: harness/run.py:146)."""
    import hashlib
    from pathlib import Path
    withheld.guard(Path(run_dir).name)
    return hashlib.sha256((Path(run_dir) / "schedule.parquet").read_bytes()).hexdigest()


def run_checks(name: str, arm: str, seed: int) -> list[str]:
    """Rule F r1 G-4: a counted run's config seed, and T4's server settings (`t4.check_settings`: the 6041-block pool
    took effect and no step-budget flag), read from its server.log."""
    from analysis.load import RUNS, load_run
    bad = [] if load_run(RUNS / name)["cfg"].get("seed") == seed else [f"config seed is not {seed}"]
    return bad + t4.check_settings(arm, t4.server_readouts(RUNS / name))


def sealed() -> list[str]:
    import analysis.load as L
    if not L.RUNS.exists():
        return []
    return sorted(d.name for d in L.RUNS.iterdir() if "_sweep_g1_" in d.name and withheld.sealed_by(d.name) == "G1")


def table() -> dict:
    """G1's computation, once, after `[RESULT] G2` (its day-drift twins are G2's runs, rule F r1 G-2):
    `AFTERSHOCK_UNSEAL=G1 uv run python -m analysis.g1 table`. Refuses while any G1 folder is sealed, before T4's or
    G2's result, when a counted run fails `run_checks`, or when a seed-1 schedule equals the seed-0 one at its rate.
    Per rate the earliest valid run counts."""
    from analysis import e1, g2, tables
    from analysis.gate import sweep_rows
    from analysis.load import RUNS
    for seal, why in ((t4.T4["seal"], "G1 compares with T4's 2048 arm"), (g2.G2["seal"], "G1's day drift reads G2's twins")):
        if not withheld.lifted(seal):
            raise withheld.WithheldError(f"{why}: '{seal}' first")
    s = sealed()
    if s:
        raise withheld.WithheldError(f"{len(s)} G1 folder(s) sealed; set AFTERSHOCK_UNSEAL=G1")
    par = G1["parents"]
    rows, later = {}, {}
    for k in ("off_s0", "off_s1"):
        r = sweep_rows(par[k])
        if not len(r):
            raise RuntimeError(f"{par[k]}: no valid run can be read")
        rows[k], later[k] = e1.first_valid(r)
    sw = {k: e1.swept_rates(par[k]) for k in par}
    stab = tables.sweep_table()
    wc = {k: t4.warm_capacity(par[k], stab) for k in ("on_s0", "on_s1")}
    # validity of the counted seed-1 runs, and that seed 1 changed the schedule at every rate (rule F r1 G-4)
    counted = {"off_s1": list(rows["off_s1"].run),
               "on_s1": sorted(set(stab[(stab.parent == par["on_s1"]) & stab.valid.astype(bool)].run)
                               - set(wc["on_s1"]["later_valid"]))}
    problems = {n: b for k, names in counted.items() for n in names
                if (b := run_checks(n, "off_2048" if k.startswith("off") else "on_2048", G1["seed"]))}
    s0 = {float(r["rate"]): r["run"] for r in rows["off_s0"].to_dict("records")}
    for r in rows["off_s1"].to_dict("records"):
        x = float(r["rate"])
        if x in s0 and schedule_hash(RUNS / r["run"]) == schedule_hash(RUNS / s0[x]):
            problems.setdefault(r["run"], []).append(f"schedule identical to seed 0's at {x}")
    if problems:
        raise ValueError(f"counted runs that fail G1's checks: {problems}")
    tw, _ = e1.first_valid(sweep_rows(g2.G2["parents"]["off"]))     # G2's mode-off twins: seed 0, G1's day
    ref = {float(r["rate"]): float(r["throughput"]) for r in rows["off_s0"].to_dict("records")}
    drift = {x: math.nan for x in G1["twin_rates"]}
    drift.update({float(r["rate"]): float(r["throughput"]) / ref[float(r["rate"])] - 1
                  for r in tw.to_dict("records") if float(r["rate"]) in ref and float(r["rate"]) in drift})
    floors = t4.shared_floors(rows["off_s0"], rows["off_s1"], sw["off_s0"], sw["off_s1"])
    own = {"s0": t4.floor(rows["off_s0"], sw["off_s0"]), "s1": t4.floor(rows["off_s1"], sw["off_s1"])}
    res = {"outcome": outcome(floors, own, {"s0": wc["on_s0"]["mu"], "s1": wc["on_s1"]["mu"]}, drift),
           "floors": floors, "own_floor": own, "warm": wc,
           "later_valid_not_counted": {**later, **{k: wc[k]["later_valid"] for k in wc}},
           "rows": {k: v.to_dict("records") for k, v in rows.items()}}
    res["against_seen"] = {k: {"floor_rel_e1": own[k]["mu_sat_off"] / G1["seen"]["floor_e1"] - 1,
                               "warm_minus_reg13": wc[f"on_{k}"]["mu"] - G1["seen"]["warm_reg13"]} for k in own}
    res["decision"] = res["outcome"]["decision"]
    return res


if __name__ == "__main__":
    import sys
    from pprint import pprint
    if sys.argv[1:] == ["table"]:
        pprint(table(), width=120, sort_dicts=False)
    else:
        sys.exit("usage: python -m analysis.g1 table")
