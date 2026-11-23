"""A28 (docs/open-questions.md): in the closed-loop cache-on sweeps at the 2048 budget on two workload seeds (T4's seed 0,
G1's seed 1), does each run's keep-up reading follow its schedule's 120 s planned surges? A reading of seen runs, 0 GPU
h, after decisions.md 2026-11-22 [RESULT] G1 (second-seed capacities). Nothing is judged: the level is K4's surge level
(`k4.K4["surge_level"]`, 3.4 req/s, registered 2026-11-17, before any of these runs) and the keep-up reading is
`t4.warm_capacity`'s per-rate `keeps_up`, both fixed before this file. The surges are read from each run's
`schedule.parquet`, the plan written before the run, with the 120 s window and 5 s grid of `gate.SURGE`, over
[SURGE start_s, the run's duration].

    uv run python -m analysis.a28 > docs/probe/a28/a28_<commit>_<stamp>.txt
"""
from __future__ import annotations

import math
import pprint
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import g1, k4, t4
from analysis.gate import SURGE, rolling_rate

A28 = {"parents": {"s0": g1.G1["parents"]["on_s0"], "s1": g1.G1["parents"]["on_s1"]},
       "level": k4.K4["surge_level"], "near": k4.K4["calm_level"]}


def surges(t_planned: np.ndarray, duration_s: float, level: float, near: float) -> dict:
    """120 s planned session rate on SURGE's grid over [start_s, duration_s]: its peak and time, the seconds at or
    above `level` and `near` (grid points x grid step), and the mean planned rate over the whole run."""
    t = np.asarray(t_planned, float)
    t = t[t < duration_s]
    g = np.arange(SURGE["start_s"], duration_s + 1e-9, SURGE["grid_s"])
    x = rolling_rate(t, g, SURGE["window_s"])
    return {"n": int(len(t)), "mean_rate": len(t) / duration_s, "peak": float(x.max()), "t_peak": float(g[x.argmax()]),
            "s_at_level": float(SURGE["grid_s"] * (x >= level).sum()),
            "s_at_near": float(SURGE["grid_s"] * (x >= near).sum())}


def run_surges(run_dir: Path, duration_s: float, p: dict = A28) -> dict:
    s = pd.read_parquet(run_dir / "schedule.parquet")
    s = s[~s.burst.astype(bool)]
    return surges(s.t_planned.to_numpy(float), duration_s, p["level"], p["near"])


def agree(rows: list[dict]) -> dict:
    """Rows agree when a run kept up exactly when its schedule had no second at or above the level."""
    bad = [r for r in rows if bool(r["keeps_up"]) == (r["s_at_level"] > 0)]
    return {"n": len(rows), "n_agree": len(rows) - len(bad),
            "disagree": [(r["seed"], r["rate"], r["keeps_up"], r["peak"]) for r in bad]}


def table(p: dict = A28) -> dict:
    from analysis import e1, tables
    from analysis.load import RUNS
    import yaml
    sw = tables.sweep_table()
    rows = []
    for seed, parent in p["parents"].items():
        wc = t4.warm_capacity(parent, sw)
        keep, _ = e1.first_valid(sw[(sw.parent == parent) & sw.valid.astype(bool)])
        for r in keep.to_dict("records"):
            d = RUNS / r["run"]
            dur = float(yaml.safe_load((d / "config.yaml").read_text())["workload"]["duration_s"])
            m = wc["margins"][float(r["rate"])]
            rows.append({"seed": seed, "rate": float(r["rate"]), "run": r["run"], "keeps_up": bool(m["keeps_up"]),
                         "tp_over_ref": m["tp_over_ref"], **run_surges(d, dur, p)})
    return {"level": p["level"], "near": p["near"], "warm": {s: t4.warm_capacity(q, sw)["mu"] for s, q in p["parents"].items()},
            "rows": rows, "agree": agree(rows)}


if __name__ == "__main__":
    pprint.pprint(table(), width=120, sort_dicts=False)
