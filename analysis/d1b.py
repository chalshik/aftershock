"""D1b (docs/decisions.md [REG] D1b (C2, cap above the cold batch)): REG-19's burst runs at λ = 2.0 on seeds 15–17
with the load generator's session cap at 40. D1's caps (25–29) hold both the population in service and, under a
binding cap, the running batch (in flight <= cap - x_cold × gap = cap - 8.75, `d1.first_order`) below the cold state's
own batch (N ≈ 30, REG-22 batch 2's end windows). At 40 the population is still bounded (REG-19's uncapped runs held far
more in service; `feasibility`), but the in-flight bound, 31.25, is above the cold batch: the cap no longer constrains
the cold state's running batch. D1b asks whether a cap that bounds the population but not the cold batch still prevents
the stuck state. (The KV gauge's 0.95 point, 38.72, is the cap at which the cold batch fits, by construction of
`d1.first_order`; so "the pool can fill" and "the cold batch fits" are one condition here and are not separated.)

Each capped run is labelled by D1's rule (`d1.label`, D1's thresholds) with one change, fixed before any cap-40 run
(`label40`): a run that `d1.label` calls `partial` (cold reuse, no backlog at the cap) is `stuck` when the server's own
line is backlogged by the same test (`d1._backlogged` on the server's waiting gauge). At cap 40 the backlog may sit at
the server, since the cap need not bind.

Outcome (`outcome`, over the seeds with a valid cap-40 run, in the order 15, 16, 17; D1's label on the same seed):
  too_few               n < 2;
  batch_bound           k_bracket >= 2, a bracket being D1 cut (recovered or recovered_cap_draining) and D1b stuck on
                        the same seed: the cap prevented the stuck state only where it also held the running batch below
                        the cold state's;
  stuck_without_d1_cut  k_stuck >= 2 but k_bracket < 2: stuck at 40, with D1 not cut on those seeds; reported, no reading;
  cut_at_40             k_cut >= 2: bounding the population alone prevented it; C2's mechanism wording goes to the author;
  mixed                 otherwise.
Not separated: running-batch pressure from idle-prefix eviction (40 sessions' prefixes exceed the pool, BELIEF), pool
fill from batch fit (above), and harness drift (`d1.drift_readout`, reported) for a cut at 40.
"""
from __future__ import annotations

import math
from pathlib import Path

import yaml

from analysis import d1
from analysis.stats import cp_upper

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"

D1B = {
    "seeds": (15, 16, 17),     # D1's first three seeds, in D1's order (fixed before any cap-40 run)
    "cap": 40,                 # above d1.first_order's cap_kv_095 (38.72) at every seed; chosen by the author, 2026-11-15
    "most": 2,                 # of 3
    "feasible_in_service": 40, # rule G: the uncapped runs' mean sessions in service over the end window must exceed it
    "feasible_k": 2,           # in at least this many of the three uncapped runs
    "conf": 0.95,              # one-sided Clopper-Pearson (rule H)
    # Prediction only (BELIEF; decisions.md [REG] D1b, "Prediction"): P(label) per valid seed, seeds independent.
    "prediction": {"stuck": 0.65, "partial": 0.03, "rewarming": 0.05, "recovered": 0.08,
                   "recovered_cap_draining": 0.04, "server_only": 0.03, "bad_baseline": 0.04, "unclear": 0.08},
}


def config_name(seed: int) -> str:
    return f"burst_l1pl350cl_r2.0_w3000_s{seed}_cap40"


def run_pattern(seed: int) -> str:
    """The cap-40 runs' folders. The name ends in `_cap40`, so D1's pattern (`*_…_cap`) and REG-19's do not match it
    (tests/test_d1b.py)."""
    return f"*_{config_name(seed)}"


def config_text(seed: int) -> str:
    if seed not in D1B["seeds"]:
        raise ValueError(f"seed {seed} is not one of D1b's")
    return (f"# D1b (docs/decisions.md [REG] D1b (C2, cap above the cold batch)): REG-19's burst run at seed\n"
            f"# {seed} with the session cap at {D1B['cap']}, the only change; D1's capped run of this seed differs only\n"
            f"# in the cap. Written by `python -m analysis.d1b configs`. Runs only after `[DECISION] ... D1b starts`.\n"
            f"extends: burst_l1pl350cl_r2.0_w3000_s{seed}.yaml\n"
            f"name: {config_name(seed)}\n"
            f"workload:\n"
            f"  sessions:\n"
            f"    cap: {D1B['cap']}\n")


def resolved(seed: int) -> dict:
    """The resolved config `config_text(seed)` gives, without writing it."""
    from harness.config import _merge, load_config, with_defaults
    from harness.run import expand
    raw = yaml.safe_load(config_text(seed))
    parent = raw.pop("extends")
    return expand(with_defaults(_merge(load_config(CONFIGS / parent), raw)))[0]


def write_configs(configs_dir: Path | None = None) -> list[Path]:
    """Writes the three cap-40 configs as new files; refuses if any exists (a used config is never edited)."""
    d = configs_dir or CONFIGS
    paths = [d / f"{config_name(sd)}.yaml" for sd in D1B["seeds"]]
    if any(p.exists() for p in paths):
        raise FileExistsError([str(p) for p in paths if p.exists()])
    for p, sd in zip(paths, D1B["seeds"]):
        p.write_text(config_text(sd))
    return paths


def first_order() -> dict:
    """D1's first-order KV bound at cap 40 (prediction readout; BELIEF as a model)."""
    return d1.first_order(D1B["cap"])


def capped_run(seed: int) -> tuple[dict | None, list[str]]:
    """The seed's counted cap-40 run (the earliest valid one) and the names of any later valid ones, as `d1.capped_run`.
    Its resolved config must equal `resolved(seed)` (run id and source path aside)."""
    from analysis.load import find_runs
    runs = sorted(find_runs(run_pattern(seed)), key=lambda r: r["dir"].name)
    if not runs:
        return None, []
    r = runs[0]
    got = {k: v for k, v in r["cfg"].items() if k not in ("run_id", "source_config")}
    diff = d1._diff_keys(got, resolved(seed))
    if diff:
        raise ValueError(f"{r['dir'].name}: config differs from D1b's at {diff}")
    return r, [x["dir"].name for x in runs[1:]]


def row(seed: int) -> dict | None:
    """One seed: the counted cap-40 run, D1's label on it and D1's readouts. None if the seed has no valid run. A run
    that `d1.harness_fault` flags is labelled unclear and stays in n (D1's DM2)."""
    from analysis.c2 import eviction_readout, last_burst_end, levels
    from analysis.load import RUNS, load_run, verdict
    run, later = capped_run(seed)
    if run is None:
        return None
    un = load_run(RUNS / d1.D1["uncapped"][seed])
    lab = label40(run)
    sp, sp_un = d1.spans(run), d1.spans(un)
    sl = d1.slots(run, D1B["cap"])
    fault = d1.harness_fault(sl)
    lr = d1.load_reading(sp, sp_un, lab["reuse"]["base"], run["cfg"]["verdict"], last_burst_end(run))
    return {"seed": seed, "cap": D1B["cap"], "run": run["dir"].name, "later_valid": later,
            "uncapped": un["dir"].name, "label": "unclear" if fault else lab["label"], "harness_fault": fault,
            "label_values": lab, "verdict": verdict(run)["verdict"], "load_reading": lr, "onset": d1.onset_class(lr),
            "spans": sp, "uncapped_spans": sp_un, "slots": sl, "waits": d1.waits(run), "levels": levels(run),
            "eviction": eviction_readout(run), "first_order": first_order()}


def label40(run: dict) -> dict:
    """D1's label with D1b's one change (module docstring): `partial` becomes `stuck` when the server's line is
    backlogged by `d1._backlogged`'s test (mean above its baseline mean + recovered_q_excess and not draining within
    drain_horizon_s), read on `load._level`'s waiting mean and slope over D1's baseline and end windows."""
    from analysis.load import _level
    lab = d1.label(run)
    v, w = run["cfg"]["verdict"], lab["windows"]
    lb, le = _level(run, *w["base"]), _level(run, *w["end"])
    server = {"base": {"mean": lb["q"]}, "end": {"mean": le["q"], "slope": le["q_slope"]}}
    nan = any(isinstance(x, float) and math.isnan(x) for x in (lb["q"], le["q"]))
    sb = None if nan else d1._backlogged(server, v)
    out = {**lab, "label_d1": lab["label"], "server_backlogged": sb}
    if lab["label"] == "partial" and sb:
        out["label"] = "stuck"
    return out


def outcome(labels: dict, labels_d1: dict) -> dict:
    """The registered decision (module docstring). `labels`: seed -> D1b's label of its counted cap-40 run;
    `labels_d1`: seed -> D1's label on that seed (missing if D1 has no valid run there)."""
    bad = sorted({lab for lab in list(labels.values()) + list(labels_d1.values()) if lab not in d1.LABELS})
    if bad:
        raise ValueError(f"labels outside d1.LABELS: {bad}")
    n = len(labels)
    k_stuck = sum(lab == "stuck" for lab in labels.values())
    k_cut = sum(lab in d1.CUT for lab in labels.values())
    k_bracket = sum(lab == "stuck" and labels_d1.get(sd) in d1.CUT for sd, lab in labels.items())
    most = D1B["most"]
    if n < most:
        dec = "too_few"
    elif k_bracket >= most:
        dec = "batch_bound"
    elif k_stuck >= most:
        dec = "stuck_without_d1_cut"
    elif k_cut >= most:
        dec = "cut_at_40"
    else:
        dec = "mixed"
    ub = (lambda k: cp_upper(k, n, D1B["conf"])) if n else (lambda k: math.nan)
    return {"decision": dec, "n": n, "k_stuck": k_stuck, "k_cut": k_cut, "k_bracket": k_bracket,
            "cp_upper_stuck": ub(k_stuck), "cp_upper_cut": ub(k_cut),
            "pairs": [{"seed": sd, "d1": labels_d1.get(sd), "d1b": lab,
                       "bracket": bool(lab == "stuck" and labels_d1.get(sd) in d1.CUT)} for sd, lab in labels.items()],
            "counts": {lab: list(labels.values()).count(lab) for lab in sorted(set(labels.values()))}}


def predicted() -> dict:
    """The prediction on D1b's outcome: exact over the seeds' (D1 label, D1b label) pairs, each drawn independently from
    D1's registered prediction (harness_fault counted as unclear, as `d1.predicted_matrix`) and D1B["prediction"]."""
    import itertools
    pb = D1B["prediction"]
    pa = dict(d1.D1["prediction"])
    pa["unclear"] = pa.get("unclear", 0.0) + pa.pop("harness_fault", 0.0)
    for p in (pa, pb):
        if abs(sum(p.values()) - 1) > 1e-9:
            raise ValueError("a prediction does not sum to 1")
    seeds = D1B["seeds"]
    pair = [(a, b, pa[a] * pb[b]) for a in pa for b in pb]
    dist: dict[str, float] = {}
    for combo in itertools.product(pair, repeat=len(seeds)):
        pr = math.prod(c[2] for c in combo)
        d = outcome({sd: c[1] for sd, c in zip(seeds, combo)}, {sd: c[0] for sd, c in zip(seeds, combo)})["decision"]
        dist[d] = dist.get(d, 0.0) + pr
    return dict(sorted(dist.items()))


def feasibility() -> dict:
    """Rule G (module docstring; condition written in the draft before this was computed): the mean sessions in
    service over the end window of the three uncapped REG-19 burst runs on D1b's seeds."""
    from analysis.k1 import sessions_in_service, windows
    from analysis.load import RUNS, load_run
    rows = []
    for sd in D1B["seeds"]:
        un = load_run(RUNS / d1.D1["uncapped"][sd])
        e0, e1 = windows(un["cfg"])["end"]
        rows.append({"seed": sd, "run": un["dir"].name, "end": (e0, e1),
                     "in_service_end": sessions_in_service(un, e0, e1)})
    k = sum(r["in_service_end"] > D1B["feasible_in_service"] for r in rows)
    return {"rows": rows, "k_above": k, "passes": bool(k >= D1B["feasible_k"])}


def _sealed() -> list[str]:
    from analysis import withheld
    from analysis.load import RUNS
    return [p.name for sd in D1B["seeds"] for p in sorted(RUNS.glob(run_pattern(sd))) if withheld.sealed_by(p.name)]


def table() -> dict:
    """D1b's registered computation, run once after the three cap-40 runs and after D1's [RESULT]:
    `AFTERSHOCK_UNSEAL=D1b python -m analysis.d1b table`. Refuses while any cap-40 folder is sealed (find_runs would skip
    it and every seed would read as not run) and before D1's [RESULT] heading exists (D1's labels are read through D1's
    lifted seal, never by unsealing D1). Reported beside the decision: `not_run`, `harness_faults`,
    `decision_underlying` (on each row's `label_values["label"]`), and REG-20's drift rerun (`d1.drift_readout`)."""
    from analysis import withheld
    sealed = _sealed()
    if sealed:
        raise withheld.WithheldError(f"{len(sealed)} cap-40 folder(s) sealed by D1b; the registered computation sets "
                                     f"AFTERSHOCK_UNSEAL=D1b")
    if not withheld.lifted("[RESULT] D1 (C2, session cap)"):
        raise withheld.WithheldError("D1's [RESULT] is not in decisions.md; D1b is computed after it")
    if "D1" in withheld.unsealed_ids():
        raise withheld.WithheldError("D1b's computation does not unseal D1")
    rows, not_run = [], []
    for sd in D1B["seeds"]:
        r = row(sd)
        (not_run if r is None else rows).append(sd if r is None else r)
    labels_d1 = {}
    for sd in D1B["seeds"]:
        r1 = d1.row(sd)
        if r1 is not None:
            labels_d1[sd] = r1["label"]
    labels = {r["seed"]: r["label"] for r in rows}
    return {"rows": rows, "not_run": not_run, "outcome": outcome(labels, labels_d1),
            "harness_faults": [r["seed"] for r in rows if r["harness_fault"]],
            "decision_underlying": outcome({r["seed"]: r["label_values"]["label"] for r in rows}, labels_d1)["decision"],
            "drift": d1.drift_readout()}


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "configs":
        print("\n".join(str(p) for p in write_configs()))
    elif len(sys.argv) == 2 and sys.argv[1] == "predicted":
        print(json.dumps({"predicted": predicted(), "first_order": first_order()}, indent=1))
    elif len(sys.argv) == 2 and sys.argv[1] == "feasibility":
        print(json.dumps(feasibility(), indent=1, default=str))
    elif len(sys.argv) == 2 and sys.argv[1] == "table":
        print(json.dumps(table(), indent=1, default=str))
    else:
        raise SystemExit("usage: python -m analysis.d1b configs | predicted | feasibility | table")
