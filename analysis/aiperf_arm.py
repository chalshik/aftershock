"""TR arm D, the AIPerf default-release arm (docs/drafts/tr_draft2.md; draft 2 of the arm:
docs/drafts/aiperf_default_draft2.md; not registered). REG-19's burst runs at lambda = 2.0 on seeds 15-19 with one
change: each later turn is released at max(planned time, previous turn's end) (`workload.sessions.release:
max_planned`, harness/loadgen.py release_mode), the release AIPerf's trace replay uses without `--open-loop-strict`.
Each burst run has REG-19's control on the same schedule with the same change, ALWAYS run (TR, the author's choice of
draft 1's GAP 1; draft 2's option (a)): the plan is fixed and the seal stays closed until the [RESULT]. The label rule is
unchanged: the control decides a seed only when its burst run is cold (`k1.open_seed`); for a warm burst run the
control's own end-window reading is a readout (`control_end`), never judged.

Per seed (`default_seed`): `k1.open_seed` applied unchanged, its labels renamed (`RENAME`); windows `k1.windows`
(baseline [300, 900) s, end [3480, 4080) s on these configs), thresholds the configs' own (`verdict`) and
`gate.SURGE["kv_onset"]`, as k1 reads them. Across seeds (`outcome`, first match over the counted seeds; `unclear` and
`not_run` leave n): too_few, pending_controls, overload_default, default_sees, default_blind, mixed.
Readouts, never judged: k1's mechanism and readouts (inside each row), the release audit (`release_audit`: lateness,
share released at the previous end, and the two code checks that must read 0), the end-window completion rate, and the
pairing with T1's U arm (`paired_u`, exact one-sided McNemar).

Nothing here reads the environment or a run folder until `table` is called, and `table` refuses while any of the arm's
folders is sealed (analysis/withheld.py, "AIPERF").
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis import d1, k1, t1, withheld

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"

# Every constant of the arm, with its source. Thresholds of the per-seed reading are not here: k1 reads them from each
# run's config (`verdict`: stuck_hit_drop, recovered_hit_drop, recovered_q_excess; configs/stage2_base.yaml:17-20) and
# from gate.SURGE["kv_onset"] (0.9), all inherited, none set by this arm.
AIPERF = {
    "id": "TR",                                       # the seal's registration id (analysis/withheld.py): TR's
    "seal": "[RESULT] TR (three turn-release arms)",  # the decisions.md heading that lifts the seal
    "seeds": k1.REG20["closed_seeds"],                # 15-19: REG-19's, shared by REG-20 and T1 (draft 1, Design)
    "kinds": ("burst", "stage2_control"),             # a burst run and its same-schedule control per seed
    "parent": "{kind}_l1pl350cl_r2.0_w3000_s{seed}",  # REG-19's configs (configs/burst_l1pl350cl_r2.0_w3000_s15.yaml)
    "name": "{kind}_l1pl350dr_r2.0_w3000_s{seed}",    # "dr": default release; outside `*_l1pl350cl_*` (gate.SURGE)
    "release": "max_planned",                         # harness/loadgen.py RELEASE_MODES
    "min_n": t1.T1["most"],                           # 3: too_few below it (draft 1, kd.outcome rule 1: T1's floor)
    "sees_k": k1.REG20["refute_k"],                   # 3: default_sees at k_stuck + k_lost + k_cold_control >= 3
    "max_attempts": 3,                                # draft 1, Design, Reruns: 3 invalid attempts -> lost_default
    "lateness_quantiles": (0.5, 0.9, 0.99),           # release_audit's readout quantiles; a readout, never judged
    # Prediction (draft 1, Prediction; BELIEF inputs, outcome DERIVED by `predicted`)
    "prediction": {
        "burst": {"stuck": 0.50, "recovered": 0.45, "partial": 0.05},   # per counted seed, burst run's reading
        "control_cold_given_cold": 0.05,              # REG-19's five controls all recovered (docs/model.md:112)
        # TR (out-of-n terms, BELIEF): a NaN window or a queue stop 0.02 per seed; three invalid attempts 0.005 (the
        # stuck closed-loop runs have passed their checks: REG-19's ten runs and T1's arm U so far, rule-B evidence
        # only once T1's [RESULT] is in)
        "unclear": 0.02,
        "lost": 0.005,
    },
}

RENAME = {"recovered_open": "recovered_default", "partial_open": "partial_default", "stuck_open": "stuck_default",
          "overload_open": "overload_default", "lost_open": "lost_default"}
LABELS = ("recovered_default", "partial_default", "stuck_default", "overload_default", "pending_control",
          "cold_control", "unclear", "lost_default", "not_run", "release_fault")
# TR rule F r1: an overload seed (control cold too) is out of n, counted and reported (M6), as is a run whose release
# audit fires (M7: a code fault, not a finding)
OUT_OF_N = ("unclear", "not_run", "overload_default", "release_fault")
SEES = ("stuck_default", "lost_default", "cold_control")           # draft 1, kd.outcome rule 4 (as REG-20's k_stuck)
DEFAULT_COLD = tuple(RENAME.get(x, x) for x in k1.OPEN_COLD)       # the burst run read cold (paired_u's D side)
OUTCOMES = ("too_few", "pending_controls", "default_sees", "default_blind", "mixed")


# ---------------------------------------------------------------- configs

def config_name(kind: str, seed: int) -> str:
    if kind not in AIPERF["kinds"]:
        raise ValueError(f"kind {kind!r} is not one of {AIPERF['kinds']}")
    if seed not in AIPERF["seeds"]:
        raise ValueError(f"seed {seed} is not one of {AIPERF['seeds']}")
    return AIPERF["name"].format(kind=kind, seed=seed)


def parent_name(kind: str, seed: int) -> str:
    config_name(kind, seed)
    return AIPERF["parent"].format(kind=kind, seed=seed)


def run_pattern(kind: str, seed: int) -> str:
    return f"*_{config_name(kind, seed)}"


def config_text(kind: str, seed: int) -> str:
    """REG-19's config of the seed and kind, renamed, with `workload.sessions.release` set; nothing else changes."""
    what = "burst run" if kind == "burst" else "control (always run: TR, the author's choice)"
    return (f"# TR arm D, AIPerf default release (docs/drafts/tr_draft2.md; not registered), seed {seed}, {what}.\n"
            f"# REG-19's config of the seed with one change: each later turn is released at max(planned time, previous\n"
            f"# turn's end), AIPerf's default trace replay (harness/loadgen.py release_mode). Written by\n"
            "# `python -m analysis.aiperf_arm configs`. Runs only after the author's [DECISION] that starts TR.\n"
            f"extends: {parent_name(kind, seed)}.yaml\n"
            f"name: {config_name(kind, seed)}\n"
            f"workload:\n"
            f"  sessions:\n"
            f"    release: {AIPERF['release']}\n")


def resolved(kind: str, seed: int) -> dict:
    from harness.config import _merge, load_config, with_defaults
    from harness.run import expand
    raw = yaml.safe_load(config_text(kind, seed))
    parent = raw.pop("extends")
    return expand(with_defaults(_merge(load_config(CONFIGS / parent), raw)))[0]


def write_configs(configs_dir: Path | None = None) -> list[Path]:
    """All ten configs as new files; refuses if any exists (a used config is never edited)."""
    d = configs_dir or CONFIGS
    pairs = [(k, sd) for sd in AIPERF["seeds"] for k in AIPERF["kinds"]]
    paths = [d / f"{config_name(k, sd)}.yaml" for k, sd in pairs]
    if any(p.exists() for p in paths):
        raise FileExistsError([str(p) for p in paths if p.exists()])
    for p, (k, sd) in zip(paths, pairs):
        p.write_text(config_text(k, sd))
    return paths


# ---------------------------------------------------------------- readouts of one run

def release_audit(run: dict) -> dict:
    """Per later session turn sent (no burst row): lateness t_released - t_planned, and the share released at the
    previous turn's end (that end later than the planned time: the default branch). Two code checks that must read 0
    on a max_planned run, with exact float equality (both sides are the same recorded numbers): `mismatch`, turns whose
    t_released is not max(t_planned, previous t_end), and `before_prev_end`, turns sent before the previous turn ended.
    A turn whose previous turn has no t_end (it failed; the harness then uses the failure time) is left out of both."""
    from analysis.margin import is_burst_request
    q = run["requests"]
    q = q[~is_burst_request(run)].sort_values(["session_id", "turn"]).copy()
    g = q.groupby("session_id")
    q["prev_end"], q["prev_turn"] = g.t_end.shift(), g.turn.shift()
    s = q[(q.turn > 0) & q.t_sent.notna()]
    late = (s.t_released.astype(float) - s.t_planned.astype(float)) if len(s) else pd.Series(dtype=float)
    known = s.prev_end.notna() & (s.prev_turn == s.turn - 1)
    want = np.maximum(s.t_planned.astype(float), s.prev_end.astype(float))
    mismatch = int((known & (s.t_released.astype(float) != want)).sum()) if len(s) else 0
    early = int((known & (s.t_sent.astype(float) < s.prev_end.astype(float))).sum()) if len(s) else 0
    at_end = known & (s.prev_end.astype(float) > s.t_planned.astype(float))
    return {"later_sent": int(len(s)),
            "lateness_mean_s": float(late.mean()) if len(s) else math.nan,
            "lateness_q_s": {str(x): float(late.quantile(x)) if len(s) else math.nan
                             for x in AIPERF["lateness_quantiles"]},
            "share_at_prev_end": float(at_end.sum() / known.sum()) if known.sum() else math.nan,
            "mismatch": mismatch, "before_prev_end": early}


def release_fault(audit: dict, mechanism: dict | None) -> bool:
    """A code fault, not a finding: a release that is not the rule, a turn sent while its previous one ran, or a
    nonzero held-prefix overlap share (`k1.mechanism`; 0 by construction under max_planned, draft 1, Readouts)."""
    ov = (mechanism or {}).get("overlap", math.nan)
    return audit["mismatch"] > 0 or audit["before_prev_end"] > 0 or (not math.isnan(ov) and ov != 0)


def default_seed(burst: dict | None, control: dict | None, lost: bool = False, control_lost: bool = False) -> dict:
    """`k1.open_seed`, labels renamed (RENAME); a cold burst run whose control was invalid max_attempts times is
    `cold_control` (TR rule F r1 B1, as k1's "control unreadable"); a burst run whose release audit fires is
    `release_fault` (out of n, M7). Also the release audit and the end-window completion rate (readouts)."""
    from analysis.gate import completion_rate
    row = k1.open_seed(burst, control, lost)
    row["label"] = RENAME.get(row["label"], row["label"])
    if row["label"] == "pending_control" and control_lost:
        row["label"], row["control_lost"] = "cold_control", True
    if burst is not None:
        a, b = k1.windows(burst["cfg"])["end"]
        row["release"] = release_audit(burst)
        row["release_fault"] = release_fault(row["release"], row.get("mechanism"))
        if row["release_fault"]:
            row["label_before_fault"], row["label"] = row["label"], "release_fault"
        row["completion_end"] = {"rate": completion_rate(burst, a, b), "lam": float(burst["cfg"]["workload"]["rate"])}
        if control is not None and row.get("control") is None:
            # TR: the control always runs; for a burst run not read cold its reading is a readout only
            c = k1.cold_backlogged(control, k1.windows(burst["cfg"]), burst["cfg"]["verdict"])
            row["control_end"] = {"run": control["dir"].name, "cold_backlogged": c["cold_backlogged"],
                                  "recovered": c["recovered"]}
    return row


# ---------------------------------------------------------------- the decision

def decide(labels: list[str]) -> str:
    """draft 1, kd.outcome, first match over the counted seeds (`unclear`, `not_run` leave n)."""
    counted = [x for x in labels if x not in OUT_OF_N]
    n = len(counted)
    if n < AIPERF["min_n"]:
        return "too_few"
    if "pending_control" in counted:
        return "pending_controls"
    if sum(x in SEES for x in counted) >= AIPERF["sees_k"]:
        return "default_sees"
    if all(x == "recovered_default" for x in counted):
        return "default_blind"
    return "mixed"


def outcome(labels: list[str]) -> dict:
    """The decision, n, the label counts, and per counted label (and the `sees` sum) k of n with its one-sided 95 %
    Clopper-Pearson upper bound (rule H)."""
    from analysis.stats import cp_upper
    bad = [x for x in labels if x not in LABELS]
    if bad:
        raise ValueError(f"not labels of the AIPerf arm: {bad}")
    counted = [x for x in labels if x not in OUT_OF_N]
    n, k = len(counted), Counter(counted)
    k_sees = sum(k[x] for x in SEES)
    cells = {x: {"k": k[x], "n": n, "upper": cp_upper(k[x], n) if n else None} for x in LABELS if x not in OUT_OF_N}
    cells["sees"] = {"k": k_sees, "n": n, "upper": cp_upper(k_sees, n) if n else None}
    return {"decision": decide(labels), "n": n, "k_sees": k_sees, "counts": dict(Counter(labels)), "cells": cells,
            "k_overload": Counter(labels)["overload_default"], "k_release_fault": Counter(labels)["release_fault"]}


def paired_u(d_labels: dict, u_cold: dict) -> dict:
    """Readout: the arm against T1's U arm (REG-19's config on the same stack), paired by seed. D cold = the burst run
    read cold (DEFAULT_COLD); U cold = `k1.cold_backlogged` on its own windows (T1's `u_ok` end window). A seed with
    D out of n or U None leaves the pairs. b = cold only in U, c = cold only in D; `k1.mcnemar_one_sided(b, c)`."""
    pairs = {sd: (bool(u_cold[sd]), d_labels[sd] in DEFAULT_COLD) for sd in d_labels
             if d_labels[sd] not in OUT_OF_N and u_cold.get(sd) is not None}
    b = sum(u and not d for u, d in pairs.values())
    c = sum(d and not u for u, d in pairs.values())
    return {"pairs": {sd: {"u_cold": u, "d_cold": d} for sd, (u, d) in pairs.items()}, "b": b, "c": c,
            "p_one_sided": k1.mcnemar_one_sided(b, c)}


# ---------------------------------------------------------------- the prediction

def seed_dist(pred: dict | None = None) -> dict:
    """One seed's label distribution from the BELIEF inputs: stuck splits into stuck_default and overload_default by
    the control's chance to be cold; the in-n labels are scaled by 1 - unclear - lost."""
    p = pred or AIPERF["prediction"]
    bu = p["burst"]
    if abs(sum(bu.values()) - 1) > 1e-9:
        raise ValueError("the burst prediction does not sum to 1")
    keep = 1 - p["unclear"] - p["lost"]
    cc = p["control_cold_given_cold"]
    out = {"stuck_default": keep * bu["stuck"] * (1 - cc), "overload_default": keep * bu["stuck"] * cc,
           "recovered_default": keep * bu["recovered"], "partial_default": keep * bu["partial"],
           "unclear": p["unclear"], "lost_default": p["lost"]}
    return {x: v for x, v in out.items() if v > 0}


def predicted(n_seeds: int | None = None, pred: dict | None = None) -> dict:
    """Exact over n_seeds independent seeds (default: the five), every pattern of `seed_dist` through `decide`.
    `no_sentence` is the mass of the outcomes that change no sentence (mixed, too_few; and
    pending_controls, which the prediction cannot reach, as every cold burst run has its control)."""
    n = len(AIPERF["seeds"]) if n_seeds is None else n_seeds
    cats = seed_dist(pred)
    dist = dict.fromkeys(OUTCOMES, 0.0)
    for combo in itertools.product(cats, repeat=n):
        dist[decide(list(combo))] += math.prod(cats[c] for c in combo)
    return {"seed": cats, "outcome": dist,
            "no_sentence": sum(dist[x] for x in ("mixed", "too_few", "pending_controls"))}


# ---------------------------------------------------------------- the registered computation (never run here)

def _check_cfg(run: dict, kind: str, seed: int) -> None:
    got = {k: v for k, v in run["cfg"].items() if k not in ("run_id", "source_config")}
    diff = d1._diff_keys(got, resolved(kind, seed))
    if diff:
        raise ValueError(f"{run['dir'].name}: config differs from the arm's {config_name(kind, seed)} at {diff}")


def _run(kind: str, seed: int) -> tuple[dict | None, int]:
    """(the one valid run or None, its invalid attempts), by `k1._runs` (a second valid run raises)."""
    r, n_bad = k1._runs(run_pattern(kind, seed))
    if r is not None:
        _check_cfg(r, kind, seed)
    return r, n_bad


def sealed() -> list[str]:
    import analysis.load as L
    if not L.RUNS.exists():
        return []
    return sorted(d.name for d in L.RUNS.iterdir()
                  if "_l1pl350dr_" in d.name and withheld.sealed_by(d.name) == AIPERF["id"])


def table() -> dict:
    """Arm D's part of TR's registered computation (`tr.table`, run with AFTERSHOCK_UNSEAL=TR). Refuses while any of its folders is sealed. The pairing with T1's U arm is read only
    once T1's seal has lifted (else `paired_u` is reported as withheld)."""
    s = sealed()
    if s:
        raise withheld.WithheldError(f"{len(s)} arm D folder(s) sealed; set AFTERSHOCK_UNSEAL={AIPERF['id']}")
    rows = {}
    for sd in AIPERF["seeds"]:
        b, n_bad = _run("burst", sd)
        c, n_bad_c = _run("stage2_control", sd)
        rows[sd] = {"seed": sd, "invalid_attempts": {"burst": n_bad, "control": n_bad_c},
                    **default_seed(b, c, lost=b is None and n_bad >= AIPERF["max_attempts"],
                                   control_lost=c is None and n_bad_c >= AIPERF["max_attempts"])}
    labels = {sd: r["label"] for sd, r in rows.items()}
    if t1.sealed():
        pu = {"withheld": "T1's U runs are sealed until [RESULT] T1"}
    else:
        u_cold = {}
        for sd in AIPERF["seeds"]:
            u, _ = t1.arm_run("U", sd, None)
            u_cold[sd] = None if u is None else k1.cold_backlogged(u)["cold_backlogged"]
        pu = paired_u(labels, u_cold)
    return {"outcome": outcome(list(labels.values())), "rows": rows, "paired_u": pu,
            "release_faults": [sd for sd, r in rows.items() if r.get("release_fault")]}


if __name__ == "__main__":
    import json
    import sys
    cmd = sys.argv[1:]
    if cmd == ["table"]:
        print(json.dumps(table(), indent=1, default=str))
    elif cmd == ["predicted"]:
        print(json.dumps(predicted(), indent=1))
    elif cmd == ["configs"]:
        print("\n".join(str(p) for p in write_configs()))
    else:
        sys.exit("usage: python -m analysis.aiperf_arm table | predicted | configs")
