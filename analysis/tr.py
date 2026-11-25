"""TR, three turn-release arms in one queue after T1's (docs/drafts/tr_draft2.md; not registered).

Arm O (`analysis/armo.py`): REG-20 arm A rerun at λ = 2.0, seeds 15-19, judged on O alone. Arm D
(`analysis/aiperf_arm.py`): REG-19's burst runs with AIPerf's default release (`max_planned`). Arm H, here: REG-20 arm
A's open-loop burst runs at λ = TR["h_lam"] = 2.73, the burst unchanged (as REG-20 arm B at 2.5), seeds 15-19, each with
its own same-schedule no-burst control, always run. Every burst run of every arm has its control, always run, in one
launcher plan (`plan`), so no label is read before TR's [RESULT].

H per seed (`h_seed`), first match:
  not_run / lost_open      no valid burst run (lost after TR["max_attempts"] invalid attempts: counted, cold);
  control_not_run / control_lost   no valid control (out of n);
  control_unclear / control_cold / control_partial   the control, read with `k1.cold_backlogged` on the burst run's
                           windows and verdict rules, is NaN, cold-backlogged, or not recovered (out of n: the schedule
                           itself does not stay warm at 2.73, so the seed cannot say what the burst did);
  otherwise `k1.open_seed`'s label (recovered_open, partial_open, stuck_open, unclear; overload_open cannot occur).
`decide_h`, first match: cell_unstable (control_cold + control_partial >= unstable_k over all five seeds: 2.73 is not a
stable open-loop cell on these seeds; rule F r1 B4); too_few (n < min_n counted); headroom_explains (stuck_open +
lost_open >= fail_k); headroom_excluded (no counted seed stuck or lost, and recovered_open >= min_n, partial_open
allowed but not counted toward it; rule F r1 B5, the review's A1 option as fixed in round 2, N1 option a); mixed.

Joint (`joint`), O and H, a summary line only, never judged and with no consequence of its own (rule F r1 B2):
lead_fails (O lead_fails or H headroom_explains); lead_holds (O replicates and H headroom_excluded); lead_unsettled.

Nothing here reads a run folder except through `k1.find_runs` (sealed folders are skipped until TR's [RESULT]).
"""
from __future__ import annotations

import itertools
import json
import math
from collections import Counter
from pathlib import Path

import yaml

from analysis import aiperf_arm, armo, k1, t1, withheld

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
PLAN = REPO / "docs/probe/tr_ops/plan.json"
KINDS = ("burst", "stage2_control")

TR = {
    "id": "TR",
    "seal": "[RESULT] TR (three turn-release arms)",
    "seeds": t1.SEEDS,                         # 15-19, as arms O and D (REG-19's seeds)
    # DERIVED: 2.0 x 3.0 / 2.2 = 2.727, rounded to 2.73: the closed arm's λ over its warm capacity (2.0 / 2.2, T4's
    # closed-loop warm capacity, decisions.md 2026-11-21 [RESULT] T4) applied to the open-loop warm capacity 3.0
    # (μ_warm, decisions.md 2026-11-07 [RESULT] REG-8; one seed, 0.5 steps: the true value lies in [3.0, 3.5)).
    # Kept after rule F r1 B5 as the outside review's requested point. Chosen by the author at 19:00 UTC
    # 2026-11-23 (decisions.md [DECISION] (author) The turn-release batch ...).
    "h_lam": 2.73,
    "h_parent_lam": k1.REG20["a_lam"],         # 2.0: H's configs extend REG-20 arm A's of the seed
    "h_suffix": "trh",                         # keeps H out of k1.open_pattern (formats λ .1f, ends at the seed)
    "max_attempts": 3,                         # the launcher's, as T1 and arm O
    "min_n": 3,                                # as arms O and D
    "fail_k": k1.REG20["refute_k"],            # 3: as O's lead_fails, REG-20's refutation count
    "h_seeds": 5,                              # chosen under the 2026-11-23 pre-approval (5 seeds, not 3)
    "wait_min": 110,                           # T1's plan: a 4080 s run plus start-up and copy-back
    "run_h": 1.25,                             # GPU h per run, T1's [REG] cost (FILE docs/decisions.md:24955; the
                                               # [REG] heading is :24850)
    "gb_per_run": 0.239,                       # DERIVED: T1's pre-launch free disk 12,430,552 (P-15) -> 10,755,812 kB
                                               # (U-19) over 7 runs (PROBE(docs/probe/t1_ops/prelaunch_T1_*.txt))
    "free_after_t1_kb": 10516563,              # DERIVED: 10,755,812 - 239,249 (U-19's own run)
    "unstable_k": 3,                           # cell_unstable: >= 3 of 5 seeds with a cold or partial control (B4)
    "burst_rate": 2.4, "burst_s": 180.0,       # the trigger, inherited (FILE configs/burst_l1pl350cl_r2.0_w3000_s15.yaml)
    # H's per-seed prediction (BELIEF inputs; outcome DERIVED by `predicted_h`)
    "h_prediction": {
        # out of n through the control: cold or not recovered at 2.73 = 0.91 of a one-seed warm capacity. A28: warm
        # capacity follows the draw's surges (11 of 12 sweep runs); K4: closed-loop schedules with a 120 s peak >= 3.4
        # at 2.2 tipped in 4 of 5. No open-loop control above 2.5 has run.
        "control_cold": 0.20, "control_partial": 0.10, "control_lost": 0.005, "control_unclear": 0.005,
        "burst_unclear": 0.01,
        # conditional on a counted seed: arm A recovered 5 of 5 at 2.0 and arm B 3 of 3 at 2.5 (decisions.md
        # 2026-11-16 [RESULT] REG-20); at 2.73 the drain margin is thinner (docstring of `drain_margin`)
        "given_counted": {"recovered_open": 0.70, "partial_open": 0.12, "stuck_open": 0.15, "lost_open": 0.03},
    },
}
H_LABELS = ("recovered_open", "partial_open", "stuck_open", "unclear", "lost_open", "not_run",
            "control_not_run", "control_lost", "control_unclear", "control_cold", "control_partial")
H_OUT = ("unclear", "not_run", "control_not_run", "control_lost", "control_unclear", "control_cold", "control_partial")
H_FAIL = ("stuck_open", "lost_open")
H_OUTCOMES = ("cell_unstable", "too_few", "headroom_explains", "headroom_excluded", "mixed")
H_UNSTABLE = ("control_cold", "control_partial")
JOINT = ("lead_fails", "lead_holds", "lead_unsettled")


# ---------------------------------------------------------------- H's configs

def config_name(kind: str, seed: int) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {KINDS}")
    if seed not in TR["seeds"]:
        raise ValueError(f"seed {seed} is not one of TR's")
    return f"{kind}_l1pl350ol_r{TR['h_lam']:.2f}_w3000_s{seed}_{TR['h_suffix']}"


def parent_name(kind: str, seed: int) -> str:
    return k1.open_pattern(TR["h_parent_lam"], seed, control=kind == "stage2_control")[2:]


def run_pattern(kind: str, seed: int) -> str:
    return f"*_{config_name(kind, seed)}"


def config_text(kind: str, seed: int) -> str:
    what = "control (always run)" if kind == "stage2_control" else "burst run"
    return (f"# TR arm H (docs/drafts/tr_draft2.md; not registered), seed {seed}, {what}: REG-20 arm A's open-loop\n"
            f"# {kind} at λ = {TR['h_lam']} (2.0 x 3.0 / 2.2), the only change; the burst itself is unchanged.\n"
            "# Written by `python -m analysis.tr configs`. Runs only after the author's [DECISION] that starts TR.\n"
            f"extends: {parent_name(kind, seed)}.yaml\n"
            f"name: {config_name(kind, seed)}\n"
            "workload:\n"
            f"  rate: {TR['h_lam']}\n")


def write_configs(configs_dir: Path | None = None) -> list[Path]:
    d = configs_dir or CONFIGS
    items = [(k, sd) for sd in TR["seeds"] for k in KINDS]
    paths = [d / f"{config_name(k, sd)}.yaml" for k, sd in items]
    if any(p.exists() for p in paths):
        raise FileExistsError([str(p) for p in paths if p.exists()])
    for p, (k, sd) in zip(paths, items):
        p.write_text(config_text(k, sd))
    return paths


def resolved(kind: str, seed: int) -> dict:
    from harness.config import _merge, load_config, with_defaults
    from harness.run import expand
    raw = yaml.safe_load(config_text(kind, seed))
    parent = raw.pop("extends")
    return expand(with_defaults(_merge(load_config(CONFIGS / parent), raw)))[0]


def check_config(run: dict | None, kind: str, seed: int) -> None:
    from analysis.d1 import _diff_keys
    if run is None:
        return
    got = {k: v for k, v in run["cfg"].items() if k not in ("run_id", "source_config")}
    diff = _diff_keys(got, resolved(kind, seed))
    if diff:
        raise ValueError(f"{run['dir'].name}: config differs from arm H's {kind} at {diff}")


def drain_margin(capacity: float = 3.0, outstanding: float | None = None) -> dict:
    """DERIVED, a check of the inherited windows at 2.73 (rule F r1 M11): the burst on top of 2.73 leaves about
    (λ + burst rate - capacity) x 180 s outstanding (383 at capacity 3.0, not arm A's 238), cleared at capacity - λ,
    against the 2400 s from the burst's end (1080 s) to the end window (3480 s). 1419 s at 3.0; at a per-seed capacity
    of 2.9, 401 / 0.17 = 2361 s: the drain is not assured at capacity <= 2.9, and a slow drain reads `partial_open`
    (counted; under B5 it does not block headroom_excluded)."""
    lam = TR["h_lam"]
    out = (lam + TR["burst_rate"] - capacity) * TR["burst_s"] if outstanding is None else outstanding
    head = capacity - lam
    return {"outstanding": out, "headroom": head, "drain_s": out / head if head > 0 else math.inf,
            "gap_s": 3480 - 1080}


# ---------------------------------------------------------------- H per seed

def h_seed(burst: dict | None, control: dict | None, lost: bool = False, control_lost: bool = False) -> dict:
    if burst is None:
        return {"label": "lost_open" if lost else "not_run", "control": None}
    if control is None:
        return {"label": "control_lost" if control_lost else "control_not_run", "run": burst["dir"].name,
                "control": None}
    c = k1.cold_backlogged(control, k1.windows(burst["cfg"]), burst["cfg"]["verdict"])
    ctl = {"run": control["dir"].name, "cold_backlogged": c["cold_backlogged"], "recovered": c["recovered"]}
    if c["cold_backlogged"] is None:
        return {"label": "control_unclear", "run": burst["dir"].name, "control": ctl}
    if c["cold_backlogged"]:
        return {"label": "control_cold", "run": burst["dir"].name, "control": ctl}
    if not c["recovered"]:
        return {"label": "control_partial", "run": burst["dir"].name, "control": ctl}
    out = k1.open_seed(burst, control)
    if out["label"] not in H_LABELS:
        raise ValueError(f"{burst['dir'].name}: label {out['label']} beside a recovered control")
    return {**out, "control": ctl}


def h_row(seed: int) -> dict:
    hb, nb = k1._runs(run_pattern("burst", seed))
    hc, nc = k1._runs(run_pattern("stage2_control", seed))
    check_config(hb, "burst", seed)
    check_config(hc, "stage2_control", seed)
    return {"seed": seed, "invalid_attempts": {"burst": nb, "control": nc},
            **h_seed(hb, hc, lost=hb is None and nb >= TR["max_attempts"],
                     control_lost=hc is None and nc >= TR["max_attempts"])}


# ---------------------------------------------------------------- verdicts

def decide_h(labels: list[str]) -> str:
    bad = [x for x in labels if x not in H_LABELS]
    if bad:
        raise ValueError(f"not H labels: {bad}")
    if sum(x in H_UNSTABLE for x in labels) >= TR["unstable_k"]:
        return "cell_unstable"
    c = [x for x in labels if x not in H_OUT]
    if len(c) < TR["min_n"]:
        return "too_few"
    k_fail = sum(x in H_FAIL for x in c)
    if k_fail >= TR["fail_k"]:
        return "headroom_explains"
    # rule F r2 N1 (option a): partial_open is allowed but does not count toward the min_n recovered seeds
    if k_fail == 0 and sum(x == "recovered_open" for x in c) >= TR["min_n"]:
        return "headroom_excluded"
    return "mixed"


def outcome_h(labels: list[str]) -> dict:
    from analysis.stats import cp_upper
    c = [x for x in labels if x not in H_OUT]
    n, k = len(c), Counter(c)
    cells = {x: {"k": k[x], "n": n, "upper": cp_upper(k[x], n) if n else None}
             for x in ("recovered_open", "partial_open", "stuck_open", "lost_open")}
    return {"decision": decide_h(labels), "n": n, "counts": dict(Counter(labels)), "cells": cells,
            "k_unstable": sum(x in H_UNSTABLE for x in labels)}


def h_consequence(h: str) -> str:
    """TR rule F r1 B2, B5: H's own consequence. headroom_excluded: §4.4 adds that the open arm does not stick at the
    closed arm's warm-capacity ratio; headroom_explains: the author words the K1 sentence with H's counts (H alone
    never withdraws the lead); cell_unstable: no §4.4 sentence, counts reported; otherwise no sentence."""
    return {"headroom_excluded": "s44_headroom_excluded", "headroom_explains": "author_words_k1_with_h_counts",
            "cell_unstable": "no_sentence_cell_unstable"}.get(h, "no_sentence")


def joint(o: str, h: str) -> str:
    if o not in armo.OUTCOMES or h not in H_OUTCOMES:
        raise ValueError(f"not O / H outcomes: {o}, {h}")
    if o == "lead_fails" or h == "headroom_explains":
        return "lead_fails"
    if o == "replicates" and h == "headroom_excluded":
        return "lead_holds"
    return "lead_unsettled"


# ---------------------------------------------------------------- prediction (exact over independent seeds)

def h_seed_dist(p: dict | None = None) -> dict:
    p = {**TR["h_prediction"], **(p or {})}
    outs = {"control_cold": p["control_cold"], "control_partial": p["control_partial"],
            "control_lost": p["control_lost"], "control_unclear": p["control_unclear"], "unclear": p["burst_unclear"]}
    keep = 1 - sum(outs.values())
    if keep < 0 or abs(sum(p["given_counted"].values()) - 1) > 1e-9:
        raise ValueError("H prediction inputs do not sum")
    d = {**outs, **{x: keep * v for x, v in p["given_counted"].items()}}
    return {x: v for x, v in d.items() if v > 0}


def predicted_h(p: dict | None = None, n_seeds: int | None = None) -> dict:
    n = n_seeds or TR["h_seeds"]
    s = h_seed_dist(p)
    d = dict.fromkeys(H_OUTCOMES, 0.0)
    for combo in itertools.product(s, repeat=n):
        d[decide_h(list(combo))] += math.prod(s[c] for c in combo)
    p_out = sum(v for x, v in s.items() if x in H_OUT)
    p_n_lt = sum(math.comb(n, j) * (1 - p_out) ** j * p_out ** (n - j) for j in range(TR["min_n"]))
    p_un = sum(v for x, v in s.items() if x in H_UNSTABLE)
    p_unst = sum(math.comb(n, j) * p_un ** j * (1 - p_un) ** (n - j) for j in range(TR["unstable_k"], n + 1))
    return {"seed": s, "outcome": d, "p_out_per_seed": p_out, "p_n_lt_min": p_n_lt, "p_unstable": p_unst,
            "n_seeds": n}


def predicted() -> dict:
    """O (`armo.predicted`), H at 5 seeds and at 3, D (`aiperf_arm.predicted`), and the joint summary line with O and H
    independent (BELIEF: they share the stack, the day and the seeds; rule F r1 M12). Every number in the [REG]'s Prediction section is printed by `python -m analysis.tr predicted`."""
    o = armo.predicted()
    h5, h3 = predicted_h(n_seeds=5), predicted_h(n_seeds=3)
    j = dict.fromkeys(JOINT, 0.0)
    for (a, pa), (b, pb) in itertools.product(o["outcome"].items(), h5["outcome"].items()):
        j[joint(a, b)] += pa * pb
    po = armo.seed_dist()
    o_out = po.get("unclear", 0.0)
    o_n_lt = sum(math.comb(5, i) * (1 - o_out) ** i * o_out ** (5 - i) for i in range(armo.ARMO["min_n"]))
    d = aiperf_arm.predicted()
    return {"o": {"outcome": o["outcome"], "outcome_at_colleague_k": o["outcome_at_colleague_k"],
                  "u_reading": o["u_reading"], "p_mcnemar_le_alpha": o["p_mcnemar_le_alpha"], "p_n_lt_min": o_n_lt,
                  "consequence": o["consequence"]},
            "h": h5, "h_3_seeds": h3, "d": {"outcome": d["outcome"], "no_sentence": d["no_sentence"]},
            "joint": j}


# ---------------------------------------------------------------- the queue

# Per seed three pairs (O, H, D), the arm order rotated across seeds; within a pair the burst run first at 15, 17, 19
# and the control first at 16, 18, so neither arm nor kind sits always at a fixed place in the day.
ARM_ORDER = {15: "OHD", 16: "HDO", 17: "DOH", 18: "ODH", 19: "DHO"}
BURST_FIRST = {15: True, 16: False, 17: True, 18: False, 19: True}


def arm_config(arm: str, kind: str, seed: int) -> str:
    return {"O": armo.config_name, "H": config_name, "D": aiperf_arm.config_name}[arm](kind, seed)


def queue() -> list[dict]:
    items = []
    for sd in TR["seeds"]:
        kinds = KINDS if BURST_FIRST[sd] else KINDS[::-1]
        for arm in ARM_ORDER[sd]:
            for kind in kinds:
                name = arm_config(arm, kind, sd)
                items.append({"key": f"{arm}{'c' if kind == 'stage2_control' else ''}-{sd}", "run_name": name,
                              "config": f"configs/{name}.yaml", "wait_min": TR["wait_min"]})
    return items


def plan() -> dict:
    """The launcher plan: T1's format and checks, after T1's gate and [RESULT] headings and with T1's queue lock."""
    t1p = json.loads((REPO / "docs/probe/t1_ops/plan.json").read_text())
    return {
        "id": "TR",
        "commit_file": "docs/probe/tr_ops/launch_commit.txt",
        "state_dir": "docs/probe/tr_ops/state",
        "probe_dir": "docs/probe/tr_ops",
        "reg_pat": "^## \\d{4}-\\d{2}-\\d{2} · \\[REG\\] TR \\(three turn-release arms\\):",
        "approve_pat": "^## \\d{4}-\\d{2}-\\d{2} · \\[DECISION\\] \\(author, pre-approved\\) TR starts",
        "halt_pat": "^## \\d{4}-\\d{2}-\\d{2} · \\[(DECISION|CORR)\\] \\(author(, pre-approved)?\\) TR halted",
        "after_pats": [*t1p["after_pats"], "^## \\d{4}-\\d{2}-\\d{2} · \\[RESULT\\] T1 \\(rescue, two arms\\)"],
        "locks": [*t1p["locks"], "docs/probe/t1_ops/state/queue.lock"],
        "items": queue(),
        "vm": t1p["vm"],
        "prelaunch": t1p["prelaunch"],
        "allowed": ("Allowed reads while diagnosing a stop: 'tmux has-session -t run'; the folder names under runs/; a "
                    "run's checks.json (valid, failed); in a launch log only the lines holding 'run_id' or '[run] "
                    "checks'. TR's 30 runs stay sealed until '[RESULT] TR (three turn-release arms)' "
                    "(analysis/withheld.py)."),
        "stop": t1p["stop"],
    }


def plan_text() -> str:
    return json.dumps(plan(), indent=1, ensure_ascii=False) + "\n"


def cost() -> dict:
    n = len(queue())
    return {"runs": n, "gpu_h": n * TR["run_h"], "gpu_h_max": n * TR["max_attempts"] * TR["run_h"],
            "disk_gb": n * TR["gb_per_run"], "disk_gb_max": n * TR["max_attempts"] * TR["gb_per_run"],
            "free_after_t1_gb": TR["free_after_t1_kb"] / 1e6,
            "slack_gb": (TR["free_after_t1_kb"] - 2097152) / 1e6 - n * TR["gb_per_run"],
            "queue_h_at_wait": n * TR["wait_min"] / 60}


# ---------------------------------------------------------------- the registered computation

def sealed() -> list[str]:
    import analysis.load as L
    if not L.RUNS.exists():
        return []
    return sorted(d.name for d in L.RUNS.iterdir() if withheld.sealed_by(d.name) == TR["id"])


def table() -> dict:
    """TR's registered computation, once after the queue: O (`armo.table`), H, D (`aiperf_arm.table`) and the joint
    label. Refuses while any TR folder is sealed (run with AFTERSHOCK_UNSEAL=TR)."""
    s = sealed()
    if s:
        raise withheld.WithheldError(f"{len(s)} TR folder(s) sealed; set AFTERSHOCK_UNSEAL={TR['id']}")
    o = armo.table()
    rows = {sd: h_row(sd) for sd in TR["seeds"]}
    h = outcome_h([r["label"] for r in rows.values()])
    d = aiperf_arm.table()
    return {"o": o, "h": {"outcome": h, "rows": rows, "drain_margin": drain_margin(),
                          "consequence": h_consequence(h["decision"])}, "d": d,
            "joint_summary": joint(o["outcome"]["decision"], h["decision"])}


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "predicted"
    if cmd == "configs":
        print("\n".join(str(p) for p in write_configs()))
    elif cmd == "plan":
        PLAN.parent.mkdir(parents=True, exist_ok=True)
        PLAN.write_text(plan_text())
        print(PLAN)
    elif cmd == "predicted":
        print(json.dumps({"predicted": predicted(), "cost": cost(), "drain_margin": drain_margin()}, indent=1,
                         default=str))
    elif cmd == "table":
        print(json.dumps(table(), indent=1, default=str))
    else:
        raise SystemExit(f"unknown command {cmd}")
