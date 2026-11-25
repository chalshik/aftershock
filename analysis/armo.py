"""TR arm O (docs/drafts/tr_draft2.md; not registered; draft 2 of the arm: docs/drafts/t1_open_arm_draft2.md):
REG-20 arm A's open-loop burst runs at λ = 2.0 on seeds 15-19, rerun under a new name (`..._t1o`) in TR's queue, which
starts after T1's queue and its [RESULT], on the same stack (kernel 6.8.0-149, driver and kernel packages on hold since
2026-11-15). The pairs with T1's U are sequential, not interleaved: the same stack, about a day later. Never writes into
runs/; nothing here reads the environment.

Judged (O alone; no T1 folder is read): per seed `k1.open_seed` on the seed's `_t1o` burst run and its `_t1o` control
(always run, in TR's one plan, so no O label is read before the [RESULT]; the label rule reads it only when O is cold), with `lost_open` after ARMO["max_attempts"]
invalid attempts; k1's labels and thresholds unchanged. `decide_o`, first match over the counted seeds (`unclear` and
`not_run` and `overload_open` leave n): too_few (n < min_n); pending_controls; lead_fails (stuck_open + lost_open + cold_control >= fail_k, REG-20's refute_k 3, chosen under the
author's 2026-11-23 pre-approval of the batch; the colleague's 2 is a readout); replicates (every counted seed `recovered_open`); mixed.

A registered readout, computed only once "[RESULT] T1 (rescue, two arms)" is in decisions.md (`u_ready`): U is T1's
same-stack readout, `t1.u_end_readout` on T1's counted U run (`t1.arm_run`): `k1.cold_backlogged` on [3480, 4080) against
U's own baseline; out of n on no valid U run, `u_fault`, baseline pf < min_baseline_pf, or a NaN input. P is not read.
`u_reading`, first match: too_few; stack_confounded (U cold in <= confounded_k); same_stack (>= same_stack_k seeds with U
cold and O `recovered_open`); weak. Beside it: REG-20's `k1.k1_outcome` and McNemar p on (U, O); each O run against its
REG-20 arm A twin; the `platform` strings of U, O and arm A runs.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from pathlib import Path

import yaml

from analysis import k1, t1, withheld

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"
KINDS = ("burst", "stage2_control")

# Every constant of the verdict, with its source. "draft 1" is docs/drafts/t1_open_arm_draft1.md.
ARMO = {
    "id": "TR",                                     # the seal id (analysis/withheld.py): TR's, one [RESULT]
    "seal": "[RESULT] TR (three turn-release arms)",  # the decisions.md heading that lifts the seal
    "seeds": t1.SEEDS,                              # 15-19: T1's U seeds = REG-19's = k1.REG20["closed_seeds"]
    "lam": k1.REG20["a_lam"],                       # 2.0: REG-20 arm A's rate
    "suffix": "t1o",                                # draft 1, Design: keeps O out of k1.open_pattern (ends at the seed)
    # draft 1, Reruns; the launcher's max_attempts, T1's [REG] ("three attempts per run"): after this many invalid
    # attempts and no valid run, the seed is `lost_open` (REG-20 used 2: one rerun, k1._open_row)
    "max_attempts": 3,
    "min_n": 3,                                     # draft 1, Code needed: too_few below 3 counted seeds
    # Chosen under the author's 2026-11-23 pre-approval of the batch (decisions.md [DECISION] (author) The turn-release batch ...):
    # REG-20's refute_k 3 (FILE analysis/k1.py:33), the registered K1 rule O replicates; the workshop colleague's 2
    # (FILE docs/paper/reviews/panel_20261122b_workshop/colleague.md:122) is reported beside it
    # (`decision_at_colleague_k`), never judged. Set before any TR run (rule A5).
    "fail_k": k1.REG20["refute_k"],
    "colleague_k": 2,
    # U reading (after T1's [RESULT]): the colleague's "at <= 2 of 5, the contrast is confounded with the stack"
    # (colleague.md:121) and "at >= 4 of 5" (colleague.md:121), applied to seeds with U cold and O recovered
    "confounded_k": 2,
    "same_stack_k": 4,
    # Prediction (draft 1 §Prediction; BELIEF inputs, outcome DERIVED by `predicted`)
    "prediction": {
        "p_o_recovered_all": 0.85,    # O recovered in 5 of 5 (colleague.md:122); per seed its fifth root
        "p_u_cold": 0.95,             # U cold-backlogged per seed (full packet, FILE …full/colleague.md:103; the
                                      # same 0.95 as T1's registered U readout prediction, 0.95^5 = 0.774)
        # TR: a cold O run's control cold too (overload_open). BELIEF: REG-19's five closed controls at λ = 2.0 all
        # recovered (docs/model.md:112); no open-loop control at 2.0 has run
        "p_control_cold_given_cold": 0.05,
        # TR: out of n (unclear: a NaN window; not_run: < 3 invalid attempts and no valid run). BELIEF: no REG-20 or T1
        # run so far read unclear; 0.02 per seed allows for a scrape gap or a queue stop
        "p_out_o": 0.02,
        "p_out_u": 0.04,              # TR: T1's registered bad_baseline 0.04 for U (decisions.md [REG] T1, Prediction)
    },
}
OUTCOMES = ("too_few", "pending_controls", "lead_fails", "replicates", "mixed")   # judged, O alone
U_OUTCOMES = ("too_few", "stack_confounded", "same_stack", "weak")                                 # the U readout
O_LABELS = ("recovered_open", "partial_open", "stuck_open", "overload_open", "pending_control", "cold_control",
            "unclear", "lost_open", "not_run")      # k1.open_seed's labels
# out of n: k1.k1_outcome's two, and (TR rule F r1 M6) an overload seed, its control cold too: counted, reported
O_OUT = ("unclear", "not_run", "overload_open")
O_FAIL = ("stuck_open", "lost_open", "cold_control")   # REG-20's k_stuck (k1.k1_outcome): lead_fails counts these


# ---------------------------------------------------------------- configs

def config_name(kind: str, seed: int) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind {kind!r} is not one of {KINDS}")
    if seed not in ARMO["seeds"]:
        raise ValueError(f"seed {seed} is not one of arm O's")
    return f"{parent_name(kind, seed)}_{ARMO['suffix']}"


def parent_name(kind: str, seed: int) -> str:
    """REG-20 arm A's config of the seed: `k1.open_pattern` without its `*_` prefix."""
    return k1.open_pattern(ARMO["lam"], seed, control=kind == "stage2_control")[2:]


def run_pattern(kind: str, seed: int) -> str:
    return f"*_{config_name(kind, seed)}"


def config_text(kind: str, seed: int) -> str:
    """REG-20 arm A's config of the seed (burst or control), extended and renamed; nothing else changes."""
    what = "control" if kind == "stage2_control" else "burst run"
    cond = ("# Always run (TR): it decides the seed only if its burst partner reads cold-backlogged (analysis/k1.py).\n"
            if kind == "stage2_control" else "")
    return (f"# TR arm O (docs/drafts/tr_draft2.md; not registered), seed {seed}: REG-20 arm A's open-loop\n"
            f"# {what}, renamed `_{ARMO['suffix']}`; nothing else changes. Written by `python -m analysis.armo configs`.\n"
            + cond +
            "# Runs only after the author approves TR's [REG].\n"
            f"extends: {parent_name(kind, seed)}.yaml\n"
            f"name: {config_name(kind, seed)}\n")


def write_configs(configs_dir: Path | None = None) -> list[Path]:
    """The ten new files (burst and control per seed); refuses if any exists (a used config is never edited)."""
    d = configs_dir or CONFIGS
    items = [(k, sd) for sd in ARMO["seeds"] for k in KINDS]
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
    """A counted O run's config must equal `resolved(kind, seed)` (run id and source path aside), as `t1.arm_run`."""
    from analysis.d1 import _diff_keys
    if run is None:
        return
    got = {k: v for k, v in run["cfg"].items() if k not in ("run_id", "source_config")}
    diff = _diff_keys(got, resolved(kind, seed))
    if diff:
        raise ValueError(f"{run['dir'].name}: config differs from arm O's {kind} at {diff}")


# ---------------------------------------------------------------- per seed

def o_row(seed: int) -> dict:
    """The seed's O row: `k1.open_seed` on the one valid `_t1o` burst run (`k1._runs`: a second valid one raises) and
    its control; `lost_open` when there is no valid burst run after max_attempts invalid ones."""
    ob, n_bad = k1._runs(run_pattern("burst", seed))
    oc, n_bad_c = k1._runs(run_pattern("stage2_control", seed))
    check_config(ob, "burst", seed)
    check_config(oc, "stage2_control", seed)
    row = {"seed": seed, "invalid_attempts": n_bad, "control_invalid_attempts": n_bad_c,
           **k1.open_seed(ob, oc, lost=ob is None and n_bad >= ARMO["max_attempts"])}
    if row["label"] == "pending_control" and oc is None and n_bad_c >= ARMO["max_attempts"]:
        row["label"], row["control_lost"] = "cold_control", True      # TR rule F r1 B1: never left pending
    return row


def u_status(seed: int, u_end: dict) -> dict:
    """U's reading at the seed from `t1.u_end_readout`'s output: cold True/False, or None with the reason it is out."""
    if seed in u_end["seeds"]:
        return {"cold": bool(u_end["seeds"][seed]["cold_backlogged"]), "out": None}
    if seed in u_end["none"]:
        return {"cold": None, "out": "u_nan"}
    return {"cold": None, "out": u_end["excluded"].get(seed, "no U row")}


def pair(seed: int, o: dict, u: dict) -> dict:
    """One seed's (O label, U cold) and whether it counts."""
    lab = o["label"]
    if lab not in O_LABELS:
        raise ValueError(f"not an O label: {lab!r}")
    out = [x for x in (f"o_{lab}" if lab in O_OUT else None, u["out"]) if x]
    return {"seed": seed, "o": lab, "u_cold": u["cold"], "counted": not out, "out": out}


# ---------------------------------------------------------------- across seeds: the judged verdict, O alone

def decide_o(labels: list[str], fail_k: int | None = None) -> str:
    """O's verdict, first match over the counted seeds (`O_OUT` leave n). No T1 run is read. `fail_k` defaults to
    ARMO["fail_k"] (REG-20's refute_k 3; the colleague's 2 is a readout)."""
    bad = [x for x in labels if x not in O_LABELS]
    if bad:
        raise ValueError(f"not O labels: {bad}")
    fk = ARMO["fail_k"] if fail_k is None else fail_k
    c = [x for x in labels if x not in O_OUT]
    if len(c) < ARMO["min_n"]:
        return "too_few"
    if "pending_control" in c:
        return "pending_controls"
    if sum(x in O_FAIL for x in c) >= fk:
        return "lead_fails"
    if all(x == "recovered_open" for x in c):
        return "replicates"
    return "mixed"


def outcome_o(labels: list[str]) -> dict:
    """O's verdict, n, and k of n with its one-sided 95 % Clopper-Pearson upper bound per counted label and for
    O_FAIL (rule H); beside it the verdict at the colleague's 2 (a readout). O's runs are not pooled with REG-20's."""
    from analysis.stats import cp_upper
    c = [x for x in labels if x not in O_OUT]
    n, k = len(c), Counter(c)
    cells = {x: {"k": k[x], "n": n, "upper": cp_upper(k[x], n) if n else None} for x in O_LABELS if x not in O_OUT}
    kf = sum(k[x] for x in O_FAIL)
    cells["fail"] = {"k": kf, "n": n, "upper": cp_upper(kf, n) if n else None}
    return {"decision": decide_o(labels), "n": n, "counts": dict(Counter(labels)), "cells": cells,
            "k_overload": Counter(labels)["overload_open"],
            "decision_at_colleague_k": decide_o(labels, ARMO["colleague_k"])}


# ---------------------------------------------------------------- the U readout (after T1's [RESULT])

def counts(pairs: list[dict]) -> dict:
    c = [p for p in pairs if p["counted"]]
    return {"n": len(c),
            "k_o_cold": sum(p["o"] in k1.OPEN_COLD for p in c),
            "k_o_recovered": sum(p["o"] == "recovered_open" for p in c),
            "k_u_cold": sum(p["u_cold"] is True for p in c),
            "k_same_stack": sum(p["u_cold"] is True and p["o"] == "recovered_open" for p in c)}


def u_reading(pairs: list[dict]) -> str:
    """The U readout's label, first match (draft 1's U rules without its O rule, which is now `decide_o`)."""
    k = counts(pairs)
    if k["n"] < ARMO["min_n"]:
        return "too_few"
    if k["k_u_cold"] <= ARMO["confounded_k"]:
        return "stack_confounded"
    if k["k_same_stack"] >= ARMO["same_stack_k"]:
        return "same_stack"
    return "weak"


def outcome_u(pairs: list[dict]) -> dict:
    from analysis.stats import cp_upper
    k = counts(pairs)
    n = k["n"]
    cells = {x: {"k": k[x], "n": n, "upper": cp_upper(k[x], n) if n else None}
             for x in ("k_o_cold", "k_o_recovered", "k_u_cold", "k_same_stack")}
    return {"label": u_reading(pairs), "n": n, "cells": cells,
            "out_of_n": {p["seed"]: p["out"] for p in pairs if not p["counted"]}}


def reg20_style(pairs: list[dict], o_rows: dict) -> dict:
    """Beside the U readout, never deciding: REG-20's registered labels and McNemar p on (U, O) by seed
    (`k1.k1_outcome`; U's None leaves the seed out, as there)."""
    return k1.k1_outcome([p["u_cold"] for p in pairs], [o_rows[p["seed"]] for p in pairs])


def u_ready(decisions: Path | None = None) -> bool:
    """The U readout is computed only once T1's [RESULT] heading is in decisions.md (TR: O's verdict must not wait for
    or depend on T1's runs, and the readout must not read them while they are sealed)."""
    return withheld.lifted(t1.T1["seal"], decisions)


# ---------------------------------------------------------------- prediction

def seed_dist(p: dict | None = None, n_seeds: int | None = None) -> dict:
    """One O seed's label distribution (BELIEF inputs): recovered with p_o_recovered_all ** (1 / n), else cold, split
    into stuck_open and overload_open by p_control_cold_given_cold; `unclear` (out of n) with p_out_o."""
    p = {**ARMO["prediction"], **(p or {})}
    n = n_seeds or len(ARMO["seeds"])
    rec = p["p_o_recovered_all"] ** (1 / n)
    keep, cc = 1 - p["p_out_o"], p["p_control_cold_given_cold"]
    out = {"recovered_open": keep * rec, "stuck_open": keep * (1 - rec) * (1 - cc),
           "overload_open": keep * (1 - rec) * cc, "unclear": p["p_out_o"]}
    return {x: v for x, v in out.items() if v > 0}


def predicted(p: dict | None = None, n_seeds: int | None = None) -> dict:
    """Exact over independent seeds: O's verdict (`decide_o`, at fail_k 3 and at colleague_k 2) from `seed_dist`; and the
    U readout (`u_reading`) and P(REG-20's McNemar p <= alpha), U cold per seed p_u_cold, out of n p_out_u."""
    p = {**ARMO["prediction"], **(p or {})}
    n = n_seeds or len(ARMO["seeds"])
    o = seed_dist(p, n)
    d_fail, d_refute = dict.fromkeys(OUTCOMES, 0.0), dict.fromkeys(OUTCOMES, 0.0)
    for combo in itertools.product(o, repeat=n):
        pr = math.prod(o[c] for c in combo)
        d_fail[decide_o(list(combo))] += pr
        d_refute[decide_o(list(combo), ARMO["colleague_k"])] += pr
    u = {True: (1 - p["p_out_u"]) * p["p_u_cold"], False: (1 - p["p_out_u"]) * (1 - p["p_u_cold"]),
         None: p["p_out_u"]}
    states = [((ol, uc), po * pu) for (ol, po), (uc, pu) in itertools.product(o.items(), u.items()) if po * pu > 0]
    d_u, p_mcnemar, d_c = dict.fromkeys(U_OUTCOMES, 0.0), 0.0, {}
    for combo in itertools.product(states, repeat=n):
        pr = math.prod(w for _, w in combo)
        ps = [pair(i, {"label": ol}, {"cold": uc, "out": None if uc is not None else "u_out"})
              for i, ((ol, uc), _) in enumerate(combo)]
        ul = u_reading(ps)
        d_u[ul] += pr
        cq = consequence(decide_o([ol for (ol, _), _ in combo]), ul)    # O's own consequence (B2-B3)
        d_c[cq] = d_c.get(cq, 0.0) + pr
        c = [(x["u_cold"], x["o"] in k1.OPEN_COLD) for x in ps if x["counted"]]
        if c and k1._p("mcnemar", sum(a for a, _ in c), len(c), c) <= k1.REG20["alpha"]:
            p_mcnemar += pr
    return {"seed": o, "p_o_recovered_per_seed": p["p_o_recovered_all"] ** (1 / n), "outcome": d_fail,
            "outcome_at_colleague_k": d_refute, "u_reading": d_u, "p_mcnemar_le_alpha": p_mcnemar,
            "consequence": d_c, "inputs": p}


# ---------------------------------------------------------------- readouts (never deciding)

def twin(seed: int, o: dict) -> dict:
    """O against its REG-20 arm A run of the same seed: reuse and waiting at the end window, the same computation
    (`k1.cold_backlogged` on its own windows), the same-stack repeat of arm A."""
    a, _ = k1._runs(k1.open_pattern(ARMO["lam"], seed))
    if a is None or o.get("reuse") is None:
        return {"arm_a": a["dir"].name if a else None, "o": o.get("run")}
    r = k1.cold_backlogged(a)
    return {"arm_a": a["dir"].name, "o": o["run"],
            "reuse_end": {"arm_a": r["reuse"]["end"], "o": o["reuse"]["end"]},
            "q_end": {"arm_a": r["end"]["q"], "o": o["end"]["q"]},
            "label": {"arm_a": k1.open_seed(a, None)["label"], "o": o["label"]}}


def platforms(u_runs: dict, o_runs: dict, arm_a: dict) -> dict:
    """The env.json `platform` of every U, O and arm A run, by seed; `one_stack` True only when all present runs share
    one string and none lacks it (draft 2: O no longer runs beside U, so "one stack" is read, not assumed)."""
    got = {name: {sd: (t1._platform(r) if r is not None else None) for sd, r in d.items() if r is not None}
           for name, d in (("u", u_runs), ("o", o_runs), ("arm_a", arm_a))}
    vals = [v for d in got.values() for v in d.values()]
    return {**got, "one_stack": bool(vals) and None not in vals and len(set(vals)) == 1}


# ---------------------------------------------------------------- the registered computation

def sealed() -> list[str]:
    import analysis.load as L
    if not L.RUNS.exists():
        return []
    return sorted(d.name for d in L.RUNS.iterdir()
                  if d.name.endswith(f"_{ARMO['suffix']}") and withheld.sealed_by(d.name) == ARMO["id"])


def o_rows() -> dict:
    return {sd: o_row(sd) for sd in ARMO["seeds"]}


def u_table(rows: dict) -> dict:
    """The U readout on O's rows, or a note while T1's [RESULT] is not in decisions.md (or a T1 folder is sealed)."""
    if not u_ready() or t1.sealed():
        return {"withheld": f"computed only after '{t1.T1['seal']}'"}
    u_runs = {sd: t1.arm_run("U", sd, None)[0] for sd in ARMO["seeds"]}
    u_end = t1.u_end_readout(u_runs)
    pairs = [pair(sd, rows[sd], u_status(sd, u_end)) for sd in ARMO["seeds"]]
    o_runs = {sd: (k1._runs(run_pattern("burst", sd))[0]) for sd in ARMO["seeds"]}
    arm_a = {sd: k1._runs(k1.open_pattern(ARMO["lam"], sd))[0] for sd in ARMO["seeds"]}
    return {"reading": outcome_u(pairs), "pairs": pairs, "u_end": u_end, "reg20_style": reg20_style(pairs, rows),
            "platforms": platforms(u_runs, o_runs, arm_a)}


def table() -> dict:
    """Arm O's part of TR's registered computation (`tr.table`), once after the queue and any controls. Refuses while
    an O folder is sealed. O's verdict reads no T1 folder; the U readout is added only after T1's [RESULT]."""
    s = sealed()
    if s:
        raise withheld.WithheldError(f"{len(s)} arm O folder(s) sealed; set AFTERSHOCK_UNSEAL={ARMO['id']}")
    rows = o_rows()
    out = outcome_o([r["label"] for r in rows.values()])
    u = u_table(rows)
    return {"outcome": out, "o": rows, "twins": {sd: twin(sd, rows[sd]) for sd in ARMO["seeds"]}, "u": u,
            "consequence": consequence(out["decision"], u.get("reading", {}).get("label"))}


def consequence(o: str, u: str | None) -> str:
    """TR rule F r1 B2-B3: O's own consequence. Appendix F's "same stack" sentence needs O `replicates` AND the U
    readout `same_stack`; `replicates` beside another U reading answers only "the open arm still recovers on today's
    stack" (a repeat of arm A); `lead_fails` withdraws the lead; anything else changes no sentence."""
    if o == "replicates":
        return "appendix_f_same_stack" if u == "same_stack" else "arm_a_repeats_today"
    return "lead_withdrawn" if o == "lead_fails" else "no_sentence"


if __name__ == "__main__":
    import json
    import sys
    cmd = sys.argv[1:]
    if cmd == ["table"]:
        print(json.dumps(table(), indent=1, default=str))
    elif cmd == ["predicted"]:
        print(json.dumps(predicted(), indent=1, default=str))
    elif cmd == ["configs"]:
        print("\n".join(str(p) for p in write_configs()))
    else:
        sys.exit("usage: python -m analysis.armo table | predicted | configs")
