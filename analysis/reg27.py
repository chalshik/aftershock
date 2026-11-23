"""REG-27 (docs/drafts/reg27_reg24_seeds_draft1.md and later drafts): REG-24's design, C1's conditions 5 and 6 at L1,
λ = 2.0, on four more seeds, so that "in one seed" (REG-24, seed 43, `confirms`) becomes k of n.

Everything per seed is REG-24's code (`analysis.confirm`): configs (`burst_cfg`, `control_cfg`), phases, the control's
and the burst run's outcomes, the pair outcome, readouts. Only the seed changes, and the drop is REG-24's 9000 s for
every seed. One departure from REG-24, for blindness: there is no `step` between a seed's control and its burst run;
both are always made, and `confirm.pair_outcome` still gives `control_not_warm` when the control is not warm, so the
pair's label is the one REG-24's map gives.

The seeds come from a rule committed before any candidate schedule was computed (`fd6a10e`), run once
(PROBE(docs/probe/reg27_draft/picks_fd6a10e_20261117T1953Z.txt)); `picks` re-runs it as a test.

The count (`count`, rule F r1 B1 and B3): over the four new seeds, a pair counts only beside a `control_warm` control
(whatever its burst label: with no `step`, a burst run can follow a control that is not warm, and REG-24's
`pair_outcome` would then keep some burst labels and turn others into `control_not_warm`). Among those, k = pairs
labelled `confirms`; n = k plus pairs whose label is a failure of condition 5 or 6 (`FAILS`). Not counted, reported
beside k of n with their labels: a control not warm or missing (the burst label kept as a readout), a missing burst
run, a bad baseline, an instrument gap, two builds, and `drop_inconclusive` (condition 6 inconclusive, PROTOCOL §3:
it also covers a drop too short for that seed's drain, a miss of the drain models, not of the system). Seed 43 is
pooled only in the readout `with_43` (stated assumption: the five seeds differ in the seed alone; rule H); it is biased
upward (43's hit is why REG-27 exists) and stays out of the C1 sentence.

Outcome (`outcome`), on the four new seeds:
  too_few             n < 3;
  replicates          n >= 3 and k = n;
  mixed               n >= 3 and 0 < k < n;
  does_not_replicate  n >= 3 and k = 0.
"""
from __future__ import annotations

from analysis import confirm as C
from analysis.stats import cp_upper

REG27 = {
    "seeds": (61, 68, 72, 79),     # the committed rule's picks, in seed order (fd6a10e)
    "n_picks": 4,
    "drop_s": 9000.0,              # REG-24's drop (author, 2026-11-14), fixed for every seed
    "reg24_seed": 43,
    "min_n": 3,                    # fewer informative new seeds than this: too_few
}

FAILS = ("not_persistent", "reformed", "emptied_in_watch", "stuck_not_cold", "no_return")
NOT_COUNTED = ("control_not_warm", "control_missing", "burst_missing", "bad_baseline", "inconclusive",
               "not_same_build", "drop_inconclusive")


def picks(n: int = REG27["n_picks"], drop_s: float = REG27["drop_s"], rows_fn=None) -> list[int]:
    """The committed rule: in seed order over `confirm.candidate_seeds()`, 43 left out, the first `n` seeds whose burst
    schedule passes REG-24's W, E and F at `drop_s` (`confirm.design_row(...)["ok"]`). Schedules only. Configs written
    after the rule ran (REG-27's own, which use the picked seeds) are left out of the used set by `used_seeds`'s
    pattern for REG-24's two families, so the rule gives the same picks once they exist."""
    rows_fn = rows_fn or C.design_row
    out = []
    for sd in C.candidate_seeds():
        if sd == REG27["reg24_seed"]:
            continue
        if rows_fn(sd, drop_s)["ok"]:
            out.append(sd)
            if len(out) == n:
                break
    return out


def tilt(seed: int, drop_s: float = REG27["drop_s"]) -> float:
    """Readout from the schedule: the end window's planned λp minus the return's end's (positive favours `confirms`)."""
    lam = C.design_row(seed, drop_s)["lam_p"]
    return float(lam["end"] - lam["return_end"])


def classed(pair: str, control: str | None) -> str:
    """The label a seed is counted under: `control_missing` with no control label, `control_not_warm` beside a control
    that is not `control_warm` (whatever the pair label), else the pair label. An unknown label raises (a new label must
    be classed before it is counted)."""
    lab = "control_missing" if control is None else ("control_not_warm" if control != "control_warm" else pair)
    if lab != "confirms" and lab not in FAILS and lab not in NOT_COUNTED:
        raise ValueError(f"unclassed label {lab!r}")
    return lab


def count(rows: dict[int, dict]) -> dict:
    """k, n and the bounds over seeds given as {seed: {"pair": pair label, "control": control label}}; every seed's
    classed label is returned beside them (`labels`), so the not-counted ones are reported with k of n."""
    lab = {s: classed(r["pair"], r["control"]) for s, r in rows.items()}
    k = sum(l == "confirms" for l in lab.values())
    n = k + sum(l in FAILS for l in lab.values())
    return {"k": k, "n": n, "labels": lab, "not_counted": sorted(s for s, l in lab.items() if l in NOT_COUNTED),
            "upper": cp_upper(k, n) if n else float("nan"),
            "lower": 1 - cp_upper(n - k, n) if n else float("nan")}


def outcome(rows: dict[int, dict], p: dict = REG27) -> str:
    c = count(rows)
    if c["n"] < p["min_n"]:
        return "too_few"
    if c["k"] == c["n"]:
        return "replicates"
    return "does_not_replicate" if c["k"] == 0 else "mixed"


def surge_flag(ctl: dict | None) -> bool | None:
    """K4's readout: True when the control was not warm (`control_cold_persistent` or `control_cold_transient`), the
    event REG-17-19 registered as `surge_reading_wrong` for a schedule with no 120 s planned peak >= 3.2 (every pick's
    highest is 2.725-3.000). None with no control."""
    if ctl is None:
        return None
    return ctl["outcome"] in ("control_cold_persistent", "control_cold_transient")


def _sealed_present() -> list[str]:
    """REG-27's folders under runs/ that the seal still withholds (rule F r1 B2: a sealed folder would read as missing
    and turn every seed into `control_missing`)."""
    import fnmatch
    from analysis import withheld
    from analysis.load import RUNS
    pats = [r[1] for r in withheld.SEALS if r[0] == "REG-27"]
    return sorted(p.name for p in RUNS.glob("*") if any(fnmatch.fnmatchcase(p.name, q) for q in pats)
                  and withheld.sealed_by(p.name) == "REG-27")


def _readout(fn, *a):
    """A readout never stops the count (REG-24's RE-O8 rule): an exception is reported in its place."""
    try:
        return fn(*a)
    except Exception as e:   # noqa: BLE001 — reported, never decisive
        return {"error": f"{type(e).__name__}: {e}"}


def table() -> dict:
    """REG-27's result, computed once after the queue's end entry, as `AFTERSHOCK_UNSEAL=REG-27 uv run python -m
    analysis.reg27 reg27`; it raises while any of its folders is still sealed. First the labels: per new seed
    `confirm.table(seed)` (pair and control labels, both runs' tables, baseline pair), then the count and outcome over
    the new seeds. Then the readouts, each guarded: the tilt, the K4 surge flag, whether `after_empty`'s control
    reference is a warm control, the order test's line on the burst run (the order clause's [REG] names REG-24's and
    D1's runs only), and the pooled readout with seed 43 (REG-24's table)."""
    sealed = _sealed_present()
    if sealed:
        raise RuntimeError(f"REG-27's folders are sealed ({len(sealed)}); set AFTERSHOCK_UNSEAL=REG-27")
    tables = {sd: C.table(sd) for sd in REG27["seeds"]}
    rows = {sd: {"pair": t["outcome"], "control": (t.get("control") or {}).get("outcome")} for sd, t in tables.items()}
    c, o = count(rows), outcome(rows)
    from analysis import order
    out = {}
    for sd, t in tables.items():
        b = t.get("burst")
        out[sd] = {**rows[sd], "classed": c["labels"][sd], "burst": (b or {}).get("outcome"),
                   "tilt": _readout(tilt, sd), "surge_flag": surge_flag(t.get("control")),
                   "after_empty_ref_warm": rows[sd]["control"] == "control_warm",
                   "order": _readout(lambda r: order.table([r])[0], b["run"]) if b else None, "table": t}
    r24 = _readout(C.table, REG27["reg24_seed"])
    w43 = (_readout(count, {**rows, REG27["reg24_seed"]: {"pair": r24["outcome"],
                                                          "control": (r24.get("control") or {}).get("outcome")}})
           if "error" not in r24 else r24)
    return {"rows": out, "count": c, "outcome": o, "with_43": w43, "reg24_outcome": r24.get("outcome"),
            "params": REG27}


if __name__ == "__main__":
    import sys
    fns = {"picks": picks, "reg27": table}
    print(fns[sys.argv[1]]())
