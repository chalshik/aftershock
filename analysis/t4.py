"""T4 (registered: the [REG] T4 (step budget) entry in docs/decisions.md): does vLLM's step budget place the cell?

Closed-loop runs of setup L1 at 350 W, seed 0, 900 s per rate, a fresh server per run, on one stack: E1's cache-off
rates and REG-13's cache-on grid (0.2 steps), each at vLLM 0.32.0's default step budget (2048 tokens, open-questions
A10) and at `--max-num-batched-tokens 8192` (Valse and Jannesari's vLLM cell). Both arms pin the KV pool at 6041
blocks (96,656 tokens), so the budget is the only difference. Runs are launched one rate at a time, the two arms
interleaved per rate and the order alternating between rates (rule F r1, T4-4): `T4["order"]`.

Judged (`outcome`; rule F r1 T4-1, T4-3, T4-5):
- each arm's cold floor (`floor`): `e1.floor` (E1's rule), else, when that rule gives no floor because rows fall
  behind without filling the pool or too few do, the registered fallback `floor_throughput` (the mean throughput of the
  saturated tail judged on throughput alone), else `above_grid` when at most one rate falls behind, else undefined;
- `delta` = floor_8192 / floor_2048 - 1 over the rates both arms saturate (`shared_floors`): `neutral` if
  |delta| <= `band`, `floor_up` / `floor_down` otherwise (an `above_grid` 8192 floor is `floor_up`);
- each arm's warm capacity (`warm_capacity`): `warm_same`, `warm_up`, `warm_down` on the grid;
- the cell at lambda = 2.0 in each arm (`label_cell`): `cell_holds` iff 2.0 >= `cell_margin` x that arm's floor and
  2.0 < its warm capacity.
- `decision`, in order: `no_floor` (the 2048 floor, the shared comparison or the 8192 cell undefined), `no_warm`
  (either warm capacity undefined), `baseline_moved` (the 2048 cell fails today), `cell_moves` (the 8192 cell fails on
  its floor side with `floor_up`, or on its warm side with `warm_down`: the budget's move broke it), `cell_edge` (the
  8192 cell fails otherwise), `budget_neutral` (`neutral` and `warm_same`), `budget_moves`.
Reported, never judged: per run the profiled block count the override replaced, the logged pool, the compile ranges,
the scheduler overrides and the step-token histogram; each arm's lam_sat; the keep-up margins at each cache-on rate;
the 2048 arm against E1's 1.798 and REG-13's 2.4 (seen runs, before the 2026-11-15 reboot), as a yardstick of
day-to-day change. One seed, one run per rate (rule H): `delta` has no interval.
"""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

import numpy as np

from analysis import withheld

REPO = Path(__file__).resolve().parent.parent
_ARMS = ("off_2048", "off_8192", "on_2048", "on_8192")
T4 = {
    "parents": {"off_2048": "sweep_t4_l1pl350cl_pc_off_b2048", "off_8192": "sweep_t4_l1pl350cl_pc_off_b8192",
                "on_2048": "sweep_t4_l1pl350cl_pc_on_b2048", "on_8192": "sweep_t4_l1pl350cl_pc_on_b8192"},
    "rates": {"off": ("1.5", "2", "2.5", "3", "3.5", "4"), "on": ("2", "2.2", "2.4", "2.6", "2.8", "3")},
    "budget": {"2048": 2048, "8192": 8192},
    "band": 0.03,              # |delta| at or below this is `neutral`
    "cell_lambda": 2.0,        # P0's cell (REG-19)
    "cell_margin": 1.05,       # the cell needs lambda >= 1.05 x the floor (the panel's T4 criterion)
    "pool_blocks": 6041,       # 96,656 tokens / 16, the pool every default-pool run logged
    "sat_frac": 0.95,          # REG-8's: a row falls behind when throughput < 0.95 x planned (fallback floor too)
    "min_rows": 2,             # REG-8's floor_min_runs (fallback floor too)
    "seen": {"floor_e1": 1.798, "warm_reg13": 2.4},   # decisions.md 2026-11-14 [RESULT] E1; REG-13 (seen)
    "seal": "[RESULT] T4 (step budget)",
}


def _order() -> tuple:
    """(arm, rate) launch order: per rate both budgets, 8192 first at the 1st, 3rd, 5th rate of each sweep and 2048
    first at the others; cache off first, then cache on."""
    out = []
    for kind in ("off", "on"):
        for i, r in enumerate(T4["rates"][kind]):
            pair = (f"{kind}_8192", f"{kind}_2048") if i % 2 == 0 else (f"{kind}_2048", f"{kind}_8192")
            out += [(a, r) for a in pair]
    return tuple(out)


T4["order"] = _order()

PREDICTED = {  # BELIEF, registered (ledger row 40); decision DERIVED by hand from the parts, see the [REG]
    "delta": {"neutral": 0.55, "floor_up": 0.40, "floor_down": 0.05},
    "warm": {"warm_same": 0.60, "warm_up": 0.20, "warm_down": 0.20},
    "decision": {"budget_neutral": 0.30, "budget_moves": 0.52, "cell_moves": 0.09, "cell_edge": 0.03,
                 "baseline_moved": 0.03, "no_floor": 0.02, "no_warm": 0.01},
}
_OVERRIDE = re.compile(r"Overriding num_gpu_blocks=(\d+) with num_gpu_blocks_override=(\d+)")
_POOL = re.compile(r"GPU KV cache size: ([\d,]+) tokens")


def floor_throughput(rows, swept=None) -> dict:
    """Registered fallback (rule F r1 T4-5, r2 T4-R2-2) when `e1.floor` gives no floor for a reason other than a hole.
    `holes` when a swept rate has no valid row (nothing is averaged over a gap). Else lam_sat is the lowest rate from
    which every row falls behind (throughput < sat_frac x planned, KV not used) and the floor their mean, with at least
    min_rows rows. `above_grid` only when every row below the top one keeps up (at most the top row falls behind, a
    monotone tail): the server kept up past the grid's useful range. Anything else is `fallback_unclear`."""
    s = rows[np.isfinite(rows[["rate", "throughput", "planned_rate"]]).all(axis=1)].sort_values("rate")
    have = set(np.round(s.rate.to_numpy(float), 6))
    out = {"mu_sat_off": math.nan, "lam_sat": math.nan, "n": 0}
    if swept is not None:
        missing = [float(x) for x in swept if round(float(x), 6) not in have]
        if missing:
            return {**out, "status": "holes", "missing_rates": missing}
    behind = (s.throughput < T4["sat_frac"] * s.planned_rate).to_numpy()
    k = len(behind)
    while k > 0 and behind[k - 1]:
        k -= 1
    tail = s.iloc[k:]
    out.update(lam_sat=float(tail.rate.min()) if len(tail) else math.nan, n=len(tail))
    if len(tail) >= T4["min_rows"]:
        return {**out, "mu_sat_off": float(tail.throughput.mean()), "status": "throughput_only"}
    if len(s) and not behind[:-1].any():
        return {**out, "status": "above_grid"}
    return {**out, "status": "fallback_unclear"}


def floor(rows, swept) -> dict:
    """An arm's floor: `e1.floor`; on a status other than ok, tail_ok or holes, `floor_throughput`."""
    from analysis import e1
    f = e1.floor(rows, swept=swept)
    if f["status"] in ("ok", "tail_ok", "holes", "no_rows"):
        return f
    return {**floor_throughput(rows, swept), "e1_status": f["status"]}


def shared_floors(rows_a, rows_b, swept_a, swept_b) -> dict:
    """`floor` of two sweeps over the swept rates both share, then both recomputed from the later of the two lam_sat
    (`k2.shared_floors`, generalised to two parents). If either arm needs the fallback, both use `floor_throughput`.
    A floor undefined in either pass stays undefined."""
    sw = tuple(sorted(set(map(float, swept_a)) & set(map(float, swept_b))))
    cut = lambda r, lo: r[r.rate.isin(sw) & (r.rate >= lo - 1e-9)]
    first = {"a": floor(cut(rows_a, -math.inf), sw), "b": floor(cut(rows_b, -math.inf), sw)}
    if any(f["status"] == "throughput_only" for f in first.values()):   # one rule for both arms
        floor_ = lambda r, s: {**floor_throughput(r, s), "forced": True}
        first = {"a": floor_(cut(rows_a, -math.inf), sw), "b": floor_(cut(rows_b, -math.inf), sw)}
    else:
        floor_ = floor
    if any(math.isnan(f["mu_sat_off"]) for f in first.values()):
        return {**first, "from": math.nan, "first": first}
    lo = max(first["a"]["lam_sat"], first["b"]["lam_sat"])
    keep = tuple(r for r in sw if r >= lo - 1e-9)
    return {"a": floor_(cut(rows_a, lo), keep), "b": floor_(cut(rows_b, lo), keep), "from": lo, "first": first}


def label_floor(delta: float, band: float, above_grid_8192: bool = False) -> str:
    if above_grid_8192:
        return "floor_up"
    if not math.isfinite(delta):
        return "undefined"
    return "neutral" if abs(delta) <= band + 1e-12 else ("floor_up" if delta > 0 else "floor_down")


def label_warm(w2048: float, w8192: float) -> str:
    if not (math.isfinite(w2048) and math.isfinite(w8192)):
        return "warm_undefined"
    return "warm_same" if abs(w8192 - w2048) < 1e-9 else ("warm_up" if w8192 > w2048 else "warm_down")


def label_cell(floor_: dict, warm: float, p: dict = T4) -> str:
    """floor_: a `floor` result. An `above_grid` floor fails the cell (the server kept up past 2.0 with caching off)."""
    if floor_.get("status") == "above_grid":
        return "cell_moves"
    f = floor_["mu_sat_off"]
    if not (math.isfinite(f) and math.isfinite(warm)):
        return "cell_undefined"
    lam = p["cell_lambda"]
    return "cell_holds" if (lam >= p["cell_margin"] * f - 1e-12 and lam < warm) else "cell_moves"


def outcome(floors: dict, own: dict, warm: dict, p: dict = T4) -> dict:
    """floors: `shared_floors(off_2048 rows, off_8192 rows, ...)` (a = 2048, b = 8192); own: {"2048": `floor`,
    "8192": `floor`} on each arm's swept rates; warm: {"on_2048": mu, "on_8192": mu} from `warm_capacity`."""
    fa, fb = floors["a"]["mu_sat_off"], floors["b"]["mu_sat_off"]
    delta = fb / fa - 1 if math.isfinite(fa) and math.isfinite(fb) and fa > 0 else math.nan
    above = own["8192"].get("status") == "above_grid"
    out = {"floor_2048": fa, "floor_8192": fb, "from": floors.get("from"), "delta": delta,
           "floor_label": label_floor(delta, p["band"], above),
           "warm_2048": warm["on_2048"], "warm_8192": warm["on_8192"],
           "warm_label": label_warm(warm["on_2048"], warm["on_8192"]),
           "own_floor": {k: (v["mu_sat_off"], v["status"]) for k, v in own.items()},
           "cell_2048": label_cell(own["2048"], warm["on_2048"], p),
           "cell_8192": label_cell(own["8192"], warm["on_8192"], p)}
    moved = not (out["floor_label"] == "neutral" and out["warm_label"] == "warm_same")
    lam, f8 = p["cell_lambda"], own["8192"]
    fails_floor = above or (math.isfinite(f8["mu_sat_off"]) and lam < p["cell_margin"] * f8["mu_sat_off"] - 1e-12)
    fails_warm = math.isfinite(warm["on_8192"]) and lam >= warm["on_8192"]
    out["budget_broke_cell"] = bool((fails_floor and out["floor_label"] == "floor_up")
                                    or (fails_warm and out["warm_label"] == "warm_down"))
    if not math.isfinite(own["2048"]["mu_sat_off"]) or out["floor_label"] == "undefined" \
            or out["cell_8192"] == "cell_undefined" and out["warm_label"] != "warm_undefined":
        out["decision"] = "no_floor"
    elif out["warm_label"] == "warm_undefined":
        out["decision"] = "no_warm"
    elif out["cell_2048"] != "cell_holds":
        out["decision"] = "baseline_moved"
    elif out["cell_8192"] != "cell_holds":
        out["decision"] = "cell_moves" if out["budget_broke_cell"] else "cell_edge"   # rule F r2, T4-R2-1
    else:
        out["decision"] = "budget_moves" if moved else "budget_neutral"
    return out


def keeps_up(row: dict, rule: dict) -> bool:
    """The configs' keep-up rule on one cache-on row (`load.capacity`'s test)."""
    return bool(row["throughput"] >= rule["min_throughput_frac"] * row[rule["against"]]
                and row["q_slope_per_s"] <= rule["max_q_slope"])


def warm_capacity(parent: str, sw=None) -> dict:
    """The configs' monotone keep-up capacity over the earliest valid run per rate (later valid runs listed), as
    `load.capacity` with every swept rate. `mu` is 0.0 ("below the grid") only when the lowest swept rate has a valid
    run and it fails the rule (rule F r1, T4-2); NaN (with `at_least` and `hole`) when the chain ends at a rate with
    no valid run, so a hole never reads as a capacity; the top rate is reported `censored` (at least that)."""
    from analysis import e1, tables
    from analysis import load as L
    sw = tables.sweep_table() if sw is None else sw
    mine = sw[(sw.parent == parent) & sw.valid.astype(bool)]
    swept = sorted(float(x) for x in e1.swept_rates(parent))
    out = {"mu": math.nan, "censored": False, "later_valid": [], "margins": {}}
    if not len(mine):
        return out
    keep, out["later_valid"] = e1.first_valid(mine)
    rule = json.loads(keep.capacity_rule.iloc[0])
    if len(set(keep.capacity_rule)) != 1:
        raise ValueError(f"{parent}: runs registered different capacity rules")
    by = {float(r["rate"]): r for r in keep.to_dict("records")}
    out["margins"] = {r: {"tp_over_ref": float(x["throughput"] / x[rule["against"]]), "q_slope": float(x["q_slope_per_s"]),
                          "keeps_up": keeps_up(x, rule)} for r, x in by.items()}
    mu = L.capacity(keep, **{k: rule[k] for k in ("min_throughput_frac", "max_q_slope", "against")}, monotone=True,
                    swept_rates=swept)
    if math.isfinite(mu):
        nxt = [r for r in swept if r > mu + 1e-9]
        if nxt and nxt[0] not in by:          # the chain ended at a rate with no valid run: unknown above mu
            out.update(at_least=float(mu), hole=nxt[0])
            return out
        out.update(mu=float(mu), censored=not nxt)
    elif swept[0] in by and not keeps_up(by[swept[0]], rule):
        out["mu"] = 0.0
    return out


def step_tokens(run_dir: Path, a: float, b: float, budget: int) -> dict:
    """Reported (rule F r1, T4-9): from `vllm:iteration_tokens_total` over [a, b), steps, mean tokens per step and the
    share of steps above half the budget (the histogram's edges permitting). Empty when not recorded or on a reset."""
    from analysis.c2 import _buckets
    path = Path(run_dir) / "metrics.jsonl"
    first = last = None
    if not path.exists():
        return {}
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if "values" not in s or s["t"] < a:
                continue
            if s["t"] >= b:
                break
            v = s["values"]
            cur = {"b": _buckets(v, "vllm:iteration_tokens_total"),
                   "sum": sum(x for k, x in v.items() if k.startswith("vllm:iteration_tokens_total_sum")),
                   "count": sum(x for k, x in v.items() if k.startswith("vllm:iteration_tokens_total_count"))}
            if last and (cur["count"] < last["count"] or any(cur["b"].get(k, 0.0) < x for k, x in last["b"].items())):
                return {"reset": True}
            first = cur if first is None else first
            last = cur
    if not first or not last or last["count"] <= first["count"]:
        return {}
    n = last["count"] - first["count"]
    d = {k: last["b"][k] - first["b"].get(k, 0.0) for k in last["b"]}
    half = max((e for e in d if e <= budget / 2), default=None)
    return {"steps": n, "mean_tokens": (last["sum"] - first["sum"]) / n,
            "share_above_half_budget": (1 - d[half] / n) if half is not None else math.nan, "half_edge": half}


def server_readouts(run_dir: Path) -> dict:
    """Reported only: the profiled block count the override replaced, the logged pool, and what engine_config reads."""
    from analysis import engine_config
    withheld.guard(Path(run_dir).name)
    log = Path(run_dir) / "server.log"
    text = log.read_text(errors="replace") if log.exists() else ""
    ov, pool = _OVERRIDE.search(text), _POOL.search(text)
    es = engine_config.engine_settings(run_dir)
    return {"profiled_blocks": int(ov.group(1)) if ov else None, "override_blocks": int(ov.group(2)) if ov else None,
            "pool_tokens": int(pool.group(1).replace(",", "")) if pool else None,
            "compile_ranges_endpoints": es.get("compile_ranges_endpoints"),
            "non_default_args": es.get("non_default_args"), "scheduler_overrides": es.get("scheduler_overrides")}


def check_settings(arm: str, srv: dict) -> list[str]:
    """Validity of a counted run's server settings (rule F r1, T4-8): the pool override 6041 took effect, and the
    budget flag is 8192 in the 8192 arm and absent in the 2048 arm. Returns the problems (empty when fine)."""
    bad = []
    if srv.get("override_blocks") != T4["pool_blocks"] or srv.get("pool_tokens") != T4["pool_blocks"] * 16:
        bad.append(f"pool {srv.get('override_blocks')} blocks / {srv.get('pool_tokens')} tokens")
    args = srv.get("non_default_args")
    if not isinstance(args, dict):
        return bad + ["server args not parsed from server.log"]
    want = arm.endswith("8192")
    got = args.get("max_num_batched_tokens")
    try:
        got = int(got) if got is not None else None
    except (TypeError, ValueError):
        pass
    if want and got != T4["budget"]["8192"]:
        bad.append(f"max_num_batched_tokens {got!r}, not 8192")
    if not want and got is not None:
        bad.append(f"max_num_batched_tokens {got!r} set in the 2048 arm")
    return bad


def sealed() -> list[str]:
    import analysis.load as L
    runs = L.RUNS
    if not runs.exists():
        return []
    return sorted(d.name for d in runs.iterdir() if "_sweep_t4_" in d.name and withheld.sealed_by(d.name) == "T4")


def table() -> dict:
    """T4's registered computation, once after the queue: `AFTERSHOCK_UNSEAL=T4 uv run python -m analysis.t4 table`.
    Refuses while any T4 folder is sealed, and when a counted run's server settings are not the registered ones.
    Per rate the earliest valid run counts (`e1.first_valid`; later ones are listed). Labels and the decision first;
    readouts after, each guarded."""
    from analysis import e1, tables
    from analysis.gate import sweep_rows
    from analysis.load import RUNS
    s = sealed()
    if s:
        raise withheld.WithheldError(f"{len(s)} T4 folder(s) sealed; set AFTERSHOCK_UNSEAL=T4")
    par = T4["parents"]
    rows, later = {}, {}
    for k in ("off_2048", "off_8192"):
        r = sweep_rows(par[k])
        if not len(r):
            raise RuntimeError(f"{par[k]}: no valid run can be read")
        rows[k], later[k] = e1.first_valid(r)
    sw = {k: e1.swept_rates(par[k]) for k in par}
    stab = tables.sweep_table()
    wc = {k: warm_capacity(par[k], stab) for k in ("on_2048", "on_8192")}
    counted = {k: list(rows[k].run) for k in rows}
    counted.update({k: sorted(set(stab[(stab.parent == par[k]) & stab.valid.astype(bool)].run)
                              - set(wc[k]["later_valid"])) for k in wc})
    srv, problems = {}, {}
    for k, names in counted.items():
        for n in names:
            srv[n] = server_readouts(RUNS / n)
            bad = check_settings(k, srv[n])
            if bad:
                problems[n] = bad
    if problems:
        raise ValueError(f"counted runs with settings other than registered: {problems}")
    floors = shared_floors(rows["off_2048"], rows["off_8192"], sw["off_2048"], sw["off_8192"])
    own = {"2048": floor(rows["off_2048"], sw["off_2048"]), "8192": floor(rows["off_8192"], sw["off_8192"])}
    warm = {k: wc[k]["mu"] for k in wc}
    res = {"outcome": outcome(floors, own, warm), "floors": floors, "own_floor": own, "warm": wc,
           "later_valid_not_counted": {**{k: v for k, v in later.items()}, **{k: wc[k]["later_valid"] for k in wc}},
           "rows": {k: v.to_dict("records") for k, v in rows.items()}, "server": srv}
    res["against_seen"] = {"floor_2048_rel_e1": own["2048"]["mu_sat_off"] / T4["seen"]["floor_e1"] - 1,
                           "warm_2048_minus_reg13": warm["on_2048"] - T4["seen"]["warm_reg13"]}
    try:
        from analysis.load import load_run
        res["steps"] = {}
        for k, names in counted.items():
            for n in names:
                r = load_run(RUNS / n)
                wl = r["cfg"]["workload"]
                res["steps"][n] = step_tokens(RUNS / n, wl["warmup_s"], wl["duration_s"], T4["budget"][k[-4:]])
    except Exception as e:   # a readout never takes the result with it
        res["steps_error"] = f"{type(e).__name__}: {e}"[:300]
    res["decision"] = res["outcome"]["decision"]
    return res


if __name__ == "__main__":
    import sys
    from pprint import pprint
    if sys.argv[1:] == ["table"]:
        pprint(table(), width=120, sort_dicts=False)
    else:
        sys.exit("usage: python -m analysis.t4 table")
