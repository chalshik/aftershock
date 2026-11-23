"""K2, stage E0 (draft, not registered): the predicted widening of the window from 8 to 16 turns per session, and the
gate that decides whether the 16-turn shock matrix (E3) is registered (docs/experiment-plan.md "Stage 2, revised", step E;
decisions.md 2026-11-12 [DECISION] "K2 staged"). Only `observed` reads run folders (E1's and E2's, under their seals);
the prediction's inputs are the session shape and schedules in the configs and two reported numbers of REG-8 (`E0["floor_8"]`, `E0["warm_best_8"]`, docs/model.md M2).

Window ratio of a setup: its closed-loop warm capacity (the tip rule, `tip_warm`, since the 2026-11-15 amendment; the
configs' ratio rule, `tables.sweep_capacity`, is reported) over its
closed-loop cold floor (`gate.cold_floor` on its closed-loop cache-off sweep). The gate is on the ratio of ratios
R = ratio_16 / ratio_8, so the 8-turn absolutes cancel and the prediction needs no E1 result.

The prediction is a per-request cost model, cost = a + b x (tokens to prefill), at the stationary turn mix:
  - cold (every turn misses): the mean prompt, system + user + (user + reply)(T - 1)/2 tokens;
  - warm (every later turn hits its previous prompt): the mean uncached prompt, (system + user + (T - 1)(user + reply))/T.
Two models, as in REG-21 (`k3.REG21["costs"]`): proportional (a = 0) and a fixed part per request. The fixed model's
(a, b) solve two equations, cost(cold mix at 8 turns) = 1 / 1.681 and cost(warm mix at 8 turns) = 1 / 4.047, so it is
REG-8's two capacities rewritten (0.180 s + 0.135 ms per token). Known blind spot, stated in the draft: neither model
has a KV pool, and at 16 turns each session's live context is about a third larger at the same session count, so the
closed-loop warm capacity may lose more than either says (BELIEF).
"""
from __future__ import annotations

import math

E0 = {
    "floor_8": 1.681,        # req/s, REG-8's open-loop cache-off floor (docs/model.md M2; RUN(REG-8, gate.reg8_zone))
    "warm_best_8": 4.047,    # req/s, REG-8's best cache-on completion rate (M2; RUN(..._pc_on_r4.5, load.summarize))
    "warm_cl_8": 2.4,        # req/s, REG-13's closed-loop warm capacity by the configs' rule (M2; decisions [RESULT])
    "warm_grid_8": (2.0, 2.4, 2.6, 2.8, 3.0),            # REG-13's rates (configs/sweep_l1pl350cl_pc_on.yaml)
    "warm_grid_16": (2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2),  # E2 (configs/sweep_l1t16pl350cl_pc_on.yaml); 2.2 added
                                                          # after the E0/E1 review (author, 2026-11-12)
    "floor_rates": (2.0, 2.5, 3.0),   # both floors over these common rates only (author, decisions.md 2026-11-14)
    "surge_window_s": 120,            # as REG-13's readout (gate.R13["surge_window_s"])
    "on_8": "configs/sweep_l1pl350cl_pc_on.yaml", "on_16": "configs/sweep_l1t16pl350cl_pc_on.yaml",
}


def planned_table(cfg_path: str) -> dict:
    """Per swept rate of a sessions sweep: the realized planned rate over [warmup_s, duration_s) (as
    `load.summarize`'s planned_rate) and the highest planned rate over consecutive surge_window_s windows (as
    `gate.surge_max`), from the config's own schedule (`harness.workload.build_schedule`, the config's seed). Reads no
    run folder: under closed-loop turns t_planned is the schedule's plan, not the release."""
    from pathlib import Path

    from harness.run import REPO, expand, load_config
    from harness.workload import build_schedule
    out = {}
    for c in expand(load_config(REPO / cfg_path if not Path(cfg_path).is_absolute() else Path(cfg_path))):
        wl = c["workload"]
        df, _ = build_schedule(wl, c["seed"], c.get("trigger"))
        a, b, w = float(wl["warmup_s"]), float(wl["duration_s"]), float(E0["surge_window_s"])
        t = df.t_planned.to_numpy(float)
        starts = [x for x in _grid(a, b - w, w)]
        out[float(wl["rate"])] = {"planned": float(((t >= a) & (t < b)).sum() / (b - a)),
                                  "surge": max(float(((t >= x) & (t < x + w)).sum() / w) for x in starts)}
    return out


def _grid(a: float, last: float, step: float) -> list[float]:
    n = int(math.floor((last - a) / step + 1e-9)) + 1
    return [a + k * step for k in range(max(n, 0))]


def shape(cfg_sessions: dict) -> dict:
    return {k: int(cfg_sessions[k]) for k in ("system_tokens", "user_tokens", "reply_tokens", "turns")}


def cold_tokens(s: dict, turns: int | None = None) -> float:
    T = turns or s["turns"]
    return s["system_tokens"] + s["user_tokens"] + (s["user_tokens"] + s["reply_tokens"]) * (T - 1) / 2


def warm_tokens(s: dict, turns: int | None = None) -> float:
    T = turns or s["turns"]
    return (s["system_tokens"] + s["user_tokens"] + (T - 1) * (s["user_tokens"] + s["reply_tokens"])) / T


def token_gap(s: dict, turns: int | None = None) -> float:
    """Cold over warm prefill tokens per request at the stationary mix (decisions.md 2026-11-12, design choices,
    item 4: 3.48, 6.19, 10.9 at 4, 8 and 16 turns of L1)."""
    return cold_tokens(s, turns) / warm_tokens(s, turns)


def calibrate_fixed(s: dict, floor: float | None = None, warm_best: float | None = None) -> tuple[float, float]:
    """(a, b) with a + b x cold_tokens(8) = 1 / floor and a + b x warm_tokens(8) = 1 / warm_best."""
    floor, warm_best = floor or E0["floor_8"], warm_best or E0["warm_best_8"]
    c, w = cold_tokens(s, 8), warm_tokens(s, 8)
    b = (1 / floor - 1 / warm_best) / (c - w)
    return 1 / floor - b * c, b


def predict(s: dict, turns: int = 16) -> dict:
    """For each cost model: warm_T / warm_8, floor_T / floor_8, and R = (warm ratio) / (floor ratio)."""
    out = {}
    for name, (a, b) in {"prop": (0.0, 1.0), "fixed": calibrate_fixed(s)}.items():
        cost = lambda L: a + b * L
        warm = cost(warm_tokens(s, 8)) / cost(warm_tokens(s, turns))
        floor = cost(cold_tokens(s, 8)) / cost(cold_tokens(s, turns))
        out[name] = {"a": a, "b": b, "warm_ratio": warm, "floor_ratio": floor, "R": warm / floor,
                     "warm_cl_T": E0["warm_cl_8"] * warm, "floor_T_open": E0["floor_8"] * floor}
    return out


def gate_factor(c8: float | None = None, grid8: tuple | None = None) -> float:
    """g: the measured R must reach it before the 16-turn matrix runs. A capacity read by the configs' rule is the
    highest passing grid rate, so the true 8-turn capacity may lie up to one grid step above the measured one; a
    measured R >= 1 + step / c8 then still implies a true R >= 1. The 16-turn capacity is rounded down the same way,
    which can only lower the measured R (conservative). Floors are means of saturated runs, not rounded. Not covered:
    a lucky pass at one grid rate (REG-13's 2.4 passed at 0.9503 against 0.95) and noise in the floors."""
    c8, grid8 = c8 or E0["warm_cl_8"], grid8 or E0["warm_grid_8"]
    above = [r for r in grid8 if r > c8 + 1e-9]
    if not above:
        return math.inf   # capacity at the grid's top: its true value is unbounded above
    return 1 + (min(above) - c8) / c8


def decide(warm8: float, f8: float, warm16: float, f16: float, planned8: dict, planned16: dict,
           low16_measured: bool = True, grid8: tuple | None = None, grid16: tuple | None = None,
           hole16: bool = False, released16: float | None = None,
           released_top8: float | None = None) -> dict:
    """The E0 gate. `planned8` and `planned16` map each grid rate to its realized planned rate (`planned_table`).
    Labels:
      'widens'          R >= g and the warm side gained in realized planned rate: warm16's planned rate is above the
                        planned rate of the 8-turn grid rate next above warm8 (author, decisions.md 2026-11-14, rule F
                        e0e1 r2 item 1: seed 0's schedules differ between the sweeps); the 16-turn matrix may be
                        registered;
      'floor_only'      R >= g but that warm clause fails: the window widened only because the floor fell; no matrix;
      'does_not_widen'  R <= 1 even with warm16 rounded up to the next 16-turn grid rate (R x next16 / warm16 <= 1;
                        author, 2026-11-14, item 3): K2's first half refuted for L1 at sweep cost; no matrix;
      'unresolved'      otherwise below g; no matrix, decided case by case and reported;
      'missing'         an input is undefined for want of data (no valid run where one is needed, or a floor the
                        rules leave undefined); no matrix, reported; decided case by case.
    `hole16`: the 16-turn capacity chain ended at a swept rate with no valid run, not at a failing one (rule F e0e1 r2,
    T4): 'missing'.
    `released16`, `released_top8` (the tip-rule amendment, rule F e0tip r1, T1): the released rate (`released_rate`)
    of warm16's own run, and of the 8-turn run at the first swept rate above warm8 (the run where the 8-turn warm state
    tipped). When both are given, the warm clause also needs released16 > released_top8, so a 16-turn run whose release
    lag kept the load it actually offered at or below the load the 8-turn system tipped at cannot read 'widens'. Both
    sides are released rates: the planned upper end is not used here, since the 8-turn sweep's own runs released well
    below their planned rates (the first version compared released16 with the planned end; draft 2 corrects it).
    A 16-turn capacity below its grid (the lowest rate measured and failed, `low16_measured`) bounds R above by its
    value at that rate (`R_upper`): 'does_not_widen' if that bound is <= 1, else 'unresolved'."""
    grid16 = grid16 or E0["warm_grid_16"]
    g = gate_factor(warm8, grid8)
    out = {"g": g, "warm_ratio": warm16 / warm8 if warm8 else math.nan,
           "floor_ratio": f16 / f8 if f8 else math.nan}
    if hole16 or any(math.isnan(x) for x in (warm8, f8, f16)):
        return {**out, "R": math.nan, "label": "missing"}
    if math.isnan(warm16):
        if not low16_measured:
            return {**out, "R": math.nan, "label": "missing"}
        R_up = measured_R(warm8, f8, min(grid16), f16)
        return {**out, "R": math.nan, "R_upper": R_up, "label": "does_not_widen" if R_up <= 1 else "unresolved"}
    R = measured_R(warm8, f8, warm16, f16)
    grid8, grid16 = grid8 or E0["warm_grid_8"], grid16
    next8 = min([r for r in grid8 if r > warm8 + 1e-9], default=None)
    top8 = planned8[next8] if next8 is not None else math.inf      # the 8-turn capacity's planned upper end
    next16 = min([r for r in grid16 if r > warm16 + 1e-9], default=None)
    R_up = R * next16 / warm16 if next16 is not None else math.inf  # warm16 rounded up one grid step
    out.update(planned_warm16=planned16[warm16], planned_top8=top8, R_round_up=R_up, released_warm16=released16,
               released_top8=released_top8)
    if (released16 is None) != (released_top8 is None):
        raise ValueError("released16 and released_top8 are given together or not at all")
    if R >= g:
        warm_gain = planned16[warm16] > top8 and (released16 is None or released16 > released_top8)
        lab = "widens" if warm_gain else "floor_only"
    else:
        lab = "does_not_widen" if R_up <= 1 else "unresolved"
    return {**out, "R": R, "label": lab}


def gate(R: float, g: float | None = None) -> str:
    """R alone against g (no warm-side condition): kept for the prediction table; the registered rule is `decide`."""
    g = gate_factor() if g is None else g
    if math.isnan(R):
        return "nan"
    return "widens" if R >= g else ("does_not_widen" if R <= 1 else "unresolved")


def measured_R(warm8: float, floor8: float, warm16: float, floor16: float) -> float:
    return (warm16 / floor16) / (warm8 / floor8)


# The tip rule (author, 2026-11-15; decisions.md 2026-11-15 [DECISION] (author), E0's warm capacities by the tip rule).
# With closed-loop turns a turn is released at the previous turn's end plus its gap, so release lag builds up over a
# session's turns. Requests planned near the end of the window are then released, or end, after it, and the keep-up
# ratio (throughput >= 0.95 x planned rate) falls with an empty queue: REG-13's healthy 8-turn runs read 0.974 (2.0) and
# 0.9503 (2.4). The lag scales with the turns per session, so the ratio rule leans against the 16-turn sweep. The tip
# rule judges both warm capacities, 8 and 16 turns, by whether the warm state held instead:
#   a run passes iff it is valid, its queue slope is <= the max_q_slope its config registered (unchanged), and it has
#   no KV onset anywhere in the run, drain included (`gate.kv_onset`: the first trailing 60 s mean KV usage >= 0.9,
#   registered SURGE settings, undefined); rule F e0tip r1, T3, option (i): an onset just after the end of arrivals
#   is a tip too;
#   the capacity is the monotone chain over the swept rates, as `load.capacity(monotone=True, swept_rates=...)`: the
#   highest rate that passes along with every lower swept rate, a swept rate with no valid run ending the chain.
# The ratio rule's capacities are still computed and reported (`observed()["warm_ratio_rule"]`); they decide nothing.

def tip_pass(row) -> bool:
    """One sweep run under the tip rule. `row` holds valid, q_slope_per_s, max_q_slope and kv_onset. A NaN slope fails
    (as in `load.capacity`); a NaN onset means the 60 s mean KV usage never reached 0.9 in the whole run (`tip_rows`
    refuses a run whose KV gauge is missing, so NaN never stands for "not measured")."""
    if not bool(row["valid"]):
        return False
    slope_ok = bool(row["q_slope_per_s"] <= row["max_q_slope"])     # False for a NaN slope
    return slope_ok and math.isnan(float(row["kv_onset"]))


def tip_capacity(rows, swept_rates) -> float:
    """The tip rule's capacity: the highest swept rate that passes along with every lower swept rate. `rows` are the
    sweep's valid runs (one per rate); a swept rate with no valid run ends the chain. Two valid runs at one rate, or a
    rate that was not swept, raise: the registration says which run counts (as `load.capacity`)."""
    rates = [round(float(r), 6) for r in rows["rate"]]
    if len(set(rates)) != len(rates):
        raise ValueError("two valid runs at one rate; the registration says which counts")
    swept = sorted(round(float(r), 6) for r in swept_rates)
    unknown = set(rates) - set(swept)
    if unknown:
        raise ValueError(f"rates that were not swept: {sorted(unknown)}")
    by_rate = {r: tip_pass(row) for r, (_, row) in zip(rates, rows.iterrows())}
    mu = math.nan
    for r in swept:
        if not by_rate.get(r, False):
            break
        mu = float(r)
    return mu


def swept_of(parent: str, sw) -> list[float]:
    """The swept rates of a parent: the union of `sweep.rates` saved in every run of it, valid or not
    (`tables.sweep_capacity`'s rule)."""
    import json
    mine = sw[sw.parent == parent]
    swept = sorted({x for v in mine.swept_rates for x in json.loads(v)})
    if not swept:
        raise ValueError(f"{parent}: no run saved its sweep.rates")
    return swept


def released_rate(run: dict) -> float:
    """Requests released in [warmup_s, duration_s) per second of that span (rule F e0tip r1, T1): the load the closed
    loop actually offered in the measured window. A turn with no release time (turn 0, or open loop) is released at its
    planned time."""
    wl, q = run["cfg"]["workload"], run["requests"]
    a, b = float(wl["warmup_s"]), float(wl["duration_s"])
    rel = q.t_released.fillna(q.t_planned) if "t_released" in q else q.t_planned
    return float(((rel >= a) & (rel < b)).sum() / (b - a))


def kv_readout(run: dict) -> dict:
    """The KV gauge (rule F e0tip r1, T4 and O4): scrapes in [warmup_s, duration_s), and over the whole run, drain
    included (the span the tip rule's onset is read over, T3 (i)), the peak raw value and the seconds with the raw value
    >= SURGE["kv_onset"] (scrape intervals summed). Raises if the gauge is absent or has fewer than
    SURGE["kv_min_periods"] scrapes in [warmup_s, duration_s), so a missing gauge can never read as "never tipped"."""
    import numpy as np
    from analysis.gate import SURGE
    m, wl = run["metrics"], run["cfg"]["workload"]
    name = "vllm:kv_cache_usage_perc"
    if not len(m) or name not in m:
        raise ValueError(f"{run['dir'].name}: no KV gauge; the tip rule cannot read it")
    t, k = m.t.to_numpy(float), m[name].to_numpy(float)
    inw = (t >= float(wl["warmup_s"])) & (t < float(wl["duration_s"])) & np.isfinite(k)
    if inw.sum() < SURGE["kv_min_periods"]:
        raise ValueError(f"{run['dir'].name}: {int(inw.sum())} KV scrapes in the window, fewer than "
                         f"{SURGE['kv_min_periods']}")
    ok = np.isfinite(k)
    tt, kk = t[ok], k[ok]
    dt = np.diff(tt, append=tt[-1])
    return {"kv_scrapes": int(inw.sum()), "kv_peak": float(kk.max()),
            "kv_s_above": float(dt[kk >= SURGE["kv_onset"]].sum())}


def tip_rows(parent: str, sw):
    """The parent's valid runs with the tip rule's inputs: `load.summarize`'s rate, valid and q_slope_per_s (from
    `sw`, a `tables.sweep_table`), the config's registered max_q_slope, `gate.kv_onset` over the whole run, and the
    readouts: the keep-up ratio, `released_rate`, `kv_readout`, and the 120 s surge peaks (`gate.surge_run`'s
    peak_before and peak_tip; rule F e0tip r1, T2)."""
    import json
    import pandas as pd
    from analysis import load as L
    from analysis.gate import kv_onset, surge_run
    mine = sw[(sw.parent == parent) & sw.valid.astype(bool)]
    out = []
    for _, s in mine.iterrows():
        run = L.load_run(L.RUNS / s["run"])
        sr = surge_run(run)
        out.append({"run": s["run"], "rate": float(s["rate"]), "valid": bool(s["valid"]),
                    "q_slope_per_s": float(s["q_slope_per_s"]),
                    "max_q_slope": float(json.loads(s["capacity_rule"])["max_q_slope"]),
                    "kv_onset": kv_onset(run), "end_s": float(run["cfg"]["workload"]["duration_s"]),
                    "keep_up_ratio": float(s["throughput"] / s["planned_rate"]),
                    "planned_rate": float(s["planned_rate"]), "released_rate": released_rate(run),
                    **kv_readout(run), "surge_peak_before": sr["peak_before"], "surge_peak_tip": sr["peak_tip"]})
    df = pd.DataFrame(out)
    if not len(df):
        return df
    if df.max_q_slope.nunique() != 1:
        raise ValueError(f"{parent}: runs registered different max_q_slope: {sorted(df.max_q_slope.unique())}")
    df["tip_pass"] = [tip_pass(r) for _, r in df.iterrows()]
    return df.sort_values("rate").reset_index(drop=True)


def tip_warm(parent: str, sw) -> dict:
    """A cache-on sweep's warm capacity by the tip rule, with its rows and the ratio rule's capacity (reported)."""
    from analysis.tables import sweep_capacity
    rows = tip_rows(parent, sw)
    try:                                  # reported only: it must never stop the gate (rule F e0tip r1, O1)
        ratio = sweep_capacity(parent, sw)
    except ValueError as e:
        ratio = f"error: {e}"
    return {"tip": tip_capacity(rows, swept_of(parent, sw)) if len(rows) else math.nan,
            "ratio_rule": ratio, "rows": rows, "swept": swept_of(parent, sw)}


E0_SWEEPS = {"off_8": "configs/sweep_l1pl350cl_pc_off.yaml", "off_16": "configs/sweep_l1t16pl350cl_pc_off.yaml"}


def closed_loop_floors(s: dict, turns: int = 16, onset_8: float = 2.0, onset_16: float = 1.5,
                       rates: tuple | None = None) -> dict:
    """Model F's closed-loop floors on the replay of E1's and E2's cache-off sweeps (analysis/e1.py `replay_rows`,
    seed 0, at REG-8's floor): the mean throughput over the rates from each sweep's onset of saturation up, as
    `gate.cold_floor` averages them. Under model F a closed-loop sweep serves a mix tilted to short turns, so its floor
    lies above the open-loop one by an amount that grows with the overload, and R moves. The onsets are inputs, not
    read from the replay: the replay has no KV pool, and its stand-in for "full" (throughput < 0.95 x planned) already
    fires at 1.5 req/s at 8 turns, below REG-8's measured onset of 2.0 (`gate.cold_floor` on KV). Defaults: 2.0 at 8
    turns (REG-8) and 1.5 at 16 turns (the first grid rate above the predicted 16-turn floor of 1.26-1.36). When the
    real onsets are known, the comparison is redone at them (`by_onset`). With `rates`, each sweep's replay keeps only
    those rates first (E0's common floor rates, author 2026-11-14)."""
    from analysis import e1
    base, out = predict(s, turns), {}
    for name, cost in e1.E1["costs"].items():
        rows = {k: e1.replay_rows(p, E0["floor_8"], cost, 0) for k, p in E0_SWEEPS.items()}
        if rates is not None:
            rows = {k: r[r.rate.round(6).isin([round(x, 6) for x in rates])].reset_index(drop=True)
                    for k, r in rows.items()}
        fl = {k: {float(lo): float(r[r.rate >= lo - 1e-9].throughput.mean()) for lo in r.rate}
              for k, r in rows.items()}
        ratio = fl["off_16"][onset_16] / fl["off_8"][onset_8]
        out[name] = {"floor_cl_8": fl["off_8"][onset_8], "floor_cl_16": fl["off_16"][onset_16],
                     "floor_ratio_cl": ratio, "R_cl": base[name]["warm_ratio"] / ratio, "by_onset": fl}
    return out


E0_PARENTS = {"warm_8": "sweep_l1pl350cl_pc_on", "off_8": "sweep_l1pl350cl_pc_off",
              "warm_16": "sweep_l1t16pl350cl_pc_on", "off_16": "sweep_l1t16pl350cl_pc_off"}


def predict_measured(s: dict, turns: int = 16) -> dict:
    """The prediction on the gate's own scale, in realized planned rate (rule F e0e1 r2, T2; author 2026-11-14): the
    true 8-turn capacity lies in [planned8(2.4), planned8(2.6)) of seed 0's schedules; each model scales both ends by
    its warm ratio, and the 16-turn sweep reads the highest grid rate whose planned rate is at or below that. Floors:
    model F's closed-loop floors over E0's common rates, from 2.0. Then `decide`, at each end ("lo", "hi")."""
    base = predict(s, turns)
    fl = closed_loop_floors(s, turns, onset_8=min(E0["floor_rates"]), onset_16=min(E0["floor_rates"]),
                            rates=E0["floor_rates"])
    p8, p16 = planned_table(E0["on_8"]), planned_table(E0["on_16"])
    w8 = E0["warm_cl_8"]
    next8 = min(r for r in E0["warm_grid_8"] if r > w8 + 1e-9)
    out = {}
    for name in base:
        out[name] = {}
        for end, c8 in (("lo", p8[w8]["planned"]), ("hi", p8[next8]["planned"])):
            c16 = c8 * base[name]["warm_ratio"]
            read = max([r for r in E0["warm_grid_16"] if p16[r]["planned"] <= c16 + 1e-9], default=math.nan)
            d = decide(w8, fl[name]["floor_cl_8"], read, fl[name]["floor_cl_16"],
                       {r: v["planned"] for r, v in p8.items()}, {r: v["planned"] for r, v in p16.items()})
            out[name][end] = {"warm_cl_8_planned": c8, "warm_cl_16_planned_true": c16, "warm_cl_16_read": read, **d}
    return out


def observed() -> dict:
    """The measured gate, from the four closed-loop sweeps (valid runs only): warm capacities by the tip rule
    (`tip_warm`; author, 2026-11-15, in place of the configs' ratio rule, which is reported as `warm_ratio_rule`),
    floors by `e1.floor` (`gate.cold_floor`, REG-8's settings, with the registered tail fallback). Never call before
    E1 and E2 have run under approved registrations."""
    from analysis.gate import sweep_rows
    from analysis.tables import sweep_table
    sw = sweep_table()
    for k in ("warm_8", "warm_16", "off_8", "off_16"):     # rule F e0tip r1, T5: a forgotten unseal is an error
        if not (sw.parent == E0_PARENTS[k]).any():
            raise ValueError(f"{E0_PARENTS[k]}: no run can be read (sealed, or none run); set AFTERSHOCK_UNSEAL=E0")
    tips = {k: tip_warm(E0_PARENTS[k], sw) for k in ("warm_8", "warm_16")}
    if [round(x, 6) for x in tips["warm_16"]["swept"]] != [round(x, 6) for x in E0["warm_grid_16"]]:   # O2
        raise ValueError(f"E2-on's swept rates {tips['warm_16']['swept']} are not E0's grid {E0['warm_grid_16']}")
    warm = {k: v["tip"] for k, v in tips.items()}
    r16 = tips["warm_16"]["rows"]
    at = r16[r16.rate.round(6) == round(warm["warm_16"], 6)] if not math.isnan(warm["warm_16"]) else r16.iloc[0:0]
    released16 = float(at.released_rate.iloc[0]) if len(at) else None
    r8, w8 = tips["warm_8"]["rows"], warm["warm_8"]
    next8 = min([r for r in E0["warm_grid_8"] if math.isnan(w8) or r > w8 + 1e-9], default=None)
    at8 = r8[r8.rate.round(6) == round(next8, 6)] if next8 is not None else r8.iloc[0:0]
    released_top8 = float(at8.released_rate.iloc[0]) if len(at8) else None
    if released16 is not None and released_top8 is None:
        raise ValueError(f"no valid 8-turn run at {next8}, the rate above warm8: the released clause cannot be read")
    rows = {k: common_rows(sweep_rows(E0_PARENTS[k])) for k in ("off_8", "off_16")}
    floor = shared_floors(rows["off_8"], rows["off_16"])
    mine = sw[(sw.parent == E0_PARENTS["warm_16"]) & sw.valid.astype(bool)]
    low16_measured = bool((mine.rate == min(E0["warm_grid_16"])).any())
    w16 = warm["warm_16"]
    ended_at = min([r for r in E0["warm_grid_16"] if math.isnan(w16) or r > w16 + 1e-9], default=None)
    hole16 = ended_at is not None and not bool((mine.rate.round(6) == round(ended_at, 6)).any())
    p8 = {r: v["planned"] for r, v in planned_table(E0["on_8"]).items()}
    p16 = {r: v["planned"] for r, v in planned_table(E0["on_16"]).items()}
    d = decide(warm["warm_8"], floor["off_8"]["mu_sat_off"], warm["warm_16"], floor["off_16"]["mu_sat_off"],
               p8, p16, low16_measured, hole16=hole16, released16=released16,
               released_top8=released_top8)
    return {"warm": warm, "warm_ratio_rule": {k: v["ratio_rule"] for k, v in tips.items()},
            "tip_rows": {k: v["rows"].to_dict("records") for k, v in tips.items()},
            "floor": floor, "low16_measured": low16_measured, "hole16": hole16, **d}


def common_rows(rows):
    """A cache-off sweep's rows at E0's common floor rates only (author, decisions.md 2026-11-14)."""
    return rows[rows.rate.round(6).isin([round(x, 6) for x in E0["floor_rates"]])]


def shared_floors(rows8, rows16) -> dict:
    """Both floors over the saturated rates the two sweeps share (author, 2026-11-14: "the saturated rates the two
    sweeps share (2.0, 2.5, 3.0)"). First pass: `e1.floor` on each sweep's common-rate rows, swept rates given, so a hole
    is "holes". If both are defined, each is recomputed over the common rates at or above the later of the two lam_sat,
    so neither averages a rate the other leaves out. A floor that is undefined in either pass stays undefined."""
    from analysis import e1
    sw = E0["floor_rates"]
    first = {"off_8": e1.floor(rows8, swept=sw), "off_16": e1.floor(rows16, swept=sw)}
    if any(math.isnan(f["mu_sat_off"]) for f in first.values()):
        return {**first, "from": math.nan}
    lo = max(first["off_8"]["lam_sat"], first["off_16"]["lam_sat"])
    keep = tuple(r for r in sw if r >= lo - 1e-9)
    cut = lambda r: r[r.rate >= lo - 1e-9]
    return {"off_8": e1.floor(cut(rows8), swept=keep), "off_16": e1.floor(cut(rows16), swept=keep), "from": lo}


def flag16() -> bool:
    """The one flag E2 reads between its two sweeps: True iff the 16-turn cache-off floor over E0's common rates is
    defined (`e1.floor` status "ok" or "tail_ok"). Raises, through `e1.sweep_rows` being empty, when the sweep cannot
    be read, so a forgotten AFTERSHOCK_UNSEAL=E0 is an error, not a False."""
    from analysis import e1
    from analysis.gate import sweep_rows
    rows = sweep_rows(E0_PARENTS["off_16"])
    if not len(rows):
        raise ValueError(f"{E0_PARENTS['off_16']}: no valid run can be read (sealed, or none valid)")
    return e1.floor(common_rows(rows), swept=E0["floor_rates"])["status"] in ("ok", "tail_ok")


if __name__ == "__main__":
    import json
    import sys

    if len(sys.argv) == 2 and sys.argv[1] == "observed":
        print(json.dumps(observed(), indent=1, default=str))
    elif len(sys.argv) == 2 and sys.argv[1] == "flag":
        print(json.dumps({"floor16_defined": flag16()}))
    else:
        raise SystemExit("usage: python -m analysis.k2 observed | flag")
