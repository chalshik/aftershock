"""The Stage 2 figures (docs/experiment-plan.md, "Outputs and done criteria"), from run folders through `analysis/`.
Never writes into runs/; images and the table go to analysis/figures/.

    uv run --extra analysis --frozen python -m analysis.figures all | boundary | matched | loopcut

1. `boundary_table`: the boundary map, one row per registered cell, "k of n seeds" with one-sided 95 % Clopper-Pearson
   bounds (rule H), taken from the registered table functions (REG-19, REG-20, D1, D1b); each row quotes the
   decisions.md [RESULT] heading its numbers belong to. A table that raises is recorded in its row, never filled in.
2. `matched_arms`: one seed, REG-20 arm A (open-loop turns, λ = 2.0) against REG-19's burst run (closed-loop turns),
   on one time axis with t = 0 at the burst's start: later-turn reuse (`k1.reuse`'s statistic via `c2.reuse_in` on
   `k1.later_turns`), KV usage and the server's waiting line.
3. `loop_cut`: one seed, REG-19's uncapped burst run, D1's capped run and D1b's cap-40 run, the same three panels; the
   waiting panel adds the load generator's cap queue (`d1.cap_queue`) for the capped runs, since under a cap the
   backlog can sit there rather than at the server (decisions.md 2026-11-17 [RESULT] D1b). There is no caching-off
   burst run: the figure says so (decisions.md 2026-11-14 [DECISION] (author), C2's caching-off control).
4. `model_vs_measurement`: withheld until K3b's [RESULT] heading exists.

Seeds are chosen by a rule fixed here, before any figure was drawn: the lowest seed with a counted valid run in every
arm. The counted run of a pattern is its earliest valid one (run ids start with their launch time), as
`d1.capped_run` and `e1.first_valid`; later valid runs are listed, never plotted. Invalid runs (checks.json) are never
read: `load.find_runs` with valid_only. Sealed runs are skipped by `load.find_runs` (analysis/withheld.py).

The per-run labels in the legends are each registration's own per-run readout (`load.verdict` and `k1.cold_backlogged`
for REG-19's runs, `k1.cold_backlogged` for arm A, `d1.label`, `d1b.label40`), not the per-seed outcomes of the
tables, which also read the controls.
"""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from analysis import confirm, d1, d1b, k1, reg30, withheld
from analysis import load as L
from analysis.c2 import reuse_in
from analysis.gate import REG19, SURGE
from analysis.k5 import K5
from analysis.stats import cp_upper
from analysis.tables import REPO, provenance

OUT = Path(__file__).resolve().parent / "figures"
DECISIONS = withheld.DECISIONS          # the log, or in the public release its heading index (withheld.py)

FIG = {
    # Seeds. Candidates in ascending order; the rule takes the lowest with a counted valid run in every arm.
    "matched_seeds": tuple(sorted(k1.REG20["closed_seeds"])),   # REG-20's paired seeds 15-19 (arm A = REG-19's seeds)
    "matched_lam": k1.REG20["a_lam"],                           # arm A's λ, 2.0 (REG-20)
    "loopcut_seeds": tuple(sorted(d1b.D1B["seeds"])),           # D1b ran seeds 15-17 only; D1 and REG-19 ran 15-19
    # Figure 3 (the return to warm): REG-27's seeds; the lowest with both runs valid is drawn. Seed 43 (REG-24) is
    # left out because it chose the design (colleague.md §6 item 3). Set before any drawing (decisions.md 2026-11-21).
    "rewarm_seeds": (61, 68, 72, 79),
    "seed_rule": "lowest seed with a counted valid run in every arm",
    "counted_run": "earliest valid run of the pattern (d1.capped_run, e1.first_valid)",
    # Smoothing. The trailing window and grid of K5's signals (analysis/k5.py K5: roll_s 30 s, grid_s 5 s), the one
    # registered smoothing of these gauges; reuse uses the same window, over later turns by first-token time.
    "roll_s": K5["roll_s"],
    "grid_s": K5["grid_s"],
    "min_turns": 1,                     # c2.reuse_in's default: a window with no later turn is NaN, not 0
    "kv_onset": SURGE["kv_onset"],      # 0.9, gate.SURGE: KV usage the stuck readout requires (k1 kv_full); a line
    "conf": REG19["conf"],              # 0.95 one-sided (rule H)
    "kv_col": "vllm:kv_cache_usage_perc",
    "waiting_col": "vllm:num_requests_waiting",
    # Presentation only (no quantity depends on these).
    "figsize": (11.0, 8.5), "dpi": 150,
    # The workshop's two-column render sets each figure at column width (about 3.3 in): a variant drawn at that size,
    # so the labels print at "column_font" points (panel 2026-11-23, B9 and the reader). Presentation only.
    "column_figsize": (3.4, 4.3), "column_font": 7,
    # One colour per role across the three figures (panel 2026-11-22, R8.4): the arm that stays cold is red in every
    # figure, the arm that stays or returns warm blue, the looser cap orange. Arms not listed take the default cycle.
    "arm_color": {"closed": "tab:red", "burst": "tab:red", "uncapped": "tab:red",
                  "open": "tab:blue", "control": "tab:blue", "d1": "tab:blue", "d1b": "tab:orange"},
    "caching_off_note": "caching off: no burst run; answered by feasibility, decisions.md 2026-11-14",
    # decisions.md [RESULT] headings each boundary row belongs to (the text after "## <date> · "; matched as a prefix).
    "results": {"REG-19": "[RESULT] REG-19 (P0 at λ = 2.0",
                "K1-A": "[RESULT] K1 arm A, from REG-20's table",
                "REG-20": "[RESULT] REG-20: K1 `supports`",
                "D1": "[RESULT] D1 (C2, session cap)",
                "D1b": "[RESULT] D1b (C2, cap above the cold batch)"},
    "k3b_result": "[RESULT] K3b (model F on K4's tipped runs)",
}


# ---------------------------------------------------------------- provenance and sources

def heading(prefix: str, decisions: Path | None = None) -> str | None:
    """The full decisions.md heading line that is this [RESULT] (`withheld.lifted`'s test), or None if absent."""
    p = decisions or DECISIONS
    if not withheld.lifted(prefix, p):
        return None
    return next(line[3:] for line in p.read_text().splitlines()
                if line.startswith("## ") and line.split(" · ", 1)[-1].startswith(prefix))


def _committed(path: Path) -> bool:
    r = subprocess.run(["git", "ls-files", "--error-unmatch", str(path.relative_to(REPO))], cwd=REPO,
                       capture_output=True, text=True)
    return r.returncode == 0 and not subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)], cwd=REPO).returncode


def footer(run_ids: list[str]) -> str:
    pv = provenance()
    code = "" if _committed(Path(__file__)) else " · analysis/figures.py not committed"
    return (f"commit {pv['commit']}{' (dirty)' if pv['dirty'] else ''}{code} · trailing {FIG['roll_s']:g} s, "
            f"grid {FIG['grid_s']:g} s · runs: " + ", ".join(run_ids))


# ---------------------------------------------------------------- bounds

def cp_lower(k: int, n: int, conf: float | None = None) -> float:
    """One-sided Clopper-Pearson lower bound, 1 - cp_upper(n - k, n) (as `d1._cp_lower`); NaN for n = 0."""
    return math.nan if not n else 1.0 - cp_upper(n - k, n, conf or FIG["conf"])


def _bounds(k: int | None, n: int | None) -> dict:
    if k is None or not n:
        return {"cp_upper": math.nan, "cp_lower": math.nan}
    return {"cp_upper": cp_upper(k, n, FIG["conf"]), "cp_lower": cp_lower(k, n)}


# ---------------------------------------------------------------- 1. the boundary map

def _row(cell: dict, src: str, k=None, n=None, what: str = "stuck", detail: str = "", error: str | None = None) -> dict:
    return {**cell, "k": k, "n": n, "counted": what, **_bounds(k, n), "detail": detail,
            "source": heading(FIG["results"][src]) or f"(heading not found: {FIG['results'][src]})", "error": error}


CELLS = {
    "reg19": {"cell": "L1 closed-loop, burst", "lam": 2.0, "turns": "closed", "cap": None, "registration": "REG-19"},
    "k1_closed": {"cell": "L1 closed-loop, burst (K1 readout)", "lam": 2.0, "turns": "closed", "cap": None,
                  "registration": "REG-20"},
    "k1_open_a": {"cell": "L1 open-loop, burst (arm A)", "lam": 2.0, "turns": "open", "cap": None,
                  "registration": "REG-20"},
    "k1_open_b": {"cell": "L1 open-loop, burst (arm B)", "lam": 2.5, "turns": "open", "cap": None,
                  "registration": "REG-20"},
    "d1": {"cell": "L1 closed-loop, burst, session cap", "lam": 2.0, "turns": "closed", "cap": "25-29 (per seed)",
           "registration": "D1"},
    "d1b": {"cell": "L1 closed-loop, burst, cap 40", "lam": 2.0, "turns": "closed", "cap": "40", "registration": "D1b"},
}


def _err(e: Exception) -> str:
    return f"{type(e).__name__}: {e}"


def boundary_rows(tables: dict) -> list[dict]:
    """The rows from the tables' return values (`tables` maps reg19 / reg20 / d1 / d1b to a dict or an Exception)."""
    rows = []
    t = tables["reg19"]
    if isinstance(t, Exception):
        rows.append(_row(CELLS["reg19"], "REG-19", error=_err(t)))
    else:
        ctl = sum(r["control_verdict"] == "recovered" for r in t["rows"])
        rows.append(_row(CELLS["reg19"], "REG-19", t["k"], t["n"], "pass (stuck, whole-window test)",
                         f"decision {t['outcome']}; k_clear {t['k_clear']}; k_c1 {t['k_c1']}; REG-18 per-span test "
                         f"{t['k_18']} of {t['n']}; controls recovered {ctl} of {len(t['rows'])}; table cp_upper "
                         f"{t['cp_upper']:.3f}"))
    t = tables["reg20"]
    if isinstance(t, Exception):
        for c in ("k1_closed", "k1_open_a", "k1_open_b"):
            rows.append(_row(CELLS[c], "REG-20", error=_err(t)))
    else:
        a = t["a"]
        rows.append(_row(CELLS["k1_closed"], "K1-A", a["k_closed"], a["n"], "cold-backlogged",
                         f"decision {a['outcome']} ({a['test']} p {a['p']:.3f})"))
        labs = [r["label"] for r in t["open_a"] if r["label"] not in ("unclear", "not_run")]
        rows.append(_row(CELLS["k1_open_a"], "K1-A", a["k_open"], a["n"], "cold (stuck, overload, lost or pending)",
                         f"recovered_open {labs.count('recovered_open')} of {len(labs)}; partial_open "
                         f"{labs.count('partial_open')}; mechanism shown in each: {a['mechanism']}"))
        lb = [r["label"] for r in t["open_b"] if r["label"] not in ("unclear", "not_run")]
        kb = sum(x in k1.OPEN_COLD for x in lb)
        mech = all((r.get("mechanism") or {}).get("shown") for r in t["open_b"] if r["label"] == "recovered_open")
        rows.append(_row(CELLS["k1_open_b"], "REG-20", kb, len(lb), "cold (stuck, overload, lost or pending)",
                         f"outcome {t['b']}; recovered_open {lb.count('recovered_open')} of {len(lb)}; "
                         f"mechanism shown in each recovered: {mech}; no closed arm at 2.5"))
    t = tables["d1"]
    if isinstance(t, Exception):
        rows.append(_row(CELLS["d1"], "D1", error=_err(t)))
    else:
        rows.append(_row(CELLS["d1"], "D1", t["k_stuck"], t["n"], "stuck",
                         f"decision {t['decision']}; cut {t['k_cut']} of {t['n']} (lower bound "
                         f"{cp_lower(t['k_cut'], t['n']):.3f}); not cut upper bound {t['cp_upper_not_cut']:.3f}; "
                         f"below_reg19 {t['below_reg19']}; McNemar p {t['mcnemar_p']:.3f}; onset {t['onset_split']}"))
    t = tables["d1b"]
    if isinstance(t, Exception):
        rows.append(_row(CELLS["d1b"], "D1b", error=_err(t)))
    else:
        o = t["outcome"]
        rows.append(_row(CELLS["d1b"], "D1b", o["k_stuck"], o["n"], "stuck",
                         f"decision {o['decision']}; k_bracket {o['k_bracket']}; cut {o['k_cut']}; labels "
                         f"{o['counts']}; seeds 15-17 only"))
    return rows


def _call(f):
    try:
        return f()
    except Exception as e:   # recorded in the row, never replaced by a number
        return e


def boundary_table(tables: dict | None = None, out: Path | None = None) -> list[dict]:
    """The boundary map. Runs the registered tables (minutes) unless `tables` is given; writes boundary_table.md."""
    if tables is None:
        from analysis.gate import reg19_table
        tables = {"reg19": _call(reg19_table), "reg20": _call(k1.reg20_table), "d1": _call(d1.table),
                  "d1b": _call(d1b.table)}
    rows = boundary_rows(tables)
    write_boundary_md(rows, (out or OUT) / "boundary_table.md")
    return rows


def _fmt(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else (f"{x:.3f}" if isinstance(x, float) else str(x))


def write_boundary_md(rows: list[dict], path: Path) -> None:
    pv = provenance()
    lines = ["# Boundary map (Stage 2)", "",
             f"Generated by `analysis.figures.boundary_table` at commit {pv['commit']}"
             f"{' (dirty)' if pv['dirty'] else ''}. Each cell: k of n seeds, one-sided {FIG['conf']:.0%} "
             "Clopper-Pearson bounds (rule H). Cells are not pooled; the closed-loop rows share runs (REG-19's).", "",
             "| Cell | λ | Turns | Cap | Counted | k of n | upper | lower | Detail | Source | Error |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for r in rows:
        kn = "—" if r["k"] is None else f"{r['k']} of {r['n']}"
        lines.append(f"| {r['cell']} | {r['lam']} | {r['turns']} | {_fmt(r['cap'])} | {r['counted']} | {kn} | "
                     f"{_fmt(r['cp_upper'])} | {_fmt(r['cp_lower'])} | {r['detail']} | {r['source']} | "
                     f"{_fmt(r['error'])} |")
    lines += ["", FIG["caching_off_note"] + " (no row).", ""]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines))


# ---------------------------------------------------------------- seeds and runs

def counted_run(pattern: str) -> tuple[dict | None, list[str]]:
    """The earliest valid run of `pattern` and the names of later valid ones; (None, []) if none."""
    runs = sorted(L.find_runs(pattern), key=lambda r: r["dir"].name)
    return (runs[0], [r["dir"].name for r in runs[1:]]) if runs else (None, [])


def pick_seed(arms: dict, seeds, seed: int | None = None) -> tuple[int, dict, dict]:
    """(seed, {arm: run}, {arm: later valid names}) for the lowest seed in `seeds` (or `seed`, if given) with a counted
    valid run in every arm. `arms` maps an arm name to a function seed -> run-folder pattern. Seeds are tried in
    ascending order and loading stops at the first that has all arms."""
    for sd in ([seed] if seed is not None else sorted(seeds)):
        got, later = {}, {}
        for arm, pat in arms.items():
            run, more = counted_run(pat(sd))
            if run is None:
                break
            got[arm], later[arm] = run, more
        else:
            return sd, got, later
    raise ValueError(f"no seed in {[seed] if seed is not None else sorted(seeds)} has a valid run in every arm "
                     f"{list(arms)}")


# ---------------------------------------------------------------- series

def burst_start(run: dict) -> float:
    return float(run["cfg"]["trigger"]["at_s"])


def _trailing_mean(t: np.ndarray, v: np.ndarray, grid: np.ndarray, roll: float) -> np.ndarray:
    """Mean of the scrapes with t in (g - roll, g] at each grid time g; NaN where there is none."""
    ok = ~np.isnan(v)
    t, v = t[ok], v[ok]
    cs = np.concatenate([[0.0], np.cumsum(v)])
    i1, i0 = np.searchsorted(t, grid, "right"), np.searchsorted(t, grid - roll, "right")
    n = i1 - i0
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[i1] - cs[i0]) / np.maximum(n, 1), np.nan)


SERIES_COLS = ("t", "t_rel", "reuse", "n_turns", "kv", "waiting", "cap_queue")


def series(run: dict, t0: float | None = None) -> pd.DataFrame:
    """One row per grid time t (every grid_s from roll_s): the trailing roll_s window (t - roll_s, t] of
      reuse     Σ cached / Σ previous-turn prompt over ok later turns by first token (`c2.reuse_in` on
                `k1.later_turns`: `k1.reuse`'s statistic), NaN with no later turn;
      n_turns   the later turns in that window;
      kv, waiting  the mean of the scrapes of the KV usage and waiting gauges;
      cap_queue sessions waiting at the load generator's cap (`d1.cap_queue`, at t), NaN for an uncapped run;
    and t_rel = t - t0 (default: the burst's start, the trigger's at_s)."""
    t0 = burst_start(run) if t0 is None else float(t0)
    m = run["metrics"].sort_values("t")
    q = run["requests"]
    tmax = max(float(m.t.max()) if len(m) else 0.0, float(q.t_end.max()) if len(q) and "t_end" in q else 0.0)
    grid = np.arange(FIG["roll_s"], tmax + 1e-9, FIG["grid_s"])
    turns = k1.later_turns(run) if len(q) else None
    roll = FIG["roll_s"]
    if turns is not None:
        ft = np.sort(turns.t_first_token.to_numpy(float))
        n_turns = np.searchsorted(ft, grid, "right") - np.searchsorted(ft, grid - roll, "right")
        # reuse_in reads [a, b); the window (g - roll, g] differs only on ties at the edges (float first-token times)
        ru = np.array([reuse_in(turns, g - roll, g, FIG["min_turns"]) if c else math.nan
                       for g, c in zip(grid, n_turns)])
    else:
        n_turns, ru = np.zeros(len(grid), int), np.full(len(grid), np.nan)
    gauge = lambda col: (_trailing_mean(m.t.to_numpy(float), m[col].to_numpy(float), grid, roll)
                         if col in m else np.full(len(grid), np.nan))
    cq = d1.cap_queue(run, grid) if len(q) else np.full(len(grid), np.nan)
    return pd.DataFrame({"t": grid, "t_rel": grid - t0, "reuse": ru, "n_turns": n_turns, "kv": gauge(FIG["kv_col"]),
                         "waiting": gauge(FIG["waiting_col"]), "cap_queue": cq})[list(SERIES_COLS)]


# ---------------------------------------------------------------- per-run labels (each registration's own readout)

def _cb(run: dict) -> str:
    r = k1.cold_backlogged(run)
    return {True: "cold-backlogged", False: "recovered" if r["recovered"] else "partial", None: "unclear"}[
        r["cold_backlogged"]]


def run_label(arm: str, run: dict) -> str:
    if arm in ("closed", "uncapped"):
        return f"verdict {L.verdict(run)['verdict']}; K1 {_cb(run)}"
    if arm == "open":
        return f"K1 {_cb(run)}"
    if arm == "d1":
        return f"d1.label {d1.label(run)['label']}"
    if arm == "d1b":
        return f"d1b.label40 {d1b.label40(run)['label']}"
    raise ValueError(arm)


# ---------------------------------------------------------------- plotting

def _plot(panels: dict, runs: dict, labels: dict, title: str, path_stem: Path, note: str | None = None,
          paper_names: dict | None = None, spans: list | None = None, vlines: list | None = None,
          xlabel: str | None = None, figsize: tuple | None = None, font: float | None = None) -> list[Path]:
    """The provenance figure; with `paper_names` ({arm: legend name}), the preprint's version of the same data: no
    title, run ids, analysis labels, note or commit footer (those stay in the provenance figure and the caption's
    source mark). Presentation only: the series are the same.
    `spans` [(a, b, grey)] replaces the default shading (the burst and K1's end window); `vlines` [(t, style, label)]
    marks times on every panel; `xlabel` replaces the default x label; `figsize` and `font` (points) redraw it at
    another size. With none of them the figure is unchanged."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if font is not None:
        with plt.rc_context({"font.size": font, "axes.labelsize": font, "xtick.labelsize": font - 1,
                             "ytick.labelsize": font - 1, "lines.linewidth": 0.8}):
            return _plot(panels, runs, labels, title, path_stem, note, paper_names, spans, vlines, xlabel, figsize)
    any_run = next(iter(runs.values()))
    if spans is None:
        trig = any_run["cfg"]["trigger"]
        win = k1.windows(any_run["cfg"])
        t0 = burst_start(any_run)
        spans = [(0, float(trig["duration_s"]), "0.85"), (win["end"][0] - t0, win["end"][1] - t0, "0.93")]
    fig, ax = plt.subplots(3, 1, sharex=True, figsize=figsize or FIG["figsize"])
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    cq_ax = None
    for i, (arm, df) in enumerate(panels.items()):
        c = FIG["arm_color"].get(arm, colors[i % len(colors)])
        lab = paper_names[arm] if paper_names else f"{arm}: {runs[arm]['dir'].name} ({labels[arm]})"
        ax[0].plot(df.t_rel, df.reuse, color=c, lw=1, label=lab)
        ax[1].plot(df.t_rel, df.kv, color=c, lw=1)
        ax[2].plot(df.t_rel, df.waiting, color=c, lw=1)
        if df.cap_queue.notna().any():
            # Sessions, not requests: on their own right-hand axis (panel 2026-11-22, R8.4), so the two units never
            # share a scale. Presentation only.
            if cq_ax is None:
                cq_ax = ax[2].twinx()
                cq_ax.set_ylabel("cap queue (sessions, dashed)")
            cq_ax.plot(df.t_rel, df.cap_queue, color=c, lw=1, ls="--",
                       label=f"{paper_names[arm] if paper_names else arm}: cap queue (sessions)")
    for a in ax:
        for lo, hi, grey in spans:
            a.axvspan(lo, hi, color=grey, zorder=0)
    for t, style, lab in vlines or []:
        for i, a in enumerate(ax):
            a.axvline(t, color="0.2", lw=0.9, ls=style, label=lab if i == 0 else None)
    ax[1].axhline(FIG["kv_onset"], color="0.4", lw=0.8, ls=":")
    ax[0].set_ylabel("follow-up hit rate")
    ax[1].set_ylabel("KV usage")
    ax[2].set_ylabel("server queue (requests)")
    ax[2].set_xlabel(xlabel or ("seconds from the burst's start (grey: burst; light grey: "
                     + ("the judged end window)" if paper_names else "verdict end window)")))
    # Legends sit below the panels, so no data is hidden behind them (presentation only).
    all_ax = list(ax) + ([cq_ax] if cq_ax is not None else [])
    handles = [h for a in all_ax for h in a.get_legend_handles_labels()[0]]
    names = [n for a in all_ax for n in a.get_legend_handles_labels()[1]]
    fig.legend(handles, names, fontsize=plt.rcParams["font.size"] if figsize else 7, loc="lower left",
               bbox_to_anchor=(0.01, 0.02), frameon=False)
    if not paper_names:
        ax[0].set_title(title, fontsize=10)
    if note and not paper_names:
        ax[0].text(0.99, 0.04, note, transform=ax[0].transAxes, ha="right", va="bottom", fontsize=8,
                   bbox={"facecolor": "white", "edgecolor": "0.5"})
    if not paper_names:
        fig.text(0.01, 0.005, footer([r["dir"].name for r in runs.values()]), fontsize=6, va="bottom")
    # One legend line is about 1.6 font sizes high; at the column size the legend gets that much room per line, as a
    # share of the figure's height, plus its 0.02 offset, so it sits below the x label.
    per_name = plt.rcParams["font.size"] * 1.6 / 72 / figsize[1] if figsize else 0.017
    fig.tight_layout(rect=(0, (0.0 if paper_names else 0.03) + (0.02 if figsize else 0) + per_name * len(names), 1, 1))
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    out = [path_stem.with_suffix(".png"), path_stem.with_suffix(".pdf")]
    for p in out:
        fig.savefig(p, dpi=FIG["dpi"])
    plt.close(fig)
    return out


# ---------------------------------------------------------------- 2. the matched-arm figure

MATCHED_ARMS = {"open": lambda sd: k1.open_pattern(FIG["matched_lam"], sd),   # REG-20 arm A
                "closed": k1.closed_pattern}                                   # REG-19's burst run


def matched_arms(seed: int | None = None, out: Path | None = None, draw: bool = True) -> dict:
    sd, runs, later = pick_seed(MATCHED_ARMS, FIG["matched_seeds"], seed)
    panels = {arm: series(r) for arm, r in runs.items()}
    labels = {arm: run_label(arm, r) for arm, r in runs.items()}
    files = (_plot(panels, runs, labels, f"Matched arms, L1 λ = {FIG['matched_lam']}, seed {sd}: open-loop turns "
                   f"(REG-20 arm A) against closed-loop turns (REG-19)", (out or OUT) / f"matched_arms_s{sd}")
             + _plot(panels, runs, labels, "", (out or OUT) / f"matched_arms_s{sd}_paper",
                     paper_names={"open": "open-loop turns", "closed": "closed-loop turns"})
             + _plot(panels, runs, labels, "", (out or OUT) / f"matched_arms_s{sd}_column",
                     paper_names={"open": "open-loop turns", "closed": "closed-loop turns"},
                     xlabel="seconds from the burst's start", figsize=FIG["column_figsize"], font=FIG["column_font"])
             if draw else [])
    return {"seed": sd, "runs": {a: r["dir"].name for a, r in runs.items()}, "later_valid": later,
            "labels": labels, "series": panels, "files": [str(p) for p in files]}


# ---------------------------------------------------------------- the return-to-warm figure (Figure 3)

REWARM_ARMS = {"burst": lambda sd: reg30.REG30["runs"][sd],         # REG-27's (or REG-24's) burst run, pinned
               "control": lambda sd: reg30.REG30["controls"][sd]}   # its same-seed control, no burst


def rewarm_fig(seed: int | None = None, out: Path | None = None, draw: bool = True) -> dict:
    """A long-watch burst run beside its same-seed control on one clock, seconds from the drop to λ = 1 (each run's
    own drop start from `confirm.phases`). Shaded: the burst, the drop. Lines: REG-30's re-warm onset and the drain
    (`reg30.rewarm`), so the figure carries that registration's numbers, not new ones."""
    sd, runs, later = pick_seed(REWARM_ARMS, FIG["rewarm_seeds"], seed)
    ph = {arm: confirm.phases(r["cfg"]) for arm, r in runs.items()}
    panels = {arm: series(r, t0=ph[arm]["drop"][0]) for arm, r in runs.items()}
    rw = reg30.rewarm(runs["burst"], control=runs["control"])
    b = ph["burst"]
    tr = runs["burst"]["cfg"]["trigger"]
    we = b["drop"][0]
    spans = [(float(tr["at_s"]) - we, b["burst_end"] - we, "0.85"), (0.0, b["drop"][1] - we, "0.93")]
    vlines = [(rw["onset_s"], ":", "reuse back at its recovered level (re-warm onset)"),
              (rw["drained_s"], "--", "total line (the owed backlog) drained")]
    labels = {"burst": f"{rw['label']}, lead {rw['lead_s']:+.0f} s", "control": "control"}
    xl = "seconds from the drop to λ = 1 (grey: burst; light grey: the drop; then the return to λ = 2.0)"
    files = (_plot(panels, runs, labels, f"Return to warm, L1 closed-loop λ = 2.0, seed {sd}: burst run (REG-27) and "
                   f"control; REG-30 onset and drain", (out or OUT) / f"rewarm_s{sd}", spans=spans, vlines=vlines,
                   xlabel=xl)
             + _plot(panels, runs, labels, "", (out or OUT) / f"rewarm_s{sd}_paper", spans=spans, vlines=vlines,
                     xlabel=xl, paper_names={"burst": "burst run", "control": "same-seed control, no burst"})
             + _plot(panels, runs, labels, "", (out or OUT) / f"rewarm_s{sd}_column", spans=spans, vlines=vlines,
                     xlabel=XL_COLUMN, paper_names={"burst": "burst run", "control": "same-seed control, no burst"},
                     figsize=FIG["column_figsize"], font=FIG["column_font"])
             if draw else [])
    return {"seed": sd, "runs": {a: r["dir"].name for a, r in runs.items()}, "later_valid": later,
            "labels": labels, "rewarm": {k: rw[k] for k in ("onset_s", "drained_s", "lead_s", "n_onset", "label")},
            "series": panels, "files": [str(p) for p in files]}


# The column variant's x label: the long one does not fit 3.4 in; the caption names the shading.
XL_COLUMN = "seconds from the drop to λ = 1"


# ---------------------------------------------------------------- 3. the loop-cut figure

LOOPCUT_ARMS = {"uncapped": k1.closed_pattern,   # REG-19's burst run
                "d1": d1.run_pattern,            # D1's per-seed healthy-peak cap
                "d1b": d1b.run_pattern}          # D1b's cap 40


def loop_cut(seed: int | None = None, out: Path | None = None, draw: bool = True) -> dict:
    sd, runs, later = pick_seed(LOOPCUT_ARMS, FIG["loopcut_seeds"], seed)
    panels = {arm: series(r) for arm, r in runs.items()}
    labels = {arm: run_label(arm, r) for arm, r in runs.items()}
    files = (_plot(panels, runs, labels, f"Loop cut, L1 closed-loop λ = 2.0, seed {sd}: uncapped (REG-19), "
                   f"session cap (D1), cap 40 (D1b)", (out or OUT) / f"loop_cut_s{sd}", note=FIG["caching_off_note"])
             + _plot(panels, runs, labels, "", (out or OUT) / f"loop_cut_s{sd}_paper",
                     paper_names={"uncapped": "uncapped", "d1": "session cap at the healthy peak", "d1b": "cap of 40"})
             if draw else [])
    return {"seed": sd, "runs": {a: r["dir"].name for a, r in runs.items()}, "later_valid": later,
            "labels": labels, "series": panels, "note": FIG["caching_off_note"], "files": [str(p) for p in files]}


# ---------------------------------------------------------------- 4. model against measurement

def model_vs_measurement() -> dict:
    """Withheld until decisions.md holds K3b's [RESULT]. It will plot, per quantity (served mix, token work, completion
    rate; session completion rate as a readout), model F's predicted value beside the measured one with its registered
    tolerance band: K3b's rows (`analysis.k3b.table`, K4's tipped runs at λ = 2.2) and REG-21's rows
    (`analysis.k3w.reg21_table`), one marker per run, inside/outside the band as each table judged it."""
    if not withheld.lifted(FIG["k3b_result"]):
        raise withheld.WithheldError(f"model_vs_measurement waits for decisions.md's {FIG['k3b_result']!r} heading")
    raise NotImplementedError("K3b's [RESULT] is in; write the plot against k3b.table and k3w.reg21_table")


# ---------------------------------------------------------------- CLI

def _summary(res: dict) -> str:
    return f"seed {res['seed']}: runs {res['runs']}; labels {res['labels']}; later valid {res['later_valid']}; " \
           f"files {res['files']}"


def main(argv: list[str]) -> int:
    what = argv[0] if argv else "all"
    if what not in ("all", "boundary", "matched", "loopcut", "rewarm"):
        raise SystemExit("usage: python -m analysis.figures all | boundary | matched | loopcut | rewarm")
    print(f"# figures {what} | {provenance()}")
    rc = 0
    if what in ("all", "matched"):
        print("matched_arms:", _summary(matched_arms()))
    if what in ("all", "loopcut"):
        print("loop_cut:", _summary(loop_cut()))
    if what in ("all", "rewarm"):
        print("rewarm:", _summary(rewarm_fig()))
    if what in ("all", "boundary"):
        rows = boundary_table()
        for r in rows:
            kn = "—" if r["k"] is None else f"{r['k']} of {r['n']}"
            print(f"boundary: {r['cell']} λ {r['lam']}: {r['counted']} {kn} (upper {_fmt(r['cp_upper'])}, lower "
                  f"{_fmt(r['cp_lower'])}); {r['detail']}; source {r['source']}; error {_fmt(r['error'])}")
            rc |= int(r["error"] is not None)
        print("boundary: wrote", OUT / "boundary_table.md")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
