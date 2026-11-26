"""Three descriptive figures for the 2026-11-25 rewrite (author's change list, part 1 items 2-4). 0 GPU h, runs
already seen, nothing judged: no label, threshold or count here decides anything, and every quantity is computed
by a committed function named below.

    uv run --extra analysis --frozen python -m analysis.rewrite_figs all | reuse | sessions | capacity

1. `reuse_by_x`: later-turn reuse against idle age in units of the free queue's turnover time, x = A / F
   (`m.ages`: idle age from the previous turn's end to this turn's first token, A the newly allocated tokens over
   it, F the free pool at its start), pooled per arm and x bucket (`m.profile`), for three sets of cold runs after
   the burst: REG-19's five closed-loop burst runs and REG-20 arm A's five open-loop burst runs (reading M's arms,
   `m.M["arms"]`), and D1b's three cap-40 runs (`b1.ARMS["cold"]`, the runs B1 read by think gap). M's buckets and
   window are reused unchanged; a bucket is drawn when it holds at least FIGS["min_turns"] turns (presentation).
   B1 read the cap-40 runs by the drawn think gap, a different axis; this figure puts them on M's axis.
2. `sessions`: sessions in service over time (`d1.in_service_steps`) for one seed's uncapped burst run (REG-19),
   D1's capped run, D1b's cap-40 run and REG-20 arm A's open-loop burst run, with the band of session counts whose
   latest prompts fill the 96,656-token pool: pool / longest prompt (3968 tokens, turn 8) to pool / mean prompt
   (3072 tokens): 24.4 to 31.5 sessions (DERIVED; prompts 2176 + 256 k, k = 0..7, FILE(configs/l1_base.yaml)).
   Seed: the lowest in 15-17 with a counted valid run in all four arms (`figures.counted_run`).
3. `capacity`: completions per second (`load.summarize` `throughput`) against the swept rate for seed 0's four
   sweeps at 350 W: open-loop turns cache on and off (REG-8), closed-loop turns cache on (REG-13) and off (E1).
   Shaded: the closed-loop window from the cache-off saturated rate (E1's 1.798 req/s, `e1.observed`) to the
   cache-on keep-up rate on that schedule (2.4, REG-13); the line marks λ = 2.0.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

from analysis import b1, d1, k1, m
from analysis import load as L
from analysis.figures import OUT, counted_run
from analysis.tables import provenance

FIGS = {
    "min_turns": 20,                       # presentation: buckets with fewer turns are not drawn (listed in the table)
    "pool_tokens": m.M["pool_tokens"],     # 96,656
    "prompt_longest": 2176 + 256 * 7,      # 3968 tokens, turn 8 (FILE(configs/l1_base.yaml): 2048 + 128 + 256 k)
    "prompt_mean": 2176 + 256 * 3.5,       # 3072 tokens
    "session_seeds": (15, 16, 17),
    "sweeps": {"open, cache on": "*_sweep_l1pl350_pc_on_r*", "open, cache off": "*_sweep_l1pl350_pc_off_r*",
               "closed, cache on": "*_sweep_l1pl350cl_pc_on_r*", "closed, cache off": "*_sweep_l1pl350cl_pc_off_r*"},
    "window": (1.798, 2.4),                # E1's closed-loop cache-off rate; REG-13's keep-up rate (seed 0)
    "lam": 2.0,
    # One colour per role, as figures.FIG["arm_color"] (validated set, dataviz validator: CVD and normal-vision PASS;
    # the orange is below 3:1 contrast, so every series is also direct-labelled and has its own line style).
    "color": {"closed": "#d62728", "open": "#1f77b4", "cap40": "#e08214", "cap": "#6a3d9a"},
    "figsize": (6.5, 4.0), "dpi": 200,
    "column_figsize": (3.4, 2.6), "column_font": 7,   # the workshop's one-column variant (review 2026-11-26): drawn at print size
    "sessions_ymax": 90,                   # presentation: the uncapped run is clipped and its peak annotated
}
ARM_LABEL = {"closed": "closed-loop turns", "open": "open-loop turns",   # REG-19, REG-20 arm A, D1b, D1 (no ids in the paper)
             "cap40": "closed-loop, session cap 40", "cap": "closed-loop, cap at the control's peak"}


def save(fig, path, foot: str) -> None:
    """KEY.pdf/png with the provenance footer; KEY_paper.png without it, the copy the paper build uses."""
    t = fig.text(0.01, 0.005, foot, fontsize=5, color="#777777")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=FIGS["dpi"])
    t.remove()
    fig.savefig(path.with_name(path.stem + "_paper.png"), dpi=FIGS["dpi"])


def footer(run_ids: list[str]) -> str:
    pv = provenance()
    return f"commit {pv['commit']}{' (dirty)' if pv['dirty'] else ''} · analysis/rewrite_figs.py · {len(run_ids)} runs"


def _x_mid(bk: str) -> float:
    lo, hi = (float(v) for v in bk.strip("[)").split(","))
    return math.sqrt(max(lo, 0.25) * hi) if math.isfinite(hi) else 24.0   # log-midpoint; [0, .5) at 0.35; [16, inf) at 24


def gap_series(pooled: dict) -> tuple[list[float], list[float]]:
    """(x, reuse) over every x bucket from the first drawn to the last, NaN at a bucket that is not drawn
    (fewer than FIGS["min_turns"] turns), so a line drawn through it breaks there instead of crossing it."""
    xs, ys = [], []
    for bk in m.X_BUCKETS:
        ok = pooled[bk]["n"] >= FIGS["min_turns"] and np.isfinite(pooled[bk]["reuse"])
        xs.append(_x_mid(bk))
        ys.append(float(pooled[bk]["reuse"]) if ok else math.nan)
    drawn = [i for i, y in enumerate(ys) if np.isfinite(y)]
    if not drawn:
        return [], []
    return xs[drawn[0]:drawn[-1] + 1], ys[drawn[0]:drawn[-1] + 1]


def reuse_by_x() -> dict:
    """Per arm: pooled reuse and turns per bucket (`m.profile`), per-run profiles, and the runs read."""
    arms = {"closed": m.M["arms"]["closed"], "open": m.M["arms"]["open"], "cap40": b1.ARMS["cold"]}
    out = {}
    for arm, pat in arms.items():
        runs = L.find_runs(pat)
        qs, names = [], []
        for r in runs:
            q = m.ages(r)
            if q is not None:
                qs.append(q)
                names.append(r["dir"].name)
        out[arm] = {"runs": names, "pooled": m.profile(qs),
                    "per_run": {n: m.profile([q]) for n, q in zip(names, qs)}}
    return out


def sessions(seed: int | None = None) -> dict:
    pats = {"closed": lambda s: b1.ARMS["uncapped"].replace("s1[5-9]", f"s{s}"),
            "cap": lambda s: f"*_burst_l1pl350cl_r2.0_w3000_s{s}_cap",
            "cap40": lambda s: f"*_burst_l1pl350cl_r2.0_w3000_s{s}_cap40",
            "open": lambda s: k1.open_pattern(FIGS["lam"], s)}
    for s in ([seed] if seed else FIGS["session_seeds"]):
        runs = {a: counted_run(p(s))[0] for a, p in pats.items()}
        if all(runs.values()):
            series = {a: d1.in_service_steps(r) for a, r in runs.items()}
            band = (FIGS["pool_tokens"] / FIGS["prompt_longest"], FIGS["pool_tokens"] / FIGS["prompt_mean"])
            return {"seed": s, "runs": {a: r["dir"].name for a, r in runs.items()}, "series": series, "band": band,
                    "peaks": {a: int(v[1].max()) for a, v in series.items()}}
    raise ValueError("no seed with all four arms")


def capacity() -> dict:
    rows = {}
    for name, pat in FIGS["sweeps"].items():
        rs = sorted((L.summarize(r) for r in L.find_runs(pat)), key=lambda x: x["rate"])
        rows[name] = [{"run": x["run"], "rate": float(x["rate"]), "throughput": x["throughput"],
                       "planned_rate": x["planned_rate"]} for x in rs]
    return rows


# ---------------------------------------------------------------- drawing (presentation only)

def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e5e5e5", "grid.linewidth": 0.6})
    return plt


def draw_reuse(res: dict, out: Path, column: bool = False) -> Path:
    """The paper's figure, or with `column` the one-column variant KEY_column.png: same data, drawn at 3.4 in with
    7 pt type and the legend alone (no direct labels, which do not fit), for the workshop paper."""
    plt = _plt()
    if column:
        plt.rcParams.update({"font.size": FIGS["column_font"]})
    fig, ax = plt.subplots(figsize=FIGS["column_figsize"] if column else FIGS["figsize"])
    fs = FIGS["column_font"] if column else 8
    styles = {"closed": "-o", "open": "--s", "cap40": ":^"}
    # direct labels: (bucket index, dx, dy points) placed by hand away from the other lines (presentation only)
    where = {"open": (2, 8, 4), "cap40": (2, 8, -2), "closed": (6, -150, 14)}
    for arm in ("open", "cap40", "closed"):          # closed last, so its points sit on top at reuse 0
        p = res[arm]["pooled"]
        pts = [(_x_mid(bk), p[bk]["reuse"], p[bk]["n"]) for bk in m.X_BUCKETS
               if p[bk]["n"] >= FIGS["min_turns"] and np.isfinite(p[bk]["reuse"])]
        if not pts:
            continue
        x, y, n = zip(*pts)
        # The line breaks at a bucket that is not drawn (LTB review 2026-11-26: the open-loop line crossed the
        # undrawn 1-2 bucket): every bucket gets a slot, NaN where it is not drawn, so no segment spans a gap.
        gx, gy = gap_series(p)
        ax.plot(gx, gy, styles[arm], color=FIGS["color"][arm], lw=1.5 if column else 2,
                ms=(5 if column else 7) if arm == "closed" else (4 if column else 6),
                label=ARM_LABEL[arm], zorder=3 if arm == "closed" else 2)
        if column:   # the bucket's turn count above (or, for the cap arm, below) each point (review 2026-11-26)
            dy = {"open": 4, "cap40": -9, "closed": 12}[arm]   # three heights, so labels at reuse 0 do not overlap
            for xi, yi, ni in zip(x, y, n):
                ax.annotate(f"{ni:,}", (xi, yi), xytext=(0, dy), textcoords="offset points", ha="center",
                            fontsize=fs - 1.5, color="#333333")
            continue
        i, dx, dy = where[arm]
        bx = _x_mid(m.X_BUCKETS[i])
        by = p[m.X_BUCKETS[i]]["reuse"] if np.isfinite(p[m.X_BUCKETS[i]]["reuse"]) else 0.0
        ax.annotate(ARM_LABEL[arm].split(" (")[0], (bx, by), xytext=(dx, dy), textcoords="offset points",
                    fontsize=8, color="#333333")
    ax.axvline(1.0, color="#888888", lw=1, ls="-")
    ax.text(1.03, 1.02, "one turnover time", fontsize=fs, color="#555555", va="bottom")
    ax.set_xscale("log")
    ax.set_xticks([0.35, 0.7, 1.4, 2.8, 5.7, 11.3, 24])
    ax.set_xticklabels(["<0.5", "0.5–1", "1–2", "2–4", "4–8", "8–16", "≥16"])
    ax.set_ylim(-0.03, 1.08)
    if column:
        ax.set_ylim(-0.14, 1.15)   # room for the cap arm's counts below its points and the top count (LTB review)
        ax.set_xlabel("idle time, in turnover times of the free pool")
        ax.set_ylabel("follow-up hit rate")
        ax.legend(loc="center right", fontsize=fs, frameon=False)
        fig.tight_layout()
        path = out / "reuse_by_x_column.png"
        fig.savefig(path, dpi=300)
        fig.savefig(path.with_suffix(".pdf"))   # vector, for the workshop build (review 2026-11-26)
        plt.close(fig)
        plt.rcParams.update({"font.size": 9})
        return path
    ax.set_xlabel("idle time before the turn, in turnover times of the free pool (x = A / F)")
    ax.set_ylabel("follow-up hit rate (share of previous prompt cached)")
    ax.legend(loc="center right", fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "reuse_by_x.pdf"
    save(fig, path, footer(sum((res[a]["runs"] for a in res), [])))
    plt.close(fig)
    return path


def draw_sessions(res: dict, out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGS["figsize"])
    lo, hi = res["band"]
    ax.axhspan(lo, hi, color="#cccccc", alpha=0.6, lw=0, label=f"sessions whose prompts fill the pool ({lo:.0f}–{hi:.0f})")
    ax.axvspan(0, 180, color="#f2f2f2", lw=0)
    ax.text(5, 84, "burst", fontsize=8, color="#555555")
    styles = {"closed": "-", "cap": "-.", "cap40": ":", "open": "--"}
    for arm in ("closed", "cap40", "cap", "open"):
        t, n = res["series"][arm]
        ax.step(t - 900.0, n, where="post", ls=styles[arm], color=FIGS["color"][arm], lw=1.6, label=ARM_LABEL[arm])
    ymax = FIGS["sessions_ymax"]
    peak = res["peaks"]["closed"]
    if peak > ymax:   # the uncapped run leaves the axis: say where it goes instead of flattening the others
        t, n = res["series"]["closed"]
        t_out = float(t[np.argmax(n >= ymax)]) - 900.0     # where the line leaves the axis
        ax.annotate(f"uncapped closed-loop run: {peak} sessions by the window's end", (t_out, ymax - 1),
                    xytext=(t_out + 250, ymax - 8), fontsize=8, color="#333333",
                    arrowprops={"arrowstyle": "-", "color": "#999999", "lw": 0.8})
    ax.set_ylim(0, ymax)
    ax.set_xlim(-600, 3180)
    ax.set_xlabel("seconds from the burst's start")
    ax.set_ylabel("sessions in service")
    ax.legend(loc="upper right", fontsize=7.5, frameon=False, bbox_to_anchor=(1.0, 0.86))
    ax.set_title(f"seed {res['seed']}", fontsize=9, loc="left")
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "sessions_in_service.pdf"
    save(fig, path, footer(list(res["runs"].values())))
    plt.close(fig)
    return path


def draw_capacity(res: dict, out: Path) -> Path:
    plt = _plt()
    fig, ax = plt.subplots(figsize=FIGS["figsize"])
    a, b = FIGS["window"]
    ax.axvspan(a, b, color="#eeeeee", lw=0)
    ax.text(a + 0.02, 0.15, "closed-loop\nvulnerable range", fontsize=8, color="#555555")   # the caption's term (review 2026-11-26)
    ax.axvline(FIGS["lam"], color="#888888", lw=1)
    top = max(x["rate"] for v in res.values() for x in v)
    ax.plot([0, top], [0, top], color="#bbbbbb", lw=1)
    ax.text(top * 0.80, top * 0.86, "completions = planned", fontsize=7, color="#888888", rotation=0)
    style = {"open, cache on": ("--s", "open", "none"), "open, cache off": ("--s", "open", None),
             "closed, cache on": ("-o", "closed", "none"), "closed, cache off": ("-o", "closed", None)}
    for name, rows in res.items():
        if not rows:
            continue
        mk, arm, face = style[name]
        x = [r["rate"] for r in rows]
        y = [r["throughput"] for r in rows]
        ax.plot(x, y, mk, color=FIGS["color"][arm], lw=1.6, ms=5, mfc=face or FIGS["color"][arm], label=name)
        ax.annotate(name, (x[-1], y[-1]), xytext=(4, -3), textcoords="offset points", fontsize=7, color="#333333")
    ax.set_xlabel("planned rate λ (requests/s)")   # the caption's term (review 2026-11-26)
    ax.set_ylabel("completions/s")
    ax.set_xlim(0.8, top + 0.9)
    ax.set_ylim(0, None)
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    path = out / "capacity.pdf"
    save(fig, path, footer([r["run"] for v in res.values() for r in v]))
    plt.close(fig)
    return path


def main(argv: list[str]) -> int:
    import json
    what = argv[1] if len(argv) > 1 else "all"
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    if what in ("all", "reuse"):
        r = reuse_by_x()
        draw_reuse(r, OUT)
        draw_reuse(r, OUT, column=True)
        res["reuse"] = {a: {"runs": v["runs"], "pooled": v["pooled"]} for a, v in r.items()}
    if what in ("all", "sessions"):
        r = sessions()
        draw_sessions(r, OUT)
        res["sessions"] = {k: v for k, v in r.items() if k != "series"}
    if what in ("all", "capacity"):
        r = capacity()
        draw_capacity(r, OUT)
        res["capacity"] = r
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
