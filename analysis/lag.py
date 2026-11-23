"""A5: how much of the client's time to first token the server's own TTFT does not see (REG-26).

The server's `vllm:time_to_first_token_seconds` runs from the renderer's `arrival_time` to the API server's receipt
of the first output (open-questions A5). The client's, here, runs from `t_sent` to `t_first_token`, the first SSE
chunk with a choice (FILE(harness/loadgen.py:102,118-119)); vLLM sends that chunk for the first token even when its
text is empty (SOURCE(vLLM 0.32.0 entrypoints/openai/completion/serving.py:OpenAIServingCompletion, l. 373-379)).
Their difference bounds from above everything outside the server's span: the TCP connect and HTTP receipt before
rendering, the chunk's detokenization, the SSE write, the network and the client's receive lag.

The server keeps only a histogram, so the comparison is between aggregates over the same requests:
- **The label** reads each run's whole span, first scrape to last: the mean gap (client mean minus Δsum / Δcount)
  and the shift in the share at or under 1.0 s (the server's Δbucket{le="1.0"} / Δcount minus the client's share),
  which is the quantity percent fast uses. A scrape's counters are read up to `scrape_timeout_s` after its stamp
  (FILE(harness/recorder.py:121-123)), on the API server's loop, so window boundaries drift under load; over a whole
  run that drift moves at most the few requests at the two ends.
- **60 s windows** are a readout only (where in a run a gap appears). They are biased where TTFT trends, by about
  the trend times the scrape delay (REG-26 rule F round 1, trace 1), so they never set the label.

Usage: `python -m analysis.lag` prints REG-26's table and its label.
"""
from __future__ import annotations

import json
import math
import sys

import numpy as np
import pandas as pd

from analysis import load

SUM, COUNT = "vllm:time_to_first_token_seconds_sum", "vllm:time_to_first_token_seconds_count"
BUCKET = "vllm:time_to_first_token_seconds_bucket"

# Fixed by REG-26 before any of these runs' client-server differences were computed: every valid, unsealed L1 run
# (`*l1pl350*`) on 2026-11-14, listed by name and checks.json validity only. Left out: 20261109-134250_stage2_
# control_l1pl350cl_r2.4_s6 (invalid, accounting) and 20261113-033842_stage2_control_l1pl350cl_r2.0_w3000_s18
# (no checks.json).
A5 = {
    "runs": (
        "20261107-075951_sweep_l1pl350_pc_off_r1", "20261107-082035_sweep_l1pl350_pc_off_r1.5",
        "20261107-083730_sweep_l1pl350_pc_off_r2", "20261107-085650_sweep_l1pl350_pc_off_r2.5",
        "20261107-091752_sweep_l1pl350_pc_off_r3", "20261107-093854_sweep_l1pl350_pc_off_r3.5",
        "20261107-095958_sweep_l1pl350_pc_off_r4", "20261107-103010_sweep_l1pl350_pc_on_r2",
        "20261107-104614_sweep_l1pl350_pc_on_r3", "20261107-110219_sweep_l1pl350_pc_on_r3.5",
        "20261107-111833_sweep_l1pl350_pc_on_r4", "20261107-113511_sweep_l1pl350_pc_on_r4.5",
        "20261107-115311_sweep_l1pl350_pc_on_r5", "20261108-211646_stage2_control_l1pl350_r2.4",
        "20261108-220248_burst_l1pl350_r2.4_s1", "20261109-034255_stage2_control_l1pl350cl_r2.4",
        "20261109-042828_burst_l1pl350cl_r2.4_s1", "20261109-055910_stage2_control_l1pl350cl_r2.4_s2",
        "20261109-080203_burst_l1pl350cl_r2.4_s2", "20261109-085121_stage2_control_l1pl350cl_r2.4_s3",
        "20261109-093609_burst_l1pl350cl_r2.4_s3", "20261109-102614_stage2_control_l1pl350cl_r2.4_s4",
        "20261109-111316_burst_l1pl350cl_r2.4_s4", "20261109-120308_stage2_control_l1pl350cl_r2.4_s5",
        "20261109-125300_burst_l1pl350cl_r2.4_s5", "20261109-142813_stage2_control_l1pl350cl_r2.4_s6",
        "20261109-151300_burst_l1pl350cl_r2.4_s6", "20261109-190221_sweep_l1pl350cl_pc_on_r2",
        "20261109-191845_sweep_l1pl350cl_pc_on_r2.4", "20261109-193526_sweep_l1pl350cl_pc_on_r2.6",
        "20261109-195349_sweep_l1pl350cl_pc_on_r2.8", "20261109-201246_sweep_l1pl350cl_pc_on_r3",
        "20261110-055201_stage2_control_l1pl350cl_r2.2_s2", "20261110-063647_burst_l1pl350cl_r2.2_s2",
        "20261110-085730_confirm_l1pl350cl_r2.2_s12", "20261110-122001_stage2_control_l1pl350cl_r2.2_long_s14",
        "20261110-174432_stage2_control_l1pl350cl_r2.0_s2", "20261110-182920_burst_l1pl350cl_r2.0_s2",
        "20261111-093312_stage2_control_l1pl350cl_r2.0_s4", "20261111-101805_burst_l1pl350cl_r2.0_s4",
        "20261111-110919_stage2_control_l1pl350cl_r2.0_s3", "20261111-115409_burst_l1pl350cl_r2.0_s3",
        "20261111-124527_stage2_control_l1pl350cl_r2.0_s5", "20261111-133019_burst_l1pl350cl_r2.0_s5",
        "20261111-142155_stage2_control_l1pl350cl_r2.0_s6", "20261111-150642_burst_l1pl350cl_r2.0_s6",
        "20261112-050920_stage2_control_l1pl350cl_r2.0_w3000_s15", "20261112-061951_burst_l1pl350cl_r2.0_w3000_s15",
        "20261112-073516_stage2_control_l1pl350cl_r2.0_w3000_s16", "20261112-084533_burst_l1pl350cl_r2.0_w3000_s16",
        "20261112-103423_stage2_control_l1pl350cl_r2.0_w3000_s17", "20261112-135538_burst_l1pl350cl_r2.0_w3000_s17",
        "20261113-165516_stage2_control_l1pl350cl_r2.0_w3000_s18", "20261113-180511_burst_l1pl350cl_r2.0_w3000_s18",
        "20261113-192110_stage2_control_l1pl350cl_r2.0_w3000_s19", "20261113-203108_burst_l1pl350cl_r2.0_w3000_s19",
        "20261114-021512_sweep_l1pl350cl_pc_off_r1.5", "20261114-023230_sweep_l1pl350cl_pc_off_r2",
        "20261114-025146_sweep_l1pl350cl_pc_off_r2.5", "20261114-031247_sweep_l1pl350cl_pc_off_r3",
        "20261114-033348_sweep_l1pl350cl_pc_off_r3.5", "20261114-035448_sweep_l1pl350cl_pc_off_r4",
    ),
    "slo_s": 1.0,          # the Stage-2 TTFT SLO; vLLM's histogram has a bucket edge at exactly 1.0
    "flag_gap_s": 0.10,    # a run's whole-span mean gap above this (10 % of the SLO) flags
    "flag_share": 0.01,    # a run's share at or under the SLO lower on the client's side by more than this flags
    "win_s": 60.0,         # readout windows
    "min_n": 20,           # a readout window is kept with at least this many first tokens on each side
}


def _bucket_le(path, le: str) -> tuple[np.ndarray, np.ndarray]:
    """(t, cumulative count at or under `le`) per scrape, summed over label sets, from a raw metrics.jsonl."""
    key = f'le="{le}"'
    ts, vs = [], []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        s = json.loads(line)
        if "values" not in s:
            continue
        v = sum(x for k, x in s["values"].items() if k.startswith(BUCKET + "{") and key in k)
        ts.append(s["t"]), vs.append(v)
    return np.asarray(ts, float), np.asarray(vs, float)


def _series(run: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(t, sum, count, bucket at the SLO) per scrape, sorted; raises if a counter goes down."""
    m = run["metrics"]
    if not len(m) or SUM not in m or COUNT not in m:
        raise ValueError("metrics missing: the server's TTFT histogram")
    m = m.dropna(subset=[SUM, COUNT]).sort_values("t")
    t, s, c = (m[k].to_numpy(float) for k in ("t", SUM, COUNT))
    bt, bv = run["slo_bucket"]
    b = np.interp(t, bt, bv) if len(bt) else np.full(len(t), np.nan)   # same scrapes; interp only aligns the order
    for name, x in (("sum", s), ("count", c), ("bucket", b)):
        if (np.diff(x) < 0).any():
            raise ValueError(f"the server's TTFT {name} goes down: a restart inside the run")
    return t, s, c, b


def _client(run: dict) -> tuple[np.ndarray, np.ndarray]:
    """(first-token time, client TTFT from t_sent) of requests with a first token. A first token without t_sent
    raises: the record is broken."""
    r = run["requests"]
    if not len(r):
        return np.array([]), np.array([])
    r = r[r.t_first_token.notna()]
    if r.t_sent.isna().any():
        raise ValueError("a request has a first token but no t_sent")
    return r.t_first_token.to_numpy(float), (r.t_first_token - r.t_sent).to_numpy(float)


def whole(run: dict, slo_s: float = A5["slo_s"]) -> dict:
    """The run's whole span, (first scrape, last scrape]: counts on both sides, mean TTFT on both sides, the mean
    gap, both shares at or under the SLO and their difference (server minus client; positive = the client sees
    fewer fast requests)."""
    t, s, c, b = _series(run)
    ft, cl = _client(run)
    inside = (ft > t[0]) & (ft <= t[-1])
    n_s, n_c = c[-1] - c[0], int(inside.sum())
    out = {"n_server": int(n_s), "n_client": n_c, "server_mean": math.nan, "client_mean": math.nan,
           "gap": math.nan, "server_share": math.nan, "client_share": math.nan, "share_shift": math.nan}
    if n_s > 0:
        out["server_mean"] = (s[-1] - s[0]) / n_s
        out["server_share"] = (b[-1] - b[0]) / n_s
    if n_c > 0:
        out["client_mean"] = float(cl[inside].mean())
        out["client_share"] = float((cl[inside] <= slo_s).mean())
    out["gap"] = out["client_mean"] - out["server_mean"]
    out["share_shift"] = out["server_share"] - out["client_share"]
    return out


def windows(run: dict, win_s: float = A5["win_s"], min_n: int = A5["min_n"]) -> pd.DataFrame:
    """Readout only. One row per window bounded by the scrapes at or before consecutive multiples of `win_s`,
    a window shorter than win_s / 2 (the run's first and last) dropped: a, b, n_server, n_client, server_mean,
    client_mean, gap, and `kept` (both counts >= min_n). A kept=False row has gap NaN."""
    t, s, c, _ = _series(run)
    ft, cl = _client(run)
    grid = np.arange(math.ceil(t[0] / win_s) * win_s, t[-1] + 1e-9, win_s)
    idx = np.unique(np.searchsorted(t, grid, side="right") - 1)
    idx = idx[idx >= 0]
    rows = []
    for i0, i1 in zip(idx[:-1], idx[1:]):
        a, bb = t[i0], t[i1]
        if bb - a < win_s / 2:
            continue
        n_s = int(c[i1] - c[i0])
        inside = (ft > a) & (ft <= bb)
        n_c = int(inside.sum())
        kept = n_s >= min_n and n_c >= min_n
        sm = (s[i1] - s[i0]) / n_s if n_s else math.nan
        cm = float(cl[inside].mean()) if n_c else math.nan
        rows.append({"a": a, "b": bb, "n_server": n_s, "n_client": n_c, "server_mean": sm, "client_mean": cm,
                     "gap": (cm - sm) if kept else math.nan, "kept": kept})
    return pd.DataFrame(rows, columns=["a", "b", "n_server", "n_client", "server_mean", "client_mean", "gap", "kept"])


def summary(name: str, run: dict) -> dict:
    """A run's line: the whole-span numbers, then the readout windows' median and max gap, where the max falls, the
    server mean's change from the previous kept window there, and how many windows were not kept."""
    w0 = whole(run)
    w = windows(run)
    k = w[w.kept]
    out = {"run": name, **w0, "windows": len(k), "unkept": int((~w.kept).sum()) if len(w) else 0,
           "win_gap_median": math.nan, "win_gap_max": math.nan, "win_max_at": math.nan, "win_slope_at_max": math.nan}
    if len(k):
        i = k.gap.to_numpy().argmax()
        out.update(win_gap_median=float(k.gap.median()), win_gap_max=float(k.gap.iloc[i]),
                   win_max_at=float(k.a.iloc[i]),
                   win_slope_at_max=float(k.server_mean.iloc[i] - k.server_mean.iloc[i - 1]) if i else math.nan)
    return out


def label(table: pd.DataFrame, flag_gap_s: float = A5["flag_gap_s"], flag_share: float = A5["flag_share"]) -> str:
    """REG-26's outcome, on the whole-span columns only. `no_data` if the table is empty; `flag` if any run has a
    gap above flag_gap_s, a share shift above flag_share, or either one NaN (a run whose two sides cannot be
    compared is not passed); `small` otherwise."""
    if not len(table):
        return "no_data"
    bad = (table.gap.isna() | table.share_shift.isna() | (table.gap > flag_gap_s) | (table.share_shift > flag_share))
    return "flag" if bad.any() else "small"


def run_dict(name: str) -> dict:
    d = load.RUNS / name
    run = load.load_run(d)            # refuses a sealed run (analysis/withheld.py)
    run["slo_bucket"] = _bucket_le(d / "metrics.jsonl", "1.0")
    return run


def report(runs: tuple = A5["runs"]) -> dict:
    tab = pd.DataFrame([summary(n, run_dict(n)) for n in runs])
    return {"label": label(tab), "table": tab}


if __name__ == "__main__":
    out = report()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 30)
    g = out["table"]
    print(g.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(json.dumps({"label": out["label"], "runs": len(g), "gap_max": float(g.gap.max()),
                      "gap_median": float(g.gap.median()), "share_shift_max": float(g.share_shift.max()),
                      "share_shift_median": float(g.share_shift.median()),
                      "win_gap_max": float(g.win_gap_max.max()), "flag_gap_s": A5["flag_gap_s"],
                      "flag_share": A5["flag_share"]}))
    sys.exit(0)
