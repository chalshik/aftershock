"""B1 (C2, draft): is the per-turn loss of a prefix set by the drawn think gap, on the free queue's clock?

A reading of runs already reported, retrospective by design (author, decisions.md 2026-11-17 [DECISION] (author) C1's
order clause ... C2 goes on with B1). Its statistic is fixed here before it is computed on any run; `table` refuses
until decisions.md holds the heading `[REG] B1`.

Per later session turn k (ok, previous turn present, first token in the watch W = [burst end, burst end + watch)):
  gap   g = t_planned[k] - t_planned[k-1]: the schedule's own drawn think gap. With closed-loop turns the harness
          releases turn k at the previous turn's end plus exactly this gap (FILE(harness/loadgen.py:150-178)), so g is
          fixed by the seed, not by the server.
  reuse r = min(cached_tokens, previous prompt) / previous prompt (the graded share of `k1.reuse`; the recorded reply
          is not cached, so the reusable prefix is the previous prompt).
Within each 300 s span of W (by first token), turns with g < short_s and turns with long_s < g <= long_max_s are
compared, token-weighted; the spans are combined with Mantel-Haenszel weights n_s * n_l / (n_s + n_l), so a span's
overall state (cold or warm) cannot make the comparison by itself. The ratio is short over long; a cluster bootstrap
over sessions gives its bounds.
"""
import math

import numpy as np
import pandas as pd

from analysis import withheld
from analysis.gate import burst_end
from analysis.k1 import later_turns
from analysis.margin import is_burst_request

B1 = {
    "short_s": 1.0,          # the cold clock: R_M = 0.05 * 96,656 / 5,164.76 = 0.94 s (DERIVED, report_fable.md B)
    "long_s": 5.0,           # the mean drawn gap
    "long_max_s": 25.0,      # the warm prediction is flat up to 25 s
    "span_s": 300.0,         # D1's span length
    "min_turns": 100,        # per bucket over W, else too_few
    "no_var_reuse": 0.01,    # both buckets below this: no_variance (all misses)
    "ratio_gap": 2.0,        # gap_bound: ratio at least this ...
    "lb_gap": 1.0,           # ... and its one-sided 95 % lower bound above this
    # flat: the one-sided 95 % upper bound of the ratio below ratio_gap (rule F r1, item 1: the effect gap_bound
    # needs is excluded; a [0.8, 1.25] band had no power at ~900 short turns and reuse ~0.1)
    "miss_tokens": 16,       # a full miss: cached_tokens below one block (readout only)
    "boot_n": 2000,
    "boot_seed": 20261117,
    "profile_edges": [0.0, 1.0, 2.0, 5.0, 10.0, 25.0, math.inf],
}

# Run patterns. Judged: D1b's cap-40 runs and REG-19's no-burst controls. Reported only: D1's capped runs and
# REG-19's uncapped burst runs (reuse 0.00 there: no variance to read).
ARMS = {
    "cold": "*_burst_l1pl350cl_r2.0_w3000_s1[5-7]_cap40",
    "control": "*_stage2_control_l1pl350cl_r2.0_w3000_s1[5-9]",
    "capped": "*_burst_l1pl350cl_r2.0_w3000_s1[5-9]_cap",
    "uncapped": "*_burst_l1pl350cl_r2.0_w3000_s1[5-9]",
}
BURST_OF_CONTROL = "*_burst_l1pl350cl_r2.0_w3000_s{seed}"   # a control is read over its burst partner's window
REG_HEADING = "[REG] B1"


def turns(run: dict) -> pd.DataFrame:
    """Later turns with their drawn gap g and graded reuse r (no window applied)."""
    q = later_turns(run).copy()
    nb = run["requests"][~is_burst_request(run)].sort_values(["session_id", "turn"])
    q["gap"] = q.t_planned - nb.groupby("session_id").t_planned.shift().reindex(q.index)
    q["r_num"] = np.minimum(q.cached_tokens, q.prev_prompt)
    q["gap_sent"] = q.t_sent - q.prev_end            # readout: should equal gap (release mechanics)
    q["wait_ttft"] = q.t_first_token - q.t_sent      # readout: server wait + prefill, per bucket (a hit shortens
    #                                                    the prefill, so it is also read among full misses only)
    return q[q.prev_prompt > 0]


def window(run: dict, partner: dict | None = None) -> tuple[float, float]:
    src = partner or run
    a = burst_end(src)
    return a, a + float(src["cfg"]["verdict"]["watch_s"])


COLS = ["n_s", "n_l", "num_s", "den_s", "num_l", "den_l"]


def _cells(q: pd.DataFrame, a: float, b: float, p: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per session and span, the six sums of COLS for turns with first token in [a, b): an array
    (sessions, spans, 6) and the session ids. The bootstrap resamples its first axis."""
    q = q[(q.t_first_token >= a) & (q.t_first_token < b)]
    n_sp = max(1, int(math.ceil((b - a) / p["span_s"])))
    sess = np.array(sorted(q.session_id.unique()))
    arr = np.zeros((len(sess), n_sp, len(COLS)))
    if not len(q):
        return arr, sess
    si = np.searchsorted(sess, q.session_id.to_numpy())
    sp = np.minimum(((q.t_first_token.to_numpy() - a) // p["span_s"]).astype(int), n_sp - 1)
    short = (q.gap < p["short_s"]).to_numpy()
    long_ = ((q.gap > p["long_s"]) & (q.gap <= p["long_max_s"])).to_numpy()
    num, den = q.r_num.to_numpy(dtype=float), q.prev_prompt.to_numpy(dtype=float)
    for k, (m, v) in enumerate([(short, 1.0), (long_, 1.0), (short, num), (short, den), (long_, num), (long_, den)]):
        np.add.at(arr[:, :, k], (si[m], sp[m]), (v[m] if isinstance(v, np.ndarray) else v))
    return arr, sess


def ratio(st: np.ndarray) -> dict:
    """st: (spans, 6) sums. Mantel-Haenszel-weighted short and long reuse over spans holding both buckets."""
    n_s, n_l, num_s, den_s, num_l, den_l = (st[:, k] for k in range(6))
    m = (n_s > 0) & (n_l > 0) & (den_s > 0) & (den_l > 0)
    if not m.any():
        return {"short": math.nan, "long": math.nan, "ratio": math.nan}
    w = n_s[m] * n_l[m] / (n_s[m] + n_l[m])
    rs = float((w * num_s[m] / den_s[m]).sum() / w.sum())
    rl = float((w * num_l[m] / den_l[m]).sum() / w.sum())
    return {"short": rs, "long": rl, "ratio": rs / rl if rl > 0 else (math.nan if rs == 0 else math.inf)}


def _boot(arr: np.ndarray, p: dict) -> np.ndarray:
    """Cluster bootstrap over sessions: each draw weights every session by how often it was picked."""
    rng = np.random.default_rng(p["boot_seed"])
    n = arr.shape[0]
    out = np.empty(p["boot_n"])
    for i in range(p["boot_n"]):
        wts = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        out[i] = ratio(np.tensordot(wts, arr, axes=1))["ratio"]
    return out


def label(est: dict, n_s: int, n_l: int, boot: np.ndarray, p: dict = B1) -> str:
    """too_few, no_variance, gap_bound, flat or unclear (in that order). gap_bound: ratio >= ratio_gap and the
    one-sided 95 % lower bound > lb_gap; flat: the one-sided 95 % upper bound < ratio_gap."""
    if n_s < p["min_turns"] or n_l < p["min_turns"]:
        return "too_few"
    if est["short"] < p["no_var_reuse"] and est["long"] < p["no_var_reuse"]:
        return "no_variance"
    b = boot[~np.isnan(boot)]   # an infinite ratio (long reuse 0) stays: it counts as above any bound
    if not len(b) or math.isnan(est["ratio"]):
        return "unclear"
    lb95 = float(np.quantile(b, 0.05, method="inverted_cdf"))
    ub95 = float(np.quantile(b, 0.95, method="inverted_cdf"))
    if est["ratio"] >= p["ratio_gap"] and lb95 > p["lb_gap"]:
        return "gap_bound"
    if ub95 < p["ratio_gap"]:
        return "flat"
    return "unclear"


def profile(q: pd.DataFrame, a: float, b: float, p: dict = B1) -> list[dict]:
    """Reported, never judged: by gap bucket, token-weighted reuse, turn count, mean t_first_token - t_sent, and the
    same among full misses (cached_tokens < miss_tokens). Buckets are [lo, hi) below long_s and (lo, hi] from long_s
    on, so (5, 10] and (10, 25] together are the judged long bucket."""
    q = q[(q.t_first_token >= a) & (q.t_first_token < b)]
    e = p["profile_edges"]
    out = []
    for lo, hi in zip(e[:-1], e[1:]):
        m = q[(q.gap > lo) & (q.gap <= hi)] if lo >= p["long_s"] else q[(q.gap >= lo) & (q.gap < hi)]
        miss = m[m.cached_tokens < p["miss_tokens"]]
        out.append({"gap": (lo, hi), "n": len(m),
                    "reuse": float(m.r_num.sum() / m.prev_prompt.sum()) if len(m) else math.nan,
                    "wait_ttft": float(m.wait_ttft.mean()) if len(m) else math.nan,
                    "n_miss": len(miss), "wait_ttft_miss": float(miss.wait_ttft.mean()) if len(miss) else math.nan})
    return out


def row(run: dict, partner: dict | None = None, p: dict = B1) -> dict:
    a, b = window(run, partner)
    q = turns(run)
    arr, _ = _cells(q, a, b, p)
    st = arr.sum(axis=0)
    est = ratio(st)
    n_s, n_l = int(st[:, 0].sum()), int(st[:, 1].sum())
    boot = _boot(arr, p) if min(n_s, n_l) >= p["min_turns"] else np.array([])
    ok = boot[~np.isnan(boot)]   # inf kept: no interpolation in the quantiles
    qw = q[(q.t_first_token >= a) & (q.t_first_token < b)]
    return {"run": run["dir"].name, "W": (a, b), "n_short": n_s, "n_long": n_l, **est,
            "lb95": float(np.quantile(ok, 0.05, method="inverted_cdf")) if len(ok) else math.nan,
            "ci90": (float(np.quantile(ok, 0.05, method="inverted_cdf")), float(np.quantile(ok, 0.95, method="inverted_cdf"))) if len(ok) else (math.nan, math.nan),
            "label": label(est, n_s, n_l, boot, p), "profile": profile(q, a, b, p),
            "gap_sent_minus_gap": float((qw.gap_sent - qw.gap).median()) if len(qw) else math.nan,
            "residency": residency(run, a, b), "queue_dist": queue_dist(run, a, b)}


def queue_dist(run: dict, a: float, b: float, max_edge: float = 5.0) -> dict:
    """Reported, never judged (rule F r1, item 6): the share of requests whose queue time over W fell at or below each
    histogram edge up to max_edge s, from the raw `vllm:request_queue_time_seconds` buckets (first scrape at or after
    a to the last before b). Empty where not recorded or where a bucket went down (a restart)."""
    import json
    from analysis.c2 import _buckets
    path = run["dir"] / "metrics.jsonl"
    if not path.exists():
        return {}
    first = last = None
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            if "values" not in s or s["t"] < a:
                continue
            if s["t"] >= b:
                break
            cur = _buckets(s["values"], "vllm:request_queue_time_seconds")
            if last and any(cur.get(k, 0.0) < v for k, v in last.items()):
                return {}
            first = cur if first is None else first
            last = cur
    if not first or not last:
        return {}
    d = {k: last[k] - first.get(k, 0.0) for k in last}
    tot = d.get(math.inf, 0.0)
    return {k: v / tot for k, v in sorted(d.items()) if k <= max_edge} if tot > 0 else {}


def residency(run: dict, a: float, b: float) -> dict:
    """Reported, never judged: mean sampled idle-before-evict, queue time and inference time over W (from the raw
    metrics.jsonl histograms' _sum and _count; NaN where not recorded)."""
    from analysis.order import _hist_sum_count
    path = run["dir"] / "metrics.jsonl"
    out = {}
    for k, name in (("idle_before_evict_s", "vllm:kv_block_idle_before_evict_seconds"),
                    ("queue_s", "vllm:request_queue_time_seconds"),
                    ("inference_s", "vllm:request_inference_time_seconds")):
        s, n = _hist_sum_count(path, name, a, b) if path.exists() else (math.nan, math.nan)
        out[k] = s / n if not math.isnan(n) and n > 0 else math.nan
    return out


def outcome(cold: list[str], control: list[str]) -> dict:
    """Judged over D1b's cold runs and REG-19's controls (labels of `label`)."""
    usable = lambda ls: [x for x in ls if x not in ("too_few", "no_variance")]
    c, k = usable(cold), usable(control)
    gb, fl = cold.count("gap_bound"), cold.count("flat")
    cf, cg = control.count("flat"), control.count("gap_bound")
    if len(c) < 2 or len(k) < 4:
        d = "undefined"
    elif gb >= 2 and cf >= 4:
        d = "supports_aging"
    elif gb >= 2 and cg >= 2:
        d = "not_specific"
    elif fl >= 2:
        d = "flat_cold"
    else:
        d = "mixed"
    return {"decision": d, "cold_gap_bound": gb, "cold_flat": fl, "n_cold": len(c),
            "control_flat": cf, "control_gap_bound": cg, "n_control": len(k)}


def _registered() -> bool:
    return withheld.lifted(REG_HEADING)


def table() -> dict:
    """The registered computation, once: refuses until decisions.md holds `[REG] B1`."""
    if not _registered():
        raise RuntimeError(f"B1 is not registered: decisions.md has no heading '{REG_HEADING}' (computed only after it)")
    from analysis.load import find_runs
    rows = {k: [row(r) for r in find_runs(pat)] for k, pat in ARMS.items() if k != "control"}
    ctl = []
    for r in find_runs(ARMS["control"]):
        seed = r["dir"].name.rsplit("_s", 1)[1]
        partner = find_runs(BURST_OF_CONTROL.format(seed=seed))
        if len(partner) != 1:   # rule F r1, item 8: no window without exactly one valid partner
            ctl.append({"run": r["dir"].name, "label": "too_few", "reason": f"{len(partner)} valid burst partners"})
            continue
        ctl.append(row(r, partner[0]))
    rows["control"] = ctl
    return {"rows": rows, "outcome": outcome([x["label"] for x in rows["cold"]], [x["label"] for x in ctl]),
            "params": {k: v for k, v in B1.items() if k != "profile_edges"}}


if __name__ == "__main__":
    import json
    import sys
    if sys.argv[1:] != ["table"]:
        sys.exit("usage: python -m analysis.b1 table")
    print(json.dumps(table(), indent=1, default=str))
