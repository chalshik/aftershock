"""Reading M (docs/drafts/m_draft1.md; not registered; 0 GPU h): a later turn's prefix reuse against its idle age in
units of the free queue's turnover, in REG-19's five closed-loop burst runs and REG-20 arm A's five open-loop burst
runs, after the burst. It tests §5's sentence: "every later turn then waits longer than the free queue takes to turn
over, the analogue of an LRU cache's characteristic time (Che et al. 2002), and misses".

Per later turn k of a session (`b1.turns`: ok turns, not burst requests, previous turn ok):
  idle age  a = t_first_token[k] - t_end[k-1]   (s; the previous turn's blocks enter the free queue at t_end[k-1];
                                                  the lookup happens before the first token, so a is an upper bound
                                                  by the prefill time). a < 0: `held` (the prefix was still in use).
  x = A(t_end[k-1], t_first_token[k]) / F(t_end[k-1]), the idle age in units of the free queue's turnover: A is the
      number of newly allocated tokens over the interval, from the raw counters (prompt_tokens_total -
      prefix_cache_hits_total + generation_tokens_total, linearly interpolated between scrapes); F the free pool at
      the start, (1 - kv_cache_usage_perc) x C, C = 96,656 tokens (t4.T4["pool_blocks"] x 16); F <= 0 counts as
      x = inf; A < 0 (a >= 0; counter timing) is bucket `neg_A`, never `held`. Under LRU with no
      reuse from the free queue, a block freed at s is evicted once A > F(s): x > 1 means the prefix is past its
      characteristic time. A counter that goes down (a restart) makes the run undefined for M (as in every reading of a cumulative counter).
  reuse     min(cached_tokens, prev_prompt) / prev_prompt, token-weighted (as `k1.reuse`).
Window: turns with t_end[k-1] >= burst end and t_first_token[k] < burst end + watch_s. Buckets of x: `held`,
[0, .5), [.5, 1), [1, 2), [2, 4), [4, 8), [8, 16), [16, inf). Reuse is pooled per arm and bucket over the arm's runs
(assumption, stated: within an arm, seeds are exchangeable given x; per-run values are reported beside it). Readout,
not judged: t_C = F / (dA/dt) per 60 s span; reuse by a in seconds (`profile_seconds`); REG-19's no-burst controls on their partner's window (x < 1 expected).

Labels, first match (`label`; thresholds set here, before any M number is computed, rule A5):
  undefined          fewer than M["min_runs"] runs per arm with defined inputs;
  closed_not_outside fewer than M["outside_share"] of the closed arm's later turns at x >= 1, or their pooled reuse
                     > M["outside_reuse"] (from seen readouts both are expected to pass: this label tests F's
                     measurement, not §5's premise);
  arm_differs        a matched bucket where the arms' reuse differs by >= M["diff"]; a bucket is matched with
                     >= min_turns turns in each arm and >= min_spread open runs holding >= min_run_turns of them;
  separated          no matched bucket;
  common_curve       matched buckets all within diff, and the open arm's [1, 2) bucket (matched-size, spread) reuses
                     >= M["fall"] less than its [0, .5): a crossing at x in [1, 2);
  agree_outside      matched buckets all within diff and all at x >= 2 (both arms miss alike past the turnover time;
                     no crossing is tested);
  mixed              otherwise.
Readouts (never judged): per-run k of n with Clopper-Pearson bounds (`per_run`), censored turns (`censored`), f_zero
counts, t_C per span, reuse by a in seconds, reuse by x per 60 s span inside the open drain (`drain_profile`), REG-19's
no-burst controls. M does not separate K1's working-set alternative: x >> 1 with no reuse follows from any cause of
the cold state (rule F r1, section 2).
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from analysis import b1, k1, t4, withheld
from analysis.gate import burst_end
from analysis.margin import is_burst_request

M = {
    "arms": {"closed": b1.ARMS["uncapped"],                       # REG-19's five burst runs
             "open": "*_burst_l1pl350ol_r2.0_w3000_s1[5-9]"},     # REG-20 arm A's five (k1.open_pattern at 2.0)
    "pool_tokens": t4.T4["pool_blocks"] * 16,                    # 96,656
    "edges": (0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, math.inf),
    "min_runs": 4,           # of 5 per arm
    # per bucket and arm, for a bucket to be matched: B1's min_turns (FILE analysis/b1.py:33), kept after rule F r1
    # derived 75-310 pooled open turns at x >= 16 (moving it either way would pick the outcome; it is set here, before
    # any M number, rule A5). With it, matched-or-not mostly measures the open drain's size (the [REG] says so).
    "min_turns": 100,
    "min_run_turns": 20,     # = min_turns / 5: a run "holds" a bucket with at least this many turns (rule F r1 item 5)
    "min_spread": 3,         # a matched bucket needs >= 3 of 5 open runs holding it (rule F r1 item 5)
    "outside_share": 0.9,    # of the closed arm's later turns at x >= 1
    "outside_reuse": 0.1,    # their pooled reuse at most this (REG-19's burst runs: 0.000, B1 `no_variance`)
    # |reuse difference| in a matched bucket. Basis: above the largest reuse change with age seen inside one cold
    # state, B1's 0.25-0.28 at gaps < 1 s against 0.000-0.003 at 5-25 s (decisions.md 2026-11-17 [RESULT] B1); a
    # difference this size at matched x is not residual age spread within a bucket.
    "diff": 0.3,
    "fall": 0.5,             # open arm's drop from [0, .5) to [1, 2) (rule F r1 item 2: at the crossing, not pooled)
    "drain_s": 600.0,        # drain-span readout: [e, e + 600), k1.REG20's mechanism window length
    "span_s": 60.0,          # t_C readout span
}
NEW = ("vllm:prompt_tokens_total", "vllm:prefix_cache_hits_total", "vllm:generation_tokens_total")
KV = "vllm:kv_cache_usage_perc"
LABELS = ("undefined", "closed_not_outside", "arm_differs", "separated", "common_curve", "agree_outside", "mixed")
REG_HEADING = "[REG] M (idle age against the turnover time)"
X_BUCKETS = tuple(f"[{a:g},{b:g})" for a, b in zip(M["edges"][:-1], M["edges"][1:]))
BUCKETS = ("held", "neg_A") + X_BUCKETS


# ---------------------------------------------------------------- per run

def allocated(run: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """(t, cumulative newly allocated tokens) at the scrapes, or None if a counter is missing or goes down."""
    m = run["metrics"]
    if any(c not in m for c in NEW) or KV not in m:
        return None
    m = m.dropna(subset=list(NEW))
    p, h, g = (m[c].to_numpy(float) for c in NEW)
    for c in (p, h, g):
        if len(c) < 2 or np.any(np.diff(c) < 0):
            return None
    return m.t.to_numpy(float), p - h + g


def ages(run: dict) -> pd.DataFrame | None:
    """The run's later turns in the window with a, F, A, x and their bucket; None when the inputs are undefined.
    Every turn in the window gets a bucket (none is dropped): a < 0 `held`; A < 0 (a >= 0) `neg_A` (counter timing,
    rule F r1 T3); F <= 0 counts as x = inf, in [16, inf), flagged `f_zero`."""
    al = allocated(run)
    if al is None:
        return None
    t, cum = al
    m = run["metrics"].dropna(subset=[KV])
    e = burst_end(run)
    q = b1.turns(run)
    q = q[(q.prev_end >= e) & (q.t_first_token < e + float(run["cfg"]["verdict"]["watch_s"]))].copy()
    q["run"] = run["dir"].name
    q["a"] = q.t_first_token - q.prev_end
    q["F"] = (1 - np.interp(q.prev_end, m.t.to_numpy(float), m[KV].to_numpy(float))) * M["pool_tokens"]
    q["A"] = np.interp(q.t_first_token, t, cum) - np.interp(q.prev_end, t, cum)
    q["f_zero"] = q.F <= 0
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.where(q.f_zero, math.inf, q.A / q.F.where(q.F > 0))
    q["x"] = np.where(q.a < 0, np.nan, x)
    q["bucket"] = bucket(q.a.to_numpy(float), q.x.to_numpy(float))
    return q


def bucket(a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """`held` (a < 0), `neg_A` (a >= 0, x < 0), else the x bucket; x = inf goes to [16, inf)."""
    out = np.full(len(a), None, dtype=object)
    out[a < 0] = "held"
    neg = (a >= 0) & (x < 0)
    out[neg] = "neg_A"
    ok = (a >= 0) & (x >= 0)
    idx = np.searchsorted(np.array(M["edges"]), x[ok], side="right") - 1
    idx = np.minimum(idx, len(M["edges"]) - 2)                   # x = inf into the last bucket
    out[ok] = [X_BUCKETS[i] for i in idx]
    return out


def censored(run: dict) -> dict:
    """Rule F r1 5c, a readout: later turns whose previous turn ended in the window but which have no ok first token
    inside it (still waiting at the run's end, or not ok). These are the longest waits; they are not in `ages`."""
    r = run["requests"]
    r = r[~is_burst_request(run)].sort_values(["session_id", "turn"]).copy()
    g = r.groupby("session_id")
    r["prev_end"], r["prev_turn"] = g.t_end.shift(), g.turn.shift()
    e = burst_end(run)
    end = e + float(run["cfg"]["verdict"]["watch_s"])
    w = r[(r.turn > 0) & (r.prev_turn == r.turn - 1) & (r.prev_end >= e) & (r.prev_end < end)]
    inside = (w.status == "ok") & (w.t_first_token < end)
    return {"in_window": int(len(w)), "censored": int((~inside).sum())}


def turnover(run: dict) -> list[dict]:
    """Readout: t_C = F / allocation rate per span after the burst (NaN when nothing was allocated)."""
    al = allocated(run)
    if al is None:
        return []
    t, cum = al
    m = run["metrics"].dropna(subset=[KV])
    e = burst_end(run)
    out = []
    for a in np.arange(e, e + float(run["cfg"]["verdict"]["watch_s"]), M["span_s"]):
        b = a + M["span_s"]
        rate = (np.interp(b, t, cum) - np.interp(a, t, cum)) / M["span_s"]
        w = (m.t >= a) & (m.t < b)
        f = (1 - float(m[KV][w].mean())) * M["pool_tokens"] if w.any() else math.nan
        out.append({"a": float(a), "free": f, "rate": rate, "t_c": f / rate if rate > 0 else math.nan})
    return out


def drain_profile(q: pd.DataFrame, e: float) -> list[dict]:
    """Rule F r1 item 6, a readout never judged: within one open run's drain [e, e + drain_s), reuse by x bucket
    per 60 s span (x varies there at a nearly fixed population, the only place M can speak to the working-set
    alternative)."""
    out = []
    for a in np.arange(e, e + M["drain_s"], M["span_s"]):
        s = q[(q.t_first_token >= a) & (q.t_first_token < a + M["span_s"])]
        out.append({"a": float(a), "buckets": {bk: _pool(s, s.bucket == bk) for bk in BUCKETS
                                               if (s.bucket == bk).any()}})
    return out


# ---------------------------------------------------------------- per arm, and the label

def _pool(q: pd.DataFrame, mask) -> dict:
    s = q[mask]
    den = float(s.prev_prompt.sum())
    return {"n": int(len(s)), "reuse": float(s.r_num.sum()) / den if den > 0 else math.nan}


def profile(qs: list[pd.DataFrame]) -> dict:
    """Pooled turns and token-weighted reuse per bucket, and the count of f_zero turns."""
    q = pd.concat(qs) if qs else pd.DataFrame(columns=["bucket", "r_num", "prev_prompt", "f_zero"])
    out = {bk: _pool(q, q.bucket == bk) for bk in BUCKETS}
    out["f_zero"] = int(q.f_zero.sum()) if len(q) else 0
    return out


def profile_seconds(qs: list[pd.DataFrame]) -> dict:
    """Readout: pooled reuse by idle age a in seconds, on b1's profile edges (B1's own binning)."""
    q = pd.concat(qs) if qs else pd.DataFrame(columns=["a", "r_num", "prev_prompt"])
    e = b1.B1["profile_edges"]
    return {f"[{lo:g},{hi:g})": _pool(q, (q.a >= lo) & (q.a < hi)) for lo, hi in zip(e[:-1], e[1:])}


def _spread(q: pd.DataFrame, bk: str) -> int:
    """How many runs contribute at least min_run_turns turns to bucket bk."""
    c = q[q.bucket == bk].groupby("run").size()
    return int((c >= M["min_run_turns"]).sum())


def matched_buckets(qc: pd.DataFrame, qo: pd.DataFrame) -> list[str]:
    """x buckets with >= min_turns pooled turns in each arm and >= min_spread open runs each holding >= min_run_turns
    of them (rule F r1 item 5: one large-shock seed cannot carry a match alone)."""
    pc, po = profile([qc]), profile([qo])
    return [bk for bk in X_BUCKETS if pc[bk]["n"] >= M["min_turns"] and po[bk]["n"] >= M["min_turns"]
            and _spread(qo, bk) >= M["min_spread"]]


def per_run(closed: list[pd.DataFrame], open_: list[pd.DataFrame], diffs: dict) -> dict:
    """Rule H readouts: k of n closed runs with share at x >= 1 >= outside_share; per matched bucket, k of n open runs
    (with >= min_run_turns there) whose reuse differs from the pooled closed reuse in the pooled difference's
    direction. Each with its one-sided 95 % Clopper-Pearson bounds (lower = 1 - upper of the complement)."""
    from analysis.stats import cp_upper

    def kn(k, n):
        return {"k": k, "n": n, "upper": cp_upper(k, n) if n else None, "lower": 1 - cp_upper(n - k, n) if n else None}
    k_out = sum((q.x >= 1).mean() >= M["outside_share"] for q in closed if len(q))
    pc = profile(closed)
    sign = {}
    for bk, d in diffs.items():
        rs = [profile([q])[bk] for q in open_ if (q.bucket == bk).sum() >= M["min_run_turns"]]
        sign[bk] = kn(sum(np.sign(r["reuse"] - pc[bk]["reuse"]) == np.sign(d) for r in rs), len(rs))
    return {"closed_outside": kn(int(k_out), len([q for q in closed if len(q)])), "matched_sign": sign}


def label(closed: list[pd.DataFrame], open_: list[pd.DataFrame]) -> dict:
    if len(closed) < M["min_runs"] or len(open_) < M["min_runs"]:
        return {"label": "undefined", "n_runs": {"closed": len(closed), "open": len(open_)}}
    qc, qo = pd.concat(closed), pd.concat(open_)
    pc, po = profile(closed), profile(open_)
    out_c = _pool(qc, qc.x >= 1)
    share = out_c["n"] / len(qc) if len(qc) else 0.0
    res = {"n_runs": {"closed": len(closed), "open": len(open_)}, "closed_outside": {**out_c, "share": share},
           "profile": {"closed": pc, "open": po}}
    if share < M["outside_share"] or not out_c["reuse"] <= M["outside_reuse"]:
        return {"label": "closed_not_outside", **res, "per_run": per_run(closed, open_, {})}
    matched = matched_buckets(qc, qo)
    diffs = {bk: po[bk]["reuse"] - pc[bk]["reuse"] for bk in matched}
    res["matched"], res["per_run"] = diffs, per_run(closed, open_, diffs)
    if any(abs(d) >= M["diff"] for d in diffs.values()):
        return {"label": "arm_differs", **res}
    if not matched:
        return {"label": "separated", **res}
    lo, cross = po["[0,0.5)"], po["[1,2)"]
    res["open_fall"] = {"low": lo, "cross": cross, "cross_spread": _spread(qo, "[1,2)")}
    if (lo["n"] >= M["min_turns"] and cross["n"] >= M["min_turns"] and _spread(qo, "[1,2)") >= M["min_spread"]
            and lo["reuse"] - cross["reuse"] >= M["fall"]):
        return {"label": "common_curve", **res}
    if all(M["edges"][X_BUCKETS.index(bk)] >= 2.0 for bk in matched):
        return {"label": "agree_outside", **res}
    return {"label": "mixed", **res}


# BELIEF (docs/drafts/m_draft1.md, Prediction, re-derived after rule F round 1): P(a matched bucket) ≈ 0.46, set by
# the open drain's size (75-310 pooled open turns at x >= 16 against min_turns 100)
PREDICTED = {"undefined": 0.03, "closed_not_outside": 0.05, "arm_differs": 0.12, "common_curve": 0.03,
             "agree_outside": 0.28, "separated": 0.46, "mixed": 0.03}
# ---------------------------------------------------------------- the registered computation

def _registered() -> bool:
    return withheld.lifted(REG_HEADING)


def table() -> dict:
    """The registered computation, once: refuses until decisions.md holds the [REG] M heading."""
    if not _registered():
        raise RuntimeError(f"M is not registered: decisions.md has no heading '{REG_HEADING}' (computed only after it)")
    from analysis.load import find_runs
    arms, per_run, tc = {}, {}, {}
    for arm, pat in M["arms"].items():
        qs = []
        for r in find_runs(pat):
            q = ages(r)
            name = r["dir"].name
            if q is None:
                per_run[name] = {"arm": arm, "defined": False}
                continue
            qs.append(q)
            per_run[name] = {"arm": arm, "defined": True, "profile": profile([q]), "censored": censored(r)}
            if arm == "open":
                per_run[name]["drain"] = drain_profile(q, burst_end(r))
            tc[name] = turnover(r)
        arms[arm] = qs
    # readout, not judged: REG-19's no-burst controls, read on their burst partner's window (as b1), expected x < 1
    ctl = {}
    for r in find_runs(b1.ARMS["control"]):
        seed = r["dir"].name.rsplit("_s", 1)[1]
        partner = find_runs(b1.BURST_OF_CONTROL.format(seed=seed))
        q = ages({**r, "cfg": partner[0]["cfg"]}) if len(partner) == 1 else None
        ctl[r["dir"].name] = profile([q]) if q is not None else None
    return {"label": label(arms["closed"], arms["open"]), "runs": per_run, "turnover": tc, "controls": ctl,
            "seconds": {arm: profile_seconds(qs) for arm, qs in arms.items()}}


if __name__ == "__main__":
    import json
    print(json.dumps(table(), indent=1, default=str))
