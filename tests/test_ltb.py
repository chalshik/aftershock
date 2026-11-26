"""analysis.ltb: the post-hoc load-test readouts (author 2026-11-26, no-run route). Synthetic runs only."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import ltb, uniform

NAN = math.nan


def _run(rows, metrics=None, burst_tokens=2944):
    q = pd.DataFrame(rows, columns=["session_id", "turn", "t_planned", "t_released", "t_sent", "t_first_token",
                                    "t_end", "status", "prompt_tokens_planned"])
    cfg = {"slo": {"ttft_s": 1.0}, "workload": {"warmup_s": 0, "duration_s": 100},
           "trigger": {"kind": "burst", "at_s": 10, "duration_s": 5, "prompt_tokens": burst_tokens}}
    m = pd.DataFrame(metrics if metrics is not None else {"t": [0.0, 50.0, 100.0]})
    return {"dir": Path("x"), "cfg": cfg, "requests": q, "metrics": m}


def test_nearest_rank_keeps_inf_and_does_not_interpolate():
    assert ltb.nearest_rank([1, 2, 3, 4], 50) == 2
    assert ltb.nearest_rank([1, 2, 3, 4], 99) == 4
    assert ltb.nearest_rank([1, 2, math.inf, math.inf], 50) == 2
    assert ltb.nearest_rank([1, math.inf, math.inf, math.inf], 50) == math.inf
    assert math.isnan(ltb.nearest_rank([], 50))
    assert ltb.nearest_rank([5.0], 0) == 5.0


def test_open_loop_release_is_planned_time():
    r = _run([(0, 0, 1.0, None, 1.0, 1.2, 2.0, "ok", 2176), (0, 1, 3.0, None, 3.0, 3.5, 4.0, "ok", 2432)])
    lat = ltb.latency(r, 0, 10)
    assert lat["scheduled"]["n"] == lat["release"]["n"] == 2
    assert lat["scheduled"]["p50"] == pytest.approx(lat["release"]["p50"])
    assert lat["release"]["goodput"] == pytest.approx(2 / 10)


def test_closed_loop_origins_differ_and_misses_are_inf():
    # turn 1 planned at 3 but released at 6 (previous reply late); turn 2 planned at 5, never released
    r = _run([(0, 0, 1.0, 1.0, 1.0, 1.5, 5.5, "ok", 2176),
              (0, 1, 3.0, 6.0, 6.0, 6.4, 8.0, "ok", 2432),
              (0, 2, 5.0, None, None, None, None, "unfinished", 2688)])
    lat = ltb.latency(r, 0, 10)
    s, rel = lat["scheduled"], lat["release"]
    assert s["n"] == 3 and s["no_first_token"] == 1 and s["p99"] == math.inf
    assert s["goodput"] == pytest.approx(1 / 10)          # turn 1 is 3.4 s behind its schedule: a miss
    assert rel["n"] == 2                                  # the unreleased turn is in no release window
    assert rel["goodput"] == pytest.approx(2 / 10) and rel["p99"] == pytest.approx(0.5)


def test_an_unfinished_request_with_a_first_token_is_a_miss_for_goodput():
    r = _run([(0, 0, 1.0, None, 1.0, 1.2, None, "unfinished", 2176)])
    lat = ltb.latency(r, 0, 10)["scheduled"]
    assert lat["goodput"] == 0 and lat["p50"] == pytest.approx(0.2)


def test_send_lag_uses_release_time():
    r = _run([(0, 0, 1.0, 1.0, 1.01, 1.2, 2.0, "ok", 2176), (0, 1, 3.0, 6.0, 6.5, 6.6, 8.0, "ok", 2432)])
    s = ltb.send_lag(r, 0, 10)
    assert s["n"] == 2 and s["max"] == pytest.approx(0.5) and s["over_0.05s"] == 1 and s["over_1s"] == 0


def test_planned_backlog_splits_exactly_and_excludes_burst():
    r = _run([(0, 0, 1.0, 1.0, 1.0, 1.5, 9.0, "ok", 2176),          # at the server at t = 8
              (0, 1, 3.0, None, None, None, None, "unfinished", 2432),  # planned, not sent: at the client
              (1, 0, 7.0, 7.0, 7.0, 7.2, 7.9, "ok", 2176),           # finished before t
              (2, 0, 12.0, 12.0, 12.0, 12.1, 13.0, "ok", 2176),      # planned after t
              (9, 0, 6.0, 6.0, 6.0, None, None, "unfinished", 2944)])  # a burst request at the server
    b = ltb.planned_backlog(r, 8.0)
    assert b["planned_backlog"] == 2 and b["at_client"] == 1 and b["at_server"] == 1
    assert b["planned_backlog"] == b["at_client"] + b["at_server"]
    assert b["burst_at_server"] == 1 and b["owed_backlog"] == 3


def test_preemptions_from_counter_differences_and_reset_flagged():
    rows = [(0, 0, 1.0, None, 1.0, 1.2, 20.0, "ok", 2176)]
    r = _run(rows, {"t": [0.0, 10.0, 20.0, 30.0], "vllm:num_preemptions_total": [0.0, 2.0, 6.0, 6.0]})
    p = ltb.preemptions(r, 5.0, 25.0)
    assert p["preemptions"] == pytest.approx(5.0) and p["completed"] == 1
    r = _run(rows, {"t": [0.0, 10.0, 20.0], "vllm:num_preemptions_total": [0.0, 4.0, 1.0]})
    assert ltb.preemptions(r, 0.0, 20.0)["preemptions"] is None


def test_cells_are_uniforms():
    """The readout covers exactly the post-hoc readout's cells (one cell list for both)."""
    assert ltb.uniform.CELLS is uniform.CELLS and len(uniform.CELLS) == 10


def test_summary_gives_ranges_over_seeds():
    def row(p50, pre):
        lat = {"n": 10, "no_first_token": 0, "goodput": 1.0, "fast_share": 1.0, "p50": p50, "p95": p50, "p99": p50}
        return {"latency": {w: {"scheduled": lat, "release": lat} for w in ("base", "end")},
                "send_lag": {"measured": {"p99": 0.002, "max": 0.2, "over_1s": 0}},
                "backlog": {t: {"planned_backlog": 5, "at_client": 2, "at_server": 3, "burst_at_server": 0,
                                "server": {"waiting": 1.0, "running": 2.0}} for t in ("burst_end", "end_end")}
                | {"end_mean": 4.0},
                "preempt": {"whole": {"preemptions": pre, "per_completion": 0.01},
                            "end": {"preemptions": None, "per_completion": math.nan}}}
    s = ltb.summary({"cells": [{"cell": "c", "rows": [{"burst": row(0.1, 3.0)}, {"burst": row(0.3, 7.0)}]}]})
    b = s["c"]["burst"]
    assert b["ttft_release"]["end"]["p50"] == [0.1, 0.3] and b["preempt"]["whole"] == [3.0, 7.0]
    assert b["preempt"]["end"] is None and b["send_lag_max"] == 0.2 and "control" not in s["c"]


def test_overlap_profile_counts_held_turns_by_span():
    # session 0: turn 1 starts before turn 0 ends (held), in the baseline; session 1: turn 1 after (not held), after
    # the burst. Burst at 10 s for 5 s, watch 300 s.
    rows = [(0, 0, 4.0, None, 4.0, 4.2, 9.5, "ok", 2176), (0, 1, 8.0, None, 8.0, 8.1, 9.8, "ok", 2432),
            (1, 0, 16.0, None, 16.0, 16.2, 17.0, "ok", 2176), (1, 1, 20.0, None, 20.0, 20.1, 21.0, "ok", 2432)]
    r = _run(rows)
    r["requests"]["cached_tokens"] = [0, 2176, 0, 0]
    r["requests"]["prompt_tokens"] = r["requests"]["prompt_tokens_planned"]
    sp = {s["span"]: s for s in ltb.overlap_profile(r, 10.0, 5.0, 300.0)}
    assert sp["base"]["turns"] == 1 and sp["base"]["held"] == 1 and sp["base"]["held_cached_share"] == 1.0
    assert sp["base"]["reuse_held"] == 1.0
    assert sp["+0"]["turns"] == 1 and sp["+0"]["held"] == 0 and sp["+0"]["reuse_other"] == 0.0
    assert math.isnan(sp["+0"]["held_cached_share"]) and sp["burst"]["turns"] == 0


def test_paper_table_formats_ranges_and_inf():
    assert ltb.fmt_num(0.1049) == "0.10" and ltb.fmt_num(5.44) == "5.4" and ltb.fmt_num(210.7) == "211"
    assert ltb.fmt_num(math.inf) == "∞" and ltb.fmt_num(0.0) == "0"
    assert ltb.fmt_rng([0.101, 0.104]) == "0.10" and ltb.fmt_rng([104.4, 162.9]) == "104–163"
    lat = {"p50": [0.1, 0.11], "p95": [0.3, 0.3], "p99": [0.4, 0.5], "goodput": [1.9, 2.2]}
    summ = {}
    for _, c, w in ltb.PAPER_ROWS:
        summ.setdefault(c, {})[w] = {"n": 5, "ttft_release": {"base": lat, "end": lat}}
    rows = ltb.paper_table(summ)
    assert len(rows) == 2 + len(ltb.PAPER_ROWS)
    assert rows[2] == "| Closed loop | 5 | 0.10–0.11 / 0.40–0.50 | 0.10–0.11 / 0.30 / 0.40–0.50 | 1.9–2.2 |"


def test_reply_profile_durations_overlap_and_stay(monkeypatch):
    # Open loop (no t_released). Session 0: turn 1 released at 2.5, before turn 0 ends at 3.0 (overlap); session 1:
    # turn 1 released at 8.0, after turn 0 ended at 6.0. A burst request (2944 tokens) is excluded.
    rows = [(0, 0, 1.0, None, 1.0, 1.1, 3.0, "ok", 2176), (0, 1, 2.5, None, 2.5, 2.6, 4.0, "ok", 2432),
            (1, 0, 5.0, None, 5.0, 5.1, 6.0, "ok", 2176), (1, 1, 8.0, None, 8.0, 8.1, 9.0, "ok", 2432),
            (99, 0, 2.0, None, 2.0, 2.1, 30.0, "ok", 2944)]
    r = _run(rows)
    r["cfg"]["workload"]["sessions"] = {"turns": 2, "turn_gap_s": {"dist": "exponential", "mean": 5.0}}
    monkeypatch.setitem(ltb.REPLY, "stay_margin_s", 5.0)
    p = ltb.reply_profile(r, 0.0, 20.0)
    assert p["n"] == 4 and p["mean"] == pytest.approx(1.375) and p["p50"] == 1.0 and p["p90"] == 2.0
    assert p["followed"] == 2 and p["release_overlap"] == 0.5
    assert p["implied_overlap"] == pytest.approx(np.mean([1 - math.exp(-2 / 5), 1 - math.exp(-1 / 5)]))
    assert p["stay"]["n"] == 2 and p["stay"]["mean"] == pytest.approx(3.5)
    # the margin drops a session whose first turn is released too close to the window's end
    assert ltb.reply_profile(r, 0.0, 9.0)["stay"]["n"] == 1


def test_reply_profile_closed_loop_uses_release_and_skips_non_exponential_gaps():
    rows = [(0, 0, 1.0, 1.0, 1.0, 1.1, 3.0, "ok", 2176), (0, 1, 2.5, 7.0, 7.0, 7.1, 9.0, "ok", 2432)]
    r = _run(rows)
    r["cfg"]["workload"]["sessions"] = {"turns": 2, "turn_gap_s": {"dist": "fixed", "mean": 4.0}}
    p = ltb.reply_profile(r, 0.0, 200.0)
    assert p["mean"] == pytest.approx(2.0) and p["release_overlap"] == 0.0 and math.isnan(p["implied_overlap"])


def test_reply_summary_ranges_over_seeds():
    w = lambda m, ov: {"mean": m, "p50": m, "p90": m, "release_overlap": ov, "implied_overlap": ov,
                       "stay": {"mean": 30 + m}}
    res = [{"cell": "c", "rows": [{"seed": 1, "burst": {"base": w(2.0, 0.4), "end": w(2.5, 0.3)}},
                                  {"seed": 2, "burst": {"base": w(3.0, 0.5), "end": w(2.6, NAN)}}]}]
    s = ltb.reply_summary(res)["c"]["burst"]
    assert s["n"] == 2 and s["base"]["mean"] == [2.0, 3.0] and s["end"]["release_overlap"] == [0.3, 0.3]
    assert s["base"]["stay_mean"] == [32.0, 33.0]
