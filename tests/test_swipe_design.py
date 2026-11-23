"""The sustained wipe's interval bound (analysis/swipe_design.py) on hand-made request tables: no run folder is read."""
import pandas as pd
import pytest

import analysis.swipe_design as S


def reqs(rows):
    cols = ["session_id", "turn", "t_sent", "t_end", "prompt_tokens", "cached_tokens"]
    df = pd.DataFrame(rows, columns=cols)
    df["status"] = "ok"
    return df


def test_a_turn_survives_in_proportion_to_its_gap():
    # session 0: turn 0 ends at 100; turn 1 sent at 102 (gap 2 s) with 1000 cached of 1200
    df = reqs([(0, 0, 90.0, 100.0, 1000, 0), (0, 1, 102.0, 105.0, 1200, 1000)])
    h = S.hit_bound(df, every_s=5.0, a=95.0, b=200.0)
    assert h["turns"] == 1 and h["H"] == pytest.approx(1000 * (1 - 2 / 5) / 1200)
    assert S.hit_bound(df, every_s=2.0, a=95.0, b=200.0)["H"] == 0.0     # a gap of E or more never survives


def test_the_denominator_holds_every_turn_in_the_window_and_turn_0_carries_nothing():
    df = reqs([(0, 0, 100.0, 101.0, 2000, 0), (0, 1, 102.0, 104.0, 1200, 1000)])
    h = S.hit_bound(df, every_s=5.0, a=95.0, b=200.0)
    assert h["H"] == pytest.approx(1000 * 0.8 / 3200)


def test_a_missing_previous_turn_carries_no_prefix():
    df = reqs([(0, 0, 90.0, 100.0, 1000, 0), (0, 2, 101.0, 105.0, 1400, 1200)])   # turn 1 is not ok / absent
    assert S.hit_bound(df, every_s=5.0, a=95.0, b=200.0)["H"] == 0.0


def test_a_turn_sent_before_its_previous_ended_is_refused():
    df = reqs([(0, 0, 90.0, 100.0, 1000, 0), (0, 1, 99.0, 105.0, 1200, 1000)])
    with pytest.raises(ValueError):
        S.hit_bound(df, every_s=5.0, a=95.0, b=200.0)


def test_the_rule_takes_the_largest_interval_within_the_limit_in_every_control():
    rows = [{"every_s": 5.0, "H": 0.04}, {"every_s": 5.0, "H": 0.06}, {"every_s": 2.5, "H": 0.01},
            {"every_s": 2.5, "H": 0.02}]
    assert S.choose(rows) == 2.5
    assert S.choose([{"every_s": 5.0, "H": 0.05}, {"every_s": 2.5, "H": 0.0}]) == 5.0       # the limit is inclusive
    assert S.choose([{"every_s": 5.0, "H": 0.2}, {"every_s": 2.5, "H": float("nan")}]) is None


def test_the_queue_is_read_after_the_last_successful_call():
    trig = pd.DataFrame([{"success": True, "t_start": 900.0, "t_end": 900.1},
                         {"success": True, "t_start": 1197.5, "t_end": 1197.6},
                         {"success": False, "t_start": 1200.0, "t_end": 1200.1}])
    m = pd.DataFrame({"t": [1190.0, 1198.0, 1203.0, 1208.0], "vllm:num_requests_waiting": [10.0, 40.0, 50.0, 99.0]})
    q = S.queue_after_last_call({"trigger": trig, "metrics": m})
    assert q["t_last_end"] == 1197.6 and q["scrapes"] == 2 and q["queue"] == 45.0
    assert S.queue_after_last_call({"trigger": trig, "metrics": m.iloc[0:0]})["queue"] != 0.0   # NaN, never zero


def test_the_span_readouts_count_completions_and_the_owed_backlog():
    """Rule F round 2 S2, p2's case: 67 more requests owed at the last call's end than at the first call's start."""
    trig = pd.DataFrame([{"success": True, "t_start": 900.0, "t_end": 900.1},
                         {"success": True, "t_start": 1197.5, "t_end": 1197.6}])
    done = [{"t_planned": 900.2 + i, "t_end": 900.4 + i} for i in range(297)]          # 297 planned and done in it
    owed = [{"t_planned": 950.0 + i, "t_end": float("nan")} for i in range(67)]         # 67 planned in it, never done
    base = [{"t_planned": 850.0 + i, "t_end": 2000.0} for i in range(5)]                # owed at both ends: no rise
    r = S.span_readout({"trigger": trig, "requests": pd.DataFrame(done + owed + base)})
    assert r["owed_rise"] == 67 and r["span_c"] == pytest.approx(297 / 297.6) and r["reprefill_reading"]
