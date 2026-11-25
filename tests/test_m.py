"""analysis/m.py (reading M; not registered): idle age, the turnover-scaled age x, buckets, each label with a failure
control, undefined inputs, and the registration gate. Synthetic runs only (tests/test_k1.make_run); no run folder."""
import math

import numpy as np
import pandas as pd
import pytest

from analysis import m as M
from analysis import withheld as W
from tests.test_k1 import WARM, make_run

C = 96656


def with_counters(run, rate_new=5000.0, hits=0.0, reset_at=None):
    """Adds the three raw counters: prompt tokens at rate_new + hits per s, hits at `hits` per s, no generation."""
    m = run["metrics"].copy()
    t = m.t.to_numpy(float)
    m["vllm:prompt_tokens_total"] = (rate_new + hits) * t
    m["vllm:prefix_cache_hits_total"] = hits * t
    m["vllm:generation_tokens_total"] = 0.0 * t
    if reset_at is not None:
        m.loc[m.t >= reset_at, "vllm:prompt_tokens_total"] -= 1e9
    return {**run, "metrics": m}


def test_pool_and_constants():
    assert M.M["pool_tokens"] == C
    assert M.BUCKETS == ("held", "neg_A", "[0,0.5)", "[0.5,1)", "[1,2)", "[2,4)", "[4,8)", "[8,16)", "[16,inf)")


def test_ages_closed_like_run_is_outside_the_turnover_time():
    """KV 0.95 after the burst: F = 0.05 C; 5000 new tokens/s over a = 3.3 s gives x = 16500 / 4832.8 = 3.41."""
    q = M.ages(with_counters(make_run(kv_after=0.95)))
    assert q.prev_end.min() >= 1080 and q.t_first_token.max() < 4080
    assert q.a.to_numpy() == pytest.approx(3.3)
    assert q.x.to_numpy() == pytest.approx(5000 * 3.3 / (0.05 * C))
    assert set(q.bucket) == {"[2,4)"}


def test_ages_open_like_run_is_inside():
    q = M.ages(with_counters(make_run(**WARM), rate_new=2000.0))
    assert q.x.to_numpy() == pytest.approx(2000 * 3.3 / (0.5 * C)) and set(q.bucket) == {"[0,0.5)"}


def test_held_turns_and_hits_do_not_count_as_allocation():
    run = with_counters(make_run(**WARM), rate_new=2000.0, hits=50000.0)
    q = M.ages(run)
    assert q.x.to_numpy() == pytest.approx(2000 * 3.3 / (0.5 * C))      # hits leave A unchanged
    r = make_run(**WARM)
    r["requests"]["t_end"] = r["requests"]["t_planned"] + 6.0          # the previous turn ends after the next starts
    q = M.ages(with_counters(r, rate_new=2000.0))
    assert set(q.bucket) == {"held"} and q.x.isna().all()


def test_undefined_inputs():
    assert M.ages(make_run()) is None                                   # no counters
    assert M.ages(with_counters(make_run(), reset_at=2000)) is None     # a counter goes down: flagged, not hidden
    assert M.ages(with_counters(make_run())) is not None                # control


def test_turnover_readout():
    tc = M.turnover(with_counters(make_run(kv_after=0.95)))
    assert len(tc) == 50 and tc[0]["t_c"] == pytest.approx(0.05 * C / 5000)


# ---------------------------------------------------------------- rule F r1: negative A, F <= 0, censoring, drain

def test_negative_A_gets_its_own_bucket_never_held():
    assert list(M.bucket(np.array([1.0, -1.0, 1.0]), np.array([-0.05, np.nan, 0.2]))) == ["neg_A", "held", "[0,0.5)"]


def test_hits_leading_prompts_give_neg_A_not_held():
    """Rule F r1 T3: hits recorded at admission before the prompt counter moves make p - h + g dip."""
    run = with_counters(make_run(**WARM), rate_new=2000.0)
    m = run["metrics"]
    m.loc[m.t >= 2001, "vllm:prefix_cache_hits_total"] += 20000.0         # a hit jump at admission ...
    m.loc[m.t >= 2004, "vllm:prompt_tokens_total"] += 20000.0             # ... the prompt counter 3 s later
    q = M.ages(run)
    assert (q.bucket == "neg_A").sum() >= 1 and (q[q.bucket == "neg_A"].A < 0).all()
    assert "held" not in set(q.bucket) and q.bucket.notna().all()
    assert M.ages(with_counters(make_run(**WARM), rate_new=2000.0)).bucket.ne("neg_A").all()    # control


def test_f_zero_counts_as_infinite_age_not_dropped():
    q = M.ages(with_counters(make_run(kv_after=1.0)))
    assert q.f_zero.all() and set(q.bucket) == {"[16,inf)"} and len(q) > 0
    assert M.profile([q])["f_zero"] == len(q)
    q = M.ages(with_counters(make_run(kv_after=0.95)))                        # control
    assert not q.f_zero.any()


def test_censored_counts_turns_still_waiting_at_the_end():
    r = make_run(**WARM)
    assert M.censored(r)["censored"] == 0
    late = r["requests"].session_id == 500
    r["requests"].loc[late & (r["requests"].turn == 3), ["t_first_token", "status"]] = [np.nan, "unfinished"]
    c = M.censored(r)
    assert c["censored"] == 1 and c["in_window"] > 1000


def test_drain_profile_readout():
    q = M.ages(with_counters(make_run(kv_after=0.95)))
    d = M.drain_profile(q, 1080.0)
    assert len(d) == 10 and d[0]["buckets"]["[2,4)"]["n"] > 0


# ---------------------------------------------------------------- labels on synthetic turn tables

def turns(spec, run="r"):
    """spec: list of (x, reuse, n); held when x is None. prev_prompt 1000 each."""
    rows = []
    for x, r, n in spec:
        for _ in range(n):
            rows.append({"a": -1.0 if x is None else 1.0, "x": np.nan if x is None else x, "r_num": 1000 * r,
                         "prev_prompt": 1000.0, "run": run, "f_zero": False})
    q = pd.DataFrame(rows)
    q["bucket"] = M.bucket(q.a.to_numpy(float), q.x.to_numpy(float))
    return q


def arm(spec):
    return [turns(spec, run=f"r{i}") for i in range(5)]


CLOSED = arm([(20.0, 0.0, 200)])


def test_label_undefined():
    assert M.label(CLOSED[:3], CLOSED)["label"] == "undefined"
    assert M.label(CLOSED[:4], arm([(0.2, 1.0, 200)])[:4])["label"] != "undefined"    # control: 4 of 5 is enough


def test_label_closed_not_outside():
    inside = arm([(20.0, 0.0, 170), (0.2, 0.0, 30)])                      # 0.85 at x >= 1
    assert M.label(inside, arm([(0.2, 1.0, 200)]))["label"] == "closed_not_outside"
    reused = arm([(20.0, 0.2, 200)])
    assert M.label(reused, arm([(0.2, 1.0, 200)]))["label"] == "closed_not_outside"
    assert M.label(CLOSED, arm([(0.2, 1.0, 200)]))["label"] != "closed_not_outside"   # control


def test_label_separated_when_no_bucket_is_matched():
    assert M.label(CLOSED, arm([(0.2, 1.0, 200)]))["label"] == "separated"


def test_one_large_shock_seed_cannot_make_a_match():
    one = [turns([(0.2, 1.0, 200), (20.0, 0.5, 150)], run="r0")] + [turns([(0.2, 1.0, 200)], run=f"r{i}")
                                                                    for i in range(1, 5)]
    assert M.label(CLOSED, one)["label"] == "separated"                   # 150 pooled, but in one run
    three = [turns([(0.2, 1.0, 200), (20.0, 0.5, 40)], run=f"r{i}") for i in range(3)] + \
            [turns([(0.2, 1.0, 200)], run=f"r{i}") for i in range(3, 5)]
    assert M.label(CLOSED, three)["label"] == "arm_differs"              # control: 120 pooled over 3 runs


def test_label_arm_differs_and_its_per_run_counts():
    out = M.label(CLOSED, arm([(0.2, 1.0, 200), (20.0, 0.5, 30)]))
    assert out["label"] == "arm_differs"
    assert out["per_run"]["matched_sign"]["[16,inf)"]["k"] == 5 and out["per_run"]["closed_outside"]["k"] == 5
    assert M.label(CLOSED, arm([(0.2, 1.0, 200), (20.0, 0.1, 30)]))["label"] == "agree_outside"   # control: 0.1


def test_common_curve_needs_the_fall_at_one_to_two():
    closed = arm([(20.0, 0.0, 200), (1.5, 0.0, 30)])                    # 150 closed turns at [1, 2): still >= 0.9 out
    ok = arm([(0.2, 1.0, 200), (1.5, 0.1, 30)])
    assert M.label(closed, ok)["label"] == "common_curve"
    # T2: [1, 2) still high, the fall only at x >= 16; pooled x >= 1 would have fallen by 0.5
    late = arm([(0.2, 1.0, 200), (1.5, 0.95, 4), (20.0, 0.0, 30)])
    out = M.label(CLOSED, late)
    assert out["label"] == "agree_outside" and out["label"] != "common_curve"
    flat = arm([(0.2, 0.3, 200), (1.5, 0.1, 30)])                       # matched at [1, 2), fall only 0.2
    assert M.label(closed, flat)["label"] == "mixed"


def test_predicted_is_a_distribution_over_the_labels():
    assert set(M.PREDICTED) == set(M.LABELS) and sum(M.PREDICTED.values()) == pytest.approx(1.0)


def test_table_refuses_before_its_registration(tmp_path, monkeypatch):
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-24 · [DECISION] mentions [REG] M (idle age against the turnover time) in passing\n")
    monkeypatch.setattr(W, "DECISIONS", d)
    with pytest.raises(RuntimeError):
        M.table()
    d.write_text(d.read_text() + "## 2026-11-24 · [REG] M (idle age against the turnover time): x\n")
    import analysis.load as L
    monkeypatch.setattr(L, "find_runs", lambda pat, **k: [])
    assert M.table()["label"]["label"] == "undefined"


def test_profile_seconds_readout():
    q = M.ages(with_counters(make_run(**WARM), rate_new=2000.0))
    out = M.profile_seconds([q])
    assert out["[2,5)"]["n"] == len(q) and out["[2,5)"]["reuse"] == pytest.approx(1.0) and out["[0,1)"]["n"] == 0
