"""analysis/b1.py on synthetic runs (no run folder is read)."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analysis import b1

AT, DUR, WATCH = 1000.0, 60.0, 3000.0
END = AT + DUR


def make_run(reuse_of, n_sessions=600, seed=0, burst=True, name="synthetic"):
    """Closed-loop-like sessions of 8 turns starting after the burst; turn k is planned at turn k-1 + an exponential
    gap (mean 5 s) and sent then; reuse_of(gap, t, rng) gives the cached share of the previous prompt."""
    rng = np.random.default_rng(seed)
    rows, sid = [], 0
    for _ in range(n_sessions):
        planned = sent = END + rng.uniform(0, WATCH - 120)
        for k in range(8):
            g = rng.exponential(5.0) if k else 0.0
            if k:   # closed loop: planned on the schedule, sent at the previous end plus the same gap
                planned, sent = planned + g, end + g
            end = sent + 1.0
            prompt = 2176 + 256 * k
            share = reuse_of(g, sent, rng) if k else 0.0
            cached = int(round(share * (2176 + 256 * (k - 1)))) if k else 0
            rows.append({"session_id": sid, "turn": k, "t_planned": planned, "t_released": sent, "t_sent": sent,
                         "t_first_token": sent + 0.5, "t_end": end, "prompt_tokens": prompt,
                         "prompt_tokens_planned": prompt, "cached_tokens": cached, "status": "ok"})
        sid += 1
    if burst:   # burst requests: turn 0 with the burst's prompt length, never later turns
        for i in range(50):
            rows.append({"session_id": 10_000 + i, "turn": 0, "t_planned": AT + i, "t_released": None, "t_sent": AT + i,
                         "t_first_token": AT + i + 1, "t_end": AT + i + 2, "prompt_tokens": 999,
                         "prompt_tokens_planned": 999, "cached_tokens": 0, "status": "ok"})
    cfg = {"trigger": {"kind": "burst", "at_s": AT, "duration_s": DUR, "prompt_tokens": 999},
           "verdict": {"watch_s": WATCH}}
    return {"cfg": cfg, "requests": pd.DataFrame(rows), "dir": Path(name)}


P = {**b1.B1, "boot_n": 200}


def test_gap_is_the_planned_difference_and_burst_rows_are_left_out():
    r = make_run(lambda g, t, rng: 1.0, n_sessions=20)
    q = b1.turns(r)
    assert (q.turn > 0).all() and not (q.prompt_tokens_planned == 999).any()
    req = r["requests"].set_index(["session_id", "turn"])
    for _, x in q.head(30).iterrows():
        assert x.gap == pytest.approx(req.loc[(x.session_id, x.turn), "t_planned"]
                                      - req.loc[(x.session_id, x.turn - 1), "t_planned"])
    assert np.allclose(q.gap_sent, q.gap)


def test_reuse_is_graded_and_capped_at_the_previous_prompt():
    r = make_run(lambda g, t, rng: 1.0, n_sessions=5)
    r["requests"].loc[r["requests"].turn == 3, "cached_tokens"] = 10**6
    q = b1.turns(r)
    assert (q.r_num <= q.prev_prompt).all()


def test_a_prefix_lost_on_a_one_second_clock_reads_gap_bound():
    r = make_run(lambda g, t, rng: 1.0 if g < 1.0 else 0.0)
    x = b1.row(r, p=P)
    assert x["label"] == "gap_bound" and x["long"] == 0 and x["ratio"] == np.inf


def test_a_graded_decay_reads_gap_bound():
    r = make_run(lambda g, t, rng: float(np.exp(-g / 1.0)))
    x = b1.row(r, p=P)
    assert x["label"] == "gap_bound" and x["ratio"] >= 2 and x["lb95"] > 1


def test_reuse_independent_of_the_gap_reads_flat():
    r = make_run(lambda g, t, rng: float(rng.random() < 0.5), n_sessions=1500)
    x = b1.row(r, p=P)
    assert x["label"] == "flat", x


def test_all_misses_read_no_variance():
    assert b1.row(make_run(lambda g, t, rng: 0.0), p=P)["label"] == "no_variance"


def test_few_turns_read_too_few():
    assert b1.row(make_run(lambda g, t, rng: 1.0, n_sessions=30), p=P)["label"] == "too_few"


def test_a_span_level_state_does_not_make_a_gap_effect():
    """Warm first half, cold second half; reuse does not depend on the gap within a span. Short gaps are made more
    common in the warm half by the generator below, so a pooled ratio would exceed 1; the span-stratified one does not."""
    rng = np.random.default_rng(3)
    rows = []
    for sid in range(1500):
        warm = sid % 2 == 0
        t = END + (rng.uniform(0, 1400) if warm else rng.uniform(1500, 2900))
        for k in range(8):
            if k:
                t += rng.exponential(1.5 if warm else 8.0)
            cached = (2176 + 256 * (k - 1)) if (k and warm) else 0
            rows.append({"session_id": sid, "turn": k, "t_planned": t, "t_released": t, "t_sent": t,
                         "t_first_token": t + 0.5, "t_end": t + 1.0, "prompt_tokens": 2176 + 256 * k,
                         "prompt_tokens_planned": 2176 + 256 * k, "cached_tokens": cached, "status": "ok"})
    r = {"cfg": {"trigger": {"kind": "burst", "at_s": AT, "duration_s": DUR, "prompt_tokens": 999},
                 "verdict": {"watch_s": WATCH}}, "requests": pd.DataFrame(rows), "dir": Path("confound")}
    q = b1.turns(r)
    short, long_ = q[q.gap < 1], q[(q.gap > 5) & (q.gap <= 25)]
    pooled = (short.r_num.sum() / short.prev_prompt.sum()) / (long_.r_num.sum() / long_.prev_prompt.sum())
    assert pooled > 2
    x = b1.row(r, p=P)
    assert x["ratio"] == pytest.approx(1.0, abs=0.15)
    assert x["label"] != "gap_bound"


def test_a_control_is_read_over_its_partners_window():
    ctl = make_run(lambda g, t, rng: 1.0, burst=False)
    ctl["cfg"] = {"verdict": {"watch_s": WATCH}}
    partner = make_run(lambda g, t, rng: 1.0, n_sessions=1)
    assert b1.window(ctl, partner) == (END, END + WATCH)


def test_profile_buckets_cover_every_gap():
    r = make_run(lambda g, t, rng: 1.0, n_sessions=100)
    a, b = b1.window(r)
    q = b1.turns(r)
    n_in = ((q.t_first_token >= a) & (q.t_first_token < b)).sum()
    assert sum(x["n"] for x in b1.profile(q, a, b)) == n_in


@pytest.mark.parametrize("cold,control,want", [
    (["gap_bound"] * 3, ["flat"] * 5, "supports_aging"),
    (["gap_bound", "gap_bound", "flat"], ["flat"] * 4 + ["unclear"], "supports_aging"),
    (["gap_bound"] * 3, ["flat"] * 3 + ["gap_bound"] * 2, "not_specific"),
    (["flat", "flat", "gap_bound"], ["flat"] * 5, "flat_cold"),
    (["gap_bound", "flat", "unclear"], ["flat"] * 5, "mixed"),
    (["gap_bound", "too_few", "no_variance"], ["flat"] * 5, "undefined"),
    (["gap_bound"] * 3, ["flat"] * 3 + ["too_few"] * 2, "undefined"),
])
def test_outcome(cold, control, want):
    assert b1.outcome(cold, control)["decision"] == want


def test_table_refuses_before_the_registration(monkeypatch):
    monkeypatch.setattr(b1.withheld, "lifted", lambda h, *a, **k: False)
    called = []
    import analysis.load as load
    monkeypatch.setattr(load, "find_runs", lambda *a, **k: called.append(a) or [])
    with pytest.raises(RuntimeError, match=r"\[REG\] B1"):
        b1.table()
    assert not called


def test_residency_is_nan_without_metrics():
    r = make_run(lambda g, t, rng: 1.0, n_sessions=1)
    assert all(np.isnan(v) for v in b1.residency(r, 0, 10).values())


def test_a_null_at_low_reuse_reaches_flat():
    """Rule F r1, T3: reuse ~ Bernoulli(0.08) independent of the gap, at D1b-like counts, must be able to read flat."""
    labels = [b1.row(make_run(lambda g, t, rng: float(rng.random() < 0.08), n_sessions=675, seed=s), p=P)["label"]
              for s in range(3)]
    assert labels.count("flat") >= 2, labels


def test_a_loss_set_by_the_wait_alone_reads_flat():
    """Rule F r1, T4: lost iff the (unseen) wait exceeds 0.6 s, whatever the gap."""
    x = b1.row(make_run(lambda g, t, rng: float(rng.exponential(1.6) <= 0.6), n_sessions=1000), p=P)
    assert x["label"] == "flat", x


def test_waits_above_the_clock_read_no_variance():
    """Rule F r1, T2: under L with every wait above the clock every turn is lost."""
    x = b1.row(make_run(lambda g, t, rng: float(g + rng.uniform(1.4, 1.9) < 0.94)), p=P)
    assert x["label"] == "no_variance"


def test_profile_reads_waits_among_full_misses_and_the_long_edges_match():
    r = make_run(lambda g, t, rng: 1.0 if g < 1.0 else 0.0, n_sessions=200)
    a, b = b1.window(r)
    prof = {x["gap"]: x for x in b1.profile(b1.turns(r), a, b)}
    assert prof[(0.0, 1.0)]["n_miss"] == 0 and prof[(5.0, 10.0)]["n_miss"] == prof[(5.0, 10.0)]["n"]
    q = b1.turns(r)
    q = q[(q.t_first_token >= a) & (q.t_first_token < b)]
    long_n = ((q.gap > 5) & (q.gap <= 25)).sum()
    assert prof[(5.0, 10.0)]["n"] + prof[(10.0, 25.0)]["n"] == long_n


def test_a_control_without_exactly_one_partner_is_too_few(monkeypatch):
    import analysis.load as load
    ctl = make_run(lambda g, t, rng: 1.0, burst=False, name="x_stage2_control_l1pl350cl_r2.0_w3000_s18")
    monkeypatch.setattr(b1.withheld, "lifted", lambda h, *a, **k: True)
    monkeypatch.setattr(b1, "row", lambda r, partner=None, p=b1.B1: {"run": r["dir"].name, "label": "flat"})
    monkeypatch.setattr(load, "find_runs", lambda pat, *a, **k: [ctl] if "control" in pat else [])
    out = b1.table()
    assert out["rows"]["control"][0]["label"] == "too_few" and "0 valid burst partners" in out["rows"]["control"][0]["reason"]
