"""analysis/reg28.py: REG-28's two blind readings of REG-27's stuck burst runs (synthetic runs; no run folder is read)."""
import math

import pandas as pd
import pytest

from analysis import modelf, reg28, withheld


def _run(n_sessions=600, span=3900.0, a=0.0, last_drop=0, preempt=(0, 0), uncached=3000, burst=False):
    """Sessions start evenly over [a, a + span); each runs 8 turns ending within the window. `last_drop` sessions lose
    their last turn (left unfinished), lowering the observed share."""
    rows = []
    for s in range(n_sessions):
        t0 = a + span * s / n_sessions
        for k in range(8):
            if k == 7 and s < last_drop:
                continue
            t = min(t0 + 0.1 * k, a + span - 1e-3)
            rows.append({"session_id": s, "turn": k, "status": "ok", "t_planned": t0 if k == 0 else t, "t_sent": t,
                         "t_first_token": t, "t_end": t, "prompt_tokens": uncached, "prompt_tokens_planned": 2176 + 256 * k,
                         "cached_tokens": 0})
    if burst:   # one-off burst prompts, which are never session turns
        for i in range(50):
            rows.append({"session_id": 10_000 + i, "turn": 0, "status": "ok", "t_planned": a + i, "t_sent": a + i,
                         "t_first_token": a + i, "t_end": a + i, "prompt_tokens": 2944,
                         "prompt_tokens_planned": 2944, "cached_tokens": 0})
    m = pd.DataFrame({"t": [a, a + span - 1], reg28_order_col(): list(preempt)})
    cfg = {"trigger": {"kind": "burst", "at_s": a - 780.0, "duration_s": 180.0, "prompt_tokens": 2944},
           "verdict": {"watch_s": span + 600.0}, "workload": {"duration_s": a + span}}
    return {"requests": pd.DataFrame(rows), "metrics": m, "cfg": cfg,
            "dir": type("D", (), {"name": "x_confirm_l1pl350cl_r2.0_s61"})()}


def reg28_order_col():
    from analysis import order
    return order.PREEMPT


def test_predicted_share_is_model_f_sessions_over_requests():
    t = modelf.l1(2.0, 1.681, 8)
    assert reg28.predicted_share(2.0, 1.681) == pytest.approx(t.sessions / t.requests)
    assert reg28.predicted_share(2.0, 1.681) == pytest.approx(0.879, abs=1e-3)   # docs/model.md M5 inputs
    assert math.isnan(reg28.predicted_share(1.6, 1.681))                       # model F drains


def test_share_of_a_quasi_static_run_is_one():
    s = reg28.share_obs(_run(), 0.0, 3900.0)
    assert s["turns_ok"] == 600 * 8 and s["last_ok"] == 600
    assert s["obs"] == pytest.approx(1.0) and s["X"] == pytest.approx(600 * 8 / 3900.0)


def test_burst_requests_are_not_session_turns():
    s = reg28.share_obs(_run(burst=True), 0.0, 3900.0)
    assert s["turns_ok"] == 600 * 8


def test_unfinished_last_turns_lower_the_share():
    s = reg28.share_obs(_run(last_drop=60), 0.0, 3900.0)
    assert s["obs"] == pytest.approx(8 * 540 / (600 * 8 - 60))


def test_model_f_prediction_is_the_replay_at_the_fitted_mu(monkeypatch):
    """The replay's session-turn rate rises with mu; mu_X is where it meets the run's; pred is the replay's share
    there (rule F r1 B1, B2)."""
    monkeypatch.setattr(reg28.modelf_sim, "schedule", lambda cfg: None)
    monkeypatch.setattr(reg28, "replay_share",
                        lambda cfg, mu, win, sched=None, p=None, **kw: {"X": mu, "obs": 0.80 + 0.05 * mu})
    run = _run()                                            # X_obs = 4800 / 3900 = 1.2308
    s = reg28.session_share(run, 0.0, 3900.0, p={**reg28.REG28, "mu_grid": (1.0, 1.5, 0.01)})
    assert s["mu_X_status"] == "ok" and s["mu_X"] == pytest.approx(4800 / 3900, abs=1e-6)
    assert s["pred_F"] == pytest.approx(0.80 + 0.05 * 4800 / 3900)        # three equal replays: their mean
    assert s["dev_Q"] == pytest.approx(0.0) and s["nearer"] == "Q"


def test_an_error_in_one_question_leaves_the_other(monkeypatch):
    monkeypatch.setattr(reg28, "session_share", lambda *a, **k: 1 / 0)
    r = reg28.row(_run(preempt=(0, 10)), (0.0, 3900.0))
    assert "error" in r["q1"] and math.isnan(r["q1"]["dev_F"]) and r["q2"]["preemptions"] == 10


def _q1row(dev_f, dev_q):
    return {"q1": {"dev_F": dev_f, "dev_Q": dev_q, "obs": 1.0 + dev_q}, "q2": {"share": 0.04}}


def test_judge_q1_labels():
    assert reg28.judge_q1([_q1row(0.0, -0.1)] * 2)["label"] == "too_few"
    assert reg28.judge_q1([_q1row(0.01, -0.11)] * 3)["label"] == "law_holds"
    assert reg28.judge_q1([_q1row(0.11, -0.01)] * 4)["label"] == "quasi_static"
    assert reg28.judge_q1([_q1row(0.06, -0.06)] * 3)["label"] == "neither"
    assert reg28.judge_q1([_q1row(0.02, -0.02)] * 3)["label"] == "not_separated"


def test_a_deviation_at_the_tolerance_is_inside():
    tol = reg28.REG28["tol"]
    assert reg28.judge_q1([_q1row(tol, -0.2)] * 3)["label"] == "law_holds"


def test_churn_bound():
    run = _run(preempt=(100, 300))
    c = reg28.churn(run, 0.0, 3900.0)
    assert c["preemptions"] == 200
    assert c["share"] == pytest.approx(200 * 4096 / (600 * 8 * 3000))


def test_churn_is_nan_when_the_counter_goes_down():
    assert math.isnan(reg28.churn(_run(preempt=(300, 100)), 0.0, 3900.0)["share"])


def test_judge_q2_labels():
    r = lambda s: {"q2": {"share": s}}
    assert reg28.judge_q2([r(0.04)] * 2)["label"] == "too_few"
    assert reg28.judge_q2([r(0.04), r(0.05), r(0.03)])["label"] == "churn_bounded"
    assert reg28.judge_q2([r(0.04), r(0.07), r(0.03)])["label"] == "not_bounded"
    assert "churn_named" not in open(reg28.__file__).read()
    assert reg28.judge_q2([r(0.04), r(math.nan), r(0.03), r(0.02)])["n"] == 3


def test_table_refuses_before_reg27s_result(monkeypatch):
    monkeypatch.setattr(withheld, "lifted", lambda heading, decisions=None: False)
    import analysis.reg27 as r27
    monkeypatch.setattr(r27, "table", lambda: pytest.fail("reg27.table must not be called"))
    with pytest.raises(withheld.WithheldError):
        reg28.table()


def test_lifting_heading_is_reg27s_result_only():
    assert reg28.REG28["after"] == "[RESULT] REG-27"


def test_module_never_sets_the_unseal_variable():
    src = open(reg28.__file__).read()
    assert "os.environ" not in src and "putenv" not in src


def test_stuck_labels_are_reg27s_stuck_through_the_watch():
    from analysis import confirm
    assert set(reg28.STUCK_WATCH) <= set(confirm.BURST_LABELS)
    assert "reformed" not in reg28.STUCK_WATCH and "emptied_in_watch" not in reg28.STUCK_WATCH


def test_a_replay_sensitive_run_is_not_counted(monkeypatch):
    """Rule F r2 N1: a variant replay (cold from the burst's end, or with the residence) that moves pred by more than
    half its own gap 1 - pred takes the run out of Q1; inside it, pred is the mean of the three."""
    monkeypatch.setattr(reg28.modelf_sim, "schedule", lambda cfg: None)
    shift = lambda kw: 0.06 if "all_miss_from" in kw else 0.0
    monkeypatch.setattr(reg28, "replay_share",
                        lambda cfg, mu, win, sched=None, p=None, **kw: {"X": mu, "obs": 0.90 + shift(kw)})
    s = reg28.session_share(_run(), 0.0, 3900.0, p={**reg28.REG28, "mu_grid": (1.0, 1.5, 0.01)})
    assert s["replay_sensitive"] is True and math.isnan(s["pred_F"]) and s["pred_F_all"] == pytest.approx(0.90)
    assert s["replay_spread"] == pytest.approx(0.06)          # > 0.5 x (1 - 0.92)
    j = reg28.judge_q1([{"run": "a", "q1": s, "q2": {}}] + [_q1row(0.0, -0.1)] * 3)
    assert j["n"] == 3 and j["replay_sensitive"] == ["a"]


def test_the_replay_runs_on_a_reg27_config():
    """Rule F r2 N4: model F's replay on REG-27's seed-61 burst config (schedule only, no run read)."""
    from analysis import confirm
    cfg = confirm.burst_cfg(61, 9000.0)
    ph = confirm.phases(cfg)
    win = (ph["burst_end"] + 600.0, ph["watch_end"])
    lo, hi = reg28.replay_share(cfg, 1.70, win), reg28.replay_share(cfg, 1.80, win)
    assert lo["X"] < hi["X"]                                  # the session-turn rate rises with mu
    assert 0.8 < lo["obs"] < 1.0 and 0.8 < hi["obs"] < 1.0    # below the quasi-static 1


def test_pred_is_the_mean_of_three_replays_inside_the_spread(monkeypatch):
    monkeypatch.setattr(reg28.modelf_sim, "schedule", lambda cfg: None)
    shift = lambda kw: 0.02 if "all_miss_from" in kw else 0.0
    monkeypatch.setattr(reg28, "replay_share",
                        lambda cfg, mu, win, sched=None, p=None, **kw: {"X": mu, "obs": 0.90 + shift(kw)})
    s = reg28.session_share(_run(), 0.0, 3900.0, p={**reg28.REG28, "mu_grid": (1.0, 1.5, 0.01)})
    assert s["replay_sensitive"] is False and s["pred_F"] == pytest.approx((0.90 + 0.92 + 0.90) / 3)
