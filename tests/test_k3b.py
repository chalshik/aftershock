"""analysis/k3b.py: model F's predictions for K4's tipped runs (synthetic rows; no run folder is read)."""
import math

import pandas as pd
import pytest

from analysis import k3b, modelf, withheld


def test_predict_matches_model_f_at_the_open_loop_floor():
    p = k3b.predict(2.2)
    t = modelf.l1(2.2, 1.681)
    assert p["c"] == pytest.approx(t.requests * 1.681)
    assert p["served"] == pytest.approx(t.served_prompt)
    assert p["sessions"] == pytest.approx(t.sessions)
    assert p["r"] == pytest.approx(1.0, abs=1e-9)          # proportional cost: Gamma x served = 1
    assert p["served"] == pytest.approx(0.9755, abs=5e-4)   # docs/model.md M5
    assert p["sessions"] == pytest.approx(0.835, abs=1e-3)


def test_predict_drains_at_or_below_the_floor():
    assert k3b.predict(1.681)["drains"] and k3b.predict(1.5)["drains"]
    assert not k3b.predict(1.7)["drains"]


def test_grid_reads_no_run():
    g = k3b.predictions_grid()
    assert set(g) == {"1.681", "1.798"}
    assert g["1.798"]["1.8"]["drains"] is False and g["1.681"]["2.2"]["c"] == pytest.approx(k3b.predict(2.2)["c"])


def test_deviation_relative_for_c_absolute_otherwise():
    assert k3b.deviation("c", 1.80, 1.72) == pytest.approx(1.80 / 1.72 - 1)
    assert k3b.deviation("served", 0.97, 0.9755) == pytest.approx(-0.0055)
    assert math.isnan(k3b.deviation("r", math.nan, 1.0))


def _row(dev: dict, judged=True):
    return {"run": f"r{id(dev)}", "judged": judged, "dev": dev, "why_not": None if judged else "idle in W"}


def test_judge_labels_and_counts():
    ok = {"c": 0.01, "r": 0.02, "served": -0.004, "sessions": 0.03}
    far = {"c": 0.09, "r": 0.08, "served": 0.03, "sessions": -0.15}
    j = k3b.judge([_row(ok), _row(ok)])
    assert all(j[q]["label"] == "undefined" for q in k3b.QUANTITIES)   # fewer than min_runs
    j = k3b.judge([_row(ok), _row(ok), _row(far), _row(far, judged=False)])
    assert j["n_judged"] == 3 and len(j["left_out"]) == 1
    assert j["c"]["label"] == "inside" and j["c"]["k_inside"] == 2 and j["c"]["n"] == 3
    j = k3b.judge([_row(far)] * 3)
    assert j["sessions"]["label"] == "outside" and j["sessions"]["side"] == "low"
    assert j["served"]["label"] == "outside" and j["served"]["side"] == "high"


def test_a_deviation_exactly_at_the_tolerance_is_inside():
    at = {q: k3b.K3B["tol"][q] for q in k3b.QUANTITIES}
    j = k3b.judge([_row(at)] * 3)
    assert all(j[q]["label"] == "inside" for q in k3b.QUANTITIES)


def test_why_lists_every_reason():
    w = (1000.0, 1500.0)
    assert k3b._why(w, 0.0, 1.0, True, {"drains": False}) is None
    s = k3b._why(w, 0.2, 0.5, False, {"drains": True})
    for part in ("not all-miss", "idle", "not cold_backlogged", "drains"):
        assert part in s
    assert "shorter" in k3b._why((1000.0, 1200.0), 0.0, 1.0, True, {})


def _synthetic_run():
    rows = []
    for s in range(40):                       # 40 sessions, 8 turns, all ok, no reuse
        for k in range(8):
            t = 1000.0 + 10 * s + 20 * k
            rows.append({"session_id": s, "turn": k, "status": "ok", "t_planned": t, "t_sent": t,
                         "t_first_token": t + 1, "t_end": t + 2, "prompt_tokens": 2176 + 256 * k,
                         "prompt_tokens_planned": 2176 + 256 * k, "cached_tokens": 0})
    return {"requests": pd.DataFrame(rows), "cfg": {"trigger": None}}


def test_observed_on_a_synthetic_run():
    run = _synthetic_run()
    o = k3b.observed(run, 1000.0, 1400.0, u_off=5000.0)
    q = run["requests"]
    w = q[(q.t_first_token >= 1000) & (q.t_first_token < 1400)]
    assert o["served"] == pytest.approx(w.prompt_tokens.mean() / 3072)
    assert o["c"] == pytest.approx(((q.t_end >= 1000) & (q.t_end < 1400)).sum() / 400)
    last = q[(q.turn == 7) & (q.t_end >= 1000) & (q.t_end < 1400)]
    assert o["sessions"] == pytest.approx(len(last) / 400 / (1.681 / 8))
    assert o["r"] == pytest.approx(o["U"] / 5000.0)


def test_row_with_a_short_window_is_not_judged():
    run = _synthetic_run()
    run["cfg"]["workload"] = {"duration_s": 2700}
    run["dir"] = type("D", (), {"name": "x_k4_l1pl350cl_r2.2_s133"})()
    r = k3b.row(run, onset=2500.0, ref={"mean": 5000.0})
    assert r["judged"] is False and "shorter" in r["why_not"]


def test_table_refuses_before_k4s_result(monkeypatch):
    monkeypatch.setattr(withheld, "lifted", lambda heading, decisions=None: False)
    import analysis.k4 as k4
    monkeypatch.setattr(k4, "table", lambda: pytest.fail("k4.table must not be called"))
    with pytest.raises(withheld.WithheldError):
        k3b.table()


def test_module_never_sets_the_unseal_variable():
    src = open(k3b.__file__).read()
    assert "os.environ" not in src and "putenv" not in src


def _gated(monkeypatch, cb=True, lam_scale=1):
    """A synthetic tipped run with the gate's inputs patched: reuse 0, busy 1, cold_backlogged cb."""
    import analysis.k3b as K
    monkeypatch.setattr(K, "reuse", lambda run, a, b: 0.0)
    monkeypatch.setattr(K.k3w, "busy_share", lambda run, a, b: 1.0)
    monkeypatch.setattr(K, "cold_backlogged", lambda run, win, v: {"cold_backlogged": cb})
    rows = []
    n = int(750 * lam_scale)
    for s in range(n):                                   # sessions start across [0, 2700)
        for k in range(8):
            t = 2700.0 * s / n + 20 * k
            rows.append({"session_id": s, "turn": k, "status": "ok", "t_planned": t, "t_sent": t,
                         "t_first_token": t + 1, "t_end": t + 2, "prompt_tokens": 2176 + 256 * k,
                         "prompt_tokens_planned": 2176 + 256 * k, "cached_tokens": 0})
    run = {"requests": pd.DataFrame(rows), "metrics": pd.DataFrame(), "cfg": {"trigger": None, "verdict": {},
           "workload": {"duration_s": 2700}}, "dir": type("D", (), {"name": "x_k4_l1pl350cl_r2.2_s137"})()}
    return run


def test_row_passing_the_gate_is_judged(monkeypatch):
    run = _gated(monkeypatch)                            # 750 sessions x 8 / 2700 s: lam_W about 2.2 req/s
    r = k3b.row(run, onset=1500.0, ref={"mean": 5164.76})
    assert r["judged"] is True and r["why_not"] is None and r["against_corollary3"] is False
    assert set(r["dev"]) == set(k3b.QUANTITIES) and r["predicted"]["drains"] is False
    assert r["served_discriminates"] is True and not math.isnan(r["dev"]["served"])


def test_a_stuck_run_where_model_f_drains_counts_against_corollary3(monkeypatch):
    run = _gated(monkeypatch, lam_scale=0.6)             # lam_W about 1.3 req/s < 1.681
    r = k3b.row(run, onset=1500.0, ref={"mean": 5164.76})
    assert r["predicted"]["drains"] and r["judged"] is False and r["against_corollary3"] is True
    assert k3b.judge([r])["against_corollary3"] == [r["run"]]


def test_undefined_cold_backlogged_is_not_judged(monkeypatch):
    run = _gated(monkeypatch, cb=None)
    r = k3b.row(run, onset=1500.0, ref={"mean": 5164.76})
    assert r["judged"] is False and "undefined" in r["why_not"] and r["against_corollary3"] is False


def test_served_is_judged_only_where_its_band_excludes_the_untilted_mix():
    assert k3b.predict(1.9)["served"] + k3b.K3B["tol"]["served"] >= 1      # 0.989 + 0.015: not judged
    assert k3b.predict(2.2)["served"] + k3b.K3B["tol"]["served"] < 1


def test_sessions_is_reported_not_judged():
    assert "sessions" not in k3b.JUDGED
    j = k3b.judge([_row({"c": 0, "r": 0, "served": 0, "sessions": 0.5})] * 3)
    assert j["sessions"]["judged"] is False and j["c"]["judged"] is True
