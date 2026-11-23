"""A5's client-server TTFT gap (analysis/lag.py, REG-26) on hand-made runs: no run folder is read."""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import analysis.lag as L


def fake_run(server_ttft=0.3, extra=0.01, end=600.0, rate=2.0, extra_from=None, extra_late=0.0, offset=0.0,
             slope=0.0):
    """Scrapes every second from t = -3; `rate` first tokens per second from 0.25 s. The server's TTFT is
    `server_ttft` + `slope` x first-token time; the client sees `extra` more (and `extra_late` more from
    `extra_from` on). `offset`: each scrape's counters are read `offset` s after its stamp."""
    t = np.arange(-3.0, end + 1.0, 1.0)
    ft = np.arange(0.25, end, 1.0 / rate)
    srv = server_ttft + slope * ft
    lag = np.full(len(ft), extra) + (np.where(ft >= extra_from, extra_late, 0.0) if extra_from is not None else 0.0)
    seen = np.searchsorted(ft, t + offset, side="right")        # first tokens the server had counted at each read
    csum = np.concatenate([[0.0], np.cumsum(srv)])
    cfast = np.concatenate([[0.0], np.cumsum(srv <= 1.0)])
    m = pd.DataFrame({"t": t, L.COUNT: seen.astype(float), L.SUM: csum[seen]})
    reqs = pd.DataFrame({"t_first_token": ft, "t_sent": ft - srv - lag})
    return {"dir": Path("fake"), "metrics": m, "requests": reqs, "slo_bucket": (t, cfast[seen])}


def test_a_steady_run_gives_the_extra_client_time_whole_and_per_window():
    r = fake_run()
    w0 = L.whole(r)
    assert w0["n_server"] == w0["n_client"] == 1200
    assert w0["gap"] == pytest.approx(0.01) and w0["server_mean"] == pytest.approx(0.3)
    assert w0["server_share"] == 1.0 and w0["client_share"] == 1.0 and w0["share_shift"] == 0.0
    w = L.windows(r)
    assert w.kept.all() and len(w) == 10 and w.gap.to_numpy() == pytest.approx(0.01)


def test_the_share_shift_counts_requests_pushed_over_the_slo_by_client_time():
    r = fake_run(server_ttft=0.95, extra=0.1, extra_from=300.0, extra_late=0.0)   # every client TTFT 1.05 s
    w0 = L.whole(r)
    assert w0["server_share"] == 1.0 and w0["client_share"] == 0.0 and w0["share_shift"] == 1.0
    assert L.label(pd.DataFrame([w0])) == "flag"


def test_a_late_rise_in_client_lag_shows_in_the_later_windows_and_the_whole_mean():
    r = fake_run(extra_from=300.0, extra_late=0.2)
    w = L.windows(r)
    assert w[w.b <= 300].gap.to_numpy() == pytest.approx(0.01)
    assert w[w.a >= 300].gap.to_numpy() == pytest.approx(0.21)
    s = L.summary("x", r)
    assert s["win_gap_max"] == pytest.approx(0.21) and s["win_max_at"] >= 300
    assert s["gap"] == pytest.approx(0.01 + 0.2 * 600 / 1200, abs=0.002)


def test_a_scrape_delay_under_a_rising_ttft_biases_windows_but_not_the_whole_span():
    # REG-26 rule F round 1, trace 1: TTFT rising 0.01 s per s, counters read 1 s late, no real client time.
    r = fake_run(extra=0.0, slope=0.01, offset=1.0)
    w = L.windows(r)
    assert w[w.kept].gap.median() == pytest.approx(-0.01, abs=0.002)   # the window bias, about slope x offset
    assert abs(L.whole(r)["gap"]) < 0.01                                  # the whole span: only the two ends


def test_partial_first_and_last_windows_are_dropped():
    w = L.windows(fake_run(end=630.0))
    assert (w.b - w.a >= 30).all() and w.a.min() >= 0 and w.b.max() <= 600


def test_a_window_with_too_few_client_tokens_is_not_kept_and_a_run_without_client_tokens_flags():
    r = fake_run()
    r["requests"] = r["requests"][(r["requests"].t_first_token < 120) | (r["requests"].t_first_token > 180)]
    w = L.windows(r)
    assert (~w.kept).sum() == 1 and math.isnan(w[~w.kept].gap.iloc[0])
    r2 = fake_run()
    r2["requests"] = r2["requests"].iloc[0:0]
    s = L.summary("x", r2)
    assert math.isnan(s["gap"]) and L.label(pd.DataFrame([s])) == "flag"
    assert L.label(pd.DataFrame()) == "no_data"


def test_a_first_token_without_t_sent_raises():
    r = fake_run()
    r["requests"].loc[5, "t_sent"] = np.nan
    with pytest.raises(ValueError, match="no t_sent"):
        L.whole(r)


def test_the_label_flags_only_above_the_registered_lines():
    ok = [L.summary("a", fake_run(extra=0.05)), L.summary("b", fake_run(extra=0.099))]
    assert L.label(pd.DataFrame(ok)) == "small"
    assert L.label(pd.DataFrame(ok + [L.summary("c", fake_run(extra=0.11))])) == "flag"


def test_a_counter_that_goes_down_raises():
    r = fake_run()
    r["metrics"].loc[100, L.COUNT] = 0.0
    with pytest.raises(ValueError, match="goes down"):
        L.whole(r)


def test_the_registered_run_list_is_the_valid_unsealed_l1_set():
    runs = L.A5["runs"]
    assert len(runs) == 62 and len(set(runs)) == 62 and all("l1pl350" in n for n in runs)
    assert "20261109-134250_stage2_control_l1pl350cl_r2.4_s6" not in runs
    assert "20261113-033842_stage2_control_l1pl350cl_r2.0_w3000_s18" not in runs
