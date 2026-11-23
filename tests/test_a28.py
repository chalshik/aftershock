"""analysis/a28.py on synthetic schedules: the surge reader, its window, and the agreement count. No run is read."""
import numpy as np

from analysis import a28


def _steady(rate, dur, start=0.0):
    return np.arange(start, dur, 1.0 / rate)


def test_constants_are_k4s_registered_levels():
    assert a28.A28["level"] == 3.4 and a28.A28["near"] == 3.1


def test_a_steady_schedule_has_no_surge_and_a_burst_does():
    calm = a28.surges(_steady(2.0, 900), 900, 3.4, 3.1)
    assert calm["s_at_level"] == 0 and abs(calm["peak"] - 2.0) < 0.02 and abs(calm["mean_rate"] - 2.0) < 0.01
    t = np.concatenate([_steady(2.0, 900), _steady(2.0, 700, 500)])      # 4 req/s over [500, 700)
    s = a28.surges(t, 900, 3.4, 3.1)
    assert s["peak"] > 3.9 and s["s_at_level"] > 0 and 500 <= s["t_peak"] <= 820


def test_arrivals_after_the_run_end_and_before_start_s_window_do_not_count():
    t = np.concatenate([_steady(2.0, 900), _steady(10.0, 1000, 900)])   # a surge after the end
    s = a28.surges(t, 900, 3.4, 3.1)
    assert s["s_at_level"] == 0 and s["n"] == 1800 and abs(s["mean_rate"] - 2.0) < 1e-9


def test_agreement_counts_the_rule_and_its_failure_control():
    rows = [{"seed": "s0", "rate": 2.0, "keeps_up": True, "s_at_level": 0, "peak": 2.8},
            {"seed": "s0", "rate": 2.6, "keeps_up": False, "s_at_level": 55, "peak": 3.8},
            {"seed": "s0", "rate": 2.4, "keeps_up": False, "s_at_level": 0, "peak": 3.2}]
    a = a28.agree(rows)
    assert a["n"] == 3 and a["n_agree"] == 2 and a["disagree"] == [("s0", 2.4, False, 3.2)]
    rows[0]["keeps_up"] = False                                           # a run that failed with no surge: disagrees
    assert a28.agree(rows)["n_agree"] == 1
