"""analysis/reg25.py: REG-25's outcome lookup and the A21 readout, on synthetic frames (no run folder read)."""
import math

import pandas as pd
import pytest

from analysis import reg25 as R
from analysis import withheld as W


def test_outcome_rows_follow_the_registered_table():
    assert R.outcome("stuck", "recovered", 1.2, 80.0) == {"row": "stuck_control_recovered", "reading": "reprefill"}
    assert R.outcome("stuck", "recovered", 1.6, 80.0)["reading"] == "cache_loss"
    assert R.outcome("stuck", "recovered", 1.5, 80.0)["reading"] == "cache_loss"      # < 1.5 is re-prefill, strict
    assert R.outcome("stuck", "stuck", 1.6, 80.0)["row"] == "control_not_healthy"
    assert R.outcome("recovered", "slow_drain", 1.6, 80.0)["row"] == "control_not_healthy"
    assert R.outcome("recovered", "recovered", 1.6, 50.0)["reading"] == "backlog_reached_1.5Q"
    assert R.outcome("slow_drain", "recovered", 1.6, 40.0)["reading"] == "backlog_short_of_1.5Q"
    assert R.outcome("stuck", "unclear", 1.6, 80.0)["row"] == "no_reading_listed"
    assert R.outcome("bad_baseline", "recovered", 1.6, 80.0)["row"] == "no_reading_listed"
    assert R.outcome("invalid", "recovered", math.nan, math.nan)["row"] == "invalid"


def _run(counter, running):
    t = [float(x) for x in range(880, 1230, 5)]
    m = pd.DataFrame({"t": t, R.PREEMPT: [counter(x) for x in t], R.RUNNING: [running] * len(t)})
    trig = pd.DataFrame({"t_start": [900.0, 902.5, 905.0], "t_end": [900.1, 902.6, 905.1], "success": [True] * 3})
    return {"metrics": m, "trigger": trig}


def test_a21_reads_the_counter_rise_against_the_running_batch():
    out = R.a21(_run(lambda x: 2.0 if x <= 900 else 2.0 + 5.0 * min(3, math.ceil((x - 900) / 2.5)), 5.0))
    assert out["calls"] == 3 and out["running_sum"] == 15.0
    assert out["b"] == pytest.approx(905.1 + 10.0) and out["natural_rise"] == 0.0 and out["counter_down"] is False
    assert out["rise"] == 15.0


def test_a21_flags_a_counter_that_went_down():
    out = R.a21(_run(lambda x: 10.0 if x < 905 else 1.0, 5.0))
    assert out["counter_down"] is True and math.isnan(out["rise"])


def test_both_runs_are_sealed_by_reg25_until_its_result(tmp_path, monkeypatch):
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-15 · [OPS] REG-25: the queue ended (QUEUE DONE)\n")
    monkeypatch.setattr(W, "DECISIONS", d)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    assert [W.sealed_by(R.REG25[k]) for k in ("control", "wipe")] == ["REG-25", "REG-25"]
