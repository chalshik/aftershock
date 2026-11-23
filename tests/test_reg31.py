"""analysis/reg31.py (REG-31, pending): four readouts on hand-made runs and stubbed schedules. No run folder is read and
no table is computed: every table refuses while decisions.md lacks the registration, and is tested only for that."""
import math
import sys
from fnmatch import fnmatch
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from analysis import gate, k1, k4, k4e, k5, withheld
from analysis import paper_tables as PT
from analysis import reg31 as R
from analysis.reg30 import REG30

CONFIGS = Path(__file__).resolve().parent.parent / "configs"


# ---------------------------------------------------------------- settings and pinned run ids

def _config_seeds() -> dict[str, list[int]]:
    """Every integer at any depth under a key whose name contains "seed" (lists and dicts included), per YAML under
    configs/."""
    out = {}

    def walk(x, under, acc):
        if isinstance(x, dict):
            for k, v in x.items():
                walk(v, under or "seed" in str(k).lower(), acc)
        elif isinstance(x, list):
            for v in x:
                walk(v, under, acc)
        elif under and isinstance(x, int) and not isinstance(x, bool):
            acc.append(x)

    for p in sorted(CONFIGS.rglob("*.yaml")):
        acc = []
        walk(yaml.safe_load(p.read_text()), False, acc)
        out[p.name] = acc
    return out


def assert_no_config_uses(seeds) -> None:
    s = set(seeds)
    used = {name: [x for x in v if x in s] for name, v in _config_seeds().items()}
    used = {k: v for k, v in used.items() if v}
    assert not used, f"configs use calm seeds: {used}"


def test_no_config_uses_a_calm_seed():
    seeds = _config_seeds()
    assert len(seeds) > 100 and any(15 in v for v in seeds.values())   # the walk finds seeds where they are
    assert_no_config_uses(R.REG31["calm_seeds"])
    assert R.REG31["calm_seeds"] == range(200000, 202000)


def test_the_helper_would_catch_a_config_seed_in_the_range(tmp_path, monkeypatch):
    with pytest.raises(AssertionError):
        assert_no_config_uses(range(15, 16))
    for i, doc in enumerate(({"seed": 200001}, {"a": {"seeds": [1, {"x": 201999}]}}, {"k4_seed_set": {"lo": [200000]}})):
        (tmp_path / f"c{i}.yaml").write_text(yaml.safe_dump(doc))
    (tmp_path / "ok.yaml").write_text(yaml.safe_dump({"seed": 199999, "other": 200001, "s": {"seed": 202000}}))
    monkeypatch.setattr(sys.modules[__name__], "CONFIGS", tmp_path)
    found = _config_seeds()
    assert {k: sorted(v) for k, v in found.items()} == {"c0.yaml": [200001], "c1.yaml": [1, 201999], "c2.yaml": [200000],
                                                       "ok.yaml": [199999, 202000]}   # safe_dump sorts keys
    with pytest.raises(AssertionError, match="c0.yaml.*c1.yaml.*c2.yaml"):
        assert_no_config_uses(R.REG31["calm_seeds"])


def test_calm_cfg_is_reg19s_control_and_p0_duration_comes_from_the_burst_config():
    assert (CONFIGS / f"{R.REG31['calm_cfg']}.yaml").exists()
    assert R.REG31["calm_cfg"] == PT.PAIRS[("replication", 15)][0].split("_", 1)[1]
    assert R.REG31["p0_cfg"] == PT.PAIRS[("replication", 15)][1].split("_", 1)[1]
    b = gate._burst_cfg(R.REG31["p0_cfg"])
    d = R.p0_duration_s()
    assert d == b["trigger"]["at_s"] + b["trigger"]["duration_s"] + b["verdict"]["watch_s"]
    assert d == float(b["workload"]["duration_s"]) == float(gate._burst_cfg(R.REG31["calm_cfg"])["workload"]["duration_s"])


def test_settings_are_the_ones_reused():
    assert R.REG31["base_window_s"] == gate.REG19["baseline_pair"]
    assert R.REG31["lead_s"] == k5.K5["min_lead_s"]
    lo, hi = R.REG31["pool_lo"], R.REG31["pool_hi"]
    assert lo == 24.4 == round(96656 / 3968, 1) == round(REG30["low_bound"], 1)
    assert hi == 31.5 == round(96656 / 3072, 1) == round(REG30["low_bounds_reported"][1], 1)
    assert R.REG31["pool_band_label"] == "pool / longest prompt to pool / mean prompt"
    assert "mean context" not in Path(R.__file__).read_text()
    assert R.REG31["edge_clock"] == "t_sent" and R.REG31["edge_clock"] in R.CLOCKS


def test_in_service_runs_are_reg19s_closed_and_reg20_arm_a_open_bursts():
    assert sorted(R.REG31["closed_burst"]) == sorted(R.REG31["open_burst"]) == list(k1.REG20["closed_seeds"])
    for sd, rid in R.REG31["closed_burst"].items():
        assert rid == PT.PAIRS[("replication", sd)][1] and fnmatch(rid, k1.closed_pattern(sd))
    for sd, rid in R.REG31["open_burst"].items():
        assert fnmatch(rid, k1.open_pattern(k1.REG20["a_lam"], sd))


def test_lead_runs_are_k4s_tipped_surge_k4es_late_seed_and_the_alarm_runs():
    assert set(R.REG31["k5_events"]) <= set(k5.K5["runs"]) and len(R.REG31["k5_events"]) == 6
    assert sorted(R.REG31["k4_tipped"]) == [sd for sd in k4.REGISTERED["surge"] if sd != 287]
    assert sorted(R.REG31["k4_calm"]) == k4.REGISTERED["calm"]
    for sd, rid in {**R.REG31["k4_tipped"], **R.REG31["k4_calm"]}.items():
        assert fnmatch(rid, k4.run_pattern(sd))
    assert list(R.REG31["k4e_late"]) == [914] and 914 in k4e.REGISTERED["hi"]
    assert fnmatch(R.REG31["k4e_late"][914], k4e.run_pattern(914))
    assert set(R.REG31["reg27_control_seeds"]) <= set(REG30["controls"]) and 43 not in R.REG31["reg27_control_seeds"]
    n_events = len(R.REG31["k5_events"]) + len(R.REG31["k4_tipped"]) + len(R.REG31["k4e_late"])
    assert n_events == 11 and len(R.REG31["reg27_control_seeds"]) + len(R.REG31["k4_calm"]) == 9


def test_no_reg19_control_is_used_anywhere():
    text = Path(R.__file__).read_text()
    assert "control_l1pl350cl_r2.0_w3000_s1" not in text.replace(f'"{R.REG31["calm_cfg"]}"', "")
    ids = [*R.REG31["k5_events"], *R.REG31["k4_tipped"].values(), *R.REG31["k4e_late"].values(),
           *R.REG31["k4_calm"].values(), *(REG30["controls"][sd] for sd in R.REG31["reg27_control_seeds"])]
    assert not [i for i in ids if "_w3000_" in i]


@pytest.mark.parametrize("fn", ["calm_table", "in_service_table", "slow_table", "lead_table"])
def test_tables_refuse_before_the_registration(fn, monkeypatch):
    monkeypatch.setattr(withheld, "lifted", lambda heading, decisions=None: False)
    monkeypatch.setattr(R, "find_runs", lambda *a, **k: pytest.fail("a run was read"))
    monkeypatch.setattr(gate, "schedule_peak_series", lambda *a, **k: pytest.fail("a schedule was built"))
    with pytest.raises(RuntimeError, match="not registered"):
        getattr(R, fn)()


# ---------------------------------------------------------------- 1. calm_share (stubbed schedules)

def test_calm_share_counts_seeds_at_or_above_each_level(monkeypatch):
    peaks = {1: 3.0, 2: 3.2, 3: 3.3, 4: 3.4, 5: 3.5}
    seen = []

    def series(cfg, lam, seed, duration_s):
        seen.append((cfg, lam, seed, duration_s))
        g = np.arange(240.0, duration_s, 5.0)
        return g, np.where(g == 500.0, peaks[seed], 2.0)

    monkeypatch.setattr(gate, "_burst_cfg", lambda name: {"name": name})
    monkeypatch.setattr(gate, "schedule_peak_series", series)
    out = R.calm_share(2.2, range(1, 6), 1000.0, (3.2, 3.4))
    assert out["seeds"] == [1, 2, 3, 4, 5] and out["lam"] == 2.2 and out["duration_s"] == 1000.0
    assert out["levels"][3.2]["n"] == 5 and out["levels"][3.2]["k"] == 4 and out["levels"][3.2]["share"] == 0.8
    assert out["levels"][3.4]["k"] == 2 and out["levels"][3.4]["share"] == 0.4          # >= the level counts
    assert 0.4 < out["levels"][3.4]["cp_upper"] < 1.0
    assert all(c == {"name": R.REG31["calm_cfg"]} and lam == 2.2 and d == 1000.0 for c, lam, _, d in seen)


# ---------------------------------------------------------------- 2. in service at the burst's end

def _sessions(rows, at=10.0, dur=2.0):
    q = pd.DataFrame(rows, columns=["session_id", "turn", "t_planned", "t_sent", "t_end"])
    q["prompt_tokens_planned"] = np.where(q.session_id >= 90, 64, 2176)
    q["status"] = np.where(q.t_end.notna(), "ok", "unfinished")
    return {"requests": q, "dir": Path("synthetic"), "metrics": pd.DataFrame(),
            "cfg": {"trigger": {"kind": "burst", "at_s": at, "duration_s": dur, "prompt_tokens": 64},
                    "workload": {"duration_s": 20.0}, "checks": {"max_scrape_gap_s": 5.0}}}


def test_in_service_at_the_bursts_end(monkeypatch):
    monkeypatch.setattr(gate, "secondary_cb", lambda run, gap: {"backlog_s": 0.0})
    run = _sessions([[0, 0, 0.0, 0.0, 1.0], [0, 1, 2.0, 2.0, 3.0],     # session 0: [0, 3)
                     [1, 0, 1.0, 1.0, 2.0], [1, 1, 3.0, 3.0, 13.0],    # session 1: [1, 13)
                     [2, 0, 5.0, 5.0, 12.0],                           # session 2: [5, 12): ends at the burst's end
                     [3, 0, 12.0, 12.0, None],                         # session 3: starts at the burst's end
                     [4, 0, 11.0, 11.0, 11.5],                         # session 4: [11, 11.5)
                     [90, 0, 10.0, 10.0, 10.5]])                       # a burst request: not a session
    out = R.in_service_at_burst_end(run)
    assert out["burst_end_s"] == 12.0
    assert out["at_end"] == 2                       # sessions 1 and 3; session 2's end comes first at t = 12
    assert out["peak_before_end"] == 3              # sessions 1, 2 and 4 over [11, 11.5)
    # outstanding: session requests due by 12 and not ended by then: session 1's turn 1 (13) and session 3 (none)
    assert out["outstanding"] == gate.shock_size(run)["outstanding"] == 2


def test_in_service_before_any_session():
    run = _sessions([[0, 0, 15.0, 15.0, 16.0]])
    t, n = R.d1.in_service_steps(run)
    assert t[0] > 12.0 and R.d1.peak(run, before=12.0)["before"] == 0


def test_band_counts_at_or_below_lo_and_strictly_above_hi():
    out = R.band_counts([{"at_end": v} for v in (24, 25, 31, 32, 40)])
    assert out["at_or_below_lo"]["k"] == 1 and out["at_or_below_lo"]["n"] == 5     # 24 <= 24.4
    assert out["above_hi"]["k"] == 2 and out["above_hi"]["share"] == 0.4           # 32 and 40 > 31.5
    assert 0.4 < out["above_hi"]["cp_upper"] < 1.0
    assert R.band_counts([{"at_end": 31}])["above_hi"]["k"] == 0                   # inside the band


# ---------------------------------------------------------------- 3. slow baseline requests

def _pair_run(rows, trigger=True, name="b", at=900.0):
    q = pd.DataFrame(rows, columns=["t_planned", "turn", "status", "ttft", "t_sent", "t_first_token"])
    q["prompt_tokens_planned"] = np.where(q.turn < 0, 64, 2176)
    q["turn"] = q.turn.clip(lower=0)
    cfg = {"slo": {"ttft_s": 1.0}, "checks": {"max_scrape_gap_s": 5.0}, "workload": {"duration_s": 2000.0}}
    if trigger:
        cfg["trigger"] = {"kind": "burst", "at_s": at, "duration_s": 180.0, "prompt_tokens": 64}
    return {"requests": q, "cfg": cfg, "dir": Path(name), "metrics": pd.DataFrame()}


NAN = np.nan
ROWS = [[299.0, 0, "ok", 5.0, 299.0, 304.0],          # before the window
        [300.0, 1, "ok", 0.5, 300.0, 300.5],          # fast, at the window's start
        [400.0, 2, "ok", 1.0, 400.0, 401.0],          # fast: TTFT equal to the SLO is fast
        [500.0, 3, "ok", 1.2, 500.0, 501.2],          # slow, at no edge
        [600.0, 0, "unfinished", NAN, 600.0, NAN],    # slow (unfinished is a miss), no first token
        [885.0, 6, "ok", 11.0, 891.0, 902.0],         # slow: planned before the edge, sent in it, first token after
        [889.0, 4, "ok", 3.0, 889.0, 892.0],          # slow, at no edge
        [890.0, 5, "error", 0.1, 890.0, 890.1],       # slow (error), planned and sent in the edge
        [899.5, 7, "ok", 2.0, 899.5, 901.5],          # slow, at every edge
        [900.0, 8, "ok", 9.0, 900.0, 909.0]]          # after the window


def test_slow_baseline_counts_not_fast_session_requests_and_the_edge_by_three_clocks():
    burst = _pair_run(ROWS + [[905.0, -1, "ok", 9.0, 905.0, 914.0]])      # a burst request after the window
    control = _pair_run(ROWS[:3] + [[895.0, 1, "ok", 1.5, 895.0, 896.5]], trigger=False, name="c")
    out = R.slow_baseline(control, burst)
    b, c = out["burst"], out["control"]
    assert b["run"] == "b" and b["n_session"] == 8 and b["slow"] == 6 and b["at_s"] == 900.0
    assert b["edge"] == {"t_planned": 2, "t_sent": 3, "first_token_after": 2} and b["slow_no_first_token"] == 1
    assert b["edge_share_of_planned"] == 2 / 8 and b["edge_share_uniform"] == 10.0 / 600.0
    assert [r["t_planned"] for r in b["requests"]] == [500.0, 600.0, 885.0, 889.0, 890.0, 899.5]
    assert [r["turn"] for r in b["requests"]] == [3, 0, 6, 4, 5, 7]
    assert b["requests"][0]["ttft"] == 1.2 and b["requests"][1]["ttft"] is None
    assert b["requests"][1]["t_first_token"] is None and b["requests"][2]["t_sent"] == 891.0
    assert [r["status"] for r in b["requests"]][1:5] == ["unfinished", "ok", "ok", "error"]
    assert c["run"] == "c" and c["slow"] == 1 and c["at_s"] == 900.0                    # read at the burst's at_s
    assert c["edge"] == {"t_planned": 1, "t_sent": 1, "first_token_after": 0}


def test_the_bursts_at_s_must_be_the_windows_end():
    burst = _pair_run(ROWS, at=800.0)
    with pytest.raises(ValueError, match="not the baseline window's end"):
        R.slow_baseline(_pair_run(ROWS, trigger=False), burst)


def test_the_not_fast_rule_is_percent_fasts():
    run = _pair_run(ROWS)
    w, slow = R._not_fast(run, 300.0, 900.0)
    assert 1 - slow.mean() == pytest.approx(R._level(run, 300.0, 900.0)["pf"]) == 2 / 8


def test_a_burst_request_in_the_window_is_not_counted():
    run = _pair_run(ROWS + [[700.0, -1, "ok", 9.0, 700.0, 709.0]])
    out = R.slow_baseline(run, run)["burst"]
    assert out["slow"] == 6 and out["n_session"] == 8


def test_pooled_edge_sums_the_burst_runs_per_clock():
    rows = [{"slow": 6, "n_session": 8, "edge_share_of_planned": 0.25,
             "edge": {"t_planned": 2, "t_sent": 3, "first_token_after": 2}},
            {"slow": 4, "n_session": 12, "edge_share_of_planned": 0.0,
             "edge": {"t_planned": 0, "t_sent": 1, "first_token_after": 0}}]
    out = R.pooled_edge(rows)
    assert out["t_planned"]["k"] == 2 and out["t_sent"]["k"] == 4 and out["first_token_after"]["k"] == 2
    assert all(out[c]["n"] == 10 for c in R.CLOCKS) and out["t_sent"]["share"] == 0.4
    assert out["t_sent"]["share"] < out["t_sent"]["cp_upper"] < 1.0 and out["edge_clock"] == "t_sent"
    assert out["planned_reference"]["edge_share_of_planned"] == 2 / 20
    assert out["planned_reference"]["edge_share_uniform"] == 10.0 / 600.0


# ---------------------------------------------------------------- 4. running-batch lead (K5's hand-made server)

END = 1500.0


def k5_run(running=None, h=None, ttft=None, kv=None, surge=None, name="fake"):
    """test_k5's warm server (2 req/s, hit 0.84, 0.2 s TTFT, KV 0.2), scraped every second; `running` maps the scrape
    times to the running gauge (6 by default); each other argument is when that signal moves to its cold value (hit 0,
    TTFT 3 s, KV 0.95); `surge` a (from, to) pair with 4 requests/s sent."""
    t, inf = np.arange(0.0, END, 1.0), math.inf
    cold_h = t >= (h if h is not None else inf)
    per = np.where(t >= (ttft if ttft is not None else inf), 3.0, 0.2)
    m = pd.DataFrame({
        "t": t,
        k5.COL["queries"]: np.cumsum(np.full(len(t), 2.0)),
        k5.COL["hits"]: np.cumsum(np.where(cold_h, 0.0, 1.68)),
        k5.COL["prompt"]: np.cumsum(np.full(len(t), 6000.0)),
        k5.COL["cached"]: np.cumsum(np.full(len(t), 5000.0)),
        k5.COL["waiting"]: np.zeros(len(t)),
        k5.COL["running"]: running(t) if running else np.full(len(t), 6.0),
        k5.COL["ttft_count"]: np.cumsum(np.full(len(t), 2.0)),
        k5.COL["ttft_sum"]: np.cumsum(2.0 * per),
        "vllm:kv_cache_usage_perc": np.where(t >= (kv if kv is not None else inf), 0.95, 0.2),
    })
    sent = list(np.arange(0.0, END, 0.5))
    if surge:
        sent += list(np.arange(surge[0], surge[1], 0.5) + 0.25)
    reqs = pd.DataFrame({"t_sent": sorted(sent), "turn": 1, "prompt_tokens_planned": 2432})
    reqs["t_planned"] = reqs.t_sent
    return {"dir": Path(name), "cfg": {"workload": {"rate": 2.0, "duration_s": END}}, "metrics": m, "requests": reqs}


def step(*spans, hi=30.0):
    """The running gauge: hi inside each [a, b) span, 6 elsewhere."""
    return lambda t: np.where(np.any([(t >= a) & (t < b) for a, b in spans], axis=0), hi, 6.0)


TIP = dict(h=900.0, ttft=960.0, kv=950.0)


def test_batch_lead_is_latency_onset_minus_the_stretch_into_it():
    out = R.batch_lead(k5_run(running=step((900.0, END)), **TIP))
    on = R.k5.event(k5_run(running=step((900.0, END)), **TIP))["onsets"]
    assert out["why"] is None and out["t_batch"] == 900.0 == out["t_first"] and out["exceeds"]
    assert out["l_onset"] == on["L"] and out["lead"] == on["L"] - 900.0 and out["lead"] > R.REG31["lead_s"]
    assert math.isnan(out["a_onset"]) and out["lead_over_A"] == math.inf       # arrivals never surged: k5's +inf


def test_t_batch_is_the_last_unbroken_stretch_and_t_first_the_first_crossing():
    out = R.batch_lead(k5_run(running=step((400.0, 500.0), (920.0, END)), **TIP))
    assert out["t_first"] == 400.0 and out["t_batch"] == 920.0 and out["lead"] == out["l_onset"] - 920.0


def test_no_t_batch_when_the_gauge_is_not_above_the_level_at_the_latency_onset():
    out = R.batch_lead(k5_run(running=step((900.0, 950.0)), **TIP))
    assert out["l_onset"] > 950.0 and math.isnan(out["t_batch"]) and math.isnan(out["lead"])
    assert out["t_first"] == 900.0 and out["exceeds"] and math.isnan(out["lead_over_A"])


def test_lead_over_arrivals_is_a_minus_t_batch():
    out = R.batch_lead(k5_run(running=step((900.0, END)), surge=(700.0, 900.0), **TIP))
    assert 760 <= out["a_onset"] <= 800 and out["lead_over_A"] == out["a_onset"] - 900.0 < 0


def test_the_level_is_exceeded_only_strictly_above_it():
    lv = float(R.REG31["batch_level"])
    out = R.batch_lead(k5_run(running=step((900.0, END), hi=lv), **TIP))
    assert not out["exceeds"] and math.isnan(out["t_batch"]) and math.isnan(out["t_first"]) and math.isnan(out["lead"])
    assert out["n_max_after_ref"] == lv and not math.isnan(out["l_onset"])


def test_a_scrape_above_the_level_before_ref_start_is_not_t_first():
    out = R.batch_cross(k5_run(running=step((100.0, 101.0), (1000.0, END))))
    assert out["t_first"] == 1000.0 and math.isnan(out["t_batch"])                # no L onset given: no t_batch


def test_a_run_that_never_tipped_has_no_latency_onset_and_no_lead():
    out = R.batch_lead(k5_run(running=step((600.0, END))))
    assert out["why"] == "never tipped" and out["exceeds"] and out["t_first"] == 600.0
    assert math.isnan(out["l_onset"]) and math.isnan(out["t_batch"]) and math.isnan(out["lead"])


def test_batch_cross_without_the_gauge():
    run = k5_run()
    run["metrics"] = run["metrics"].drop(columns=[k5.COL["running"]])
    out = R.batch_cross(run, 960.0)
    assert not out["exceeds"] and math.isnan(out["t_batch"]) and math.isnan(out["n_max_after_ref"])


def test_lead_counts_keep_only_events_with_why_none_and_both_times():
    rows = [{"why": None, "lead": 60.0}, {"why": None, "lead": 30.0}, {"why": None, "lead": math.nan},
            {"why": "no reference", "lead": math.nan}, {"why": None, "lead": -5.0}]
    out = R.lead_counts(rows)
    assert out["n_events"] == 5 and out["n_with_lead"] == 3
    assert out["lead_above"]["k"] == 1 and out["lead_above"]["n"] == 3               # only 60 > 30


def test_alarm_counts_leave_tipped_runs_out_of_the_warm_count():
    rows = [{"kv_onset": math.nan, "exceeds": True}, {"kv_onset": math.nan, "exceeds": False},
            {"kv_onset": 1200.0, "exceeds": True}]
    out = R.alarm_counts(rows)
    assert out["n_alarm"] == 3 and out["n_tipped"] == 1
    assert out["warm_exceeding"]["k"] == 1 and out["warm_exceeding"]["n"] == 2


def test_cp_upper_log_matches_stats_and_handles_large_n():
    from analysis import reg31
    from analysis.stats import cp_upper as exact
    for k, n in [(0, 5), (3, 5), (17, 150), (0, 199)]:
        assert abs(reg31._binom_cdf_log(k, n, 0.3) - sum(__import__("math").comb(n, i) * 0.3 ** i * 0.7 ** (n - i)
                                                         for i in range(k + 1))) < 1e-12
    # the log path itself, on n where the exact path still works
    for k, n in [(0, 150), (40, 180)]:
        lo, hi = k / n, 1.0
        for _ in range(100):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if reg31._binom_cdf_log(k, n, mid) > 0.05 else (lo, mid)
        assert abs(hi - exact(k, n)) < 1e-9
    u = reg31.cp_upper(0, 2000)
    assert abs(u - (1 - 0.05 ** (1 / 2000))) < 1e-9      # k = 0: 1 - alpha^(1/n)
    assert 0.5 < reg31.cp_upper(1100, 2000) < 0.6
    assert reg31.cp_upper(2000, 2000) == 1.0
