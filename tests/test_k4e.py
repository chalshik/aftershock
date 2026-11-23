"""analysis/k4e.py (K4's edge): the pick, configs, the counting and the outcome; no run folder is read."""
import fnmatch
import math

import numpy as np
import pytest

from analysis import gate, withheld
from analysis import k4 as K
from analysis import k4e as E

R = E.REGISTERED


def test_pick_gives_the_registered_seeds():
    pk = E.pick()
    assert pk["lo"] == R["lo"] and pk["hi"] == R["hi"] and pk["order"] == [sd for sd in R["order"] if sd not in (R["positive"], R["calm"])] and not pk["scanned_more"]
    for sd, row in pk["rows"].items():
        assert row["t_cross"] == R["t_cross"][sd] and row["peak"] == pytest.approx(R["peak"][sd])


def test_classify_strata_and_windows():
    g = np.arange(240.0, 2700.0, 5.0)
    def series(peak, t_peak, pre=None):
        x = np.full(len(g), 2.5)
        x[g == t_peak] = peak
        if pre is not None:
            x[g == pre] = 3.15
        return x
    assert E.classify(g, series(3.2, 1500.0))["stratum"] == "lo"
    assert E.classify(g, series(3.3, 1500.0))["stratum"] == "hi"
    assert E.classify(g, series(3.4, 1500.0))["stratum"] is None           # K4's surge level: out of the band
    assert E.classify(g, series(3.09, 1500.0))["stratum"] is None
    assert E.classify(g, series(3.3, 1900.0))["stratum"] is None           # peak after the window
    assert E.classify(g, series(3.3, 1500.0, pre=1000.0))["stratum"] is None   # >= 3.1 before 1200 s


@pytest.mark.parametrize("seed,st", E._all_seeds())
def test_configs_are_k4s_workload_with_seed_and_name_alone(seed, st):
    got, k4cfg = E.resolved(seed, st), K.resolved(133, "surge")
    diff = sorted(k for k in set(got) | set(k4cfg) if got.get(k) != k4cfg.get(k))
    assert diff == (["name"] if seed == 133 else ["name", "seed"])
    from pathlib import Path
    f = Path(K._configs()) / f"{E.config_name(seed)}.yaml"
    assert f.read_text() == E.config_text(seed, st)


def test_the_positive_control_is_k4s_seed_137_but_for_its_name():
    got, k4cfg = E.resolved(137, "positive"), K.resolved(137, "surge")
    assert sorted(k for k in set(got) | set(k4cfg) if got.get(k) != k4cfg.get(k)) == ["name"]
    assert R["positive_t_cross"] == K.REGISTERED["t_cross"][137]


def test_the_calm_control_is_k4s_highest_calm_seed_but_for_its_name():
    got, k4cfg = E.resolved(103, "calm"), K.resolved(103, "calm")
    assert sorted(k for k in set(got) | set(k4cfg) if got.get(k) != k4cfg.get(k)) == ["name"]
    cfg = gate._burst_cfg(K.K4["base_cfg"])
    peaks = {sd: float(gate.schedule_peak_series(cfg, E.K4E["lam"], sd, K.K4["duration_s"])[1].max())
             for sd in K.REGISTERED["calm"]}
    assert max(peaks, key=peaks.get) == R["calm"] == 103


@pytest.mark.parametrize("seed", R["lo"] + R["hi"])
def test_the_run_schedule_is_the_scanned_schedule(seed):
    cfg = E.resolved(seed, "lo" if seed in R["lo"] else "hi")
    g, x = gate.schedule_peak_series(gate._burst_cfg(K.K4["base_cfg"]), E.K4E["lam"], seed, K.K4["duration_s"])
    from harness.workload import build_schedule
    s, _ = build_schedule(cfg["workload"], cfg["seed"], cfg.get("trigger"))
    y = gate.rolling_rate(s.t_planned.to_numpy(float), g, gate.SURGE["window_s"])
    assert np.allclose(x, y)


def lab(label, onset=None):
    return {"label": label, "onset": onset}


@pytest.mark.parametrize("label,onset,want", [
    ("tipped", 1600.0, "tip"), ("tipped", 1545.0, "tip"),            # t_cross 1600: the window opens 60 s early
    ("tipped", 1812.0, "tip"), ("tipped", 1813.0, "late"),           # and closes 212 s after
    ("tipped", 1539.0, "surgeless"), ("tipped", 1100.0, "surgeless"), ("tipped_early", 400.0, "surgeless"),
    ("tipped_recovered", 1500.0, "surgeless"),
    ("tipped_recovered", 1650.0, "recovered"), ("tipped_unclear", 1650.0, "unclear"),
    ("tipped_recovered", 2000.0, "late"),
    ("warm", None, "warm"), ("bad_baseline", None, "bad_baseline"), ("unclear", None, "unclear"),
])
def test_counted(label, onset, want):
    assert E.counted(lab(label, onset), 1600.0) == want


def test_seed_663s_window_opens_before_1200_s():          # rule F r2 N2
    assert E.counted(lab("tipped", 1190.0), R["t_cross"][663]) == "tip"
    assert E.counted(lab("tipped", 1180.0), R["t_cross"][663]) == "surgeless"


def _row(sd, c, pk):
    lb = c.split(":")[1] if ":" in c else {"tip": "tipped", "recovered": "tipped_recovered"}.get(c, c)
    return {"seed": sd, "class": c.split(":")[0], "label": lb, "peak": pk}


def rows(lo, hi):
    return {"lo": [_row(i, c, 3.1 + i / 100) for i, c in enumerate(lo)],
            "hi": [_row(10 + i, c, 3.3 + i / 100) for i, c in enumerate(hi)]}


TIP = {"class": "tip"}


@pytest.mark.parametrize("lo,hi,want", [
    (["warm"] * 5, ["warm"] * 5, "band_warm"),
    (["warm"] * 5, ["tip"] + ["warm"] * 4, "band_warm"),
    (["recovered"] * 2 + ["warm"] * 3, ["recovered"] + ["warm"] * 4, "band_warm"),   # recovered is counted, not a tip
    (["tip"] + ["warm"] * 4, ["tip"] + ["warm"] * 4, "mixed"),
    (["warm"] * 5, ["tip"] * 3 + ["warm"] * 2, "mixed"),
    (["tip"] * 2 + ["warm"] * 3, ["tip"] * 2 + ["warm"] * 3, "band_tips"),
    (["warm"] * 4 + ["late"], ["warm"] * 3 + ["unclear", "surgeless"], "too_few"),
    # rule F r2 N1: persistent tips outside the window block band_warm
    (["tip"] + ["warm"] * 4, ["surgeless:tipped", "late:tipped"] + ["warm"] * 3, "mixed"),
    (["warm"] * 5, ["late:tipped", "surgeless:tipped_recovered"] + ["warm"] * 3, "band_warm"),
    (["warm"] * 5, ["late:tipped", "surgeless:tipped"] + ["warm"] * 3, "mixed"),
])
def test_outcome(lo, hi, want):
    assert E.outcome(rows(lo, hi), TIP)["decision"] == want


def test_the_positive_control_is_reported_beside_the_band():
    assert E.outcome(rows(["warm"] * 5, ["warm"] * 5), TIP)["positive_ok"] is True
    assert E.outcome(rows(["warm"] * 5, ["warm"] * 5), {"class": "warm"})["positive_ok"] is False
    assert E.outcome(rows(["warm"] * 5, ["warm"] * 5), None)["positive_ok"] is None
    o = E.outcome(rows(["warm"] * 5, ["warm"] * 5), {"class": "recovered"}, {"class": "warm"})
    assert (o["positive_class"], o["calm_ok"], o["calm_class"]) == ("recovered", True, "warm")
    assert E.outcome(rows(["warm"] * 5, ["warm"] * 5), TIP, {"class": "tipped"})["calm_ok"] is False


def test_surgeless_runs_are_flagged_and_not_counted():
    o = E.outcome(rows(["surgeless"] + ["warm"] * 4, ["warm"] * 5), TIP)
    assert o["surgeless"] == [0] and o["band"]["n"] == 9 and o["decision"] == "band_warm"
    o = E.outcome(rows(["recovered"] + ["warm"] * 4, ["warm"] * 5), TIP)
    assert o["recovered"] == [0] and o["band"]["k"] == 0


def test_predicted_sums_to_one():
    assert sum(E.predicted().values()) == pytest.approx(1.0)


def test_exact_peak_is_at_least_the_grid_peak():
    for sd in (630, 813):
        r = E.exact_peak(sd)
        assert r["exact_peak"] >= r["grid_peak"] - 1e-12


def test_seal_covers_the_twelve_and_not_k4():
    rows_ = [r for r in withheld.SEALS if r[0] == "K4e"]
    assert len(rows_) == 3
    for sd, _ in E._all_seeds():
        n = f"20261119-000000_{E.config_name(sd)}"
        assert [r[0] for r in withheld.SEALS if fnmatch.fnmatchcase(n, r[1])] == ["K4e"]
    for other in ("x_k4_l1pl350cl_r2.2_s133", "x_k4_l1pl350cl_r2.2_s137", "x_k4_l1pl350cl_r2.2_s103"):
        assert not any(r[0] == "K4e" for r in withheld.SEALS if fnmatch.fnmatchcase(other, r[1]))


def test_only_its_result_lifts_it(tmp_path):
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-19 · [RESULT] K4 (the warm state ends at a surge): …\n## 2026-11-19 · [REG] K4e (K4's edge): …\n")
    assert not withheld.lifted("[RESULT] K4e (K4's edge)", d)
    d.write_text("## 2026-11-20 · [RESULT] K4e (K4's edge): band_warm …\n")
    assert withheld.lifted("[RESULT] K4e (K4's edge)", d)


def test_table_refuses_while_sealed(tmp_path, monkeypatch):
    import analysis.load as L
    (tmp_path / f"20261119-000000_{E.config_name(630)}").mkdir()
    monkeypatch.setattr(L, "RUNS", tmp_path)
    monkeypatch.setattr(withheld, "lifted", lambda *a, **k: False)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    with pytest.raises(withheld.WithheldError):
        E.table()


def test_table_assembles_bands_and_both_controls(monkeypatch):
    """The registered computation's bookkeeping, on stub runs (no run folder): the configs, classes and controls."""
    import analysis.load as L
    from analysis import d1
    monkeypatch.setattr(E, "_sealed", lambda: [])
    monkeypatch.setattr(E, "pick", lambda: {"lo": R["lo"], "hi": R["hi"], "rows": {
        sd: {"t_cross": R["t_cross"][sd], "peak": R["peak"][sd]} for sd in R["lo"] + R["hi"]}})
    stub = {sd: [{"dir": type("D", (), {"name": f"x_{E.config_name(sd)}"})(), "cfg": {"seed": sd}}]
            for sd, _ in E._all_seeds() if sd != 914}
    monkeypatch.setattr(L, "find_runs", lambda pat: stub.get(int(pat.rsplit("_s", 1)[1]), []))
    monkeypatch.setattr(d1, "_diff_keys", lambda a, b: [])
    onsets = {137: 1430.0, 103: None, 630: 1700.0, 609: 1200.0}
    monkeypatch.setattr(K, "label", lambda r, tc: {"label": "tipped" if onsets.get(r["cfg"]["seed"]) else "warm",
                                                  "onset": onsets.get(r["cfg"]["seed"])})
    monkeypatch.setattr(K, "readouts", lambda r, tc: {"tc": tc})
    t = E.table()
    assert t["not_run"] == [914]
    assert t["positive"]["class"] == "tip" and t["calm"]["class"] == "warm" and t["calm"]["t_cross"] is None
    cls = {r["seed"]: r["class"] for r in t["rows"]["lo"] + t["rows"]["hi"]}
    assert cls[630] == "tip" and cls[609] == "surgeless" and cls[650] == "warm"
    o = t["outcome"]
    assert (o["band"]["k"], o["band"]["n"], o["positive_ok"], o["calm_ok"]) == (1, 8, True, True)
    assert o["tipped_outside_window"] == [609] and o["decision"] == "mixed"
