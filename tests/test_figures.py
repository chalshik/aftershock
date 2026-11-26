"""analysis/figures.py on synthetic runs and synthetic run folders only (no real run folder is read)."""
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

import analysis.figures as F
from analysis import load as L
from analysis import withheld


def make_run(at=900.0, dur=180.0, name="synthetic", reuse_after=0.0, kv_after=0.95, q_after=50.0):
    """Sessions every 4 s, 8 turns 5 s apart; later turns reuse their whole previous prompt before the burst's end and
    `reuse_after` of it after; gauges step at the burst's end."""
    end = at + dur
    t = np.arange(0.0, 3000.0)
    m = pd.DataFrame({"t": t, "vllm:kv_cache_usage_perc": np.where(t >= end, kv_after, 0.5),
                      "vllm:num_requests_waiting": np.where(t >= end, q_after, 0.0)})
    rows = []
    for sid, t0 in enumerate(np.arange(0.0, 2900.0, 4.0)):
        for k in range(8):
            ts = t0 + 5 * k
            prev = 2176 + 256 * (k - 1)
            share = 1.0 if ts + 0.3 < end else reuse_after
            rows.append({"session_id": sid, "turn": k, "t_planned": ts, "t_sent": ts, "t_first_token": ts + 0.3,
                         "t_end": ts + 2.0, "status": "ok", "prompt_tokens": 2176 + 256 * k,
                         "prompt_tokens_planned": 2176 + 256 * k, "cached_tokens": 0 if k == 0 else int(share * prev)})
    cfg = {"trigger": {"kind": "burst", "at_s": at, "duration_s": dur, "prompt_tokens": 2944},
           "verdict": {"baseline_s": 600, "watch_s": 1200, "end_s": 600}}
    return {"cfg": cfg, "metrics": m, "requests": pd.DataFrame(rows), "dir": Path(name)}


# ---------------------------------------------------------------- the seed rule

def test_seed_rule_picks_the_lowest_common_valid_seed(monkeypatch):
    have = {"open_s15": "o15", "open_s16": "o16", "open_s17": "o17", "closed_s16": "c16", "closed_s17": "c17"}
    tried = []

    def fake(pat):
        tried.append(pat)
        return ({"dir": Path(have[pat])}, []) if pat in have else (None, [])

    monkeypatch.setattr(F, "counted_run", fake)
    arms = {"open": lambda sd: f"open_s{sd}", "closed": lambda sd: f"closed_s{sd}"}
    sd, runs, _ = F.pick_seed(arms, (17, 16, 15))
    assert sd == 16 and runs["open"]["dir"].name == "o16" and runs["closed"]["dir"].name == "c16"
    assert "open_s17" not in tried            # stops at the first seed that has every arm
    with pytest.raises(ValueError):
        F.pick_seed(arms, (15,))
    assert F.pick_seed(arms, (15, 16, 17), seed=17)[0] == 17


def _folder(root: Path, name: str, valid: bool) -> None:
    d = root / name
    d.mkdir(parents=True)
    (d / "config.yaml").write_text(yaml.safe_dump({"seed": 0}))
    (d / "checks.json").write_text(json.dumps({"valid": valid}))
    (d / "env.json").write_text("{}")


def test_invalid_runs_are_excluded_and_the_earliest_valid_run_counts(tmp_path, monkeypatch):
    _folder(tmp_path, "20990101-000000_burst_l1pl350ol_r2.0_w3000_s15", False)   # open arm invalid at seed 15
    _folder(tmp_path, "20990101-000100_burst_l1pl350cl_r2.0_w3000_s15", True)
    _folder(tmp_path, "20990101-000200_burst_l1pl350ol_r2.0_w3000_s16", True)
    _folder(tmp_path, "20990101-000300_burst_l1pl350ol_r2.0_w3000_s16", True)    # a later valid run: listed only
    _folder(tmp_path, "20990101-000400_burst_l1pl350cl_r2.0_w3000_s16", True)
    _folder(tmp_path, "20990101-000500_burst_l1pl350cl_r2.0_w3000_s16_cap", True)  # another arm's name: not matched
    monkeypatch.setattr(L, "RUNS", tmp_path)
    monkeypatch.setattr(withheld, "sealed_by", lambda name, decisions=None: None)
    sd, runs, later = F.pick_seed(F.MATCHED_ARMS, (15, 16, 17))
    assert sd == 16
    assert runs["open"]["dir"].name == "20990101-000200_burst_l1pl350ol_r2.0_w3000_s16"
    assert later["open"] == ["20990101-000300_burst_l1pl350ol_r2.0_w3000_s16"]
    assert runs["closed"]["dir"].name == "20990101-000400_burst_l1pl350cl_r2.0_w3000_s16"


# ---------------------------------------------------------------- series

def test_series_columns_and_time_axis_from_the_burst_start():
    r = make_run(at=900.0)
    s = F.series(r)
    assert tuple(s.columns) == F.SERIES_COLS
    assert np.allclose(s.t_rel, s.t - 900.0)
    assert s.t.iloc[0] == F.FIG["roll_s"] and np.allclose(np.diff(s.t), F.FIG["grid_s"])
    assert np.allclose(F.series(r, t0=100.0).t_rel, s.t - 100.0)
    assert s.cap_queue.isna().all()             # no cap: NaN, never 0


def test_series_values_follow_the_definitions():
    s = F.series(make_run(at=900.0, dur=180.0))
    before, after = s[(s.t_rel > 0) & (s.t_rel < 150)], s[s.t_rel > 300]
    assert np.allclose(before.reuse, 1.0) and np.allclose(after.reuse.dropna(), 0.0)
    assert np.allclose(before.kv, 0.5) and np.allclose(after.kv, 0.95)
    assert np.allclose(after.waiting, 50.0) and (s[s.t < 2900].n_turns > 0).all()
    assert s[s.n_turns == 0].reuse.isna().all()      # no later turn in the window: NaN, never 0


def test_series_reuse_matches_k1_reuse_on_a_window():
    from analysis.k1 import reuse
    r = make_run(at=900.0, dur=180.0, reuse_after=0.4)
    s = F.series(r)
    row = s[s.t == 1500.0].iloc[0]
    assert row.reuse == pytest.approx(reuse(r, 1500.0 - F.FIG["roll_s"], 1500.0))


def test_trailing_mean_is_nan_without_scrapes():
    t, v = np.array([0.0, 1.0, 2.0, 50.0]), np.array([1.0, 2.0, 3.0, 4.0])
    out = F._trailing_mean(t, v, np.array([2.0, 40.0, 50.0]), 30.0)
    assert out[0] == pytest.approx(2.0) and math.isnan(out[1]) and out[2] == pytest.approx(4.0)


# ---------------------------------------------------------------- model against measurement

def test_model_vs_measurement_is_withheld_before_k3b(monkeypatch):
    monkeypatch.setattr(withheld, "lifted", lambda text, decisions=None: False)
    with pytest.raises(withheld.WithheldError):
        F.model_vs_measurement()
    monkeypatch.setattr(withheld, "lifted", lambda text, decisions=None: text == F.FIG["k3b_result"])
    with pytest.raises(NotImplementedError):
        F.model_vs_measurement()


# ---------------------------------------------------------------- the boundary map

def test_boundary_rows_quote_sources_and_record_errors(tmp_path, monkeypatch):
    dec = tmp_path / "decisions.md"
    dec.write_text("## 2026-11-13 · [RESULT] REG-19 (P0 at λ = 2.0, seeds 15–19): supports\n"
                   "## 2026-11-16 · [RESULT] D1 (C2, session cap): `supports`, 5 of 5\n")
    monkeypatch.setattr(F, "DECISIONS", dec)
    reg19 = {"k": 5, "n": 5, "k_clear": 5, "k_c1": 5, "k_18": 3, "outcome": "supports", "cp_upper": 1.0,
             "rows": [{"control_verdict": "recovered"}] * 5}
    d1t = {"k_stuck": 0, "n": 5, "k_cut": 5, "decision": "supports", "cp_upper_not_cut": 0.451, "below_reg19": True,
           "mcnemar_p": 0.031, "onset_split": {}}
    rows = F.boundary_rows({"reg19": reg19, "reg20": RuntimeError("no runs"), "d1": d1t, "d1b": ValueError("x")})
    by = {r["cell"]: r for r in rows}
    r19 = by[F.CELLS["reg19"]["cell"]]
    assert (r19["k"], r19["n"], r19["cp_upper"]) == (5, 5, 1.0) and r19["cp_lower"] == pytest.approx(0.549, abs=1e-3)
    assert r19["source"].startswith("2026-11-13 · [RESULT] REG-19") and r19["error"] is None
    rd1 = by[F.CELLS["d1"]["cell"]]
    assert rd1["k"] == 0 and rd1["cp_upper"] == pytest.approx(0.451, abs=1e-3) and math.isnan(rd1["cp_lower"]) is False
    for c in ("k1_closed", "k1_open_a", "k1_open_b"):
        r = by[F.CELLS[c]["cell"]]
        assert r["k"] is None and r["error"] == "RuntimeError: no runs" and math.isnan(r["cp_upper"])
        assert r["source"].startswith("(heading not found")
    assert by[F.CELLS["d1b"]["cell"]]["error"] == "ValueError: x"
    F.write_boundary_md(rows, tmp_path / "b.md")
    text = (tmp_path / "b.md").read_text()
    assert "5 of 5" in text and "RuntimeError: no runs" in text and F.FIG["caching_off_note"] in text


def test_cp_lower_mirrors_cp_upper():
    assert F.cp_lower(5, 5) == pytest.approx(0.549, abs=1e-3)
    assert F.cp_lower(0, 5) == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(F.cp_lower(0, 0))


# ---------------------------------------------------------------- Figure 3 (the return to warm)

def test_rewarm_seeds_are_reg27s_and_pinned_in_reg30():
    from analysis import reg30
    assert F.FIG["rewarm_seeds"] == (61, 68, 72, 79) and 43 not in F.FIG["rewarm_seeds"]   # 43 chose the design
    assert all(s in reg30.REG30["runs"] and s in reg30.REG30["controls"] for s in F.FIG["rewarm_seeds"])


def test_rewarm_fig_aligns_both_runs_on_their_own_drop_and_quotes_reg30(monkeypatch):
    burst = {"cfg": {"trigger": {"kind": "burst", "at_s": 900.0, "duration_s": 180.0}}, "dir": Path("b61")}
    ctrl = {"cfg": {}, "dir": Path("c61")}
    monkeypatch.setattr(F, "pick_seed", lambda arms, seeds, seed=None: (61, {"burst": burst, "control": ctrl}, {}))
    drops = {"b61": (5580.0, 14580.0), "c61": (5500.0, 14500.0)}
    monkeypatch.setattr(F.confirm, "phases", lambda cfg: {"drop": drops["b61" if cfg is burst["cfg"] else "c61"],
                                                         "burst_end": 1080.0})
    t0s = {}
    monkeypatch.setattr(F, "series", lambda run, t0=None: t0s.setdefault(run["dir"].name, t0))
    monkeypatch.setattr(F.reg30, "rewarm", lambda run, control=None: {
        "onset_s": 2580.0, "drained_s": 2560.0, "lead_s": -20.0, "n_onset": 9.5, "label": "near_drain"})
    out = F.rewarm_fig(draw=False)
    assert t0s == {"b61": 5580.0, "c61": 5500.0}            # each run's own drop start
    assert out["rewarm"]["label"] == "near_drain" and out["rewarm"]["lead_s"] == -20.0 and out["files"] == []


def test_plot_draws_with_default_and_explicit_shading(tmp_path):
    run = make_run(name="x")
    panels, runs, labels = {"a": F.series(run)}, {"a": run}, {"a": "lab"}
    default = F._plot(panels, runs, labels, "t", tmp_path / "d")
    spans = F._plot(panels, runs, labels, "", tmp_path / "s", spans=[(0.0, 100.0, "0.85")],
                    vlines=[(50.0, ":", "mark")], xlabel="x", paper_names={"a": "arm"})
    assert all(Path(p).exists() for p in default + spans)


def test_plot_column_variant_sets_size_and_font_and_leaves_defaults(tmp_path):
    import matplotlib
    run = make_run(name="x")
    panels, runs, labels = {"a": F.series(run)}, {"a": run}, {"a": "lab"}
    before = matplotlib.rcParams["font.size"]
    out = F._plot(panels, runs, labels, "", tmp_path / "c", paper_names={"a": "arm"},
                  figsize=F.FIG["column_figsize"], font=F.FIG["column_font"])
    assert all(Path(p).exists() for p in out)
    from PIL import Image
    w, h = Image.open(out[0]).size                           # the PNG is drawn at the column size
    assert (w, h) == tuple(round(v * F.FIG["dpi"]) for v in F.FIG["column_figsize"])
    assert matplotlib.rcParams["font.size"] == before         # the font change does not leak to later figures


def test_last_planned_ignores_burst_requests():
    """Figure 1's dashed line (review 2026-11-26): the last planned session request, burst rows excluded, from the
    burst's start."""
    import pandas as pd
    from analysis import figures
    run = {"cfg": {"trigger": {"kind": "burst", "at_s": 900, "prompt_tokens": 2944}},
           "requests": pd.DataFrame({"turn": [0, 1, 0, 0], "prompt_tokens_planned": [2176, 2432, 2944, 2176],
                                     "t_planned": [100.0, 3000.0, 4000.0, 2500.0]})}
    assert figures.last_planned(run) == 2100.0


def test_reuse_line_breaks_at_an_undrawn_bucket():
    """LTB review 2026-11-26: Figure 3's open-loop line crossed the undrawn 1-2 bucket. gap_series puts NaN there."""
    from analysis import m, rewrite_figs as R
    n = {"[0,0.5)": 500, "[0.5,1)": 82, "[1,2)": 18, "[2,4)": 37, "[4,8)": 41, "[8,16)": 5, "[16,inf)": 35}
    pooled = {bk: {"n": n[bk], "reuse": 0.5} for bk in m.X_BUCKETS}
    x, y = R.gap_series(pooled)
    assert len(x) == len(y) == len(m.X_BUCKETS)        # first to last drawn bucket
    assert math.isnan(y[2]) and math.isnan(y[5])        # 18 and 5 turns: not drawn, so the line breaks
    assert all(np.isfinite(v) for i, v in enumerate(y) if i not in (2, 5))
    pooled["[16,inf)"]["n"] = 0
    x, y = R.gap_series(pooled)
    assert len(y) == 5                                  # trailing undrawn buckets are not part of the line
