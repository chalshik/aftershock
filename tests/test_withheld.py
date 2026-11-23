"""The seal on runs a pending registration withholds (analysis/withheld.py)."""
import fnmatch
import json
from pathlib import Path

import pytest

import analysis.load as L
import analysis.recheck as R
from analysis import withheld as W
from harness.run import load_config

CONFIGS = Path(__file__).resolve().parent.parent / "configs"
REG19 = [f"{k}_l1pl350cl_r2.0_w3000_s{s}" for s in (15, 16, 17, 18, 19) for k in ("stage2_control", "burst")]


@pytest.fixture
def no_result(tmp_path, monkeypatch):
    """A decisions.md with no [RESULT] of REG-19 or REG-20, and no unseal variable."""
    d = tmp_path / "decisions.md"
    d.write_text("## 2026-11-11 · [REG] REG-19 (confirmatory)\n- body mentions [RESULT] REG-19 without being a heading\n")
    monkeypatch.setattr(W, "DECISIONS", d)
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    return d


def test_reg19_seal_covers_exactly_its_ten_configs(no_result):
    sealed = sorted(c.stem for c in CONFIGS.glob("*.yaml") if W.sealed_by(f"20261112-050920_{c.stem}") == "REG-19")
    assert sealed == sorted(REG19)


def test_reg20_seal_covers_its_open_arms_controls_and_drift(no_result):
    sealed = sorted(c.stem for c in CONFIGS.glob("*.yaml") if W.sealed_by(f"20261114-000000_{c.stem}") == "REG-20")
    want = ([f"burst_l1pl350ol_r2.0_w3000_s{s}" for s in range(15, 20)]
            + [f"stage2_control_l1pl350ol_r2.0_w3000_s{s}" for s in range(15, 20)]
            + [f"{k}_l1pl350ol_r2.5_w3000_s{s}" for s in (15, 16, 17) for k in ("burst", "stage2_control")]
            + ["burst_l1pl350cl_r2.0_w3000_s15_drift"])
    assert sealed == sorted(want)


@pytest.mark.parametrize("name", ["20261110-182920_burst_l1pl350cl_r2.0_s2", "20261111-101805_burst_l1pl350cl_r2.0_s4",
                                  "20261110-085730_confirm_l1pl350cl_r2.2_s12", "20261107-075951_sweep_l1pl350_pc_off_r1",
                                  "20261112-050920_burst_l1pl350cl_r2.0_w3000_s150"])
def test_runs_on_record_are_not_sealed(no_result, name):
    assert W.sealed_by(name) is None


def test_seal_lifts_only_on_a_result_heading(no_result):
    name = "20261112-061951_burst_l1pl350cl_r2.0_w3000_s15"
    assert W.sealed_by(name) == "REG-19"                       # the body line above is not a heading
    no_result.write_text(no_result.read_text() + "\n## 2026-11-14 · [RESULT] REG-19: some, 2 of 5\n")
    assert W.sealed_by(name) is None
    assert W.sealed_by("20261114-000000_burst_l1pl350ol_r2.0_w3000_s15") == "REG-20"   # another seal stays


def test_unseal_variable_names_the_registration(no_result, monkeypatch):
    name = "20261114-000000_burst_l1pl350ol_r2.0_w3000_s15"
    monkeypatch.setenv("AFTERSHOCK_UNSEAL", "REG-19")
    assert W.sealed_by(name) == "REG-20"
    monkeypatch.setenv("AFTERSHOCK_UNSEAL", "REG-19, REG-20")
    assert W.sealed_by(name) is None


def _run_folder(root: Path, name: str) -> Path:
    d = root / name
    d.mkdir()
    (d / "config.yaml").write_text("seed: 1\n")
    (d / "env.json").write_text("{}")
    (d / "checks.json").write_text(json.dumps({"valid": True, "failed": []}))
    return d


def test_load_run_refuses_a_sealed_folder_and_find_runs_skips_it(no_result, tmp_path, monkeypatch, capsys):
    runs = tmp_path / "runs"
    runs.mkdir()
    sealed = _run_folder(runs, "20261112-061951_burst_l1pl350cl_r2.0_w3000_s15")
    _run_folder(runs, "20261110-182920_burst_l1pl350cl_r2.0_s2")
    (sealed / "requests.jsonl").write_text("this line would not parse\n")   # proof that nothing is read
    monkeypatch.setattr(L, "RUNS", runs)
    with pytest.raises(W.WithheldError):
        L.load_run(sealed)
    found = L.find_runs("*_l1pl350cl_*")
    assert [r["dir"].name for r in found] == ["20261110-182920_burst_l1pl350cl_r2.0_s2"]
    assert "skipped 1 run folder(s) sealed by REG-19" in capsys.readouterr().err
    no_result.write_text("## 2026-11-14 · [RESULT] REG-19: none\n")       # lifted: the folder is read again
    with pytest.raises(Exception) as e:
        L.load_run(sealed)
    assert not isinstance(e.value, W.WithheldError)                        # it now fails on the bad line, i.e. it reads


def test_recheck_skips_a_sealed_folder(no_result, tmp_path, monkeypatch):
    runs = tmp_path / "runs"
    runs.mkdir()
    _run_folder(runs, "20261112-061951_burst_l1pl350cl_r2.0_w3000_s15")
    _run_folder(runs, "20261110-182920_burst_l1pl350cl_r2.0_s2")
    monkeypatch.setattr(R, "RUNS", runs)
    seen = []
    monkeypatch.setattr(R, "run_checks", lambda d: seen.append(d.name) or {"valid": True, "failed": [], "checks": []})
    rows = R.recheck("*")
    assert seen == ["20261110-182920_burst_l1pl350cl_r2.0_s2"] and [r["run"] for r in rows] == seen


def test_the_repo_state_today():
    """While decisions.md has no [RESULT] REG-19 heading, REG-19's runs on record are sealed; afterwards they are not."""
    want = None if W.lifted("[RESULT] REG-19") else "REG-19"
    assert W.sealed_by("20261112-135538_burst_l1pl350cl_r2.0_w3000_s17") == want
    assert fnmatch.fnmatchcase("20261112-135538_burst_l1pl350cl_r2.0_w3000_s17", W.SEALS[0][1])


def test_e1_and_e2_seals_cover_exactly_their_sweeps(no_result):
    # A run folder is named after the config's `name`, which a rerun config keeps (it lists one rate under the same
    # name), so the seal is checked on names, not on config file stems.
    names = {load_config(c)["name"] for c in CONFIGS.glob("sweep_*.yaml")}
    sealed = {n: W.sealed_by(f"20261114-000000_{n}_r2") for n in names}
    assert {k for k, v in sealed.items() if v == "E1"} == {"sweep_l1pl350cl_pc_off"}
    assert {k for k, v in sealed.items() if v == "E0"} == {"sweep_l1t16pl350cl_pc_on", "sweep_l1t16pl350cl_pc_off"}
    assert {k for k, v in sealed.items() if v == "T4"} == {f"sweep_t4_l1pl350cl_pc_{c}_b{b}"        # T4 (step budget)
                                                          for c in ("off", "on") for b in (2048, 8192)}
    assert {k for k, v in sealed.items() if v == "G1"} == {f"sweep_g1_l1pl350cl_pc_{c}_b2048_s1"     # G1 (second seed)
                                                          for c in ("off", "on")}
    assert {k for k, v in sealed.items() if v == "G2"} == {f"sweep_g2_l1pl350cl_pc_off_b2048_{m}"   # G2 (dev mode)
                                                          for m in ("dev", "nodev")}
    assert all(v in (None, "E1", "E0", "T4", "G1", "G2") for v in sealed.values())
    assert W.sealed_by("20261109-190221_sweep_l1pl350cl_pc_on_r2") is None        # REG-13's sweep stays readable
    assert W.sealed_by("20261107-083730_sweep_l1pl350_pc_off_r2") is None         # REG-8's too


def test_e1_seal_lifts_alone(no_result):
    no_result.write_text("## 2026-11-15 · [RESULT] E1: floor 1.75\n")
    assert W.sealed_by("20261114-000000_sweep_l1pl350cl_pc_off_r2") is None
    assert W.sealed_by("20261115-000000_sweep_l1t16pl350cl_pc_off_r2") == "E0"


def test_reg25_seal_covers_exactly_the_sustained_wipe_and_its_control(no_result):
    names = {load_config(c)["name"] for c in CONFIGS.glob("*.yaml") if c.name != "base.yaml"}
    sealed = {n for n in names if W.sealed_by(f"20261115-000000_{n}") == "REG-25"}
    assert sealed == {"swipe_l1pl350cl_r2.0_s44", "stage2_control_l1pl350cl_r2.0_s44"}


def test_reg24_seal_covers_exactly_its_two_configs(no_result):
    """REG-24 (rule F round 1 on draft 1, RE5): two exact patterns, one per config. Checked on the configs' names (a run
    folder is named after the config's `name`), over every config present: exactly REG-24's two are sealed, whether or
    not their files are in the tree yet (the seal may land before or after the configs' commit, never after a run)."""
    want = {"confirm_l1pl350cl_r2.0_s43", "stage2_control_l1pl350cl_r2.0_long_s43"}
    names = {load_config(c)["name"] for c in CONFIGS.glob("*.yaml")}
    sealed = {n for n in names if W.sealed_by(f"20261115-000000_{n}") == "REG-24"}
    assert sealed == want & names
    assert all(W.sealed_by(f"20261115-000000_{n}") == "REG-24" for n in want)
    assert W.sealed_by("20261110-085730_confirm_l1pl350cl_r2.2_s12") is None              # REG-15's run
    assert W.sealed_by("20261111-000000_confirm_l1pl350cl_r2.2_long_s14") is None         # REG-16's design
    assert W.sealed_by("20261115-000000_confirm_l1pl350cl_r2.0_s143") is None
    assert W.sealed_by("20261115-000000_stage2_control_l1pl350cl_r2.0_long_s430") is None


def test_reg24_seal_lifts_only_on_its_result(no_result):
    name = "20261115-000000_confirm_l1pl350cl_r2.0_s43"
    no_result.write_text("## 2026-11-15 · [REG] REG-24: waits for [RESULT] REG-24\n## 2026-11-16 · [RESULT] REG-240: x\n")
    assert W.sealed_by(name) == "REG-24"
    no_result.write_text(no_result.read_text() + "## 2026-11-16 · [RESULT] REG-24: confirms, 1 of 1\n")
    assert W.sealed_by(name) is None and W.sealed_by("20261115-000000_stage2_control_l1pl350cl_r2.0_long_s43") is None


def test_log_path_falls_back_to_the_release_index_and_seals_without_either(tmp_path):
    """The public release has docs/results.md (the log's headings) and no decisions.md; a heading there lifts its
    seal, a missing heading or a missing index lifts nothing, and the log wins when both exist."""
    docs = tmp_path / "docs"; docs.mkdir()
    assert W.log_path(docs) == docs / "results.md" and not W.lifted("[RESULT] E1", W.log_path(docs))
    (docs / "results.md").write_text("## 2026-11-14 · [RESULT] E1: the floor\n")
    assert W.lifted("[RESULT] E1", W.log_path(docs))
    assert not W.lifted("[RESULT] E0", W.log_path(docs))                     # failure control: not in the index
    (docs / "decisions.md").write_text("# log\n")
    assert W.log_path(docs) == docs / "decisions.md" and not W.lifted("[RESULT] E1", W.log_path(docs))
    assert W.DECISIONS == W.log_path()          # the module reads whichever this tree has
