"""analysis/cotenant.py: A7's start and end process lists (temporary folders; no real run is read)."""
import json

from analysis import cotenant


def _folder(root, name, launch="", end="1, VLLM::EngineCore, 22994 MiB", valid=True, with_end=True):
    d = root / name
    d.mkdir()
    env = {} if launch == "absent" else {"gpu_processes": launch}
    (d / "env.json").write_text(json.dumps(env))
    ev = [{"event": "start"}] + ([{"event": "gpu_processes_end", "processes": end}] if with_end else [])
    (d / "events.jsonl").write_text("\n".join(json.dumps(e) for e in ev))
    (d / "checks.json").write_text(json.dumps({"valid": valid}))
    return d


def test_parse():
    assert cotenant.parse("") == []
    assert cotenant.parse(None) is None
    assert cotenant.parse("1, VLLM::EngineCore, 22994 MiB\n2, python, 300 MiB") == [
        {"pid": "1", "name": "VLLM::EngineCore", "mem": "22994 MiB"}, {"pid": "2", "name": "python", "mem": "300 MiB"}]


def test_foreign_processes_at_launch_and_end(tmp_path, monkeypatch):
    monkeypatch.setattr(cotenant, "RUNS", tmp_path)
    _folder(tmp_path, "a_clean")
    _folder(tmp_path, "b_launch", launch="9, python, 400 MiB")
    _folder(tmp_path, "c_end", end="1, VLLM::EngineCore, 22994 MiB\n7, miner, 900 MiB", valid=False)
    _folder(tmp_path, "d_old", launch="absent", with_end=False)
    _folder(tmp_path, "e_failed", launch=None, end=None)
    t = cotenant.table()
    s = t["summary"]
    assert s["folders"] == 5 and s["launch_recorded"] == 3 and s["launch_command_failed"] == 1
    assert s["launch_foreign"] == 1 and s["end_foreign"] == 1 and s["end_foreign_valid"] == 0
    assert s["end_recorded"] == 3 and s["end_command_failed"] == 1
    assert {r["run"] for r in t["foreign"]} == {"b_launch", "c_end"}
    assert s["names_at_end"] == ["VLLM::EngineCore", "miner"]


def test_an_end_list_without_vllm_is_flagged(tmp_path, monkeypatch):
    monkeypatch.setattr(cotenant, "RUNS", tmp_path)
    _folder(tmp_path, "x", end="")
    assert cotenant.table()["end_without_vllm"] == ["x"]


def test_sealed_folders_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(cotenant, "RUNS", tmp_path)
    _folder(tmp_path, "20261117-053015_k4_l1pl350cl_r2.2_s133")
    monkeypatch.delenv("AFTERSHOCK_UNSEAL", raising=False)
    # K4's seal was lifted by its [RESULT] (2026-11-17); hold it here so the test does not depend on decisions.md
    from analysis import withheld
    monkeypatch.setattr(withheld, "lifted", lambda *a, **k: False)
    assert cotenant.table()["summary"]["folders"] == 0


def test_mid_run_samples_are_read_when_present(tmp_path, monkeypatch):
    monkeypatch.setattr(cotenant, "RUNS", tmp_path)
    d = _folder(tmp_path, "m")
    (d / "gpu_procs.jsonl").write_text(
        json.dumps({"t": 1.0, "processes": [{"pid": 1, "name": "VLLM::EngineCore", "used_mib": 1.0}]}) + "\n" +
        json.dumps({"t": 61.0, "processes": [{"pid": 1, "name": "VLLM::EngineCore", "used_mib": 1.0},
                                             {"pid": 9, "name": None, "used_mib": 300.0}]}) + "\n")
    _folder(tmp_path, "n")
    t = cotenant.table()
    assert t["summary"]["mid_sampled"] == 1 and t["summary"]["mid_foreign"] == 1
    assert [r["run"] for r in t["foreign"]] == ["m"] and t["foreign"][0]["foreign_mid"] == [(9, None)]
