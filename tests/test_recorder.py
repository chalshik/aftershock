"""harness.recorder.LagTail must never crash on, or consume, a half-written line of requests.jsonl (open question A14)."""
import json

from harness.recorder import LagTail


def line(tp, ts):
    return json.dumps({"t_planned": tp, "t_sent": ts}) + "\n"


def test_partial_line_is_left_for_the_next_read(tmp_path):
    p = tmp_path / "requests.jsonl"
    full, nxt = line(0.0, 0.1), line(1.0, 1.5)
    p.write_text(full + nxt[:10])                       # the writer is partway through the second line
    tail = LagTail(p)
    assert tail.recent_max() == 0.1
    with open(p, "a") as f:
        f.write(nxt[10:])                               # ... and finishes it
    assert tail.recent_max() == 0.5                     # read whole, not from the middle
    assert tail.bad_lines == 0


def test_no_complete_line_yet(tmp_path):
    p = tmp_path / "requests.jsonl"
    p.write_text('{"t_planned": 0')
    tail = LagTail(p)
    assert tail.recent_max() is None and tail.pos == 0


def test_bad_complete_line_is_skipped_and_counted(tmp_path):
    p = tmp_path / "requests.jsonl"
    p.write_text(line(0.0, 0.2) + "{not json\n" + line(1.0, 1.1))
    tail = LagTail(p)
    assert tail.recent_max() == 0.2 and tail.bad_lines == 1


def test_unsent_requests_and_missing_file(tmp_path):
    p = tmp_path / "requests.jsonl"
    assert LagTail(p).recent_max() is None
    p.write_text(json.dumps({"t_planned": 0.0, "t_sent": None}) + "\n")
    assert LagTail(p).recent_max() is None


# --- A7: compute-process samples (recorder.gpu_processes_every_s; author, decisions.md 2026-11-17) ---
import threading
import types

import yaml

from harness import recorder as R


class _FakeNvml(types.SimpleNamespace):
    class NVMLError(Exception):
        pass


def _fake_nvml(procs, fail_list=False, fail_name=()):
    n = _FakeNvml(NVML_CLOCK_SM=0, NVML_CLOCK_MEM=1, NVML_TEMPERATURE_GPU=0)
    n.nvmlInit = lambda: None
    n.nvmlDeviceGetHandleByIndex = lambda i: "h"
    n.nvmlDeviceGetMemoryInfo = lambda h: types.SimpleNamespace(used=2**30)
    n.nvmlDeviceGetPowerUsage = lambda h: 300000
    n.nvmlDeviceGetEnforcedPowerLimit = lambda h: 350000
    n.nvmlDeviceGetClockInfo = lambda h, c: 1500
    n.nvmlDeviceGetTemperature = lambda h, c: 50
    n.nvmlDeviceGetUtilizationRates = lambda h: types.SimpleNamespace(gpu=99)
    n.nvmlDeviceGetCurrentClocksThrottleReasons = lambda h: 0

    def running(h):
        if fail_list:
            raise RuntimeError("nvml down")
        return [types.SimpleNamespace(pid=p, usedGpuMemory=m * 2**20) for p, m in procs]

    def name(pid):
        if pid in fail_name:
            raise RuntimeError("no name")
        return b"VLLM::EngineCore" if pid == 1 else b"miner"
    n.nvmlDeviceGetComputeRunningProcesses = running
    n.nvmlSystemGetProcessName = name
    return n


def test_processes_lists_pid_memory_and_name(monkeypatch):
    monkeypatch.setattr(R, "pynvml", _fake_nvml([(1, 22994), (7, 900)], fail_name=(7,)))
    assert R.Gpu().processes() == [{"pid": 1, "name": "VLLM::EngineCore", "used_mib": 22994.0},
                                   {"pid": 7, "name": None, "used_mib": 900.0}]


def test_processes_without_nvml_is_none(monkeypatch):
    monkeypatch.setattr(R, "pynvml", None)
    assert R.Gpu().processes() is None


def _run_main(tmp_path, monkeypatch, every, nvml, ticks=6):
    cfg = {"server": {"host": "127.0.0.1", "port": 1}, "recorder": {"interval_s": 0.02, "scrape_timeout_s": 0.01}}
    if every is not None:
        cfg["recorder"]["gpu_processes_every_s"] = every
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(cfg))
    monkeypatch.setattr(R, "pynvml", nvml)
    monkeypatch.setattr(R, "scrape", lambda url, timeout: (_ for _ in ()).throw(OSError("no server")))
    monkeypatch.setattr(R.signal, "signal", lambda *a: None)       # main() runs off the main thread here
    monkeypatch.setattr(R, "_stop", False)
    th = threading.Thread(target=R.main, args=(tmp_path, R.time.monotonic()))
    th.start()
    deadline = R.time.monotonic() + 5
    while R.time.monotonic() < deadline:
        m = tmp_path / "metrics.jsonl"
        if m.exists() and len(m.read_text().splitlines()) >= ticks:
            break
        R.time.sleep(0.01)
    R._stop = True
    th.join(5)
    assert not th.is_alive()


def test_main_writes_process_samples_when_the_knob_is_set(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch, 0.04, _fake_nvml([(1, 22994), (7, 900)]))
    lines = [json.loads(l) for l in (tmp_path / "gpu_procs.jsonl").read_text().splitlines()]
    assert len(lines) >= 2 and lines[0]["processes"][1] == {"pid": 7, "name": "miner", "used_mib": 900.0}
    assert all(b["t"] - a["t"] >= 0.04 - 1e-9 for a, b in zip(lines, lines[1:]))
    assert (tmp_path / "gpu.jsonl").read_text().strip()               # gpu.jsonl unchanged in schema, still written
    assert "processes" not in json.loads((tmp_path / "gpu.jsonl").read_text().splitlines()[0])


def test_no_process_file_without_the_knob(tmp_path, monkeypatch):
    _run_main(tmp_path, monkeypatch, None, _fake_nvml([(1, 1)]))
    assert not (tmp_path / "gpu_procs.jsonl").exists()


def test_a_failing_process_list_does_not_end_the_recorder(tmp_path, monkeypatch, capsys):
    _run_main(tmp_path, monkeypatch, 0.02, _fake_nvml([], fail_list=True), ticks=8)
    assert len((tmp_path / "metrics.jsonl").read_text().splitlines()) >= 8
    assert (tmp_path / "gpu_procs.jsonl").read_text() == ""
    assert "process list failed" in capsys.readouterr().err
