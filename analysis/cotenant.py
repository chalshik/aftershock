"""A7 (docs/open-questions.md): was another process using the GPU during a run? A readout over every unsealed run
folder of the compute-process lists the harness already records: at launch, before the server starts (`env.json`
`gpu_processes`, harness/env_snapshot.py:44), and at the end, after the recorder stops and before the server stops
(event `gpu_processes_end`, harness/run.py, from 2026-11-04, `677559d`). Each list is `nvidia-smi
--query-compute-apps=pid,process_name,used_memory` as text. At launch any listed process is foreign; at the end a
process whose name does not match VLLM_NAMES is foreign. Nothing between the two times is covered: a co-tenant that
starts and stops inside a run is not seen. Reads only env.json, events.jsonl and checks.json (`load._jsonl`); sealed
folders are skipped (`withheld.keep`). Judges nothing.
From the recorder change of 2026-11-17, a run whose config sets `recorder.gpu_processes_every_s` also has
`gpu_procs.jsonl` (NVML samples during the run); a sampled process whose name does not match VLLM_NAMES, or has no
name, is reported as foreign mid-run.
"""
from __future__ import annotations

import json
import re

from analysis import withheld
from analysis.load import RUNS, _jsonl

VLLM_NAMES = (r"^VLLM::", r"vllm")   # vLLM's engine core lists as "VLLM::EngineCore" (PROBE(docs/probe/20261115_open_questions/a7_midrun_20261115T073256Z.txt))


def parse(text: str | None) -> list[dict] | None:
    """Rows of an nvidia-smi compute-apps list; [] for an empty list, None when the command failed (None)."""
    if text is None:
        return None
    out = []
    for line in str(text).splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split(",")]
        out.append({"pid": parts[0], "name": parts[1] if len(parts) > 1 else "", "mem": parts[2] if len(parts) > 2 else ""})
    return out


def is_vllm(name: str) -> bool:
    return any(re.search(p, name) for p in VLLM_NAMES)


def one(run_dir) -> dict:
    env_p, ev_p, ck_p = run_dir / "env.json", run_dir / "events.jsonl", run_dir / "checks.json"
    env = json.loads(env_p.read_text()) if env_p.exists() else {}
    ck = json.loads(ck_p.read_text()) if ck_p.exists() else None
    launch = parse(env.get("gpu_processes")) if "gpu_processes" in env else "not recorded"
    end_ev = [e for e in _jsonl(ev_p) if e.get("event") == "gpu_processes_end"]
    end = parse(end_ev[-1].get("processes")) if end_ev else "not recorded"
    foreign_launch = None if not isinstance(launch, list) else [p for p in launch]
    foreign_end = None if not isinstance(end, list) else [p for p in end if not is_vllm(p["name"])]
    mid = _jsonl(run_dir / "gpu_procs.jsonl") if (run_dir / "gpu_procs.jsonl").exists() else None
    foreign_mid = None if mid is None else sorted({(p["pid"], p.get("name")) for x in mid for p in x["processes"]
                                                   if not is_vllm(p.get("name") or "")})
    return {"run": run_dir.name, "valid": (ck or {}).get("valid"), "launch": launch, "end": end,
            "mid_samples": None if mid is None else len(mid), "foreign_mid": foreign_mid,
            "foreign_launch": foreign_launch, "foreign_end": foreign_end,
            "vllm_at_end": None if not isinstance(end, list) else any(is_vllm(p["name"]) for p in end)}


def table() -> dict:
    dirs = withheld.keep(sorted(p for p in RUNS.iterdir() if p.is_dir()), "cotenant.table")
    rows = [one(d) for d in dirs]
    cnt = lambda f: sum(1 for r in rows if f(r))
    summary = {
        "folders": len(rows), "valid": cnt(lambda r: r["valid"] is True),
        "launch_recorded": cnt(lambda r: isinstance(r["launch"], list)),
        "launch_command_failed": cnt(lambda r: r["launch"] is None),
        "launch_foreign": cnt(lambda r: bool(r["foreign_launch"])),
        "end_recorded": cnt(lambda r: isinstance(r["end"], list)),
        "end_command_failed": cnt(lambda r: r["end"] is None),
        "end_foreign": cnt(lambda r: bool(r["foreign_end"])),
        "end_recorded_valid": cnt(lambda r: isinstance(r["end"], list) and r["valid"] is True),
        "end_foreign_valid": cnt(lambda r: bool(r["foreign_end"]) and r["valid"] is True),
        "end_without_vllm": cnt(lambda r: r["vllm_at_end"] is False),
        "mid_sampled": cnt(lambda r: r["mid_samples"] is not None),
        "mid_foreign": cnt(lambda r: bool(r["foreign_mid"])),
        "names_at_end": sorted({p["name"] for r in rows if isinstance(r["end"], list) for p in r["end"]}),
    }
    return {"summary": summary, "foreign": [r for r in rows if r["foreign_launch"] or r["foreign_end"] or r["foreign_mid"]],
            "end_without_vllm": [r["run"] for r in rows if r["vllm_at_end"] is False]}


if __name__ == "__main__":
    print(json.dumps(table(), indent=1, default=str))
