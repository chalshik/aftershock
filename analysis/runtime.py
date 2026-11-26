"""Wall-clock time of every run folder, for the artifact section's compute total (paper A.6; 0 GPU h, reads runs/ only).

A run's time is its first to its last `wall` stamp in `events.jsonl`: from the harness's start, through the server's
start-up, the schedule and the drain, to the validity checks; time between runs is not counted. Folders
are counted valid or invalid by `checks.json` (`valid`); a folder without either file is reported, not guessed.
"""
from __future__ import annotations

import datetime as dt
import json
import pathlib

RUNS = pathlib.Path(__file__).resolve().parents[1] / "runs"


def run_seconds(folder: pathlib.Path) -> float | None:
    """Seconds from the first to the last `wall` stamp in the run's events.jsonl, or None if there are fewer than two."""
    f = folder / "events.jsonl"
    if not f.exists():
        return None
    stamps = []
    for line in f.read_text().splitlines():
        try:
            w = json.loads(line).get("wall")
        except (json.JSONDecodeError, AttributeError):
            continue
        if isinstance(w, str):
            stamps.append(dt.datetime.fromisoformat(w))
    return (max(stamps) - min(stamps)).total_seconds() if len(stamps) > 1 else None


def is_valid(folder: pathlib.Path) -> bool | None:
    f = folder / "checks.json"
    if not f.exists():
        return None
    return bool(json.loads(f.read_text()).get("valid"))


def total_hours(root: pathlib.Path = RUNS) -> dict:
    """Run-hours over every folder in runs/: all, valid, invalid, and the folders whose time or validity is unknown."""
    out = {"folders": 0, "hours_all": 0.0, "hours_valid": 0.0, "hours_invalid": 0.0, "no_time": [], "no_checks": []}
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        out["folders"] += 1
        s, v = run_seconds(d), is_valid(d)
        if s is None:
            out["no_time"].append(d.name)
            continue
        h = s / 3600
        out["hours_all"] += h
        if v is None:
            out["no_checks"].append(d.name)
        elif v:
            out["hours_valid"] += h
        else:
            out["hours_invalid"] += h
    for k in ("hours_all", "hours_valid", "hours_invalid"):
        out[k] = round(out[k], 1)
    return out


if __name__ == "__main__":
    print(total_hours())
