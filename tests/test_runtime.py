import json

from analysis import runtime


def _run(root, name, walls, valid):
    d = root / name
    d.mkdir()
    (d / "events.jsonl").write_text("".join(json.dumps({"wall": w, "event": "x"}) + "\n" for w in walls) + "not json\n")
    if valid is not None:
        (d / "checks.json").write_text(json.dumps({"valid": valid}))


def test_total_hours_splits_valid_invalid_and_unknown(tmp_path):
    _run(tmp_path, "a", ["2026-11-12T06:00:00+00:00", "2026-11-12T07:30:00+00:00"], True)     # 1.5 h
    _run(tmp_path, "b", ["2026-11-12T08:00:00+00:00", "2026-11-12T08:30:00+00:00"], False)    # 0.5 h
    _run(tmp_path, "c", ["2026-11-12T09:00:00+00:00", "2026-11-12T10:00:00+00:00"], None)     # 1 h, no checks
    _run(tmp_path, "d", ["2026-11-12T09:00:00+00:00"], True)                                   # one stamp: no time
    out = runtime.total_hours(tmp_path)
    assert out["folders"] == 4
    assert out["hours_all"] == 3.0 and out["hours_valid"] == 1.5 and out["hours_invalid"] == 0.5
    assert out["no_checks"] == ["c"] and out["no_time"] == ["d"]


def test_run_seconds_uses_first_and_last_stamp_in_any_order(tmp_path):
    _run(tmp_path, "a", ["2026-11-12T07:00:00+00:00", "2026-11-12T06:00:00+00:00", "2026-11-12T06:30:00+00:00"], True)
    assert runtime.run_seconds(tmp_path / "a") == 3600.0
