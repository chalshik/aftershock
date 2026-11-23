import time

import pytest

from harness import trigger


def fire(trig, results):
    """Run fire() with a fake HTTP call; returns the urls called and the records written."""
    urls, recs = [], []

    def post(url, timeout_s):
        urls.append(url)
        return results(len(urls))

    orig, trigger._post = trigger._post, post
    try:
        trigger.fire(trig, "http://x", time.monotonic(), recs.append)
    finally:
        trigger._post = orig
    return urls, recs


PREEMPT = {"at_s": 0, "mode": "preempt", "request_timeout_s": 1}
IDLE = {"at_s": 0, "mode": "idle", "request_timeout_s": 1, "retry_interval_s": 0.001, "max_wait_s": 0.05}
FAILED, OK = {"status": 200, "success": False}, {"status": 200, "success": True}


def test_preempt_calls_once_with_the_flag_even_if_it_fails():
    urls, recs = fire(PREEMPT, lambda n: FAILED)
    assert urls == ["http://x/reset_prefix_cache?reset_running_requests=true"] and len(recs) == 1


def test_idle_never_sends_the_flag_and_stops_at_first_success():
    urls, recs = fire(IDLE, lambda n: OK if n == 3 else FAILED)
    assert set(urls) == {"http://x/reset_prefix_cache"} and [r["success"] for r in recs] == [False, False, True]


def test_idle_gives_up_at_max_wait():
    t = time.monotonic()
    _, recs = fire(IDLE, lambda n: FAILED)
    assert not any(r["success"] for r in recs) and time.monotonic() - t < 1


def test_idle_stops_on_an_http_error():
    urls, _ = fire(IDLE, lambda n: {"status": 500, "success": False, "error": "x"})
    assert len(urls) == 1


def cfg(**trig):
    return {"name": "x", "workload": {"warmup_s": 240, "duration_s": 3000},
            "server": {"env": {trigger.DEV_MODE_ENV: "1"}},
            "trigger": {"kind": "cache_wipe", "mode": "preempt", "at_s": 1200, **trig}}


def test_validate_accepts_a_good_trigger():
    trigger.validate(cfg())


@pytest.mark.parametrize("bad", [{"kind": "power_cap"}, {"kind": "burst"}, {"mode": "other"}, {"at_s": 100}, {"at_s": 3000},
                                 {"mode": "idle", "max_wait_s": 1800}])
def test_validate_rejects(bad):
    with pytest.raises(SystemExit):
        trigger.validate(cfg(**bad))


def burst_cfg(**trig):
    return {"name": "x", "workload": {"warmup_s": 240, "duration_s": 2520}, "server": {}, "verdict": {"watch_s": 1500},
            "trigger": {"kind": "burst", "at_s": 900, "duration_s": 120, "rate": 3.0, "prompt_tokens": 1408, **trig}}


def test_validate_accepts_a_burst_without_dev_mode():
    trigger.validate(burst_cfg())


@pytest.mark.parametrize("bad", [{"at_s": 100}, {"duration_s": 200}, {"rate": 0}, {"prompt_tokens": 0}])
def test_validate_rejects_a_bad_burst(bad):
    with pytest.raises(SystemExit):
        trigger.validate(burst_cfg(**bad))


def test_validate_requires_dev_mode():
    c = cfg()
    c["server"]["env"] = {}
    with pytest.raises(SystemExit):
        trigger.validate(c)


REPEAT = {"at_s": 0, "mode": "preempt", "request_timeout_s": 1, "repeat": {"every_s": 0.01, "span_s": 0.025}}


def test_schedule_single_and_repeated():
    assert trigger.schedule({"at_s": 900}) == [900.0]
    assert trigger.schedule({"at_s": 900, "repeat": {"every_s": 5, "span_s": 300}}) == [900.0 + 5 * k for k in range(60)]
    assert trigger.schedule({"at_s": 0, "repeat": {"every_s": 2, "span_s": 5}}) == [0.0, 2.0, 4.0]


def test_a_repeated_wipe_calls_once_per_planned_time_with_the_flag_and_does_not_stop_on_failure():
    urls, recs = fire(REPEAT, lambda n: FAILED if n == 2 else OK)
    assert len(urls) == 3 and all("reset_running_requests=true" in u for u in urls)
    assert [r["attempt"] for r in recs] == [0, 1, 2] and [r["success"] for r in recs] == [True, False, True]


@pytest.mark.parametrize("bad", [{"mode": "idle", "max_wait_s": 1, "retry_interval_s": 0.1},
                                 {"repeat": {"every_s": 0, "span_s": 300}},
                                 {"repeat": {"every_s": 5, "span_s": 2}},
                                 {"repeat": {"every_s": 5, "span_s": 5000}}])
def test_validate_rejects_a_bad_repeat(bad):
    c = cfg(**{"repeat": {"every_s": 5, "span_s": 300}, **bad})
    with pytest.raises(SystemExit):
        trigger.validate(c)


def test_validate_accepts_a_good_repeat():
    trigger.validate(cfg(repeat={"every_s": 5, "span_s": 300}))


def test_the_sustained_wipe_config_validates_with_120_calls():
    from pathlib import Path

    from harness.run import load_config
    c = load_config(Path(__file__).resolve().parent.parent / "configs" / "swipe_l1pl350cl_r2.0_s44.yaml")
    trigger.validate(c)
    times = trigger.schedule(c["trigger"])
    assert len(times) == 120 and times[0] == 900.0 and times[-1] == 1197.5
    assert c["checks"]["manipulation"]["wipe_max_hit"] == 0.25
