"""The sustained-wipe gate (analysis/wipe_gate.py) on hand-made inputs: no run folder is read."""
import analysis.wipe_gate as G

CFG = {"workload": {"output_tokens": 128, "sessions": {"system_tokens": 2048, "user_tokens": 128, "reply_tokens": 128,
                                                       "turns": 8}},
       "checks": {"expected_kv_tokens": 96656}}


def test_q_star_is_the_pool_over_a_mean_session_request():
    assert abs(G.q_star(CFG) - 96656 / (2176 + 3.5 * 256 + 128)) < 1e-9


def test_the_smallest_span_that_builds_the_backlog_is_chosen():
    d = G.decide(c_cold=1.75, controls_recovered=5, qstar=30.0)   # needs 45: 0.25 x 180 = 45 passes
    assert d["g1"] and d["D_s"] == 180.0 and d["passes"]
    d = G.decide(c_cold=1.84, controls_recovered=5, qstar=30.0)   # 0.16 x 180 = 28.8, x 300 = 48 (1.85 sits on a float edge)
    assert d["D_s"] == 300.0


def test_g1_fails_without_a_recovered_control_or_with_a_cold_rate_at_the_load():
    assert not G.decide(1.75, 0, 30.0)["passes"]
    assert not G.decide(2.0, 5, 30.0)["passes"]


def test_g2_fails_when_no_span_builds_the_backlog():
    d = G.decide(c_cold=1.95, controls_recovered=5, qstar=30.0)   # 0.05 x 600 = 30 < 45
    assert d["g1"] and d["D_s"] is None and not d["passes"]
