"""analysis/g2.py: G2's rule and validity checks on synthetic inputs (no run folder is read)."""
import math
from pathlib import Path

import pytest

from analysis import g2, t4, withheld

ALL_BEHIND = {(a, x): True for a in ("on", "off") for x in (3.0, 4.0)}


@pytest.mark.parametrize("on,off,dec", [
    ({3.0: 1.80, 4.0: 1.85}, {3.0: 1.80, 4.0: 1.85}, "dev_neutral"),
    ({3.0: 1.836, 4.0: 1.85}, {3.0: 1.80, 4.0: 1.85}, "dev_neutral"),   # +2.0 %: inclusive
    ({3.0: 1.85, 4.0: 1.90}, {3.0: 1.80, 4.0: 1.85}, "dev_moves"),      # +2.8 %, +2.7 %: both, same sign
    ({3.0: 1.84, 4.0: 1.85}, {3.0: 1.80, 4.0: 1.85}, "dev_unclear"),    # rule F r1 G-7: one rate only
    ({3.0: 1.85, 4.0: 1.80}, {3.0: 1.80, 4.0: 1.85}, "dev_unclear"),    # opposite signs
    ({3.0: 1.80}, {3.0: 1.80, 4.0: 1.85}, "dev_undefined"),
    ({3.0: 1.80, 4.0: 1.85}, {3.0: 0.0, 4.0: 1.85}, "dev_undefined"),
])
def test_rule(on, off, dec):
    assert g2.outcome(on, off, ALL_BEHIND)["decision"] == dec


def test_a_run_that_keeps_up_makes_it_undefined():
    # rule F r1 G-8: at a rate the engine keeps up with, r measures the offered load
    behind = {**ALL_BEHIND, ("off", 3.0): False}
    assert g2.outcome({3.0: 1.80, 4.0: 1.85}, {3.0: 1.80, 4.0: 1.85}, behind)["decision"] == "dev_undefined"


def _log(tmp_path, name, text):
    d = tmp_path / name
    d.mkdir()
    (d / "server.log").write_text(text)
    return d


def test_mode_is_read_from_the_server_log(tmp_path, monkeypatch):
    monkeypatch.setattr(withheld, "sealed_by", lambda name, decisions=None: None)
    on = _log(tmp_path, "a", "INFO [launcher.py:195] Route: /reset_prefix_cache, Methods: POST\n")
    off = _log(tmp_path, "b", "INFO [launcher.py:195] Route: /health, Methods: GET\n")
    assert g2.dev_routes_logged(on) and not g2.dev_routes_logged(off)
    assert g2.check_mode(on, "on") == [] and g2.check_mode(off, "off") == []
    assert g2.check_mode(off, "on") and g2.check_mode(on, "off")


def test_configs_add_only_the_mode_at_two_rates():
    from harness.config import load_config
    a = load_config(Path("configs") / "sweep_t4_l1pl350cl_pc_off_b2048.yaml")
    for par, env in ((g2.G2["parents"]["on"], {"VLLM_SERVER_DEV_MODE": "1"}), (g2.G2["parents"]["off"], {})):
        b = load_config(Path("configs") / f"{par}.yaml")
        assert (b["server"].get("env") or {}) == env and not (a["server"].get("env") or {})
        assert tuple(float(x) for x in b["sweep"]["rates"]) == g2.G2["rates"]
        strip = lambda c: {k: ({**v, "env": None} if k == "server" else v) for k, v in c.items()
                           if k not in ("name", "sweep")}
        assert strip(b) == strip(a)   # rule F r1 G-12: the whole config, not three keys
    assert all(r in tuple(float(x) for x in a["sweep"]["rates"]) for r in g2.G2["rates"])


def test_readout_is_t4_and_table_refuses_before_t4(monkeypatch):
    assert g2.G2["readout_parent"] == t4.T4["parents"]["off_2048"]
    assert any(r[0] == "G2" and r[2] == g2.G2["seal"] for r in withheld.SEALS)
    monkeypatch.setattr(withheld, "lifted", lambda h, decisions=None: False)
    with pytest.raises(withheld.WithheldError):
        g2.table()
