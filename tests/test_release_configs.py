"""Every config's resolved form (extends chain, configs/defaults.yaml beneath) equals the one the study ran with.

Comments in some configs were reworded for the release; manifests/configs.resolved.sha256 holds the sha256 of each
config's resolved form computed from the working repository's unedited files, before any rewording."""
import copy
import hashlib
import json
from pathlib import Path

from harness.config import load_config, with_defaults

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "manifests" / "configs.resolved.sha256"


def digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()


def recorded() -> dict[str, str]:
    return {name: h for h, name in (line.split("  ", 1) for line in MANIFEST.read_text().splitlines())}


def test_every_config_resolves_as_recorded():
    want = recorded()
    have = {p.name: digest(with_defaults(load_config(p))) for p in sorted((REPO / "configs").glob("*.yaml"))}
    assert set(have) == set(want)
    assert [k for k in want if have[k] != want[k]] == []


def test_a_changed_value_is_caught():
    """Failure control: one changed setting in a resolved config gives a different digest from the recorded one."""
    p = REPO / "configs" / "base.yaml"
    cfg = with_defaults(load_config(p))
    bad = copy.deepcopy(cfg)
    bad["seed"] = (bad.get("seed") or 0) + 1
    assert digest(cfg) == recorded()["base.yaml"] and digest(bad) != recorded()["base.yaml"]
