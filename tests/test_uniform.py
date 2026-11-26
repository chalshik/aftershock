"""analysis.uniform: the post-hoc one-criterion readout (author 2026-11-26)."""
import pytest

from analysis import uniform


def test_outcome_columns():
    assert uniform.outcome(True, False) == "cold"
    assert uniform.outcome(False, True) == "recovered"
    assert uniform.outcome(False, False) == "neither"
    assert uniform.outcome(None, True) == "unclear"
    assert uniform.outcome(False, None) == "unclear"


def test_outcome_refuses_both():
    with pytest.raises(ValueError):
        uniform.outcome(True, True)


def test_cells_match_the_papers_table():
    """The cells and their n are the burst cells of the papers' Table 3 plus the long-window runs."""
    n = {c[0]: len(c[3]) for c in uniform.CELLS}
    assert n["closed loop, λ 2.0 (main runs)"] == 5 and n["open loop, λ 2.5"] == 3
    assert n["closed loop, cap of 40"] == 3 and n["closed loop, long window"] == 5
    assert len(uniform.CELLS) == 10
    for name, burst, control, seeds in uniform.CELLS:
        assert "{s}" in burst and (control is None or "{s}" in control)
        assert burst.startswith("*_") and len(set(seeds)) == len(seeds)


def test_patterns_do_not_overlap():
    """A bare REG-19 pattern must not also match the capped or later-stack runs of the same seed."""
    import fnmatch
    pats = [c[1].format(s=15) for c in uniform.CELLS if 15 in c[3]]
    names = ["20261112-000000_" + p[2:] for p in pats]
    for p in pats:
        assert sum(fnmatch.fnmatch(n, p) for n in names) == 1, p
