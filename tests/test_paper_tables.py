"""analysis/paper_tables.py (no run folder is read)."""
from analysis import paper_tables as T


def test_pairs_are_the_replication_and_the_long_watch_runs():
    assert sorted(s for e, s in T.PAIRS if e == "replication") == [15, 16, 17, 18, 19]
    assert sorted(s for e, s in T.PAIRS if e == "long-watch") == [43, 61, 68, 72, 79]
    for (e, s), (c, b) in T.PAIRS.items():
        assert c.endswith(f"_s{s}") and b.endswith(f"_s{s}") and "control" in c and "control" not in b
        assert c < b                     # the control ran first
