"""Run folders whose outcome a pending registration withholds (rule 7, rule A5; docs/FABLE_PROMPT.md §3).

A run is sealed from the moment its folder exists until `docs/decisions.md` holds the [RESULT] heading of the
registration that owns it. While sealed, `load.load_run` refuses it, and `load.find_runs`, `recheck.recheck`,
`tables.arrival_table` and `engine_config.scan` skip it and say so on stderr. Nothing here reads a run folder.

Not covered, on purpose: `checks.json`, which a rerun decision may read directly (the registrations' stopping rules),
and `tools/` (checksums, validity flags, inventory), which read no outcome.

The owning registration's own computation is the one exception. It sets the environment variable
AFTERSHOCK_UNSEAL to its id (several ids separated by commas), e.g. `AFTERSHOCK_UNSEAL=REG-20 python -m analysis.k1 …`.
REG-19's registered computation needs no variable: it runs from the worktree at a7bb77a (decisions.md 2026-11-13
[DECISION], REG-19 resume), whose code predates this module.
"""
from __future__ import annotations

import fnmatch
import os
import re
import sys
from pathlib import Path

_DOCS = Path(__file__).resolve().parent.parent / "docs"
# The public release carries no working log. It has docs/results.md instead: the log's [REG], [RESULT] and [CORR]
# heading lines, verbatim (tools/build_public_release.py). Here the log always exists, so this reads decisions.md.
def log_path(docs: Path = _DOCS) -> Path:
    """The log if it exists, else the release's heading index (which may be absent too: then nothing is lifted)."""
    return docs / "decisions.md" if (docs / "decisions.md").exists() else docs / "results.md"


DECISIONS = log_path()

# (registration id, pattern over the run-folder name, text of the decisions.md heading that lifts the seal)
SEALS: tuple[tuple[str, str, str], ...] = (
    ("REG-19", "*_l1pl350cl_r2.0_w3000_s1[5-9]", "[RESULT] REG-19"),
    ("D1", "*_burst_l1pl350cl_r2.0_w3000_s1[5-9]_cap", "[RESULT] D1 (C2, session cap)"),   # the five capped burst runs
    ("D1b", "*_burst_l1pl350cl_r2.0_w3000_s1[5-7]_cap40", "[RESULT] D1b (C2, cap above the cold batch)"),
    ("T1", "*_burst_l1pl350cl_r2.0_w3000_s1[5-9]_t1p", "[RESULT] T1 (rescue, two arms)"),           # arm P
    ("T1", "*_burst_l1pl350cl_r2.0_w3000_s1[5-9]_t1u",          "[RESULT] T1 (rescue, two arms)"),  # arm U
    ("D1r", "*_burst_l1pl350cl_r2.0_w3000_s1[5-9]_rescue", "[RESULT] D1r (C2, cap as a rescue)"),  # cap from 1680 s
    ("K4", "*_k4_l1pl350cl_r2.2_s[1-5][0-9][0-9]", "[RESULT] K4 (the warm state ends at a surge)"),  # both arms
    ("K4e", "*_k4e_l1pl350cl_r2.2_s[6-9][0-9][0-9]", "[RESULT] K4e (K4's edge)"),   # both strata
    ("K4e", "*_k4e_l1pl350cl_r2.2_s137", "[RESULT] K4e (K4's edge)"),               # its positive control
    ("K4e", "*_k4e_l1pl350cl_r2.2_s103", "[RESULT] K4e (K4's edge)"),               # its calm control
    ("T4", "*_sweep_t4_l1pl350cl_pc_o*_b*_r*", "[RESULT] T4 (step budget)"),   # all four step-budget sweeps
    ("T2", "*_t2_*_l1pl350cl_off_r0.8_s1[5-7]", "[RESULT] T2 (cache-off burst control)"),   # bursts and twins
    ("G1", "*_sweep_g1_l1pl350cl_pc_o*_b2048_s1_r*", "[RESULT] G1 (second-seed capacities)"),   # both seed-1 sweeps
    ("G2", "*_sweep_g2_l1pl350cl_pc_off_b2048_*dev_r*", "[RESULT] G2 (development-mode twin)"),  # both arms
    ("P8192", "*_l1pl350cl_r2.2_b8192_s100[2-9]", "[RESULT] P8192 (P0 at an 8192-token step budget)"),  # bursts, controls
    ("REG-20", "*_l1pl350ol_r2.[05]_w3000_s1[5-9]", "[RESULT] REG-20"),
    ("REG-20", "*_burst_l1pl350cl_r2.0_w3000_s15_drift", "[RESULT] REG-20"),
    ("E1", "*_sweep_l1pl350cl_pc_off_r*", "[RESULT] E1"),            # the closed-loop cache-off sweep at 8 turns
    ("E0", "*_sweep_l1t16pl350cl_pc_o*_r*", "[RESULT] E0"),          # E2: both 16-turn sweeps, until the gate
    ("REG-25", "*_l1pl350cl_r2.0_s44", "[RESULT] REG-25"),      # the sustained wipe and its same-seed control
    ("REG-24", "*_confirm_l1pl350cl_r2.0_s43", "[RESULT] REG-24"),                 # C1's confirmation, burst run
    ("REG-24", "*_stage2_control_l1pl350cl_r2.0_long_s43", "[RESULT] REG-24"),     # and its same-schedule control
    # REG-27: REG-24's design on the four seeds of `reg27.REG27["seeds"]`, both runs per seed
    *((("REG-27", f"*_{fam}_s{sd}", "[RESULT] REG-27")
       for sd in (61, 68, 72, 79)
       for fam in ("confirm_l1pl350cl_r2.0", "stage2_control_l1pl350cl_r2.0_long"))),
)


class WithheldError(RuntimeError):
    """A sealed run folder was about to be read."""


def lifted(heading_text: str, decisions: Path | None = None) -> bool:
    """True once decisions.md has a heading that IS that result: `## <date> · <heading_text>` with no digit after it.
    A heading that only quotes the text (a [CORR], a [REG] naming the result it waits for) or names another
    registration ("[RESULT] REG-190") lifts nothing."""
    p = decisions or DECISIONS
    if not p.exists():
        return False
    pat = re.compile(r"^## \d{4}-\d{2}-\d{2} · " + re.escape(heading_text) + r"(?!\d)")
    return any(pat.match(line) for line in p.read_text().splitlines())


def unsealed_ids() -> set[str]:
    return {x.strip() for x in os.environ.get("AFTERSHOCK_UNSEAL", "").split(",") if x.strip()}


def sealed_by(run_name: str, decisions: Path | None = None) -> str | None:
    """The id of the registration that still seals this run folder, or None."""
    for reg, pattern, heading in SEALS:
        if fnmatch.fnmatchcase(run_name, pattern) and reg not in unsealed_ids() and not lifted(heading, decisions):
            return reg
    return None


def guard(run_name: str) -> None:
    reg = sealed_by(run_name)
    if reg:
        raise WithheldError(f"{run_name} is withheld until the [RESULT] of {reg} is in docs/decisions.md "
                            f"(analysis/withheld.py). Only its checks.json may be read.")


def keep(paths, what: str = "runs") -> list:
    """`paths` (run folders) without the sealed ones; one stderr line names how many were skipped and by which seal."""
    out, skipped = [], {}
    for p in paths:
        reg = sealed_by(Path(p).name)
        if reg:
            skipped[reg] = skipped.get(reg, 0) + 1
        else:
            out.append(p)
    for reg, n in sorted(skipped.items()):
        print(f"withheld: {what} skipped {n} run folder(s) sealed by {reg} (analysis/withheld.py)", file=sys.stderr)
    return out
