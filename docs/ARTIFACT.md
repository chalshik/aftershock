# The run folders, the entry point and the analysis

The procedures behind the paper's Appendix A. Section numbers refer to the paper that cites this repository by commit hash.

**What is kept.** Every run has a folder with its resolved configuration, the schedule written before the run, the per-request client log, the server metrics scraped each second, GPU samples, events, the server log, the environment (versions, kernel, driver, git commit, dirty flag) and the automatic validity checks. 314 run folders existed on 2026-11-22, 283 valid, 24 invalid and 7 without checks; invalid runs are kept and never counted. A checksum manifest (`manifests/runs.sha256`, append-only) lists the files of the run folders copied from the VM, and `python3 tools/manifest.py verify <copy>` checks any copy against it. The one folder that had existed only on the VM (the first attempt at the replication's seed-18 control, 2026-11-12) was copied and added to the manifest on 2026-11-22.

**One entry point.** `uv run python -m harness.run configs/<name>.yaml`. Every parameter is in a configuration file. A configuration extends a parent, and `configs/defaults.yaml` is merged beneath the chain; the resolved configuration is saved in the run folder. The configuration's workload seed builds the session schedule, which is written before the load starts. A configuration used by a run is never edited. A run from a dirty tree (a changed tracked file, or an untracked file under `configs/`) is invalid, and so is any run started with `--allow-dirty`.

**Pre-specification.** Each run and each analysis of existing runs has an entry in the study's append-only log (`docs/log`), written before the run or the computation: the question, a numeric prediction, the decision under each outcome, the thresholds, the stopping rule and the cost. Before it is committed, a separate review pass that sees only the entry and its code returns a trace on which the verdict would be wrong, the strongest alternative explanation, and whether any outcome changes a decision; its text is in the entry. Corrections are separate entries; old entries are never edited. A script (`tools/check_docs.py`) checks that run ids named in the documents exist, that validity claims match the checks, that the log only grew, and that no used configuration changed. It runs inside the test suite (`uv run pytest`) and at the end of each working session; there is no commit hook.

**Analysis.** Every number taken from the runs comes from a named function in `analysis/` applied to named run folders. A [RESULT] entry gives the command that produced its table, run once; some of the first entries (2026-11-04) do not. Runs whose outcome a pending pre-specification judges are sealed in the analysis code (`analysis/withheld.py`) until that registration's result is in the log. The owning pre-specification's own computation lifts the seal with an environment variable; the validity checks (`checks.json`) and the tools that hash and count run files are not sealed. The replication's table was computed by a script kept in the study's working repository (`docs/probe/20261113_reg19/ops_r3/final.sh`) that runs `analysis.gate.reg19_table` from a checkout of the commit at which the pre-specification was approved.

## The baseline before the burst, with clock, power and temperature

**The paper's Table 8, in full.** The 600 s before the burst (300 to 900 s), control / burst run, per pair; the control always ran first. SLO attainment and hit rate over the window, median TTFT in seconds, SM clock (MHz) and power (W) over busy GPU samples, mean temperature (°C) (P60).

| Pair | Seed | Gap (h) | SLO attainment | Hit rate | TTFT p50 | SM clock | Power | Temp. |
|---|---|---|---|---|---|---|---|---|
| replication | 15 | 1.2 | 1.000 / 0.984 | 0.833 / 0.833 | 0.102 / 0.103 | 1912 / 1907 | 346 / 346 | 64.5 / 65.6 |
| replication | 16 | 1.2 | 1.000 / 0.998 | 0.840 / 0.840 | 0.103 / 0.103 | 1906 / 1903 | 346 / 346 | 65.8 / 66.6 |
| replication | 17 | 3.4 | 1.000 / 0.997 | 0.838 / 0.838 | 0.102 / 0.102 | 1904 / 1896 | 347 / 347 | 67.0 / 68.2 |
| replication | 18 | 1.2 | 1.000 / 0.992 | 0.838 / 0.838 | 0.107 / 0.106 | 1900 / 1899 | 348 / 348 | 65.8 / 65.6 |
| replication | 19 | 1.2 | 1.000 / 0.990 | 0.834 / 0.834 | 0.107 / 0.107 | 1901 / 1902 | 347 / 347 | 65.7 / 65.8 |
| long-window | 43 | 4.6 | 1.000 / 0.990 | 0.840 / 0.840 | 0.102 / 0.103 | 1903 / 1905 | 347 / 347 | 65.5 / 65.0 |
| long-window | 61 | 4.5 | 1.000 / 0.984 | 0.839 / 0.839 | 0.100 / 0.101 | 1910 / 1901 | 347 / 347 | 64.8 / 67.3 |
| long-window | 68 | 28.5 | 1.000 / 0.992 | 0.837 / 0.837 | 0.102 / 0.104 | 1896 / 1898 | 347 / 346 | 67.4 / 68.0 |
| long-window | 72 | 4.5 | 1.000 / 0.993 | 0.839 / 0.839 | 0.101 / 0.102 | 1907 / 1910 | 347 / 346 | 66.1 / 65.7 |
| long-window | 79 | 4.5 | 1.000 / 0.987 | 0.835 / 0.835 | 0.103 / 0.103 | 1910 / 1908 | 347 / 347 | 64.5 / 65.1 |
