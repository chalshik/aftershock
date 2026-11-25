# aftershock

Code, configurations and analysis for a study, with pre-specified tests, of a cache-driven cold state in vLLM's prefix caching: a single vLLM 0.32.0 server on one RTX 3090 (350 W limit), serving multi-turn chat sessions. The study asks whether a short burst of new sessions can leave the server cold (no prefix-cache hits), full and backlogged after the load returns to its earlier level, where the same schedule without the burst stays healthy. Every test's question, thresholds and outcome rule were written into an append-only log, with the code committed, before its runs were read. The log's timestamps are the author's own commits, not a third-party registry, so the tests are called pre-specified.

The findings are reported in the paper *Turn Release Decides Whether a Load Test Sees Prefix-Cache Collapse: One vLLM Configuration* (preprint link to follow). The run data are on Zenodo: https://doi.org/10.5281/zenodo.24478010.

## Layout

| Path | What |
|---|---|
| `harness/` | the only entry point for a run: `uv run python -m harness.run configs/<x>.yaml` (GPU host) |
| `analysis/` | every number's computation |
| `tests/` | `uv run pytest`, against a fake vLLM server (`tests/fake_vllm.py`), no GPU |
| `configs/` | one YAML per run or sweep, each extending `base.yaml` |
| `env/` | host setup and the environment probe |
| `PROTOCOL.md` | the study's design: question, claims, what counts for and against, how runs are done |
| `docs/rules.md` | the working rules the code and docs cite (rule F, research rule 5, …) |
| `docs/definitions.md`, `docs/claims.md`, `docs/model.md` | one definition per term with its code, the claims and their tests, the models and the prediction ledger |
| `registrations/` | one card per registration: the question, runs, outcome rule, prediction and decisions fixed before the runs, then the result |
| `docs/results.md` | every registration and result heading of the study's log, in order |
| `docs/PROVENANCE.md` | the paper's numbers: for each, the run folders and the analysis function that produced it |
| `docs/EXPERIMENTS.md` | the paper's experiment names and their ids in the log |
| `docs/PREDICTIONS.md` | the probability each pre-specification gave its outcome, and each outcome beside what the paper reports |
| `docs/ARTIFACT.md` | what a run folder holds, the manifest, the entry point, the pre-specification process and the seals; the paper's Table 3 in full |
| `docs/probe/` | the inputs the code reads (seed picks fixed before their runs), and the saved outputs the paper cites by path and checksum |
| `manifests/runs.sha256` | the sha256 of every file of every run folder |

## The data

Run folders are not in git. They are published as a separate dataset, every folder kept, valid or invalid; `checks.json` in each says which and why, and only valid runs count. Put the folders under `runs/` and verify them with `python3 tools/manifest.py verify runs`. Each folder holds the resolved config, the schedule written before the run, per-request records, server metrics, GPU samples and the environment (`env.json`, with the commit the run used).

## Reproducing

- **A number in the paper:** `docs/PROVENANCE.md` gives its runs and function. The paper cites release tag `v1.1` of this repository (`v1.0` was the first release; v1.1 renumbers the docs' table and figure pointers to match the paper).
- **A registration's number:** find its registration in `registrations/` (or its heading in `docs/results.md`). Its computation is in `analysis/`, in the module named after it where there is one (`t4.py`, `k4e.py`, `reg27.py`, …; REG-19's is in `gate.py`, REG-20's in `k1.py`, REG-24's in `confirm.py`), and that module's tests in `tests/` show how it is called. Put the runs it reads under `runs/` and call it.
- **A run:** on a Linux host with an RTX 3090, `bash env/setup_vm.sh`, set the GPU power limit to 350 W, commit, then `uv run python -m harness.run configs/<x>.yaml`. A run from a dirty tree is refused.
- **Tests:** `uv sync --extra analysis && uv run pytest`.

## Notes on this release

- It is built from one commit of the study's working repository (b84bd24, 466 files). That repository also holds the study's log (`docs/decisions.md`) and its drafts; comments and docs here that cite `decisions.md` refer to that log. `registrations/` gives each registration's registered facts and result, compiled from it, and `docs/results.md` its registration and result headings verbatim. The full log is available from the author on request. A `FILE(...)` or `PROBE(...)` tag that names a path not present here refers to that commit.
- Wording in comments, docstrings, docs and some messages was edited for this release. No name, number or operator in the code changed (the build compares each file's tokens), and no config's resolved form changed (`tests/test_release_configs.py` checks every config against the working repository's).
- Left out of the code:

  - `analysis/recovered/`: recorded output of early, never-committed scripts (used only by recompute_published)
  - `analysis/recompute_published.py`: reads analysis/recovered/
  - `analysis/mechanisms.py`: REG-2's early dip readouts; not cited by the claims
  - `analysis/analysis.ipynb`: scratch notebook
  - `tests/test_mechanisms.py`: tests analysis/mechanisms.py
  - `tests/test_check_docs.py`: tests the docs checker (tools/check_docs.py), not in the release
  - `tests/test_launcher.py`: tests the GPU queue launcher (tools/launcher.py), not in the release

## License

Code (`harness/`, `analysis/`, `tests/`, `tools/`, `env/`): MIT, see `LICENSE`. Data, configs and docs: CC BY 4.0, see `LICENSE-DATA`.
