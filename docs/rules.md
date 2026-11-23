# Study rules

The rules this study ran under. Code comments and docs cite them by letter (rule F, rule H) or by number (research rule 5). `PROTOCOL.md` gathers the study's design; this file holds the working rules.

## A. Stance
1. Before any run, state the prediction as a number, the decision that changes under each outcome, and the cost in GPU hours. If no outcome changes a decision, the run is not done.
2. For every result, list at least two alternative explanations and the evidence that would separate them.
3. Keep observed, derived and conjectured statements apart.
4. Change a position only for new evidence or a better argument, and name which.
5. Never choose or widen a threshold after seeing the data it judges.
6. Log errors as `[CORR]` entries; never fix them silently.

## B. Evidence tags
Every factual statement carries one: `RUN(run_id, function)` · `SOURCE(vLLM 0.32.0 path:symbol)` · `FILE(path:line)` · `PROBE(path)` · `DERIVED` (the computation is shown) · `BELIEF` (not evidence; if it can be checked, it is listed as an open question).

## C. Decisions reserved to the author
Any GPU run; changes to the protocol, verdict code, thresholds, checks, workload or seeds; marking a run invalid beyond the automatic checks; vLLM or `uv.lock` changes; deleting anything; anything that changes a paper claim.

## E. Templates
Every run is registered with a `[REG]` entry before it runs; results use `[RESULT]`, corrections `[CORR]`. `docs/results.md` lists every registration and result heading of the study's log.

## F. Adversarial review
Before any registration, a separate review pass sees only the registration and the code diff. It builds a trace on which the verdict would be wrong, gives the strongest alternative explanation, and says whether the run can change any decision. Its findings go into the `[REG]` entry. At most two rounds per registration.

## G. Feasibility gate
Before any shock matrix, compute from existing or cheap runs whether a stuck state is possible at all (`docs/model.md`, M2). The pass condition is registered before it is computed.

## H. Statistics
Report k of n per cell with its one-sided 95 % Clopper–Pearson bound. Do not pool cells without a stated assumption. A control with a single seed is labelled as such.

## I. Provenance
Every number in the docs comes from a named function in `analysis/` applied to named run ids.

## J. Definitions
`docs/definitions.md` holds one definition per term with its code; code must match it. A term with more than one computation is marked, and the choice is the author's.

## Research rules
1. **Scope.** One trigger at a time; later triggers only after earlier ones work. The gateway stays off (`gateway.enabled: false`).
2. **Fixed choices:** a single replica, served directly; open-loop session arrivals (turn release, open- or closed-loop, is set per config and registered); no client retries or timeouts; no KV offload, and no KV sharing between replicas.
3. **Pinned versions.** vLLM is pinned in `pyproject.toml` and never upgraded during the study. Any `uv.lock` change is logged.
4. **Raw data is immutable.** Nothing in `runs/` is edited, rewritten or deleted. A failed run is kept and marked invalid in its `checks.json`.
5. **Every parameter lives in config.** No magic numbers in code. Once a run has used a config, the config is never edited; it is copied to a new file.
6. **Reproducible.** Seeds are fixed and the schedule is written before the run; a run from a dirty git tree is invalid.
7. **Pre-register** thresholds, tolerances and SLOs in config or in the log *before* looking at the results they judge.
8. **Figures** come only from run folders, through `analysis/`.
9. **Evidence, not memory** (rule B). Open questions are answered only with probe output, the pinned vLLM source, or a run id.
10. **Report honestly.** An invalid run, a surprising number or a deviation is logged.
