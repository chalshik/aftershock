# Claims

The study's claims, each with its test, what would support it, what would refute it, and its status. Created 2026-11-04; restructured 2026-11-12 (decisions.md 2026-11-12 [DECISION], Paper Plan v3). Changing a claim or its status needs the author (docs/rules.md, C).

**Structure since 2026-11-12.** C1 and C2 are now the **premise P0**: they are cited from three preprints (Lu et al., arXiv 2611.02543; Valse and Jannesari, arXiv 2611.08902; Huang and Liu, arXiv 2611.13412) and replicated here; the paper claims no novelty for them. Their tests below are unchanged and still judge every run registered against them (REG-19 included). The paper's own claims are **K1–K5**, at the end of this file. What the three preprints show is BELIEF until checked against the papers (open-questions A16).

**Paper layer.** The paper plan is `docs/paper/plan/research-plan.md` (until 2026-11-17 `docs/paper/paper-plan-v3.md`), in the paper repository `docs/paper`, which has its own git history and is excluded from this one; it replaces Paper Plan v2 and its claims C3–C9 (v2's C3 became K2, C5 became K3 and K4, C7 became K5; C4 is reduced to reporting cache reset as a weak trigger; C6, C8 and C9 are dropped). Nothing here may cite the paper plan as evidence, and it is not checked by `tools/check_docs.py`.

## C1. The symptom
**Claim:** at some load λ between μ_off and μ_warm, percent fast stays low after the trigger has ended, while the same load without a trigger stays healthy (plan, Stage 2 Purpose; PROTOCOL §2).

- **Test:** shock runs judged by `load.verdict` (PROTOCOL §3), and all six conditions of a metastable failure at one load; plus the up-then-down ramp at two dwells.
- **Supports:** at one load, `stuck` in most seeds, the control `recovered`, the confirmation run `stuck` over a 4500 s watch, and `recovered` after the drop to λ = 1 (condition 6). Or a ramp gap that stays at both dwells.
- **Refutes, for the conditions tested:** every valid trigger run at a setup's healthy loads `recovered` or `slow_drain`, none `stuck`, and that setup's ramp shows no gap (PROTOCOL §3; this rule was written after results, **[new 2026-11-04]**).
- **Metric:** percent fast is the plan's main number. **Decided 2026-11-12 (author):** the paper states C1 in hit rate and SLO share (percent fast), not completions/s; the completion rate stays a secondary readout (decisions.md 2026-11-05 and 2026-11-12 [DECISION]).
- **Status (2026-11-04):** not supported. Per cell, all stuck 0 of 3, one-sided 95 % upper bound 0.63 each (`analysis/stats.py`, `cp_upper`): **Superseded on 2026-11-15: see "Status lines of 2026-11-15" at the end of this file.**
  - default pool, 20 s gaps: wipe λ = 3, wipe λ = 3.5, burst λ = 3, burst λ = 3.5; ramp at 300 s and 600 s: no gap. Negative by the PROTOCOL rule.
  - 72,480-token pool, 20 s gaps: wipe λ = 3, burst λ = 3. **No ramp was run, so this setup is not negative by the PROTOCOL rule**; it is "no stuck run".
  - default pool, 5 s gaps: wipe and burst at λ = 3.5 and 4.5. **No ramp was run**; "no stuck run".
  - Single-seed controls in every cell. Whether the test loads were inside a real danger zone is in question (`docs/model.md` M2, M4); until the feasibility gate is run, a negative here may say only that these loads recover.

## C2. The cause
**Claim:** a sustaining loop. Cache lost → slow requests → big running batch → KV memory full → cached prefixes evicted before their session's next turn → cache stays lost (plan; PROTOCOL §2).

- **Test:** in stuck runs, the loop figure (helper metrics in the plan's order) and condition 4 (mean KV ≥ 0.80, mean N ≥ 2 × baseline, N below the 256 cap); then the controls that cut a link: prefix caching off, and a session-cap arm (a limit on the sessions in service, set at the load generator). **The session-cap arm replaces the protected-prefixes control (author, decisions.md 2026-11-12 [DECISION]);** it is not yet defined in a config or registered.
- **Supports:** stuck runs that show the order and the memory limit, and stop getting stuck when a link is cut.
- **Refutes:** stuck runs without a memory limit (a backlog, not this loop), or stuck runs that stay stuck under the session cap.
- **Status (2026-11-04):** untestable so far: there is no stuck run. Related evidence, not a test of C2: during bursts KV reached 1.00, and in the first 60 s after each burst the counter hit rate was 0.11–0.35 (a counter that includes the burst requests, which never hit, so it overstates how cold the sessions' cache was; decisions.md 2026-11-05 [CORR]), and every run then recovered within about 5 minutes (decisions.md 2026-11-02, 2026-11-03). The caching-off control has no defined form yet, and the protected-prefixes control (S4) was not investigated; it is replaced by the session-cap arm (2026-11-12). **Superseded on 2026-11-15: see "Status lines of 2026-11-15" at the end of this file.**

## Status of C1 and C2 after 2026-11-04
The two status lines above are dated 2026-11-04 and describe the open-loop setups. Later results are in `docs/status.md` and decisions.md: with closed-loop turns at setup L1 (350 W), single-seed pairs stuck with a recovered control at λ = 2.2 and 2.0 (decisions.md 2026-11-10 [RESULT] REG-14, REG-17), and the registered matrix at λ = 2.0 gave 1 of 4 (decisions.md 2026-11-11 [RESULT] REG-18). C1 is still not supported by its own test; REG-19 is under way. **C2 is no longer "untestable":** stuck runs exist, but condition 4 has not been computed on them, no loop figure exists, and neither control that cuts a link has been run, so C2 is untested. **Superseded on 2026-11-15: see "Status lines of 2026-11-15" at the end of this file.**

# Paper claims K1–K5 (since 2026-11-12)

Each is stated for the cells tested only. None has a registered test yet; every test below needs its own `[REG]`, with thresholds fixed before the data (rule A5). **Superseded on 2026-11-15: see "Status lines of 2026-11-15" at the end of this file.**

## K1. Turn semantics
**Claim:** whether a turn waits for the previous reply decides the outcome. With open-loop turns (each turn sent at its scheduled time) a slow server makes the next turn arrive while the previous one still runs, so its prefix is held and cannot be evicted. With closed-loop turns (the next turn sent a think time after the reply) every prefix has an idle, evictable interval, and the stuck state can hold.

- **Prior statement (BELIEF, open-questions A16):** Lu et al. name closed-loop sessions as a premise of the feedback (their §9), and Huang and Liu contrast agent sessions with "open-loop chat traffic" (their §2); neither tests an open-loop arm. K1 is the test of that stated premise, not a new idea. Valse and Jannesari's congestion bound assumes that each session "has at most one outstanding request" and that "sessions share no cached tokens" (their assumption A1, App. E.14); K1 relaxes the first clause and K2 the second.
- **Test:** one cell, the same seeds and the same burst, two arms that differ only in `workload.sessions.closed_loop`.
- **Supports:** the closed-loop arm passes C1's per-seed test in most seeds while the open-loop arm is `recovered`, and most open-loop session hits are served while the previous turn runs (`gate.overlap_share`).
- **Refutes:** the open-loop arm stuck at the matched cell.
- **Caution (2026-11-12 audit):** latency is measured from the planned time with open-loop turns and from the release time with closed-loop turns (`docs/definitions.md`, percent fast), so percent fast and `load.verdict` are not the same quantity in the two arms. The K1 test must be registered on readouts defined alike in both arms: hit rate, KV usage, the server's waiting line, completions per second.
- **Status (2026-11-12):** not tested at a clean load. The only related evidence is one matched pair at L1, λ = 2.4, one seed: the open-loop burst run drained with `overlap_share` 0.608 and the closed-loop one stuck (decisions.md 2026-11-09 [RESULT] REG-10 and REG-11); λ = 2.4 is at the closed-loop warm capacity, so overload explains the closed-loop arm as well (decisions.md 2026-11-10 [RESULT] REG-13). **Superseded on 2026-11-15: see "Status lines of 2026-11-15" at the end of this file.**
- **Not evidence for K1 ([CORR] 2026-11-12):** the open-loop runs that were stuck in 0 of 3 in each of 10 cells (FILE(PROTOCOL.md:105)). None of those cells is known to have been able to get stuck at all. At the default pool, M2 says the λ = 3 runs could not stay stuck under the cache-off saturation floor, and no run there could under assumption A (untested); λ = 3.5 sits at the floor (`docs/model.md` M2). At the 72,480-token pool and with 5 s gaps the cold floor is unmeasured. A recovery in a cell where a stuck state may not have been possible cannot be credited to turn semantics. This line said they were related evidence.
- **Registered test (2026-11-12; approved by the author the same day, decisions.md [DECISION]):** REG-20 (decisions.md 2026-11-12 [REG]), which replaces the pass-rate comparison above with a readout defined alike in both arms and adds a check at an equal share of the warm capacity.

## K2. Cost gap and accumulated context
**Claim:** the range of loads between the cold capacity and the warm capacity widens as sessions accumulate more context, here by the number of turns per session. (Until 2026-11-12 the claim had a second half, that the range is empty when all reuse comes from prefixes shared across sessions; it was removed as untestable in this setup, see the note below.)

- **Definitions (2026-11-12, decisions.md [CORR]):** *cache gain* is the counter hit rate (`load._level` `"hit"`) of the no-trigger control in its warm state. A workload is *shared-only* when every reusable token lies in one of a fixed set of prefixes shared across sessions (tenants), the set's total size does not change with load or backlog, and no request reuses tokens of its own session's earlier turns. *Matched cache gain* means the two warm hit rates agree within a tolerance registered beforehand. A workload that mixes a shared prompt with growing per-session context does not test the shared-only prediction.
- **Prior result to cite (BELIEF, checked through verbatim quotes, open-questions A16):** Huang and Liu's sessions carry a 2K system prompt shared by all sessions inside a 6K–37K context, and their stuck state held in 9 of 12 pairs. That workload is mixed, so it is not a counterexample; they report that the collapsed hit ratio of 0.10–0.12 is the shared prompt alone.

- **Test:** setups that differ only in the turns per session (8 at L1, then 16, then 4; decisions.md 2026-11-12 [DECISION], design choices), under closed-loop turns, each with its cold floor and warm capacity measured under closed-loop turns and a paired shock matrix. The three turn counts straddle the rule of thumb of Schroeder et al. (2006, principle vii; BELIEF, read 2026-11-12) that a partly open system behaves like an open one at up to about 5 requests per session and like a closed one from about 10.
- **Staged (2026-11-12, decisions.md [DECISION] "K2 staged"; `docs/experiment-plan.md` step E):** the range is measured first, by closed-loop sweeps with caching on and off at 8 and 16 turns, against a prediction and a gate registered before them; the 16-turn shock matrix runs only if the range widens by the registered factor. A range that does not widen refutes the first half at sweep cost.
- **Supports:** the range (warm capacity over cold floor) and the share of seeds that pass rise with the turn count.
- **Refutes:** no trend, or the opposite trend, with the turn count.
- **Shared prefixes:** A prefix shared across sessions is no longer a tested arm (decisions.md 2026-11-12 [DECISION], wording and K2): with a small set of shared prefixes and a large running batch every prefix is held by some running request and cannot be evicted, so "no stuck run" would follow from the engine's rule and could not refute anything. It is kept as a stated consequence, with the prior evidence (BELIEF): Huang and Liu report that the collapsed hit ratio of 0.10–0.12 is their shared system prompt alone, and Valse and Jannesari expect, without having shown it, that only the private suffixes are affected.
- **Status (2026-11-12):** untested. The 512-token setups were shocked with open-loop turns only, so they say nothing here. The system prompt is drawn per session (FILE(harness/workload.py:97)); a shared prompt needs a harness change (author). **Superseded on 2026-11-15: see "Status lines of 2026-11-15" at the end of this file.**

## K3. Model transfer
**Claim:** the published models place this setup's boundary and describe its stuck state within a tolerance fixed before the runs, or they do not. Quantities to predict: the drain threshold (a stuck server drains only below its all-miss capacity), the backlog slope, the mix of turns served while stuck, and the healthy-side condition.

- **Test:** the models' predictions and the tolerances registered before each run; the comparison computed once.
- **Supports / refutes:** per quantity, inside or outside its registered tolerance. Either outcome is reported.
- **Status (2026-11-12):** no registered test. Model F's equations were read from the paper and evaluated for L1 (`docs/model.md` M5; [CORR] 2026-11-12: this line said the equations were known only through summaries); everything about the model stays BELIEF (open-questions A16). Three readings of runs already seen agree with a drain threshold near the cold floor, none of them blind: the stuck completion rate is 1.03–1.08 times the cold floor (DERIVED: 1.73 / 1.681 and 1.81 / 1.681; `docs/model.md` M2, REG-8 and REG-14 rows); REG-18's seeds 3 and 6 failed the cold side only in spans planned at 1.62 and 1.76 (decisions.md 2026-11-11 [RESULT] REG-18); REG-15's 1800 s drop was short of the ≈ 3960 s a drain at the floor needs (DERIVED: 400 waiting / (1.681 − 1.58) req/s; `docs/model.md` M2, REG-15 row). **Superseded on 2026-11-15: see "Status lines of 2026-11-15" at the end of this file.**
  - Expected under the published model for this workload (computed 2026-11-12 from the model's equations, fixed 8 turns, turn cost proportional to prompt length; BELIEF as to the model, the code is `analysis/modelf.py` with tests, moved 2026-11-12): at λ = 2.0 (ρ = 1.19 against the open-loop floor) sessions complete at 0.89 of the all-miss capacity, the session backlog grows at 1.56 times the excess load, the served prompts are 1.6 % shorter than the schedule's mean, and requests complete at 1.016 times the all-miss request rate (1.71 req/s); at λ = 2.2, 0.83, 1.53, 2.5 % and 1.025 (1.72 req/s). The measured request rates (1.73–1.77 and 1.73–1.81) lie 1–5 % above these. So the request rate is a weak test, and it already leans above the model; the session completion rate and the session backlog slope are the discriminating readouts, and neither has been computed.
  - Alternatives for the first reading: the floor, measured with open-loop turns on seed 0, is off for closed-loop turns; or the spans were partly re-warming. The mean prompt length of the turns served in those spans separates them.

## K4. Lifetime of the warm state near the edge
**Claim:** close to the warm capacity the healthy state ends at an arrival surge, and the time to such a surge can be computed from the schedule alone.

- **Test:** a surge threshold fixed beforehand, then no-burst runs on new schedules chosen on both sides of it (`gate.surge_hazard` gives the schedule side).
- **Supports:** runs tip where the schedule crosses the threshold and stay warm where it does not.
- **Refutes:** tips without a surge, or surges above the threshold without a tip.
- **Status (2026-11-12):** exploratory. Every no-burst collapse so far followed a 120 s surge of 3.24–3.49 req/s (decisions.md 2026-11-10 [CORR]); that readout was found in the data and has not judged any new run. Prior work (BELIEF): Valse and Jannesari have a heuristic for the edge at which a healthy run tips by itself, with misses they report; their tests sit at 0.41–0.51 of the warm peak, ours at 0.83–1.0. **Superseded on 2026-11-15: see "Status lines of 2026-11-15" at the end of this file.**

## K5. Detection lead time (optional)
**Claim:** recomputed prefill tokens per second and the hit rate change before latency and queue length do; the contribution is the measured lead time.

- **Test:** a registered onset rule per signal, applied to stuck runs. 0 GPU h on existing runs, which makes it retrospective.
- **Status (2026-11-12):** not started; kept only if K1–K3 leave room. **Superseded on 2026-11-15: see "Status lines of 2026-11-15" at the end of this file.**

# Status lines of 2026-11-15

These replace the status lines marked "Superseded on 2026-11-15" above (author, decisions.md 2026-11-15 [DECISION] (author) on the preprint and claims.md). Each claim's test, support and refutation above are unchanged.

## C1. Status (2026-11-17)
- **Not supported by its own test** (all six conditions of PROTOCOL §3 at one load). At one load of one setup (L1, λ = 2.0), the matrix condition holds under a test written after the earlier matrix, and conditions 5 and 6 hold in one seed (REG-24). Condition 4's order clause is not met: the one judged burst run at this cell reads `does_not_follow` (0 of 1), as did REG-25's wipe run.
- **Setup L1, 350 W, closed-loop turns, λ = 2.0.** This load is 1.11 times the closed-loop cold floor (2.0 / 1.798, DERIVED; decisions.md 2026-11-14 [RESULT] E1, one seed) and 1.19 times the open-loop one (2.0 / 1.681).
  - REG-19's registered matrix on new seeds 15–19 gave `supports`, k_clear 5 of 5, one-sided 95 % Clopper–Pearson bounds 0.549 to 1.000 (RUN(REG-19's ten runs, gate.reg19_table); decisions.md 2026-11-13 [RESULT] REG-19).
  - The registered sentence: "At λ = 2.0, after the burst the server stayed stuck, cold and below its load through a 3000 s watch without the line emptying, while the same schedule without the burst stayed healthy, in 5 of 5 new seeds (15–19), under REG-19's whole-window test, written after REG-18's matrix (1 of 4 under the per-span test). Under REG-18's per-span test on the same runs: 3 of 5."
  - Each seed has a single control (rule H).
  - Earlier at the same cell: REG-17 `pass_clear` on one seed; REG-18 `some`, 1 of 4, stopped by its futility rule, no bound.
- **By condition at λ = 2.0:**
  1. The trigger ended and the load was unchanged, by design.
  2. The control `recovered` in 5 of 5 (REG-19).
  3. Most seeds stuck: holds under REG-19's test.
  4. **Levels held; order clause not met.** REG-22 read the registered levels and the four links retrospectively (C2), and REG-24's burst run held the levels at the watch's end (KV 0.950, N 30.0, 5.95× baseline). The order clause, judged literally on new runs by the order test registered on 2026-11-14 (decisions.md 2026-11-14 [REG] Condition 4's order clause …), reads `does_not_follow` on REG-24's burst run: reuse fell at 930 s, but the rise in uncached prompt tokens came at 1230 s, 310 s after the running batch's rise and 270 s after the pool's, and preemptions were 231 against 8242 requests ended ok, above the 1 % limit (DERIVED: 0.028) (RUN(20261116-182704_confirm_l1pl350cl_r2.0_s43, order.table); 0 of 1 at this cell, a single seed, upper bound 0.95). REG-25's wipe run read `does_not_follow` as well (decisions.md 2026-11-15 [RESULT] REG-25). D1's capped runs were not stuck, so they are not judged.
  5. **Persistence over a 4500 s watch: holds in one seed (REG-24).**
  6. **Two states at one load: holds in one seed (REG-24).** Wording approved by the author (2026-11-17), REG-24's registered sentence for `confirms` (RUN(20261116-135154_stage2_control_l1pl350cl_r2.0_long_s43, confirm.reg24); RUN(20261116-182704_confirm_l1pl350cl_r2.0_s43, confirm.reg24); decisions.md 2026-11-16 [RESULT] REG-24):
     > At λ = 2.0, in one seed (43), beside one no-burst control on the same arrivals: after the burst the server stayed stuck for 4500 s, its line never emptying and its completions below the planned load; its total line drained 2380 s into a 9000 s drop to λ = 1; the last 600 s of the return to λ = 2.0 were recovered against the pre-burst baseline; the control stayed warm throughout. The stuck window was planned at λp 2.068 and the recovered window at 1.903, the largest such difference among the 25 schedules that the seed rule's window test admitted (`reg24_tilt`, `n_larger` 0). The control carried both windows warm. 1 of 1 (one-sided 95 % Clopper–Pearson upper bound 1.00, lower bound 0.05). This meets C1's conditions 5 and 6 at this cell. It does not test C2's chain.
     - **Not separated:** the persistence may be held by the clients' owed backlog against the pool size rather than by C2's loop. At the watch's end 1650 turns were owed, more context than the 96,656-token pool holds (decisions.md 2026-11-16 [RESULT] REG-24, alternative 5).
     - **Limits:** one seed, one load; a 9000 s drop against PROTOCOL's 1800 s (author's ruling); the control's schedule peaked at 2.733 over 120 s, lighter than REG-19's 2.833–3.092.
     - Earlier: at λ = 2.2, REG-15's one run stayed stuck through 4500 s, but a no-burst control also went cold at that load (REG-16).
- **Open-loop setups** (the status of 2026-11-04 still stands for them): per cell, all stuck 0 of 3, one-sided 95 % upper bound 0.63 each. Whether those cells could get stuck at all is in question (`docs/model.md` M2, M4).

## C2. Status (2026-11-17)
- **Not met.** Both controls that cut a link now have a result: the session cap (D1, `supports`, and D1b, `batch_bound`, below) and caching off (D2, answered by feasibility). Neither separates C2's loop from population-driven pool pressure or a smaller effective shock, the order of the loop's signals is not judged, and no loop figure exists.
- **Read retrospectively at L1, λ = 2.0, by REG-22's line a, each batch apart, never pooled:**
  - batch 2 (REG-19's seeds): "condition 4's registered levels held at the end of the watch, and all four links were present in every 300 s span after the burst's requests had left, in 5 of 5 stuck runs at λ = 2.0 (one-sided 95 % Clopper–Pearson bounds 0.549 to 1.000); the order clause is not judged and no loop figure exists; an overload gives the same picture; the loop-cut controls are not run". Retrospective by design.
  - batch 1 (seeds 2, 4, 3, 6): the same sentence with 4 of 4 (bounds 0.473 to 1.000), with thresholds fixed after these runs were reported.
  - Every one of these runs has a same-seed control that recovered and holds the four links in no judged span (decisions.md 2026-11-13 [RESULT] REG-22 batches 1 and 2).
  - λ = 2.4 (batch 1, 5 of 5) is given only "at the closed-loop warm capacity; three of five controls did not stay healthy (REG-12)", and is not cited as support.
- **Caching off (D2): answered by feasibility, no run** (decisions.md 2026-11-14 [DECISION] (author) C2's caching-off control (D2) is answered by feasibility arithmetic, not by a run).
  - At L1 with closed-loop turns, the cache-off server shows one throughput level, rising with the load, over 1.5–4.0 req/s: 1.459 req/s at 1.5 and 1.706–1.848 at 2.0–4.0 (decisions.md 2026-11-14 [RESULT] E1; [RESULT] D2's matched capacity).
  - A stuck state at the matched load (λ_D2 = 0.80) would need a slow state below 0.80 req/s. None is on record, so no outcome of a run would change a decision (rule A1).
  - **Then tested by one run set (T2), 2026-11-21:** REG-19's burst with caching off at λ = 0.80, each burst run beside a no-burst twin, 1500 s watch, seeds 15–17. Outcome `drains`: in the 2 counted seeds (16, 17) the line emptied 338–390 s after the burst and the last 600 s showed no lasting line or completion shortfall against the twin (one-sided 95 % lower bound 0.224); seed 15 was not counted (baseline 0.945 of planned against 0.95, both runs; A27). Headroom 0.44 of the cold floor against 1.11 at the stuck cell, so this does not show the cache is necessary (decisions.md 2026-11-21 [RESULT] T2 (cache-off burst control); the paper's wording: decisions.md 2026-11-22 [DECISION] (author) Paper wording for T2 …).
  - This rests on a single seed (0) and one 900 s run per rate, with no burst (rule H). It is not "a cache-off burst run recovered".
- **The session cap (D1): `supports`** (RUN(the five capped runs, d1.table); decisions.md 2026-11-16 [RESULT] D1 (C2, session cap)). Wording approved by the author (2026-11-16; decisions.md 2026-11-16 [DECISION] (author) wording of C2, K1 and K2 …):
  > At λ = 2.0 (L1, 350 W, closed-loop turns), the session cap at each seed's healthy peak (25–29 sessions) prevented the stuck state in 5 of 5 seeds (one-sided 95 % lower bound 0.549 on the share cut), where the same burst without the cap left all five stuck (REG-19). Of these, 0 were never cold, 2 were cold only while the burst's own requests were in the system, and 3 were partly cold after the burst's exit (reuse 0.80–0.90, against 0.00 uncapped) and then cleared; no capped seed reached the stuck state's reuse or completion rate after the burst. The cap prevented the onset; whether it breaks an established stuck state is not tested by D1. The cap holds the whole population in service, so it holds down the running batch and the pool gauge together, and it moves waiting to the client; D1 does not say which link mattered, nor whether the loop runs through the prefix cache at all. The caching-off control is not run.
- **The cap above the cold batch (D1b): `batch_bound`** (RUN(the three cap-40 runs, d1b.table); decisions.md 2026-11-17 [RESULT] D1b (C2, cap above the cold batch)): at cap 40, seeds 16 and 17 were stuck where D1's cap had cut them (2 of 3), seed 15 was `rewarming` but backlogged at the cap, and none was cut. Wording approved by the author (2026-11-17):
  > A session cap at each seed's healthy peak (25–29) prevented the stuck state in 5 of 5 seeds (D1); at a cap of 40, which bounds the population in service but lets the running batch reach the cold state's (≈ 30), the burst still left the system stuck in 2 of the 3 seeds re-run (16, 17) and partly re-warming but backlogged in the third (D1b). On these seeds the cap worked by holding the running batch, not by bounding the population; that is not separated from the pool filling, nor from idle prefixes being evicted.
- **Open-loop setups** (2026-11-04, unchanged): no stuck run; during bursts KV reached 1.00, and every run recovered within about 5 minutes.

## K1–K5: which have a registered test (replaces the header line "None has a registered test yet")
Registered so far:
- **K1:** REG-20 computed: arm A `supports`, arm B `headroom_weakened` (decisions.md 2026-11-16 [RESULT] REG-20).
- **K2:** E1 and E2 measured; E0 read `floor_only` (decisions.md 2026-11-16 [RESULT] E0), so no 16-turn shock matrix.
- **K3:** REG-21, a retrospective reading of the premise its four quantities share, not a test of them.
- **K3 (2026-11-17):** K3b computed: the served mix, r and the completion rate `inside` on K4's four tipped runs (decisions.md 2026-11-17 [RESULT] K3b (model F on K4's tipped runs)). The drain threshold, the backlog slope and the healthy-side condition have no judged test.
- **K5:** REG-23 (computed; K5 leaves the claims).
- **K4:** computed 2026-11-17: `supports` (decisions.md 2026-11-17 [RESULT] K4 (the warm state ends at a surge)).

Every other test needs its own `[REG]`, with thresholds fixed before the data (rule A5).

## K1. Status (2026-11-16)
- **REG-20 is computed (decisions.md 2026-11-16 [RESULT] REG-20):**
  - **Arm A, `supports`:** at λ = 2.0, closed-loop turns stuck in 5 of 5 seeds and open-loop turns recovered in 5 of 5. Exact one-sided McNemar p = 0.031, mechanism readout shown in each (RUN(REG-19's five closed burst runs and arm A's five open runs, k1.reg20_table)).
  - **Arm B, `headroom_weakened`:** open-loop turns at λ = 2.5 recovered in 3 of 3 seeds (upper bound 0.632 on the cold rate), with the mechanism shown in each (RUN(the three arm B runs, k1.reg20_table)).
  - The drift run agrees, on one seed.
- **Limit kept:** a smaller working set in the open arm (9–11 sessions in service at λ = 2.0 and 11–13 at λ = 2.5, against 218–320 in the closed arm at 2.0; corrected 2026-11-17) is not ruled out, so held prefixes are not shown to be the cause.
- **Wording approved by the author** (2026-11-16; decisions.md 2026-11-16 [DECISION] (author) wording of C2, K1 and K2 …), REG-20's registered sentence for `supports` with `headroom_weakened`, within the limit above:
  > At L1 and λ = 2.0, after the same burst on the same five seeds, closed-loop turns left the server cold and backlogged at the end of a 3000 s watch in 5 of 5 seeds, and open-loop turns recovered in 5 of 5 (exact one-sided McNemar p = 0.031). Open-loop turns also recovered in 3 of 3 seeds at λ = 2.5, the same share of their warm capacity, after a larger shock. In every open-loop run, turns that arrived while the session's previous turn was still running reused all of their prefix.
- The claim text above is unchanged; it is stated for the cells tested only.

## K2. Status (2026-11-16)
- **The gate is computed: `floor_only`, no shock matrix.**
  - E1's 8-turn closed-loop cold floor is 1.798 req/s: status "ok", λ_sat 2.5, n 4, rates 2.5–4.0, seed 0, one run per rate. It lies inside the predicted 1.70–1.80 (RUN(E1's six runs, e1.observed); decisions.md 2026-11-14 [RESULT] E1).
  - Throughput rises with the rate, +0.084 req/s per req/s (SE 0.019), so the floor is a mean over 2.5–4.0, not a constant.
- **E0 read `floor_only` (2026-11-16, seed 0; decisions.md 2026-11-16 [RESULT] E0):** at 16 turns the cold floor fell to 1.400 req/s (1.758 at 8, over the shared rates), and the warm capacity fell one grid step to 2.2 (2.4 at 8). So R = 1.151 ≥ g = 1.083, but the warm side did not gain (RUN(the seven E2-on runs, k2.observed)).
  - The window widened only through the floor. The 16-turn shock matrix is not registered, and K2 stops at E1 + E2.
  - **Wording approved by the author** (2026-11-16; decisions.md 2026-11-16 [DECISION] (author) wording of C2, K1 and K2 …), the decision tree's sentence for "floor_only": "From 8 to 16 turns per session the window ratio grew by R = 1.151, entirely because the cold floor fell (floor ratio 0.797); the warm capacity did not rise (2.2 against 2.4). One seed, one 900 s run per rate." K2's first half is reported as: the window widens only through the cold floor. The claim text above is unchanged.
- Unchanged: the 512-token setups were shocked with open-loop turns only, so they say nothing here. The system prompt is drawn per session (FILE(harness/workload.py:97)).

## K3. Status (2026-11-15)
- **No registered test of K3's four quantities** (the drain threshold, the backlog slope, the mix of turns served, the healthy-side condition). None has a judged comparison on any run.
- **REG-21 read the premise the four share retrospectively,** and is never called a test (decisions.md 2026-11-13 [RESULT] REG-21 batches 1 and 2):
  - batch 2, on REG-19's runs: median r 1.028 (n 5, 1.023 to 1.029) at λ = 2.0;
  - batch 1: 1.021 at λ = 2.0 (n 5) and 1.020 at λ = 2.4 (n 6).
  - So the cold and backlogged engine did the token work of the cache-off sweep within 0.05.
- **The reference r used is the open-loop cache-off sweep** (1.681).
  - E1's closed-loop floor (1.798) is now known. The stuck completion rates (1.720 to 1.798 in batch 1, 1.740 to 1.758 in batch 2) lie at or just below it, and 1.02–1.07 times the open-loop floor (DERIVED: 1.720 / 1.681 and 1.798 / 1.681).
  - A reading of r against the closed-loop sweep would need its own registration. None is proposed here.
- Unchanged, BELIEF as to the model: model F's expected values at λ = 2.0 and 2.2 (`docs/model.md` M5). The session completion rate and the session backlog slope are the discriminating readouts, and neither has been judged.

## K4. Status (2026-11-15)
- **Exploratory. No registered test.** Every no-burst collapse so far followed a 120 s surge of 3.24–3.49 req/s, and a warm peak of 3.32 that did not tip overlaps that range (decisions.md 2026-11-10 [CORR]). That readout was found in the data and has judged no new run.
- Since then the ten no-burst controls at λ = 2.0 all stayed warm. Nine ran on schedules whose highest 120 s planned rate was below 3.2. These schedules were not chosen on both sides of a threshold, so they do not test K4.
- In REG-23's six no-trigger collapses the sent arrival rate peaked at 2.925–3.025 req/s before the KV onset, against planned peaks of 3.350–3.708. It never crossed 3.2 (decisions.md 2026-11-14 [RESULT] REG-23). That is a readout, not a test of K4.

## K3. Status (2026-11-17): a blind test registered, not computed
- **Registered test:** decisions.md 2026-11-17 [REG] K3b (model F on K4's tipped runs), approved by the author. On K4's runs that tip and stay cold, over W = [onset + 300 s, 2700 s): the served prompt mix against model F's tilt (±0.015, judged only where the band excludes the untilted mix), with the token work r (±0.05) and the completion rate (±5 %) as consistency checks. The session completion rate is reported only (the surge's own cohort biases it). Model F's values were fixed before any K4 run was read (PROBE(docs/probe/k3b_draft/predictions_e3f2d73.txt)). Computed once after K4's [RESULT].
- **Not tested by it:** the drain threshold (no drop in load; a stuck run below μ_off is counted against model F's Corollary 3) and the healthy-side condition (K4).

## K4. Status (2026-11-17): registered, not run
- **Registered test:** decisions.md 2026-11-17 [REG] K4 (the warm state ends at a surge): ten no-burst runs at λ = 2.2, 2700 s, on seeds picked from the schedule alone, five with a 120 s planned peak ≥ 3.4 in [1200, 1800] s (surge arm) and five with none ≥ 3.1 (calm arm); labels and outcome in `analysis/k4.py` (`docs/definitions.md`, K4).
- **Refutes, as registered (author, 2026-11-17):** at least two calm runs with an onset, or at least two surge runs warm. One calm onset gives `mixed`, not `refutes`, unlike the test line above ("tips without a surge"): the record allows a stray calm tip at up to 0.099 per run, and with one tip refuting a true claim would be refuted with p 0.354 against 0.184 with two (`k4.design_power`; decisions.md 2026-11-17 [DECISION] K4's four open points). One calm onset still blocks `supports`.
- **What it does not separate:** the edge inside 3.1–3.4; a surge of requests from one of new sessions; a 120 s peak from the few-minute rise in planned load around it (2.66–2.92 req/s against 1.89–2.25 in the calm arm at the matched spans, `k4.schedule_load`).

## K3. Status (2026-11-17, after K3b's [RESULT]): the served mix inside model F's band
- **K3b gave `inside` for each judged quantity** (RUN(K4's four tipped runs, k3b.table); decisions.md 2026-11-17 [RESULT] K3b (model F on K4's tipped runs)): on K4's four runs that tipped and stayed cold (seeds 133, 137, 175, 196; λ_W 2.07–2.43), over W = [onset + 300 s, 2700 s), the served prompt mix was inside model F's tilt in 4 of 4 runs (median deviation +0.0025 against ±0.015); r (median 1.019, ±0.05) and the completion rate (median +1.6 %, ±5 %) were inside in 4 of 4, as consistency checks. One-sided 95 % upper bound on the share outside, each: 0.527. `against_corollary3` empty.
- **Wording (author, 2026-11-17: "ok continue with recommended things"; the registered sentence for "served inside"):**
  > Registered before its runs were read, model F's tilt of the served mix held in the no-burst stuck state at λ = 2.2 on K4's four runs that tipped and stayed cold (median deviation +0.0025 against a band of ±0.015). r and the completion rate, which a saturated all-miss engine gives almost by itself, were inside too. It is one setup, one load, and four runs.
- **Not separated:** any other cause of a tilt toward first turns (for example vLLM preempting long requests in a full pool: 0.04–0.05 preemptions per s); the four runs are all from K4's surge arm. The session completion rate (reported only) was inside in 3 of 4, high in three, as the surge's own sessions predict.
- **Still without a judged test:** the drain threshold, the backlog slope and the healthy-side condition.

## K4. Status (2026-11-17, after K4's [RESULT]): `supports`
- **K4 gave `supports`** (RUN(K4's ten runs, k4.table); decisions.md 2026-11-17 [RESULT] K4 (the warm state ends at a surge)): surge arm `tipped` 4 of 5, all on time (32–79 s after the crossing), the fifth (seed 287, the marginal seed with 5 s at or above 3.4) `warm`; calm arm `warm` 5 of 5, no onset. One-sided 95 % upper bounds: surge not tipped 0.657, calm tipped 0.451; Fisher's one-sided p 0.024.
- **Wording (author, 2026-11-17: "ok continue with recommended things"; the registered sentence for `supports`):**
  > At λ = 2.2, five schedules with a 120 s planned peak ≥ 3.4 (inside a few-minute rise to 2.66–2.92 req/s) tipped and stayed cold in 4 of 5, 4 within 212 s of the crossing. Five with no peak ≥ 3.1 stayed warm in 5 of 5.
  The paper's healthy-side paragraph says a schedule-identified surge episode ends the warm state at this load.
- **Still open:** the edge within 3.1–3.4 (the one warm surge schedule is the one nearest it), the λ dependence, the 120 s peak against the few-minute rise around it, and a surge of requests against one of new sessions. The sent 120 s peak does not order the outcome (3.075 warm against 2.892 tipped).
- The claim text above is unchanged; it is stated for the cell tested only.

## C2. Status (2026-11-17, after B1's [RESULT]): idle-time aging measured, still not met
- **B1 gave `supports_aging`** (RUN(D1b's three cap-40 runs and REG-19's five controls, b1.table); decisions.md 2026-11-17 [RESULT] B1): short-gap reuse ≥ 2× long-gap reuse in 3 of 3 cold runs (one-sided 95 % lower bound 0.368); the controls `flat` 5 of 5, a ceiling check.
- **Wording (author, 2026-11-17: "continue approved"; the registered sentence for `supports_aging`):**
  > In D1b's cold capped state (L1, λ = 2.0, cap 40, seeds 15–17), a later turn's reuse of its prefix fell with the drawn think gap: 0.25–0.28 below 1 s against 0.000–0.003 at 5–25 s, in 3 of 3 runs; the clock's length is not measured by B1.
- **What it adds to C2:** the eviction link (a freed prefix lost before its session's next turn) depends on how long the prefix sat idle, per turn, in the cold state. **Not separated:** the clock's length, why the clock is short (pool pressure from the batch or from the population), preemption. The 17 s mean idle-before-evict is mostly the time a block is held by its running request; the free-queue time left is ≈ 0.6–0.7 s (open-questions A25, DERIVED, approximate; decisions.md 2026-11-17 [CORR] B1's result misread …). C2 stays **not met**: the order of the loop's signals is not judged, and the loop is not separated from pool pressure set by the number of sessions or a smaller effective shock.
- The claim text above is unchanged; it is stated for the cell tested only.

## K5. Status (2026-11-15): leaves the claims
- **REG-23 gave `no_usable_cache_lead`.** In six collapses of the warm state with no trigger, H and U led the 1.0 s TTFT breach by more than 30 s in 3 of 6 and 4 of 6, and led latency's first move by more than 30 s in 0 of 6. Each signal also raised false alarms in quiet runs (RUN(REG-23's 18 valid candidates, k5.k5_table); decisions.md 2026-11-14 [RESULT] REG-23).
- **As registered, K5 leaves the claims.** The registered limitations sentence: "it moved more than 30 s before latency in 3 of 6 collapses (H) and 4 of 6 (U), not in all", where latency is the 1.0 s breach.
- It was a retrospective reading, on runs that went cold with no trigger, not on burst runs.

## C1. Status (2026-11-20, after REG-27's [RESULT])
- **Conditions 5 and 6 at L1, λ = 2.0 hold in 4 of 4 further seeds** (61, 68, 72, 79; one-sided 95 % bounds 0.473 to 1.000; RUN(REG-27's eight runs, reg27.table); decisions.md 2026-11-20 [RESULT] REG-27), beside seed 43 (REG-24), reported apart.
- **Wording (author, 2026-11-21: "I approve line changes"; REG-27's registered sentence for `replicates`):**
  > At λ = 2.0, in 4 of 4 further seeds (61, 68, 72, 79; one-sided 95 % Clopper–Pearson lower bound 0.47), each beside one no-burst control on the same arrivals that stayed warm: after the burst the server stayed stuck for 4500 s, its line never emptying and its completions below the planned load; its total line drained 2400–2950 s into a 9000 s drop to λ = 1; the last 600 s of the return to λ = 2.0 were recovered. Seed 43 (REG-24, the seed that selected the design) is reported apart. This meets C1's conditions 5 and 6 at this cell in five seeds. It does not test C2's chain.
- **Unchanged:** C1 is still not supported by its own test: condition 4's order clause is not met (the replication's order readout, not judged, `does_not_follow` in 4 of 4). Not separated: the owed backlog (1532–1970 turns at the watch's end) against the pool.
- The claim text above is unchanged; it is stated for the cell tested only.

## K3. Status (2026-11-20, after REG-28's [RESULT])
- **REG-28 Q1 `law_holds`** (RUN(REG-27's four burst runs, reg28.table); decisions.md 2026-11-20 [RESULT] REG-28): model F's FIFO all-miss replay at each run's fitted capacity reproduced the session share, mean deviation +0.002 (±0.03), nearer than the quasi-static reading (−0.112) in 4 of 4. **Q2 `churn_bounded`:** preemption re-prefill ≤ 3.9 % of uncached prompt work in each run (≤ 0.05).
- **Wording (author, 2026-11-21: "I approve line changes"; the [RESULT]'s registered wording):**
  > A FIFO all-miss replay of model F, at the capacity fitted to each run's throughput, reproduced the share of completed work that finishes sessions, blind, in 4 of 4 stuck runs on a fixed-turn chat workload. It does not test the all-miss assumption itself.
- **Not separated:** FIFO with a growing wait gives the tilt by itself; preemption that evicts other sessions' prefixes is not bounded by Q2.
- The session share counts as a second blind test of the served mix, not a second judged quantity: K3 has one of its four quantities judged (author, 2026-11-21: "Do what is recommended"; decisions.md 2026-11-21 [DECISION] (author) B6 and P32 …).

## K4. Status (2026-11-20, after K4e's [RESULT]): the band `band_warm`
- **K4e gave `band_warm`** (RUN(K4e's twelve runs, k4e.table); decisions.md 2026-11-20 [RESULT] K4e (K4's edge)): 0 of 9 counted band schedules tipped inside the window (one-sided upper 0.283); seed 914 (exact 3.392) tipped 223 s after its crossing (`late`) and stayed cold; positive control 137 `tip`, calm control 103 `warm`.
- **Wording (author, 2026-11-21: "I approve line changes"; the registered sentence, filled, with seed 914 added):**
  > At λ = 2.2, ten schedules with 120 s planned peaks of 3.125–3.400 (exact) tipped within 212 s of their crossing and stayed cold in 0 of 9 counted (upper bound 0.28), none recovered; the tenth (exact peak 3.392) tipped 223 s after its crossing and stayed cold.
- **Still open:** an edge inside 3.1–3.4 (not locatable by this design), other λ, dose against height, the few-minute rise, requests against new sessions. K4's `supports` (2026-11-17) stands.
- The claim text above is unchanged; it is stated for the cell tested only.

## C1 and C2. Note (2026-11-21, after REG-30's [RESULT])
- **REG-30 gave `drain_bound`** (RUN(the five long-watch burst runs and their controls, reg30.table); decisions.md 2026-11-21 [RESULT] REG-30): in 5 of 5 runs later-turn reuse came back within 40 s of the total line's drain in the drop to λ = 1, with 6.7–16.2 sessions in service (the controls: 4.0–8.2). The population passed the pool's N* (24.4–31.5) within minutes of the drain, so these runs do not separate the population link from hysteresis.
- **Unchanged:** every claim's text and strength. The "owed backlog" alternative under C1 conditions 5–6 and C2 stays not separated.

## All claims. Paper wording after the panel of 2026-11-22 (author-approved; no claim status changes)
The author approved applying every panel change except new runs (decisions.md 2026-11-22 [DECISION] (author) No paper 2 …). The claim status lines above stand. What changes is how the paper states them (paper repo branch `panel-20261122`, `reviews/panel_20261122/applied.md`). Each item below names where it is stated and its evidence.
- **P0 and C1 (the reading).** The paper reads the cold regime as overload on the all-miss branch, whose line grows above the cold floor, sitting above a warm state that can last. It is "not a second steady state" (FILE(docs/definitions.md:170)). Condition 6 shows the engine is not damaged; it does not show a second equilibrium. The owed backlog is read as the carrier of the state, not a rival to the loop, and this is labelled a reading, not a result. Nothing here tests it: T1, a cap engaged inside an established state, was not run (author: no new runs).
- **P0 (headline order).** The abstract and §1 lead with the per-span test registered before any of the replication's runs, 3 of 5 (lower bound 0.189). The whole-window test, 5 of 5 (0.549), is stated second, as written after the earlier matrix's data (RUN(REG-19's ten runs, gate.reg19_table)).
- **K1 (turn release).** It is stated through Little's law. Open-loop release keeps far fewer sessions in service: (λ / 8) × 36 s = 9.0 at λ = 2.0 and 11.3 at 2.5 (DERIVED), against a measured 9–11 and 11–13. The small population is what the treatment does, not a confound. K1 is therefore a statement about what a timestamp-replay benchmark can observe. At the burst's end the open arm held 25–38 sessions in service and the closed arm 44–66 (RUN(REG-19's and REG-20 arm A's burst runs, reg31.in_service_table); decisions.md 2026-11-22 [RESULT] REG-31 (R31b–R31d)).
- **K3 (models).** It is stated as consistency: the cold state agrees with the all-miss assumption, run to run. It is not a claim that a published model predicts where the state begins. The memory-fit model M1's miss is reported: 0 of 3 on the earlier setups, and its 16-turn edge (1.73, 1.87] req/s lay below the measured (2.2, 2.4] (FILE(docs/model.md:87)).
- **K4 (exposure).** Schedule-only shares are printed beside it. A planned 120 s peak ≥ 3.4 req/s falls within a 4080 s run in 0.018 of fresh schedules at λ = 2.0, 0.11 at 2.2 and 0.48 at 2.4 (RUN(none: schedule only; reg31.calm_table); decisions.md 2026-11-22 [RESULT] REG-31 (R31a)). The surge test judged λ = 2.2 only.
- **P0 scope (calm schedules).** The warm twin is shown only on schedules without a planned 120 s peak ≥ 3.2 req/s. At λ = 2.0 such a peak falls within 4080 s in 0.081 of schedules and within 15,000 s in 0.30 (same source).
- **K5 (out of the claims).** The paper adds that the running batch gave no warning either. Its last unbroken stretch above 22 began more than 30 s before the 1.0 s breach in 0 of 10 collapses, and it passed 22 in 3 of 9 warm runs (RUN(REG-23's six, K4's four tipped runs and K4e's s914, reg31.lead_table)).
- **Run order (a threat, not a claim).** The burst runs' lower baseline percent fast, 10 of 10 pairs, is the burst's own reach into the baseline window. All 112 slow baseline requests were planned in the last 17 s before the burst and sent after it began; the controls had none (RUN(the ten pairs of paper_tables.PAIRS, reg31.slow_table)). Run order and card warmth do not account for the gap.
- **The 8192 test** stays recorded by its counts with no outcome label (author's ruling, kept).
