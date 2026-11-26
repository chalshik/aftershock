# Definitions

One entry per term: one text definition, its source, and the code that computes it. Created 2026-11-04 (decisions.md, same date).

- **Which text is given:** where files differ, the newer decisions.md entry rules (`PROTOCOL.md:3`). Older or other wordings are listed under "Other text", not deleted.
- **⚠ more than one computation:** the term is computed in more than one way, and reported numbers may come from either. **No single one has been chosen yet.**
- **Rule J (docs/rules.md):** code must match the definition given here. Where it does not, the term is marked ⚠ and the choice between computations is the author's; nothing here picks one.
- Line numbers are as of 2026-11-04. Code paths: `load` = `analysis/load.py`, `derived` = `analysis/derived.py`.

## Outcome and capacity

### percent fast ⚠ more than one computation
- **Text:** "a request is fast if its first token arrives within 1.0 s of its planned arrival and it finishes ok. Unfinished requests are misses." `docs/decisions.md:122`. It is counted by arrival time: "Each second, take the requests that **arrived** in that second and count how many met the limit." `docs/experiment-plan.md:115`.
- **Code:**
  - `load.percent_fast` (`load.py:76`): a share for each 1 s arrival bin.
  - `load._level` `"pf"` (`load.py:150`): one share pooled over every request arriving in [a, b). This one feeds `verdict` and `ramp_table`. It counts requests, not bins, and it includes burst requests.
  - **Closed-loop turns** (REG-11, decisions.md 2026-11-09 [REG]; `workload.sessions.closed_loop`): latency (`ttft`, `e2e` in `load.load_run`) is measured from the release time `t_released` instead of `t_planned`. Both computations above still bin by `t_planned`, so a turn released late, or never, counts in the window of its planned time (REG-11 review, T3). Open-loop runs are unchanged (`t_released` is None).
  - `gate.pf_by_sent` (added 2026-11-10 for REG-16, decisions.md [REG] REG-16): one share pooled over the requests **sent** in [a, b), by `t_sent`. Under closed loop a turn is served near its send time, up to 15 min after its planned time, so this computation lines pf up with the scrape-time hit rate. Requests never sent are left out (6.1 % of REG-15's requests; REG-16 review round 2, T3), which leans towards "fast" in a collapse; `gate.window_mix` reports them per window. Used only by REG-16's `cold_windows`, against a baseline pf from `_level`.

### goodput ⚠ two meanings
- **Text:** "Share of requests that met the speed limit (SLO); the goodput measure from Stage 1". `docs/experiment-plan.md:205`. In this repo it is a synonym of percent fast.
- **Other meaning:** completions per second, used outside this repo (the author's paper notes; not checkable here, BELIEF). The C1 metric, percent fast or completions/s, is the author's decision (`docs/claims.md`).
- **Code:** none of its own; see percent fast and throughput. **Decided 2026-11-12 (author):** the paper states C1 in hit rate and percent fast, not completions/s (decisions.md 2026-11-12 [DECISION]). Still write "percent fast" or "completions/s", never "goodput".

### throughput (completion rate c)
- **Text:** "completions per second inside the measured window, the plan's definition". `docs/decisions.md:97`. Written c(s) in `docs/model.md` M2.
- **Code:** `load.summarize` `"throughput"` (`load.py:98`): ok requests whose `t_end` falls in [warm-up, end of arrivals), divided by that span.
- **Other text:** the step 4 sweep used an arrival-based count instead (`docs/decisions.md:88`, superseded).

### μ_warm ⚠ more than one computation (E0's warm capacities use the tip rule, below)
- **Text:** "μ is the highest swept λ whose run is valid, with throughput ≥ 0.95 λ and waiting-queue slope ≤ 0.01 req/s², both over the measured window", with prefix caching on. `docs/decisions.md:61`.
- **Code:** `load.capacity` (`load.py:243`), on `load.summarize` rows.
- **Reference for future sweeps (item 42, decisions.md 2026-11-04; code 2026-11-05):** a sweep whose config has a `capacity:` block with `against: planned_rate` is judged against `load.summarize` `planned_rate`, the requests planned in [warm-up, end of arrivals) per second, instead of λ. Read by `tables.capacity_rule` and `tables.sweep_capacity`. Every sweep up to 2026-11-04 has no such block and keeps the 2026-10-30 rule. With `monotone: true`, μ is the highest rate that passes along with every lower swept rate (`load.capacity(monotone=True)`). First used by `configs/l1_base.yaml` (REG-7). **[2026-11-05]** A swept rate with no valid run, after the reruns REG-7 allows, ends the chain like a failing one. The swept rates are the union of `sweep.rates` saved in the parent's runs (`load.capacity(swept_rates=...)` via `tables.sweep_capacity`; decisions.md 2026-11-05 [CORR] and [DECISION]).
- **Other text:** "Highest load the server keeps up with when the cache works" (`docs/experiment-plan.md:201`).
- **Second computation (E0 only, 2026-11-15):** E0's warm₈ and warm₁₆ use the tip rule ("warm capacity by the tip
  rule", below); the rule above is still computed there and reported as `warm_ratio_rule`, deciding nothing. On
  REG-13 both give 2.4 (RUN(the five REG-13 runs, k2.tip_warm)).

### warm capacity by the tip rule (E0)
- **Text:** the highest swept rate whose cache-on run, and every lower swept rate's run, is valid, has a queue slope
  ≤ its config's registered `max_q_slope`, and has no KV onset anywhere in the run, drain included. The keep-up ratio
  is not used, because closed-loop release lag lowers it on healthy runs. A swept rate with no valid run ends the
  chain. Author, 2026-11-15 (decisions.md [DECISION] (author) E0 amendment, the tip rule).
- **Code:** `k2.tip_pass`, `k2.tip_capacity`, `k2.tip_rows`, `k2.tip_warm`; used by `k2.observed`. `k2.tip_rows` refuses
  a run without a usable KV gauge (`k2.kv_readout`).

### released rate
- **Text:** requests released in [warm-up, end of arrivals) per second of that span: the load a closed loop actually
  offered in the measured window. A request with no release time (a first turn, or open-loop turns) counts at its
  planned time. Compare `load.summarize` `planned_rate`, which counts by planned time.
- **Code:** `k2.released_rate`. Used by `k2.decide`'s released clause (E0's "widens": warm₁₆'s run must release more
  than the 8-turn run at the first rate above warm₈).

### μ_off
- **Text:** the same rule as μ_warm (`docs/decisions.md:61`), with prefix caching off.
- **Code:** `load.capacity` (`load.py:243`).
- **Other text:** "Highest load the server keeps up with when nothing comes from the cache" (`docs/experiment-plan.md:202`).

### danger zone
- **Text:** "Loads between μ\_off and μ\_warm: fine while the cache works, too much once it is lost". `docs/experiment-plan.md:203`.
- **Code:** none.
- **Other text:** the test loads are the subset "whose no-trigger control is `recovered`" (`PROTOCOL.md:61`).

## Timing windows

### τ (recovery time)
- **Text:** "time from the step until a sliding 180 s span, moved 60 s at a time, has hit rate within ±0.03 and TTFT p50 within ±0.02 s of the settled level. **τ is the larger of the up-step and down-step values.**" `docs/decisions.md:168`.
- **Code:** `load.settling_time` (`load.py:124`); settings under `tau:` in `configs/tau.yaml`.
- **Other text:** "how long the healthy server takes to settle after a normal load change that wipes nothing" (`docs/experiment-plan.md:223`).

### K
- **Text:** "watch window = max(1200 s, 5 τ), so K = 5". `docs/decisions.md:172`.
- **Code:** none. The result, `watch_s: 1500`, is written into `configs/stage2_base.yaml:13`.

### watch window
- **Text:** "Watch window = max(1200 s, 5 × 300 s) = **1500 s**" (`docs/decisions.md:192`). It is judged on "the last 600 s (`end`) against the 600 s before them (`previous`) and the baseline" (`docs/decisions.md:272`).
- **Code:** `load.verdict` (`load.py:197`): `t_end = trigger end + watch_s`; `end_s` is set in `configs/stage2_cap_base.yaml:8`.
- **Other text:** an `end_s` of 300 s in `configs/stage2_base.yaml:14`, superseded before any verdict. It is still inherited by `wipe_test_*.yaml` and `stage2_control.yaml`, none of which was judged.

### baseline window
- **Text:** "the baseline is the 600 s before the trigger *starts*". `docs/decisions.md:391`.
- **Code:** `load.verdict` (`load.py:197`).
- **Other text:** "the last 600 s before the wipe" (`docs/decisions.md:210`); the 600 s before the trigger's end (the error corrected at `:389`).

### trigger end ⚠ more than one computation (two copies of one rule)
- **Text:** no text definition. Closest: "The baseline ends where the trigger starts; the watch starts where it ends (a successful wipe, a burst, or control_t_s for a control...)" (`analysis/load.py:198-200`).
- **Code:**
  - `load.verdict` (`load.py:197`): a wipe ends at the `t_end` of the successful call in `trigger.jsonl`; a burst at `at_s + duration_s` (its planned arrivals end, not its backlog); a control at `control_t_s`.
  - `derived.trigger_times` (`derived.py:20`) is the same rule, copied so that `verdict` stays unchanged.

## Server metrics

### hit rate h ⚠ more than one computation
- **Text:** "Δ`vllm:prefix_cache_hits` ÷ Δ`vllm:prefix_cache_queries`". `docs/experiment-plan.md:125`.
- **Units (source):** tokens. Readmissions of preempted requests are excluded (`docs/open-questions.md`, S8).
- **Code:**
  - `load.rates` `hit_rate` (`load.py:66`): one ratio per scrape interval.
  - `load.summarize` `"hit_rate"` (`load.py:101`): the mean of the per-scrape ratios over the measured window.
  - `load._level` `"hit"` (`load.py:150`), `derived.hit_between` (`derived.py:13`) and `derived.thirds(how="counter")` (`derived.py:37`): the ratio of counter differences over the span. This feeds `verdict`. Since 2026-11-05, `hit_between` is also NaN on a counter decrease or no queries, like `_level` (decisions.md 2026-11-05 [CORR]).
  - `load.whole_run_hit_rate` (`load.py:108`): first scrape to last.
- **Undefined windows (REG-4, 2026-11-04):** `_level` gives NaN hit when the query counter does not rise, a cache counter goes down (restart), or the scrapes don't cover the window (`checks.max_scrape_gap_s`). Before REG-4 a window with no queries gave 0.0.
- **Different numbers for the same run:** e.g. the long run's last third, 0.6004 (mean of ratios) vs 0.6255 (counters), `docs/decisions.md:116`.

### KV usage
- **Text:** "usage is 1 − free blocks / total ... So usage counts blocks held by running requests, and saved prefixes with no running user count as free." `docs/decisions.md:244`. Source: `vllm/v1/core/block_pool.py` `get_usage`.
- **Code:** the gauge `vllm:kv_cache_usage_perc`, through `load.rates` (`load.py:70-72`); `load._level` `"kv"` (mean, `load.py:150`); `derived.kv_footprint` (`derived.py:76`).

### running batch N
- **Text:** "`vllm:num_requests_running` | How many requests share the GPU". `docs/experiment-plan.md:124`.
- **Code:** gauge; `load._level` `"n"` (mean, `load.py:150`).

### waiting Q
- **Text:** "`vllm:num_requests_waiting` | Whether requests pile up". `docs/experiment-plan.md:123`.
- **Queue slope, units:** a least-squares slope of a request count against time, so its unit is **req/s**. The docs write "req/s²" (`docs/decisions.md:21`, `:61` and later); the numbers and thresholds are unchanged, only the unit was misnamed (2026-11-04 audit).
- **Code:** gauge; `load._level` `"q"` (mean) and `"q_slope"` (least-squares slope, `load.py:150`); `load.summarize` `"q_mean"` and `"q_slope_per_s"` (`load.py:102-103`).

### preemption
- **Text:** "report preemptions per minute in the watch window, excluding the wipe's own, against the no-trigger control and the caching-off run at the same λ". `docs/decisions.md:247`. Counter: `vllm:num_preemptions` (`docs/experiment-plan.md:128`).
- **Code:** only the generic `load.rates` `num_preemptions_rate` (`load.py:63`). **No code for the registered comparison.**

### R (block idle time before eviction)
- **Text:** "`vllm:kv_block_idle_before_evict_seconds` | How long a cached prefix survives (R)". `docs/experiment-plan.md:129`. Recorded as a histogram, sampled (`--kv-cache-metrics`, `configs/base.yaml:17`).
- **Code:** `c2.histogram_delta` (REG-22, decisions.md 2026-11-13) reads the sampled histogram's buckets from the raw `metrics.jsonl`: the gain over a window, the median's bucket with both edges, the shares at or below 5 s and 30 s; a window in which any bucket goes down is flagged `reset` and not read. `load.metrics_wide` still drops the `_bucket` series. Before REG-22: none.

### total line (owed backlog)
- **Text:** "requests planned before t and not completed by t: server waiting and running, plus closed-loop turns not yet released." REG-24 (decisions.md 2026-11-15 [REG] REG-24, citing `docs/drafts/reg24_confirm_draft4.md`, "For the approval commit").
- **Code:** `gate.owed_backlog` (`gate.py:1079`), and its vectorized form `confirm.owed_series` (`confirm.py:401`), which `test_owed_series_is_gate_owed_backlog` shows equal.

### emptying ⚠ more than one computation (two copies of one rule)
- **Text:** "0 waiting at a scrape and for the next 30 s, the 30 s inside the window; a new emptying counts only after the line refilled." REG-19's rule, restated by REG-24 (rule F round 2, DK-O2).
- **Code:**
  - `gate.reg19_emptyings` (`gate.py:1857`): relative to the burst's end, over [burst end, end of arrivals).
  - `confirm.emptyings(run, a, b)` (`confirm.py:381`): absolute, over [a, b). On REG-19's window the two are equal (`test_emptyings_are_reg19s_on_reg19s_window`). REG-24 uses [1080, 14580), which leaves out a zero whose hold would cross into the return.
- **Author:** the two copies are left to the author under rule J; nothing is chosen (decisions.md 2026-11-15, the author's approval of REG-24 draft 4, open point 9). Same precedent as "trigger end ⚠".

### drained (REG-24)
- **Text:** "the first 10 s grid time in [a, b) at which the total line is at or below its level at the trigger: the token model's event, in requests." REG-24 (draft 4, "For the approval commit").
- **Code:** `confirm.drained_at` (`confirm.py:410`).

## Run verdicts

All five come from one rule, `docs/decisions.md:272-277`, restated at `PROTOCOL.md:24-28`. Code: `load.verdict` (`load.py:197-241`), with thresholds under `verdict:` in `configs/stage2_base.yaml:10-22` and `configs/stage2_cap_base.yaml:7-13`.

### bad_baseline
- **Text:** "baseline percent fast < 0.98." `docs/decisions.md:273`.
- **Code:** `load.verdict`, `min_baseline_pf`.

### recovered
- **Text:** "percent fast ≥ baseline − 0.02, hit rate ≥ baseline − 0.06, mean waiting ≤ baseline + 1." `docs/decisions.md:274`.
- **Code:** `load.verdict`.
- **Other text:** "Percent fast returns to its pre-trigger level" (`docs/experiment-plan.md:206`); `:259` (superseded).

### slow_drain
- **Text:** "any one sign of recovery: hit rate up ≥ 0.06, or percent fast up ≥ 0.05, or the line shrinking fast enough to empty within 3600 s." `docs/decisions.md:275`.
- **Code:** `load.verdict`, `climbing or draining`.
- **Other text:** `docs/experiment-plan.md:260` asks for all three signs together (superseded).

### stuck
- **Text:** "percent fast ≤ baseline − 0.10 and hit rate ≤ baseline − 0.06, with no sign of recovery." `docs/decisions.md:276`.
- **Code:** `load.verdict`.
- **Other text:** `docs/experiment-plan.md:261` requires a waiting line that is "Flat or growing". The code also accepts a line that shrinks too slowly to empty within 3600 s.

### unclear
- **Text:** "anything else." `docs/decisions.md:277`. "never rounded to another verdict" (`PROTOCOL.md:28`).
- **Also `unclear` (REG-4, 2026-11-04):** a branch that needs a value the window cannot give (NaN). `bad_baseline` and `recovered` are still decided when their own inputs exist; `slow_drain` and `stuck` are not reached if a sign of recovery can't be computed.
- **Code:** `load.verdict`.

### metastable failure
- **Wording (2026-11-12, decisions.md [DECISION], wording and K2):** the stuck regime is called *metastable*, not *bistable*. "Two states at one load" (condition 6) means a healthy steady state and a degraded regime that sustains itself while its waiting line grows; the degraded regime is not a second steady state (the line rose 76 → 120 in REG-17, decisions.md 2026-11-10 [RESULT]). "Bistable" is used only for cited systems with two steady states.
- **Text:** six conditions at one load: trigger over at the same load; control `recovered`; `stuck` in most seeds; a memory-limited loop; persistence over a 4500 s watch; two states at one load. `docs/decisions.md:281-287`, restated at `PROTOCOL.md:30-36`.
- **Code:** none. Only the per-run verdict is coded. Condition 4's levels have code (`c2.levels`, REG-22); its order clause has none. No definition here says condition 4 is met.
- **Other text:** three conditions (`docs/experiment-plan.md:265`, superseded); "Stuck ... The metastable failure" (`:207`).

## Derived quantities

Added 2026-11-04 with their code. Each is the reading that reproduces the published values (`docs/decisions.md`, 2026-11-04 "Derived figures moved into analysis/").
The original code, recovered verbatim from session transcripts, is in `analysis/recovered/` (INDEX.md there). Where its reading differs from the function below, the entry says so; the function stays the definition (decisions.md 2026-11-04, "[CORR] Derived-figure differences explained by the recovered originals").

### W (working set)
- **Text:** "sum of the latest prompt over active sessions from the schedule". `docs/decisions.md:248`.
- **Code:** `derived.working_set` (`derived.py:63`). A session counts as active from its first planned turn to its last.
- **Original reading:** the same rule sampled every 5 s from t = 600 s (every 10 s for the 5 s-gap comparison), against a 1 s grid here. This explains the ≤ 0.4 % differences in published W values.

### session median (prompt length)
- **Text:** the p50 of `prompt_len` over every scheduled **session** request of a run's whole schedule (`schedule.parquet`): warm-up included, sessions cut off at the end of arrivals included, burst requests excluded. Computed as pandas `median` (linear interpolation).
- **Value:** 1408 tokens (turn 3) in every schedule checked. It is a knife edge: the share of session requests at or below 1408 is 0.502–0.510 (decisions.md 2026-11-04 (7)), because truncated sessions add early turns. A full 8-turn session alone has median 1536, and over the measured window (t ≥ 240 s) the median is 1664 in 3 of 5 schedules.
- **Used by:** the burst size, "1408-token prompts (the session median)" (decisions.md 2026-11-02) and the `burst_*.yaml` comments ("median session prompt").
- **Code:** none; a one-line count on `schedule.parquet`.

### F (running-batch footprint)
- **Text:** no definition in the original entry (`docs/decisions.md:248-250`).
- **Code:** `derived.kv_footprint` (`derived.py:76`): mean KV usage × pool size.

### cached-share split
- **Text:** "Mean cached share of the prompt, all later turns vs only those starting after the previous turn had finished". `docs/decisions.md:293`.
- **Code:** `derived.cached_share_split` (`derived.py:93`). "Starting" here means **its first token** at or after the previous turn's end.

### overlap fraction
- **Text:** "Sent before the previous turn's first token". `docs/decisions.md:292`.
- **Code:** `derived.overlap_fraction` (`derived.py:103`).

### capacity loss
- **Text:** 1 − c(state) / c_max, the share of the best completion rate a state loses. `docs/model.md` M2.
- **Code:** none yet; DERIVED from `load.summarize` throughputs.

### post-trigger margin
- **Text:** completions − λ over a span after a trigger's end. Defined with its windows in the registration REG-1 (`docs/model.md` M2), **not yet computed**.
- **Code:** none yet; to be written only after REG-1 is approved.

## Feasibility gate for setup L1 (REG-7, decisions.md 2026-11-05)

### μ_sat,off (cold floor)
- **Text:** the mean `load.summarize` throughput of a cache-off sweep's valid, finite runs at λ ≥ λ_sat, λ_sat being the lowest swept λ whose measured-window mean `kv_cache_usage_perc` is ≥ 0.85. Every such run must be saturated (KV ≥ 0.85 and throughput < 0.95 × planned_rate). Statuses: `ok`; `unsaturated`, `not_monotone`, `too_few` (< 2 runs), all meaning "unmeasured". Threshold 0.85: author's decisions on items 38–42 (decisions.md 2026-11-04).
- **Code:** `gate.cold_floor`, inputs from `gate.sweep_rows`.
- **Other text:** the REG-5 draft (`docs/drafts/addendum_34-43.md`), never computed for the default pool (decisions.md 2026-11-04 [DECISION] REG-5).

### λ_test
- **Text:** the largest multiple of 0.5 that is ≤ 0.8 μ_warm and ≥ 1.3 μ_sat,off, stepped down by 0.5 until the cache-on sweep run at the smallest swept rate ≥ λ_test had pooled percent fast ≥ 0.98 at TTFT ≤ 1.0 s, staying ≥ 1.3 μ_sat,off; none if no value fits.
- **Code:** `gate.zone_check`, `gate.slo_feasible`, `gate.choose_lambda`; driver `gate.reg7_zone`.

### c_b (completion rate in the cold backlog) ⚠ more than one computation
- **Text:** completions per second while requests wait and the sessions' cache is cold, against the planned session rate λp of the same span.
- **Code:**
  - `gate.burst_backlog` (via `margin.post_trigger_margin`): per fixed span after a burst, with its 99 % Poisson upper bound, against both λ and the span's planned session rate; decides REG-7's gate together with the matched control span (`gate.control_backlog`).
  - `floor.sustained_capacity`: over backlogged stretches (waiting ≥ 1 for ≥ 30 s); REG-6's measure, and REG-7's secondary readout only.

### gate verdict
- **Text:** `pass`, `drains`, `control_not_healthy` or `inconclusive` (an instrument gap, or a `stuck` burst run that no span confirms), from a control and a burst run with the same seed and duration, the control judged at the burst's end (REG-7).
- **Code:** `gate.gate_verdict`.

### overlap share
- **Text:** of the session cache hits in a span (by first token), the share served while the previous turn was still running: Σ cached tokens of later turns whose first token came before the previous turn's end ÷ Σ cached tokens of all later turns.
- **Code:** `gate.overlap_share`. Related, not the same: `derived.overlap_fraction` (share of turns *sent* before the previous first token) and `derived.cached_share_split` (mean shares over the whole window).

### 120 s surge peak
- **Text:** the highest number of planned session requests (burst requests left out, by `t_planned`) in a trailing 120 s window, per second, on a 5 s grid from 240 s. For a no-burst run that tipped: the peak in the last 180 s of its warm time (`peak_tip`), and before that (`peak_before`). A readout found in the data (decisions.md 2026-11-10 [CORR]), not a threshold, except where a registration fixes one (REG-17's `surge_tip` 3.2).
- **Code:** `gate.rolling_rate`, `gate.surge_run`, `gate.surge_table`; schedule only: `gate.schedule_peak_series`, `gate.surge_hazard`, `gate.pair_schedule`.
- **Other text:** `gate.surge_max` (REG-13) takes the highest planned rate over consecutive, non-overlapping 120 s windows, and REG-16's `peak_120s_before_first_cold` over 120 s windows stepped by 60 s; both can read lower than this on the same run (REG-16: 2.78 against 3.45).

### KV onset
- **Text:** the first scrape time whose trailing 60 s mean KV usage is ≥ 0.9 (at least 30 scrapes in the window); NaN if never. Marks a tip into the cold state; warm time ends 60 s before it.
- **Code:** `gate.kv_onset`.

## Terms of the revised Stage 2 (added 2026-11-12, decisions.md [DECISION] of that date)

None has code yet; each gets its code when the run or analysis that uses it is registered. **[CORR] 2026-11-14:** that sentence is stale; most terms below now name their code in their own entry.

### cold floor (all-miss capacity)
- **Text:** the same quantity as μ_sat,off above: the completion rate of the saturated server with caching off. The preprints call it the all-miss capacity. The value on record for L1, 1.681 req/s, was measured with open-loop turns; a closed-loop value is not yet measured.
- **Code:** `gate.cold_floor`.

### margin
- **Text:** the load divided by the cold floor. L1 at λ = 2.0: 2.0 / 1.681 = 1.19 (DERIVED).
- **Code:** none.

### open-loop turns / closed-loop turns
- **Text:** open-loop: each turn is sent at its planned time, the gaps counted from the previous turn's planned time. Closed-loop: a later turn is released at the previous turn's end plus the planned gap (`workload.sessions.closed_loop: true`; FILE(configs/burst_l1pl350cl_r2.4_s1.yaml:8)). Latency is measured from the planned time in the first case and from the release time in the second (percent fast, above), so latency-based readouts are not comparable between the two.
- **Code:** `harness/workload.py`, `harness/loadgen.py`.

### cache gain
- **Text:** the counter hit rate (`load._level` `"hit"`) of a setup's no-trigger control in its warm state.
- **Code:** `load._level`.

### shared-only workload, matched cache gain
- **Text:** shared-only: every reusable token lies in one of a fixed set of prefixes shared across sessions, the set's total size does not change with load or backlog, and no request reuses tokens of its own session's earlier turns. Matched cache gain: two setups' cache gains agree within a tolerance registered beforehand. (`docs/claims.md`, K2.)
- **Code:** none. **Not used by any planned run since 2026-11-12:** the shared-only arm was dropped from K2 (decisions.md 2026-11-12 [DECISION], wording and K2); the definition is kept for the record.

### session cap, sessions in service
- **Text:** a session is in service from its first turn sent to its last reply received. The session cap is a limit on the number of sessions in service, held at the load generator; a session beyond it waits to start. Its value per seed is the highest number of sessions in service at once in the same seed's no-trigger control (decisions.md 2026-11-12 [DECISION], design choices).
- **Code (added 2026-11-12; author approved the load-generator change, decisions.md 2026-11-12 [DECISION] session cap):** the cap: `workload.sessions.cap` (closed-loop turns only; refused at startup otherwise), `harness/loadgen.py` `SessionCap` (first come first served; the slot is held from admission to the chain's end) and `session_chain`; turn 0 records `t_arrived` (reached the cap) and `t_admitted` (got a slot), and its `t_released` is the admission time, so latency is measured from admission and the wait is reported apart. Burst requests pass the cap. The value: `cap.max_sessions_in_service` (exact maximum over the run, burst requests excluded); the wait: `cap.cap_wait` (`analysis/cap.py`).

### cold-backlogged (K1 readout)
- **Text:** a burst run is cold-backlogged when, in its end window [burst end + watch − end_s, burst end + watch), against its own baseline [trigger start − baseline_s, trigger start): reuse ≤ baseline − `stuck_hit_drop`; mean waiting > baseline + `recovered_q_excess` (`load._level`); mean KV ≥ 0.9 (`gate.SURGE["kv_onset"]`, there on a 60 s rolling mean, here on the window's mean). *Recovered* (K1 sense): reuse ≥ baseline − `recovered_hit_drop` and waiting ≤ baseline + `recovered_q_excess`. Undefined (None) when any input is NaN. No latency and no request mix enter, so it means the same with open- and closed-loop turns. A control is read over its burst partner's windows. Added 2026-11-12 with REG-20 (draft; decisions.md 2026-11-12 [REG]).
- **Code:** `k1.cold_backlogged`, `k1.windows` (`analysis/k1.py`).
- **Related, not the same:** the `stuck` verdict (`load.verdict`) also needs percent fast to fall and no sign of recovery, and uses the server hit rate; cold-backlogged needs neither and uses reuse.

### reuse (prefix reuse of later turns)
- **Text:** Σ cached tokens (capped at the previous prompt) ÷ Σ previous-turn prompt tokens, over ok later session turns (turn > 0, previous turn present) whose first token falls in the window. A turn's reusable prefix is its previous prompt. Unlike the hit rate it does not fall when the served mix shifts to first turns, which can never hit (REG-20 review, T1). Added 2026-11-12.
- **Code:** `k1.reuse`, `k1.later_turns`.

### sessions in service (count)
- **Text:** the mean, on a 10 s grid, of the sessions in service (see session cap: first turn sent to last reply; a session with an unfinished or unsent turn stays in service). A readout of REG-20 only.
- **Code:** `k1.sessions_in_service`.

### uncached-token work rate U (REG-21)
- **Text:** "uncached prompt tokens per second, Σ (prompt_tokens − cached_tokens) over ok requests, session and burst, whose first token falls in the window (the first token ends a request's prefill)" FILE(docs/decisions.md:9747), decisions.md 2026-11-13 [REG] REG-21, Statistic.
- **Code:** `k3w.work_rate`. It raises if an ok request has no first-token time or no prompt length. Its values over consecutive 300 s spans are reported beside it (`reported_spans`).
- **Not the same as:** the counter-based `recomputed_prefill_tok_s` of `load.rates` (from the server's counters, all requests, per scrape interval, FILE(analysis/load.py:75)), a different computation of an uncached prefill rate (REG-21's text, FILE(docs/decisions.md:9849)); and the session uncached prefill rate (REG-22, below), which leaves the burst's requests out.

### cache-off work rate U_off (REG-21)
- **Text:** "the mean of U over REG-8's five saturated cache-off runs, pinned by id" FILE(docs/decisions.md:9748); "each over its measured window. It raises unless exactly those five are there and valid." FILE(docs/decisions.md:9748)
- **Code:** `k3w.reference`; the five run ids are `k3w.REG21["ref_runs"]`.

### work-rate ratio r (REG-21)
- **Text:** "= U over W ÷ U_off, per run. W = [burst end + 600 s, burst end + watch)" FILE(docs/decisions.md:9749).
- **Code:** `k3w.row` (`"r"`), with W from `k3w.window`.

### busy share (REG-21)
- **Text:** "at least one request waiting in at least 0.99 of the scrapes of W" FILE(docs/decisions.md:9760), "a scrape within 10 s of each end and no gap above 10 s (`load._covered`). A missing gauge is listed as "no waiting gauge in W", not as idleness." FILE(docs/decisions.md:9760)
- **Code:** `k3w.busy_share` on `vllm:num_requests_waiting`; coverage by `load._covered`.
- **What the gauge counts:** "`vllm:num_requests_waiting` is the scheduler's waiting queues, the deferred requests included; a request in the running set is not counted" FILE(docs/decisions.md:9761).

### condition-4 levels ⚠ (two computations on record; left for the author, rule J) (REG-22)
- **Text:** "in the end window, mean KV ≥ 0.80, mean N ≥ 2 × its baseline mean, maximum N < 256" FILE(docs/decisions.md:4641), decisions.md 2026-11-13 [REG] REG-22, thresholds of decisions.md 2026-11-01. A gauge value needs coverage: "scrapes within 10 s of each end of its window and no gap above 10 s (`load._covered`); otherwise the value is missing." FILE(docs/decisions.md:4642)
- **Code:** `c2.levels`, every gauge value through `c2.gauge`.
- **Second computation:** REG-15's [RESULT] computed the same three levels over the watch's last 1500 s (decisions.md 2026-11-10 [RESULT] REG-15, "Condition 4"), not the end window.
- **Never written here or anywhere under REG-22:** that condition 4 is met, holds or is passed. The order clause of PROTOCOL §3 condition 4 has no code.

### the exit (REG-22)
- **Text:** "the moment the last burst request ended. Missing if a burst request never ended, ended with an error status, or the run has no burst rows." FILE(docs/decisions.md:4643)
- **Code:** `c2.last_burst_end`.

### reuse of later turns, with a minimum of turns ⚠ (two computations) (REG-22)
- **Text:** "reuse of later turns (cached tokens over the previous prompt, by first token; at least 20 later turns, else missing)" FILE(docs/decisions.md:4645).
- **Code:** `c2.reuse_in` (with its minimum of turns); `k1.reuse` (FILE(analysis/k1.py:69)), which REG-21 prints, computes the same quantity with no minimum ("reuse (prefix reuse of later turns)", above).

### loop links and sustain (REG-22)
- **Text:** the four links over one span (decisions.md 2026-11-13 [REG] REG-22, its lines FILE(docs/decisions.md:4645) to FILE(docs/decisions.md:4648)), and "a span is judged when it starts at or after the exit." FILE(docs/decisions.md:4650)
- **Code:** `c2.links` (one span), `c2.sustain` (every span); the label by `c2.condition4`.

### session uncached prefill rate ⚠ (three computations) (REG-22)
- **Text:** "session uncached prefill (prompt tokens not served from the cache, session requests only, per second, by first token)" FILE(docs/decisions.md:4646).
- **Code:** `c2.session_uncached_rate`.
- **Other computations:** `k3w.work_rate` (U of REG-21: the burst's requests too, by first token); `load.rates`'s `recomputed_prefill_tok_s` (from the server's counters, all requests, per scrape interval, FILE(analysis/load.py:75)).

### turn spacing (REG-22)
- **Text:** "the median time between the first tokens of a session's consecutive turns (`c2.turn_spacing`)" FILE(docs/decisions.md:4725).
- **Code:** `c2.turn_spacing`.

### onset reading (REG-22)
- **Text:** "The code takes each run's own baseline and end-window values, so each onset minus its own lag is the step time it implies." FILE(docs/decisions.md:4724)
- **Code:** `c2.onset_reading`.

### Order clause (condition 4), literal
- **Text:** condition 4's "the traces follow the plan's order", judged literally on stuck runs made after its registration: onsets of reuse, the session uncached rate, N and KV from the trigger's start (REG-22's rules, KV at `min_kv`), none crossing more than `tol_s` before one the plan puts earlier; evicted sooner where measurable; reuse low and N, KV high over the end window; preemptions ≤ 0.01 per request ended ok, less a wipe's own attempt. Labels `follows`, `does_not_follow`, `undefined`. Registered 2026-11-14 (decisions.md 2026-11-14 [REG], condition 4's order clause).
- **Code:** `order.judge`, `order.table`.
- **Not the same as:** `c2.onsets` (REG-22, reported, never judged, KV level 0.90, origin from `at_s` only) or `c2.sustain`.

### signal onset (REG-23)
- **Text:** "the first grid time after the reference window from which the signal stays beyond its level for 15 s (at every grid time of that stretch)" FILE(docs/decisions.md:20311), on the 5 s grid of 30 s trailing means (A: 120 s), with the levels of REG-23's table.
- **Code:** `k5.crossings`.
- **Not the same as:** `c2.onsets` (REG-22) or `order.judge`'s onsets (condition 4's order clause).

### lead over latency, latency's first move, the queue and arrivals (REG-23)
- **Text:** "over latency, onset(L) − onset(X) (`leads`) and onset(Lr) − onset(X) (`leads_rel`); over the queue, onset(Q) − onset(X); over arrivals, onset(A) − onset(X) (`lead_over_A`), per event. Where A has no onset in an event, X counts as earlier there" FILE(docs/decisions.md:20312). A lead counts only above 30 s (over A: above 120 s).
- **Code:** `k5.event`, `k5._lead_label`.

### false alarms per hour, their upper bound, and alarm share (REG-23)
- **Text:** onsets on the quiet runs, each re-referenced on its own [240, 540) s and scanned from 540 s to the end of arrivals, per hour of scanned time, with the one-sided 95 % Poisson upper bound; the alarm share is the share of scanned grid time inside a stretch beyond the level that lasts at least 15 s FILE(docs/decisions.md:20317).
- **Code:** `k5.quiet`, `k5.in_alarm`, `k5.judge` (`stats.poisson_rate_upper`).

### client-server TTFT gap and share shift (REG-26)
- **Text:** over a run's whole span (first scrape to last), the client's mean TTFT from `t_sent` to `t_first_token`, over requests whose first token falls in the span, minus the server's mean TTFT, Δ`vllm:time_to_first_token_seconds_sum` / Δ`_count` over the same span. The share shift is the server's share at or under 1.0 s, Δ`_bucket{le="1.0"}` / Δ`_count`, minus the client's share at or under 1.0 s. Both bound from above what lies outside the server's span; neither is client lag alone. The 60 s window gaps are a readout, biased where TTFT trends (decisions.md 2026-11-14 [REG] REG-26).
- **Code:** `lag.whole`, `lag.windows`, `lag.label`.

### switched session cap, overhang (D1r; draft, not registered)
- **Text:** the session cap (above) with a switch time `cap_from_s` (seconds from the run's start; absent or 0 = the cap from the start, the earlier behaviour exactly). A session reaching the cap before the switch is admitted at once and counted, so more than `cap` may be in service. From the switch on, a session is admitted only while fewer than `cap` are in service, first come first served. The *overhang* is the sessions in service above the cap at the switch. They are not stopped; in service falls to the cap only as they end, and no slot is handed over while in service is above the cap. Burst requests pass the cap. Added 2026-11-17 (author approved the change, decisions.md 2026-11-17 [DECISION] (author) D1r goes ahead as drafted).
- **Code:** `workload.sessions.cap_from_s`, read with `.get`; `harness/loadgen.py` `SessionCap(n, from_s)`, `acquire(now)`, `release`; `session_cap`. The invariant read back from the records: `d1r.slots_switched`, `d1r.harness_fault`.

### established (D1r precondition; draft, not registered)
- **Text:** a rescue run is established when it is cold-backlogged (above, K1 readout) on the 300 s before the switch, [1380, 1680) s, against its own baseline window. A run that is not is `not_established`: reported, and left out of n.
- **Code:** `d1r.established` (`k1.cold_backlogged` on `{"base": k1.windows(cfg)["base"], "end": d1r.D1R["established"]}`).

### reuse_unpaused ⚠ (a second computation of reuse) (T1; draft 3, not registered)
- **Text:** reuse of later turns (above, "reuse") with c2's minimum of turns, leaving out the turn right after a pause: Σ min(cached, previous prompt) ÷ Σ previous prompt over ok later session turns whose first token falls in the window and whose record has no `t_paused`; missing (NaN) with fewer than `c2.C2["min_turns"]` (20) such turns. On a run without pause fields it equals `k1.reuse` only on windows with at least that many later turns; `k1.reuse` has no minimum (review A7). Name approved by the author (decisions.md 2026-11-22 [DECISION] (author) T1's design choices before rule F).
- **Code:** `t1.reuse_unpaused` (`c2.reuse_in` over `k1.later_turns` filtered on `t_paused`).
- **Other computations:** `k1.reuse` (no minimum, no filter); `c2.reuse_in` with `k1.later_turns` (the minimum, no filter), which D1, D1r and REG-22 use. Left for the author under rule J.

### slots_held ⚠ (a second computation of "sessions in service") (T1; draft 3, not registered)
- **Text:** the number of sessions holding a slot of the session cap at time t (right-continuous): a session holds a slot from its admission (else its first send) to its first pause, from each resume to its next pause, and from its last resume (or its admission, if it never paused) to its last reply (to the end of the record if a turn never ended); a session still paused holds none. It is the load generator's `SessionCap.used`, read back from the records. On a run without pauses it equals sessions in service (`d1r.in_service`). In T1 it is an input to `ws_slot`, and so to `t_fit`, not a readout (draft 3 §2, OWNER-CHANGE approved decisions.md 2026-11-22). Name approved by the author (decisions.md 2026-11-22 [DECISION] (author) Workshop title approved; `slots_held` named …).
- **Code:** `t1.slots_held`, its spans `t1.held_spans`.
- **Other computations:** `d1r.in_service`, `k1.sessions_in_service` (10 s grid mean), `cap.max_sessions_in_service`, which count a paused session as in service. Left for the author under rule J.

### ws_slot (in-slot working set) (T1; draft 3, not registered)
- **Text:** at time t, Σ over the sessions holding a slot at t (by `slots_held`'s spans) of the `prompt_tokens` of each one's latest turn sent by t; a session holding a slot that has sent nothing yet counts its turn 0's `prompt_tokens_planned`. It is the context each such session needs cached for its next turn, since a turn's reusable prefix is its previous prompt (FILE(analysis/k1.py:71)). A sent turn whose `prompt_tokens` is null counts its `prompt_tokens_planned`. Draft 3 §2; draft 4, gap 9.
- **Code:** `t1.ws_slot`.

### t_cap, t_fit (T1; draft 6, not registered)
- **Text:** `t_cap` is the first 1 s grid time at or after the switch `from_s` with `slots_held` ≤ the cap (`t1.slots_paused`'s `t_at_cap`). `t_fit` is the first 1 s grid time at or after `t_cap` at which `ws_slot` ≤ C − cap × 256 tokens (the working-set clause alone; draft 6, author 2026-11-23, rule F round 2 B1: draft 5's waiting clause timed the re-warm it was meant to precede). At L1 22 × 3,968 = 87,296 ≤ 96,656 − 22 × 256 = 91,024 (DERIVED), so t_fit = t_cap whenever the cap holds. C is the run config's `checks.expected_kv_tokens` (96,656 at L1, which the `kv_capacity` check verifies). Missing (None) when no such time comes before `duration_s`, or when t_cap or C is missing. Draft 3 §2, draft 6.
- **Code:** `t1.t_cap`, `t1.t_fit`, `t1.pool_tokens`; the constants `T1["cap"]`, `T1["ws_margin_per_slot"]`, `T1["grid_s"]`.

### drain band: inside, early, late, unjudged; band_inside, band_miss, band_too_few (T1; draft 5, not registered)
- **Text:** per seed, D_est = `t_d_est` − from_s, with `t_d_est` the gate's fluid line (`t1.fluid_drain`) read from the committed probe's row of the seed. D = `t_d` − from_s on P (`t1.drained`). *Inside* if |D / D_est − 1| ≤ `band_tol` (computed as |D − D_est| ≤ `band_tol` × D_est, the same for D_est > 0); otherwise *early* (D < D_est) or *late*. With `t_d` = ∞ the seed is *late* if from_s + (1 + `band_tol`) D_est ≤ `duration_s`, else *unjudged*; a D_est ≤ 0 or a `t_d` of None is *unjudged*. Judged seeds: the counted ones (labels 5–11) less the unjudged, n_band of them. `band_inside` if k_inside ≥ `most`; `band_miss` if n_band ≥ `most` and k_inside < `most`; `band_too_few` otherwise. Author 2026-11-22: origin from_s, all counted seeds. Draft 5 §Band.
- **Code:** `t1.drain_band`, `t1.band_outcome`; the constants `T1["band_tol"]`, `T1["most"]`.

### U's end window (T1; draft 5, not registered; a readout)
- **Text:** over every seed with a valid U run, no `u_fault` and U baseline pf ≥ `min_baseline_pf`, whatever P's label: k = the U runs `k1.cold_backlogged` on their end window [3480, 4080) (`k1.windows`) against their own baseline; a None is listed and left out of n. Reported with each U run's `env.json` `platform` beside those of K1 arm A's open runs (`k1.open_pattern(2.0, s)`, REG-20's seeds) and D1's counted capped runs (`d1.capped_run`). Never judged. Draft 5 §U's end window.
- **Code:** `t1.u_end_readout`, `t1.reference_platforms`.

### surge arm, calm arm, crossing, marginal seed (K4)
- **Text:** schedule-only properties of a no-trigger run's 120 s planned session rate (`gate.rolling_rate` over `t_planned`, 5 s grid from 240 s). The *crossing* `t_cross` is the first grid time at which it is ≥ 3.4 req/s. A *surge-arm* schedule has its crossing in [1200, 1800] s and no value ≥ 3.2 before 1200 s; a *calm-arm* schedule has no value ≥ 3.1 in [240, duration). A *marginal* surge seed spends under 30 s at or above 3.4. Added 2026-11-17 with K4 (decisions.md 2026-11-17 [REG] K4).
- **Code:** `k4.classify`, `k4.scan`, `k4.pick`, `k4.schedule_load`, `k4.MARGINAL_S`.

### K4 run labels: tipped, tipped_recovered, tipped_unclear, tipped_early, warm, bad_baseline, unclear; on time
- **Text:** the onset is read first: an onset (`gate.kv_onset`, 60 s trailing mean KV ≥ 0.9) before 600 s is `tipped_early`; an onset in [600, duration) is `tipped` if the end window [2100, 2700) is not recovered against the baseline [300, 600) by k1's rule (reuse ≥ baseline − `recovered_hit_drop`, waiting ≤ baseline + `recovered_q_excess`), `tipped_recovered` if it is, `tipped_unclear` if that is undefined. With no onset: `bad_baseline` if the baseline's percent fast is below `min_baseline_pf` or NaN, `warm` if the end window is recovered, `unclear` otherwise. *On time*: a surge-arm `tipped` run's onset lies in [t_cross, t_cross + 212 s]. Different from `load.verdict`'s labels and from k1's cold-backlogged readout, which it uses.
- **Code:** `k4.label`, `k4.outcome` (`k4.CALM_TIPS`: the labels that count as an onset against the calm arm).

## Used configs that cannot be fixed (rule 5)

Used configs are never edited. Their known defects are recorded here instead.

### `configs/stage2_base.yaml`
- **Unit:** the comment at `:21` says "req/s^2". The unit is req/s (queue slope, above; decisions.md 2026-11-04 [CORR]).
- **Verdict block (`:10-22`):** it lacks `pf_climb`, `hit_climb` and `drain_horizon_s`, which `load.verdict` needs (`load.py:197`). It keeps the retired `q_slope`, `climb` and `end_s: 300`.
- **Effect:** the runs that use it directly cannot be judged by `load.verdict`: `20261101-144335_wipe_test_preempt`, `20261101-145940_wipe_test_idle`, and the stopped `20261101-151707_stage2_control_r1.5`. The wipe tests fail even before the missing keys. They are 900 s runs with the wipe at 450 s, so the 600 s baseline and the 1500 s watch do not fit, and `verdict` raises on an empty window. None of these runs was meant to be judged (decisions.md 2026-11-01, "Live wipe test").
- Every judged Stage 2 run gets the missing keys from `configs/stage2_cap_base.yaml:7-13`.

### K3b window and quantities: W, served, r, c, sessions; served_discriminates; against_corollary3 (K3b)
- **Text:** for a K4 run labelled `tipped`, W = [onset + 300 s, duration), at least 300 s long. Over W: *c* is requests of any status completed by t_end per second; *r* is the uncached prompt tokens of ok requests by first token per second over U_off (REG-8's five cache-off runs); *served* is the mean planned prompt of ok session turns by first token over 3072 (the schedule's mean); *sessions* is the sessions whose last turn ended ok in W, per second, over 1.681 / 8. Model F's values come at λ_W (planned request rate of the sessions starting in W) and μ_off 1.681. *served_discriminates*: model F's served + 0.015 < 1 at the run's λ_W (served is judged only then). *against_corollary3*: a run that passes the gate apart from λ_W ≤ μ_off, where model F says the trap drains. Added 2026-11-17 with K3b (decisions.md 2026-11-17 [REG] K3b).
- **Code:** `k3b.window`, `k3b.observed`, `k3b.predict`, `k3b.row`, `k3b.judge` (`k3b.JUDGED`: c, r, served).

### T2 labels: drained, twin_not_drained, slow, draining, relapse, unclear; emptied_before_E (T2; draft, registered with T2)
- **Text:** windows from the burst run's config, the twin read on the same ones: B the 600 s before the burst, E the run's last `verdict.end_s` s, E1 and E2 its halves. Per window q is the mean server waiting line (`load._level`), c the session completions ok per s, p the session requests planned per s, burst requests excluded. A baseline is healthy iff q_B ≤ 1.0 and c_B ≥ 0.95 p_B (else `bad_baseline`). A twin is `drained` iff q_E ≤ q_B + `recovered_q_excess` and c_E ≥ 0.95 p_E, else `twin_not_drained`. A burst run, with lined = q_E > q_B + `recovered_q_excess` and short = c_E < 0.95 × the twin's c_E, in order: `drained` (neither); `unclear` (short without a line, or a half not finite); `draining` (lined, q_E2 < q_E1); `slow` (lined, short, q_E2 ≥ q_E1); `relapse` (lined, not short, the line emptied before E by `gate.drain_time`); else `unclear`. `emptied_before_E` is that drain flag, reported beside the label. These names differ from REG-19's `recovered` and `stuck`, which judge the hit rate and percent fast.
- **Code:** `t2.windows`, `t2.window`, `t2.healthy_baseline`, `t2.label_twin`, `t2.label_burst`, `t2.outcome`, `t2.pair_row`.

### turn release: closed, open, default (`max_planned`) (TR arm D; draft, not registered)
- **Text:** when a session's later turn is sent. *Closed* (`closed_loop: true`): at the previous turn's end plus its drawn gap. *Open* (`closed_loop: false`): at its planned time, whatever the previous turn's state. *Default* (`workload.sessions.release: max_planned`): at max(planned time, previous turn's end), the release AIPerf's trace replay uses without `--open-loop-strict`. `release_chain` (harness/checks.py) checks each against its own rule.
- **Code:** `harness.loadgen.release_mode`, `harness.loadgen.session_chain`, `harness.checks` (release_chain), `aiperf_arm.release_audit`.

### TR labels: arm O's verdict, arm H's seed labels and verdict, arm D's out-of-n rules, the joint summary (TR; draft, not registered)
- **Text:** *O* per seed is `k1.open_seed` on its `_t1o` burst run and always-run control; a cold burst run whose control failed 3 attempts is `cold_control`; lost after 3 invalid burst attempts. Verdict, first match over counted seeds (`unclear`, `not_run`, `overload_open` out of n): `too_few` (n < 3), `pending_controls`, `lead_fails` (stuck_open + lost_open + cold_control ≥ 3, REG-20's refute_k), `replicates` (all `recovered_open`), `mixed`. O's consequence (`armo.consequence`): Appendix F's same-stack sentence needs `replicates` and the U readout `same_stack` (T1's U, `armo.u_reading`: too_few, stack_confounded, same_stack, weak, computed only after T1's [RESULT]); `replicates` otherwise is "arm A repeats today". *H* per seed, first match: `not_run`/`lost_open`; `control_not_run`/`control_lost`; the control read by `k1.cold_backlogged` on the burst run's windows is NaN (`control_unclear`), cold (`control_cold`) or not recovered (`control_partial`), each out of n; else `k1.open_seed`'s label. H's verdict: `cell_unstable` (control_cold + control_partial ≥ 3 of 5), `too_few`, `headroom_explains` (stuck_open + lost_open ≥ 3), `headroom_excluded` (no counted seed stuck or lost and recovered_open ≥ 3; partial_open allowed, not counted toward the 3), `mixed`. *D*: k1's labels renamed `*_default`; a lost control beside a cold burst run is `cold_control`; `overload_default` and `release_fault` out of n. *Joint summary* (never judged): `lead_fails` (O `lead_fails` or H `headroom_explains`), `lead_holds` (O `replicates` and H `headroom_excluded`), else `lead_unsettled`.
- **Code:** `armo.o_row`, `armo.decide_o`, `armo.consequence`, `armo.u_reading`, `tr.h_seed`, `tr.decide_h`, `tr.h_consequence`, `tr.joint`; arm D: `aiperf_arm.default_seed`, `aiperf_arm.decide`.

### idle age a, turnover-scaled age x, turnover time t_C; M labels (reading M; draft, not registered)
- **Text:** for a later turn k, *a* = t_first_token[k] − t_end[k−1] (s); a < 0 is `held`. *x* = A / F: A the newly allocated tokens (Δ(prompt_tokens_total − prefix_cache_hits_total + generation_tokens_total), interpolated between scrapes) over [t_end[k−1], t_first_token[k]], F = (1 − kv_cache_usage_perc at t_end[k−1]) × 96,656 tokens; A < 0 with a ≥ 0 is bucket `neg_A`; F ≤ 0 counts as x = ∞ (`f_zero`). A counter that goes down makes the run undefined. *t_C* (readout) = F / allocation rate per 60 s span. A bucket is *matched* with ≥ 100 turns in each arm and ≥ 3 of 5 open runs holding ≥ 20 of them. Labels, first match: `undefined` (< 4 defined runs per arm), `closed_not_outside` (< 0.9 of closed-arm later turns at x ≥ 1, or their reuse > 0.1), `arm_differs` (a matched bucket with |reuse difference| ≥ 0.3), `separated` (no matched bucket), `common_curve` (matched buckets within 0.3 and the open [1, 2) bucket, matched-size, reuses ≥ 0.5 less than open [0, 0.5)), `agree_outside` (matched buckets within 0.3, all at x ≥ 2), `mixed`.
- **Code:** `m.allocated`, `m.ages`, `m.bucket`, `m.matched_buckets`, `m.label`; readouts `m.per_run`, `m.censored`, `m.turnover`, `m.drain_profile`.

## Post-hoc load-test readouts (author 2026-11-26, the no-run route; decisions.md 2026-11-26 [DECISION] (author) Post-hoc load-test readouts …)
Not pre-specified; nothing below judges a run. Cells, runs and windows are `uniform.CELLS`'s and `k1.windows`'s.

### time to first token, two origins (post hoc)
- **Text:** *from the scheduled time*: over requests planned in the window, t_first_token − t_planned. *From the release time*: over requests released in the window, t_first_token − the release time (`t_released` under a rule that records it; t_planned under open loop, which releases at the planned time; a turn never released is in no release window). No first token by the run's end = inf, a miss. p50, p95 and p99 by nearest rank.
- **Code:** `ltb.latency`, `ltb.release_time`, `ltb.nearest_rank`. Differs from `load.load_run`'s `ttft`, which uses one origin per rule (`t_released`, else `t_planned`) and is summarized with pandas' interpolating quantile over finite values.

### goodput (post hoc, this readout only) ⚠ the repo's earlier "goodput" is percent fast (above)
- **Text:** requests of a latency population (above) that finished ok with TTFT ≤ `slo.ttft_s` (1.0 s), per second of the window. Named "goodput" in the papers for the load-testing audience, always with this definition beside it.
- **Code:** `ltb.latency` `goodput` (and `fast_share`, the same count over the population).

### send lag (post hoc)
- **Text:** t_sent − the release time (as above; t_planned under open loop), over requests sent whose release time falls in the span. The harness's `client_lag` check (FILE(harness/checks.py:61)) bounds its p99 over the whole run at `checks.max_client_lag_p99_s` (0.05 s) in every valid run.
- **Code:** `ltb.send_lag`.

### planned backlog (post hoc; rule-independent)
- **Text:** at time t, session requests (burst requests excluded) planned before t and not finished by t. It reads the schedule only, which is the same under every release rule on one seed. Exact split: `at_client` (planned, not yet sent) + `at_server` (sent, not finished). With burst requests included it is `gate.owed_backlog` (the "total line", above).
- **Code:** `ltb.planned_backlog`, `ltb.backlog_mean`.
