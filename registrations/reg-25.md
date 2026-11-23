# REG-25: the wipe run

*Registered 2026-11-14 · GPU runs · claim C1 · result 2026-11-15*

| | |
|---|---|
| **Question** | Does a sustained series of preempt cache wipes, with no extra arrivals, leave the server at L1, λ = 2.0 stuck? It also supplies the order test's wipe run. |
| **Runs** | Two runs on seed 44, which no other run uses, control first: `configs/stage2_control_l1pl350cl_r2.0_s44.yaml` (no trigger) and `configs/swipe_l1pl350cl_r2.0_s44.yaml` (`cache_wipe`, `mode: preempt`, every 2.5 s for 300 s from 900 s, 120 calls); each 2700 s, closed-loop turns, 350 W; ≈ 1.6–1.7 GPU h. |
| **Outcome rule** | Each run's `load.verdict` with the verdict settings in force (`baseline_s` 600, `watch_s` 1500, `end_s` 600, `stuck_hit_drop` 0.06, `stuck_pf_drop` 0.10), the wipe run's baseline before its first call and its watch from its last call's end. For wipe `stuck` with control `recovered`, the registered readout `span_c` (completions over the wipe span, `swipe_design.span_readout`) sets the reading: ≥ 1.5 req/s, cache loss with completions held at the all-miss floor; < 1.5, "a 300 s series of preempt calls cut completions below the all-miss floor; re-prefill work, not cache loss alone, built the backlog". `owed_rise` is reported against 1.5 Q* = 45.3. The manipulation check requires the hit rate over the 300 s after the first call's end ≤ `wipe_max_hit` 0.25. |
| **Prediction** | Control `recovered` ≈ 0.9; wipe `stuck` ≈ 0.5; `stuck` with `span_c` ≥ 1.5 (the cache-loss reading) ≈ 0.1; P(`span_c` < 1.5) ≈ 0.8; the span completing at ≈ 0.3–1.1 req/s; `owed_rise` ≥ 45.3; the order line `follows` ≈ 0.6 if `stuck` (all BELIEF). |
| **Decision under each outcome** | Wipe `stuck` with control `recovered`: if `span_c` ≥ 1.5 the author decides on more seeds toward C1 for the wipe trigger; if `span_c` < 1.5 the author's choice is about the manipulation. Wipe `recovered` or `slow_drain` beside a recovered control: a 300 s series of preempt wipes did not hold, read against `owed_rise`. A control not `recovered`, or `unclear` or `bad_baseline`, gives no reading. |
| **Adversarial review** | Two rounds: round 1 NOT READY with five items (R1–R5), round 2 READY provided four (S1–S4); all applied before registration. |
| **Result** | Wipe run `stuck`, control `recovered` (1 of 1 each, one seed; upper bound 1.000). `span_c` = 0.084 req/s, below 1.5, so the registered reading is re-prefill work, not cache loss alone; `owed_rise` = 507 against 45.3. Wipe end window: pf 0.0, hit 0.0, waiting 102.5, N 30.0, KV 0.953. Order line `does_not_follow` (preemptions only). |
| **Prediction held** | Indeterminate at the outcome level (no prior was written for the joint outcome): control `recovered` (0.9), wipe `stuck` (0.5), `span_c` < 1.5 (0.8) and `owed_rise` ≥ 45.3 held; the predicted span rate of 0.3–1.1 req/s and the order line `follows` (0.6) missed. |
| **Not separated** | Cache loss from the backlog that the re-prefill outage built (registered as not separated by this run). |
| **In the paper** | The wipe run: §4.3 (P21, P22, P36), §6 (P48), Appendix D's table (P32, wipe completions), Appendix F (P51, run order); one of the controls counted in P43. |
| **Notes** | The feasibility gate passed before registration: c_cold = 1.777, Q* = 30.2, Q_300 = 67.0 passes and Q_180 = 40.2 does not, so D = 300 s. The interval rule chose no interval (H(5.0) 0.295–0.323, H(2.5) 0.158–0.182, against 0.05); the author chose 2.5 s and `wipe_max_hit` 0.25. The queue stopped after the control when the GPU was returned and resumed later that day (control 14:13, wipe run 19:31 UTC); both runs were valid on one attempt. Afterwards the author closed the wipe arm: REG-25 is not evidence that cache loss alone leaves the system stuck. |

[All registrations](README.md)
