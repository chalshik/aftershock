# D1b: the looser cap

*Registered 2026-11-15 · GPU runs · claim C2 · result 2026-11-17*

| | |
|---|---|
| **Question** | With the session cap at 40, a cap that still bounds the population in service but not the cold state's running batch (in flight ≤ 31.25, above N ≈ 30), does REG-19's burst still leave the system stuck at λ = 2.0? |
| **Runs** | Three capped burst runs at λ = 2.0 (L1, 350 W, closed-loop turns), seeds 15, 16, 17, each REG-19's burst config with `workload.sessions.cap: 40`; ≈ 3.7 GPU h. |
| **Outcome rule** | `d1b.label40`: D1's label with D1's thresholds, except that a run D1 calls `partial` is `stuck` when the server's own line is backlogged. `d1b.outcome`, against D1's label on the same seed: `too_few` (n < 2); `batch_bound` (k_bracket ≥ 2, a bracket being D1 cut and D1b stuck on the same seed); `stuck_without_d1_cut` (k_stuck ≥ 2, k_bracket < 2); `cut_at_40` (k_cut ≥ 2); otherwise `mixed`. |
| **Prediction** | `batch_bound` 0.491, `stuck_without_d1_cut` 0.227, `mixed` 0.242, `cut_at_40` 0.040; per valid seed `stuck` 0.65 (BELIEF inputs). |
| **Decision under each outcome** | `batch_bound` with D1 `supports`: C2's status may say the cap prevented the stuck state where it held the running batch below the cold state's, and not where it bounded the population alone, with both confounds in the same sentence; `cut_at_40`: read beside the drift rerun; anything else is reported and changes nothing. |
| **Adversarial review** | One round: NOT READY with eight items (U1–U8) and three optional ones (O1–O3), all applied in draft 2 before registration. |
| **Result** | `batch_bound`: seeds 16 and 17 `stuck` where D1's cap had cut them (k_bracket 2 of 3), seed 15 `rewarming`, none cut; k_stuck 2 (upper bound 0.983), k_cut 0 (upper bound 0.632). |
| **Prediction held** | yes (likeliest outcome, 0.491) |
| **Not separated** | Pool fill from batch fit; running-batch pressure from idle-prefix eviction (both registered as not separated). |
| **In the paper** | The looser cap: §4.5 (P39, P41), §6 (P49), Table 2 (P58), Appendix C.14. |
| **Notes** | The feasibility gate passed before registration: REG-19's uncapped runs on seeds 15–17 held 280.6, 245.3 and 319.5 sessions in service over the end window, above 40. All three runs were valid on one attempt. Seed 15's `rewarming` run had a rising cap queue and a backlogged server; a 4080 s run cannot say whether it would have cleared. |

[All registrations](README.md)
