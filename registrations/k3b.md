# K3b: the served-mix test

*Registered 2026-11-17 · a reading of runs already made · claim K3 · result 2026-11-17*

| | |
|---|---|
| **Question** | Once a K4 run has tipped and stayed cold (L1, 350 W, closed-loop turns, λ = 2.2, no trigger), are the served prompt mix, the uncached token work and the completion rate each inside its tolerance of model F's value at the run's own planned rate? |
| **Runs** | 0 GPU h: the K4 runs that `k4.table` labels `tipped` (s133, s137, s175, s196), each over W = [onset + 300 s, 2700 s); registered before any K4 run was read. |
| **Outcome rule** | `k3b.judge` per quantity over runs that pass the gate (reuse ≤ 0.05, a request waiting in ≥ 0.99 of the scrapes, `k1.cold_backlogged`): served (obs − pred, ±0.015, judged only where the band excludes 1.000), r (r − 1, ±0.05), c (c / c_pred − 1, ±0.05). `inside` if \|median deviation\| ≤ tolerance, else `outside` with its side (`low` or `high`); `undefined` with fewer than 3 judged runs. The session completion rate is reported only; a run with λ_W ≤ μ_off counts `against_corollary3`. |
| **Prediction** | P(at least 3 judged runs) ≈ 0.80; given that, r inside 0.85, c inside 0.85, served inside 0.65; `against_corollary3` non-empty 0.10 (BELIEF). |
| **Decision under each outcome** | Served inside (with r and c inside): K3's status records that model F's served-mix tilt held in the no-burst stuck state at λ = 2.2 on n runs, with c and r named as consistency checks; served outside: reported by side, with `high` not separated from the surge cohort; r or c outside: reported, nothing reworded automatically; `undefined`: K3 keeps no judged test. |
| **Adversarial review** | One round: two blockers and five advisories, all applied before registration. |
| **Result** | Served, r and c all `inside`, 4 of 4 judged runs each: served median deviation +0.0025 (±0.015), r +0.0188 (±0.05), c +0.0159 (±0.05); upper bound on the share outside 0.527. Sessions (reported only) inside in 3 of 4; `against_corollary3` empty. |
| **Prediction held** | yes (likeliest outcome: served inside 0.65; r and c inside 0.85 each) |
| **Not separated** | Any tilt toward first turns passes served (for example, preempting long requests when KV is full); r and c follow from the gate and are consistency checks only. |
| **In the paper** | The served-mix test: §4.3 and §6 (P31), Appendix D's table (P32). |
| **Notes** | Registered at 07:12 UTC on 2026-11-17 while K4's queue ran (two runs copied back sealed, one running); no K4 outcome had been read. Model F's values use μ_off = 1.681 (REG-8's open-loop floor); E1's 1.798 is reported only. |

[All registrations](README.md)
