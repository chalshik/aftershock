# B3's gate: the server-side batch bound

*Registered 2026-11-17 · a feasibility gate · claim C2 · result 2026-11-17*

| | |
|---|---|
| **Question** | Before drafting B3, a server-side bound on the running batch (`--max-num-seqs` 16) meant to spare the warm state and keep the pool from filling: what is the highest running N in REG-19's five warm no-burst controls over the measured window? |
| **Runs** | No new runs (0 GPU h). Reads REG-19's five no-burst controls at λ = 2.0 (`*_stage2_control_l1pl350cl_r2.0_w3000_s1[5-9]`), valid only, exactly one per seed 15–19; per control, the highest N (`vllm:num_requests_running`) over the 1 s scrapes with 240 ≤ t < 4080 s. |
| **Outcome rule** | `b3.control_row`, `b3.gate_outcome`. `undefined`: not one usable control per seed 15–19. `pass`: every highest N < 16. `fail_redraft`: 16 ≤ highest ≤ 22. `infeasible`: highest ≥ 23. N = 16 counts as reaching the flag. Readouts (mean N, share of scrapes with N ≥ 16, peak count and time, 0.999 quantile) are never judged. |
| **Prediction** | M/G/∞ with N ~ Poisson(mean N): P(pass) 0.91 at mean N 4.0, 0.68 at 4.5, 0.27 at 5.0, 0.02 at 5.5. With mean N ≈ 5: `pass` ≈ 0.3, `fail_redraft` ≈ 0.65, `infeasible` ≈ 0.05. |
| **Decision under each outcome** | `pass`: B3 is drafted at 16 for the author. `fail_redraft`: a new draft at some flag k in [highest + 1, 23], the author first. `infeasible`: B3 as designed is not drafted. |
| **Adversarial review** | One round before computation: three blocking items (the window, the gauge's source, one run per seed) and six advisories, all applied except the share rule (item 5), which the author declined. |
| **Result** | `fail_redraft`: highest running N over 240 ≤ t < 4080 s by seed 15: 22, 16: 16, 17: 22, 18: 17, 19: 19; every control reached 16 and the highest is 22 (`b3.k_max` 23). Readouts: mean N 4.93–5.31; share of scrapes with N ≥ 16 from 0.0008 (s16) to 0.0096 (s15). |
| **Prediction held** | yes (likeliest outcome: `fail_redraft` at 0.65); the model's peak distribution missed (observed peaks up to 22) |
| **Not separated** | No alternative is separated at the result; after B3 was closed, C2's loop stays not separated from pool pressure set by the population or a smaller effective shock. |
| **In the paper** | Appendix A, 'the server-side batch bound'; §6 (P49); its level of 22 is reused in §4.6 (P69). |
| **Notes** | The only flag the rule left, 23, lets the pool in use reach 23 × 4,096 / 96,656 = 0.975, the stuck state's own level, so on 2026-11-17 the author closed B3 with no server batch bound drafted at this cell, and the preprint's limitations gained a sentence on it. Registered limits: a 1 s scrape misses short peaks; one load, one run per seed. |

[All registrations](README.md)
