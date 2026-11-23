# REG-6

*Registered 2026-11-04 · GPU runs · closed without a result*

| | |
|---|---|
| **Question** | Is λ = 3 below the default pool's cache-off capacity in sustained operation? If it is, the 6 default-pool shock runs at λ = 3 were bound to recover and were not tests of C1. |
| **Runs** | Planned: `configs/long_pc_off_r3_s0.yaml` and `configs/long_pc_off_r3_s1.yaml`, λ = 3, 3600 s, judged over 600–3600 s, seeds 0 and 1, cache off, default pool, 420 W (about 2.2 GPU h). One attempt: seed 0, `20261104-134159_long_pc_off_r3_s0`, interrupted; seed 1 never started. |
| **Outcome rule** | `floor.reg6_table` (`floor.sustained_capacity`). Backlogged stretches: runs of consecutive scrapes with waiting ≥ 1 lasting ≥ 30 s; c_b = ok completions inside them ÷ their total length. Label per seed (λp = that seed's planned rate): `unclear_data` (a NaN waiting value, or a scrape gap > `checks.max_scrape_gap_s`); `sustainable_no_backlog` (stretches total < 300 s); `sustainable` (c_b ≥ 1.05 λp); `not_sustainable` (c_b < λp); `unclear` otherwise. |
| **Prediction** | Seed 0: `sustainable`, c_b 3.35–3.60. Seed 1: `sustainable_no_backlog` (≈ 0.6) or `sustainable` (≈ 0.3). Both `not_sustainable` or `unclear`: ≈ 0.1. |
| **Decision under each outcome** | Both seeds `sustainable` or `sustainable_no_backlog`: the 6 default-pool λ = 3 shock runs were not tests; μ_off = 2 is recorded as an underestimate due to the seed-0 schedule; no further shocks at the default pool, and the next setup must have its floor measured first. Either seed `not_sustainable`: the 6 runs were real tests and recovered (evidence against C1 at this load), and the floor calculation is withdrawn. Otherwise: no change to any statement. |
| **Adversarial review** | One round of adversarial review on the first draft. It showed that the draft's queue-shape rule would most likely give a split that moves no decision; the rule was replaced by c_b over backlogged stretches, NaN and gaps give `unclear_data`, a false 'comparable' claim was removed, and the decision was restricted to the 6 default-pool λ = 3 runs (the draft said 12), before the attempt. |
| **Result** | No result. The only attempt (seed 0, 420 W) stopped at about t = 3161 s of 3600 when the VM received a power-off event; it has no `checks.json` and does not count. It also ran in SW-thermal slowdown for 0.296 of its samples in [600, 3168] s, against the 0.05 allowed. Closed without a result, superseded by REG-9 at 350 W. |
| **Prediction held** | no result (not run to completion) |
| **In the paper** | Not named in the text (Appendix A: not run; repeated at 350 W as REG-9). |
| **Notes** | The author chose to repeat the question at a 350 W power limit (REG-9) rather than rerun at 420 W, where the attempt's heat share made an invalid rerun likely. The interruption time is written as about t = 3161 s in the interruption entry and t ≈ 3168 s in REG-9. |

[All registrations](README.md)
