# REG-15

*Registered 2026-11-10 · GPU runs · claim C1 · result 2026-11-10*

| | |
|---|---|
| **Question** | At λ = 2.2 with closed-loop turns (L1, 350 W): (5) is the server still `stuck` 4500 s after the burst, without having recovered on the way? (6) With λ dropped to 1 for 1800 s and returned to 2.2 for 1500 s, does the queue reach 0 in the drop, and are the last 600 s `recovered` against the original baseline? Confirmatory for C1's conditions 5 and 6 at one load, in one run. |
| **Runs** | One burst run, seed 12, `configs/confirm_l1pl350cl_r2.2_s12.yaml`: baseline 300–900 s, burst 900–1080 s, watch 1080–5580 s, drop 5580–7380 s at λ = 1, return 7380–8880 s at λ = 2.2. No separate control. Valid on the first attempt. |
| **Outcome rule** | `gate.reg15_outcome`, in order: `bad_baseline`; `not_persistent` (the watch ended `recovered` or `slow_drain`); `inconclusive` (`unclear` or a missing value); `reformed` (`stuck` at the watch's end but some 300 s window of the watch `recovered`, `gate.watch_broken`); `drop_inconclusive` (the line never reached 0 in the drop); `no_return` (the line reached 0 but the return's end is not `recovered`); `confirms` (stuck through the whole watch, the line reached 0, the return's end `recovered`: pf ≥ base − 0.02, hit ≥ base − 0.06, waiting mean ≤ base + 1). |
| **Prediction** | Run valid p ≈ 0.75; baseline healthy p ≈ 0.85; burst fills the pool p ≈ 0.85. Given a valid run with a healthy baseline: `confirms` ≈ 0.26, `not_persistent` ≈ 0.35, `reformed` ≈ 0.1, `no_return` ≈ 0.17, `drop_inconclusive` ≈ 0.08, other ≈ 0.04. |
| **Decision under each outcome** | `confirms`: draft the confirmatory matrix at λ = 2.2 on seeds 3–7. `not_persistent` or `reformed`: the matrix is not drafted; the author chooses another cell or reports a long but transient cold state. `no_return`: the author decides, no matrix before that. `drop_inconclusive`: reported, the author decides (a longer drop). `bad_baseline`: not rerun. `inconclusive`: rerun unchanged, at most twice. |
| **Adversarial review** | Two rounds. Round 1 (on a first draft with seed 2) had three blocking items (wrong judged windows; planned load confounding conditions 5 and 6; a re-formed collapse counting as persistent); round 2 found none blocking. All items were applied before the run; the author chose seed 12 by the judged-window rule. |
| **Result** | `drop_inconclusive`: the 4500 s watch `stuck` with `watch_broken` false (none of 15 windows `recovered`; end window pf 0.000, hit 0.000, queue 399.7, KV 0.949), so condition 5 holds; `line_reached_zero` false (no scrape with 0 waiting in 5580–7380 s), so condition 6 is untested, not failed. |
| **Prediction held** | no (`drop_inconclusive` had ≈ 0.08 against `not_persistent` ≈ 0.35) |
| **Not separated** | The data cannot tell a cache that would warm once the line empties from one that would not; with no same-schedule control and a return that never ran warm, persistence by history rather than overload is evidence, not proof. |
| **Corrections** | 2026-11-10: the registration assumed a 4–6 minute client lag in the drop; it was 856 s at the watch's end and 930 s at the drop's end (no result changes). 2026-11-10: the write-up named `trigger` instead of `manipulation` as the pool check, and the drop's released load was 1.582 req/s, not ≈ 1; `drop_inconclusive` stands. |
| **In the paper** | §3, P4 (1800 s at λ = 1 did not empty the line); App. C.7, P14 |
| **Notes** | The seed rule (all four judged windows within ± 5 % of 2.2; only seed 12 of 1–30 qualifies) and its tolerance were written after seeing seed 2's windows, before any run of this design. The matrix was not drafted. |

[All registrations](README.md)
