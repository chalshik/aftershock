# REG-14

*Registered 2026-11-10 · GPU runs · result 2026-11-10*

| | |
|---|---|
| **Question** | With REG-11's setup and burst, at λ = 2.2 and seed 2 (closed-loop turns, L1, 350 W), does the server hold a cold backlog after the burst (a counting span) while the same-seed control stays healthy? Registered as exploratory: a `pass` is a reason for a confirmatory matrix, not evidence for C1. |
| **Runs** | One same-seed pair (control, then burst run), seed 2, λ = 2.2, closed-loop turns, 2580 s, burst at 900 s for 180 s at 2.4 req/s with 2944-token prompts; `configs/stage2_control_l1pl350cl_r2.2_s2.yaml` and `configs/burst_l1pl350cl_r2.2_s2.yaml`. 2 of 2 valid, no rerun. |
| **Outcome rule** | `gate.reg14_gate` (= `gate.reg10_gate` with λ = 2.2, seed 2) gives REG-11's labels `pass`, `drains`, `control_not_healthy`, `stuck_unconfirmed`, `inconclusive`; a span holds only if `poisson_rate_upper(n, 300 s, 0.99)` < min(λ, λp). `gate.reg14_outcome` splits `pass`: `pass_clear` if in some counting span the control waited (waiting > 0) at ≤ 0.5 of its scrapes (`REG14["max_control_waiting"]`), else `pass_warm_edge`. `shock_comparable`: outstanding ≥ 0.5 × 46 and backlog_s ≥ 0.5 × 1499 against REG-12's seed-2 burst run. |
| **Prediction** | Both runs valid p ≈ 0.85; burst fills the pool p ≈ 0.85; control `recovered` p ≈ 0.75; burst `stuck` p ≈ 0.55. Given a valid pair: `pass` ≈ 0.35, `stuck_unconfirmed` ≈ 0.1, `drains` ≈ 0.3, `control_not_healthy` ≈ 0.15, `inconclusive` ≈ 0.1; a `pass` is `pass_clear` p ≈ 0.6; `shock_comparable` p ≈ 0.6. |
| **Decision under each outcome** | `pass_clear`: report as exploratory, one seed, and draft a confirmatory matrix at λ = 2.2 (five seeds) and C1's confirmation run. `pass_warm_edge`, `stuck_unconfirmed`, `control_not_healthy`: reported, the author decides. `drains` with `shock_comparable`: "no cold state held at λ = 2.2, seed 2, with this detector", the author chooses another seed, λ = 2.3 or a larger trigger. `inconclusive`: rerun unchanged, at most twice. |
| **Adversarial review** | Two rounds. Round 1 (on a first draft at λ = 2, seed 1) had three blocking items (the feasibility check measured the wrong quantity; the `drains` wording; the seed); round 2 (on λ = 2.2, seed 2) had two (B1, the `pass` → matrix decision; B2, `release_rate`). All were applied before the runs, B1 in a stricter uniform form (`pass_clear`); the author chose λ = 2.2, seed 2 after round 1. |
| **Result** | `pass_clear`: control `recovered`, burst `stuck`, label `pass`, all three spans count; the control's waiting share in the counting spans 0.000–0.003. Span c 1.807, 1.733, 1.753 against λp 2.207, 2.100, 2.347; queue 116 → 152 → 177; `shock_comparable` true (44 outstanding, 1499 s backlogged). |
| **Prediction held** | no (`pass_clear` had ≈ 0.21, DERIVED 0.35 × 0.6, against `drains` ≈ 0.3; at the coarser `pass` level, ≈ 0.35, it was the likeliest) |
| **Not separated** | A later turn-around of the backlog is not excluded within 1500 s; whether the cold state lasts a 4500 s watch and recovers after a drop to λ = 1 is untested; one seed, picked by the detector rule. |
| **Corrections** | 2026-11-10: the result's alternative 8 misstated C1's λ = 1 condition as persistence; it is recovery after the drop. Outcome and decision unchanged. |
| **In the paper** | App. C.7, P13; App. D, P27 (the stuck rate 1.733) |
| **Notes** | Exploratory, one seed. The seed rule (lowest seed in 1–10 whose every span limit exceeds 1.84) was written after REG-11/12's cold rates were known and before any λ = 2.2 run. After the result the author reversed the registered order: C1's confirmation run (REG-15) before the matrix. |

[All registrations](README.md)
