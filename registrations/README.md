# Registrations

One card per registration of the study, in the order they were registered. Each gives what was fixed before the runs were made or read (the question, the runs, the outcome rule with its thresholds, the prediction, the decision each outcome leads to and the adversarial review of the registration), then the result. A card whose kind is *a reading of runs already made* judged runs that had been made but not read for that question. The outcome rules are functions in `analysis/`; numbers are copied from the study's append-only log, not recomputed. The log itself is available from the author on request; `docs/results.md` lists its registration and result headings verbatim.

*Prediction held:* whether the outcome that occurred was the likeliest of the registered outcomes (*split*: arms or batches differ; *tied*: it shared the highest prior; *indeterminate*: no prior was written for the joint outcome; *not scored*: no outcome label was claimed). Each card gives the priors.

| Registration | Name in the paper | Claim | Registered | Result | Outcome | Prediction held |
|---|---|---|---|---|---|---|
| [REG-1](reg-1.md) |  |  | 2026-11-04 | 2026-11-04 | `inconclusive` | yes |
| [REG-2](reg-2.md) |  |  | 2026-11-04 | 2026-11-04 | `arrival_driven` (both runs) | split |
| [REG-4](reg-4.md) |  |  | 2026-11-04 | 2026-11-04 | no valid verdict changed | yes |
| [REG-6](reg-6.md) |  |  | 2026-11-04 | — | no result | no result |
| [REG-7](reg-7.md) |  |  | 2026-11-05 | — | no result | no result |
| [REG-8](reg-8.md) |  |  | 2026-11-06 | 2026-11-07 | no λ_test at 350 W | no |
| [REG-9](reg-9.md) |  |  | 2026-11-07 | 2026-11-09 | `no_change` | no |
| [REG-10](reg-10.md) |  |  | 2026-11-08 | 2026-11-09 | `drains` | yes |
| [REG-11](reg-11.md) |  |  | 2026-11-09 | 2026-11-09 | `pass` | no |
| [REG-12](reg-12.md) |  | C1 | 2026-11-09 | 2026-11-09 | `controls_not_healthy` | no |
| [REG-13](reg-13.md) |  |  | 2026-11-09 | 2026-11-10 | `overload` | no |
| [REG-14](reg-14.md) |  |  | 2026-11-10 | 2026-11-10 | `pass_clear` | no |
| [REG-15](reg-15.md) |  | C1 | 2026-11-10 | 2026-11-10 | `drop_inconclusive` | no |
| [REG-16](reg-16.md) |  | C1 | 2026-11-10 | 2026-11-10 | `control_cold_persistent` | no |
| [REG-17](reg-17.md) | the earlier matrix (per-span test) | P0 | 2026-11-10 | 2026-11-10 | `pass_clear` | yes |
| [REG-18](reg-18.md) | the earlier matrix (per-span test) | P0 | 2026-11-11 | 2026-11-11 | `some` (1 of 4) | yes |
| [REG-19](reg-19.md) | the replication (whole-window test) | P0, C1 | 2026-11-11 | 2026-11-13 | `supports` (5 of 5) | yes |
| [REG-20](reg-20.md) | the turn-release test | K1 | 2026-11-12 | 2026-11-16 | arm A `supports`; arm B `headroom_weakened` | split |
| [REG-22](reg-22.md) | the four-links reading | C2 | 2026-11-13 | 2026-11-13 | `loop` | yes |
| [REG-21](reg-21.md) | the token-work readings | K3 | 2026-11-13 | 2026-11-13 | `inside` | yes |
| [E1](e1.md) | the capacity sweeps | K2 | 2026-11-14 | 2026-11-14 | `ok` (floor 1.798 req/s) | yes |
| [Condition 4's order clause](condition-4s-order-clause.md) | the order test | C1 | 2026-11-14 | 2026-11-16 | `does_not_follow` | split |
| [E0 and E2](e0-and-e2.md) | the capacity sweeps | K2 | 2026-11-14 | 2026-11-16 | `floor_only` | tied |
| [REG-25](reg-25.md) | the wipe run | C1 | 2026-11-14 | 2026-11-15 | wipe run `stuck` (one seed) | indeterminate |
| [REG-23](reg-23.md) | the early-warning reading | K5 | 2026-11-14 | 2026-11-14 | `no_usable_cache_lead` | yes |
| [REG-26](reg-26.md) |  |  | 2026-11-14 | 2026-11-14 | `small` | yes |
| [D1](d1.md) | the session cap | C2 | 2026-11-15 | 2026-11-16 | `supports` | yes |
| [REG-24](reg-24.md) | the long-watch run | C1 | 2026-11-15 | 2026-11-16 | `confirms` (one seed) | yes |
| [D1b](d1b.md) | the looser cap | C2 | 2026-11-15 | 2026-11-17 | `batch_bound` | yes |
| [K4](k4.md) | the surge test | K4 | 2026-11-17 | 2026-11-17 | `supports` | yes |
| [K3b](k3b.md) | the served-mix test | K3 | 2026-11-17 | 2026-11-17 | `inside` | yes |
| [B1](b1.md) | the aging reading | C2 | 2026-11-17 | 2026-11-17 | `supports_aging` | yes |
| [B3's gate](b3s-gate.md) | the server-side batch bound | C2 | 2026-11-17 | 2026-11-17 | `fail_redraft` | yes |
| [REG-27](reg-27.md) | the long-watch replication | C1 | 2026-11-17 | 2026-11-20 | `replicates` (4 of 4) | no |
| [K4e](k4e.md) | the band test | K4 | 2026-11-17 | 2026-11-20 | `band_warm` | no |
| [REG-28](reg-28.md) | the session-share test | K3, C2 | 2026-11-18 | 2026-11-20 | `law_holds`; `churn_bounded` | yes |
| [T4](t4.md) | the step-budget sweep |  | 2026-11-21 | 2026-11-21 | `cell_moves` | no |
| [T2](t2.md) | the cache-off burst control | C2 | 2026-11-21 | 2026-11-21 | `drains` | yes |
| [REG-30](reg-30.md) | the re-warming reading | C2 | 2026-11-21 | 2026-11-21 | `drain_bound` | no |
| [G1](g1.md) | the second-schedule sweep |  | 2026-11-21 | 2026-11-22 | `seed_varies` | yes |
| [G2](g2.md) | the development-mode twin |  | 2026-11-21 | 2026-11-22 | `dev_neutral` | yes |
| [P8192](p8192.md) | the 8192 burst test | P0 | 2026-11-21 | 2026-11-22 | no label claimed (3 of 5 stuck) | not scored |
| [REG-31](reg-31.md) | the four post-review readouts |  | 2026-11-22 | 2026-11-22 | four readouts | mixed |
| [T1](t1.md) |  |  | 2026-11-23 | 2026-11-23 | `mixed` | yes |
| [TR](tr.md) |  | K1 | 2026-11-23 | 2026-11-25 | O `replicates`, H `headroom_excluded`, D `default_sees` | split |
| [M](m.md) |  | K1 | 2026-11-23 | 2026-11-23 | `separated` | yes |
| [PW](pw.md) |  | P0 | 2026-11-25 | 2026-11-25 | 0 of 10 counted (all `cut_sensitive`) | no |
