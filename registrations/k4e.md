# K4e: the band test

*Registered 2026-11-17 · GPU runs · claim K4 · result 2026-11-20*

| | |
|---|---|
| **Question** | Does a 120 s planned peak between K4's calm and surge levels (3.1–3.4 req/s) also end the warm state and keep it cold at λ = 2.2? |
| **Runs** | 12 no-burst runs at λ = 2.2 for 2700 s, ≈ 11.4 GPU h, on K4's workload with `name` and `seed` alone changed (L1, 2048-token system prompt, 8 turns, 5 s gaps, closed-loop turns, 350 W, cache on): ten band schedules picked by a rule fixed before seeds 600–999 were scanned (`lo` [3.1, 3.25): 630, 650, 663, 669, 694; `hi` [3.25, 3.4): 609, 711, 772, 813, 914; exact peaks 3.125–3.400), plus K4's surge seed 137 (positive control) and calm seed 103 (calm control). Configs `configs/k4e_l1pl350cl_r2.2_s<seed>.yaml`. |
| **Outcome rule** | `k4.label`, then `k4e.counted` and `k4e.outcome`, the ten band runs as one cell. Counted: `tip` (an onset in [t_cross − 60 s, t_cross + 212 s] that stayed cold), `recovered`, `warm`; not counted, reported: `surgeless`, `late`, `unclear`, `bad_baseline`. In order: `too_few` (n < 8); `band_warm` (in-window tips plus persistent `tipped` runs outside the window ≤ 1); `band_tips` (in-window tips ≥ 4); `mixed` otherwise. Beside it `positive_ok` (137 a `tip`) and `calm_ok` (103 `warm`). |
| **Prediction** | Tip probability per run: `lo` ≈ 0.10, `hi` ≈ 0.35; 137 ≈ 0.8; 103 warm ≈ 0.9. Outcome: `band_warm` 0.29, `mixed` 0.55, `band_tips` 0.16; `too_few` ≈ 0.05. |
| **Decision under each outcome** | `band_warm` (both controls ok): K4's sentence adds that ten schedules with exact peaks 3.10–3.40 tipped and stayed cold in k of n, so the surge that ends the warm state lies above the band at this cell. `band_tips`: the surge level in K4's sentence moves below 3.4, the tips' peaks named. `mixed`: k of n with the tips' peaks; no edge claimed. A control not ok: reported without the contrast. Any `surgeless` tip: K4's refuting case; K4 is reworded. No outcome starts another run. |
| **Adversarial review** | Two rounds (the limit) before registration. Round 1: five blocking items (pre-crossing tips, edge labels, tip criterion, timing window, pooling), all applied. Round 2: items N1–N3 (excluded tips, seed 663's window, a calm control added) and advisories, applied; round 2's fixes were not reviewed again. |
| **Result** | `band_warm`: in-window tips 0, persistent tips outside the window 1 (s914, `late`), sum 1. Band 0 of 9 counted (s914 not counted), one-sided 95 % bounds 0 to 0.283; `lo` 0 of 5, `hi` 0 of 4. s914 (exact peak 3.392) tipped 223 s after its crossing, 11 s past the window, and stayed cold. Both controls ok (137 tipped 74 s after its crossing; 103 warm). No `recovered`, `surgeless`, `unclear` or `bad_baseline`. |
| **Prediction held** | no (`band_warm` had 0.29 against `mixed` at 0.55) |
| **Not separated** | Dose (height against duration above a level), the few-minute rise around the peak, and sessions against requests are not separated; whether s914 tipped from its peak or from something later is not separated. No edge inside the band is located. |
| **In the paper** | Appendix A, 'the band test'; §4.6 (P56); s914 is among the events of the running-batch readout in §4.6 (P69). |
| **Notes** | The outcome does not hang on s914's class: had its onset fallen inside the window, the sum would still be 1 (1 of 10, upper bound 0.394). The registered gloss for `band_warm` would hide s914, so K4's sentence as approved names it beside the 0 of 9. Seed 813's exact peak equals K4's surge level 3.4; it was kept as the pick rule gave it. |

[All registrations](README.md)
