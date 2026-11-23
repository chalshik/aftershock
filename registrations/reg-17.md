# REG-17: the earlier matrix (per-span test)

*Registered 2026-11-10 · GPU runs · claim P0 · result 2026-11-10*

| | |
|---|---|
| **Question** | At λ = 2.0, seed 2, does REG-14's 180 s burst leave the server stuck through the 1500 s watch while the same schedule without the burst stays healthy? Decided by C1's per-seed verdict pair plus a measured cold side. |
| **Runs** | One same-seed pair (control, then burst run), seed 2, λ = 2.0, closed-loop turns, 2580 s; REG-14's configs with `workload.rate: 2.0` only (`configs/stage2_control_l1pl350cl_r2.0_s2.yaml`, `configs/burst_l1pl350cl_r2.0_s2.yaml`). 2 of 2 valid, no rerun. |
| **Outcome rule** | `gate.reg17_outcome`, in order: `bad_baseline`; control not `recovered` → `control_cold_surge` (tipped after a 120 s peak ≥ 3.2), `control_cold_no_surge` (tipped with no such peak), `control_unclear`, `control_warm_queue`; burst `stuck`, control `recovered` → `inconclusive` (cold side NaN), `stuck_draining` (cold side fails), `pass_clear` (cold side holds and the control waited in ≤ 0.5 of the scrapes of every span), `pass_warm_edge`; `drains` (burst `recovered` or `slow_drain`); `unclear`. Cold side (`gate.reg17_cold_side`): in every span 600–1500 s after the burst's end, point completion rate < min(2.0, the span's λp), and the end-window queue slope > 0. |
| **Prediction** | Control `recovered` ≈ 0.9; burst `stuck` ≈ 0.6; given `stuck`, the cold side holds ≈ 0.8; `pass_clear` ≈ 0.45; `drains` ≈ 0.35. Readouts if stuck: span c ≈ 1.70–1.82; end-window queue ≈ 60–200 and rising. |
| **Decision under each outcome** | `pass_clear`: draft the matrix at λ = 2.0 on the reserved seeds 3–7. `drains`: seed 2's schedule at 2.0 does not hold a cold state (0 of 1); the author chooses between λ = 2.1, another seed at 2.0, and reporting the zone as too narrow. `control_cold_no_surge`: stop, no matrix, rethink the surge reading. Every other outcome: reported, the author decides, no matrix. |
| **Adversarial review** | One round: four should-fix items (T1, close to blocking: an unclear control read as a refutation; T2: the cold side not measured; T3, T4: wording) and five minor items. All were applied before the runs. |
| **Result** | `pass_clear`: control `recovered` (waiting > 0 in 0.007, 0.003, 0.003 of the spans' scrapes, never tipped); burst `stuck` (end queue 110.8, slope +0.061 /s); cold side holds, span c 1.773, 1.733, 1.753 against λp 2.097, 2.007, 2.167. |
| **Prediction held** | yes (likeliest outcome; `pass_clear` ≈ 0.45 against `drains` ≈ 0.35) |
| **In the paper** | §4.2, P16; §4.1, P43 (controls at λ = 2.0); App. F, P51 |
| **Notes** | Proposed by an independent reviewer after REG-16 and approved by the author. Rule G's pass condition for a matrix at λ = 2.0 was stated after its inputs were seen (registered as such). One seed (rule H: 1 of 1, one-sided 95 % lower bound 0.05); 1500 s watch, no drop. |

[All registrations](README.md)
