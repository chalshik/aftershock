# REG-10

*Registered 2026-11-08 · GPU runs · result 2026-11-09*

| | |
|---|---|
| **Question** | After a 180 s burst at λ = 2.4, does the L1 server at 350 W hold a cold backlog (a counting span) while a same-seed control without the burst is healthy? Exploratory: λ was chosen after REG-8's data. |
| **Runs** | One pair at 350 W, seed 1, λ = 2.4, 2580 s, open-loop turns, dev mode, `loadgen_force_close`: control `20261108-211646_stage2_control_l1pl350_r2.4` and burst run `20261108-220248_burst_l1pl350_r2.4_s1` (burst at 900 s for 180 s at 2.4 req/s, 2944-token prompts; 422 burst requests). |
| **Outcome rule** | `gate.reg10_gate`, through `gate.reg8_gate_verdict` and `gate.gate_verdict` with REG-7's spans and settings. Labels: `stuck_unconfirmed` (`inconclusive` with the burst verdict `stuck` and the control `recovered`); `inconclusive` (every other `inconclusive`); `control_not_healthy`, `pass`, `drains` as REG-7. Registered detectable band: a span can hold only if its completion rate c ≤ 1.953, 2.190, 2.147. Readout `gate.overlap_share` over [1080, 2580) is decisive only under `drains` (> 0.5). |
| **Prediction** | The burst run fills the pool p ≈ 0.85. Given a valid pair: `pass` ≈ 0.25, `drains` ≈ 0.45, `control_not_healthy` ≈ 0.1, `stuck_unconfirmed` ≈ 0.1, `inconclusive` ≈ 0.1. Overall `pass` ≈ 0.2. |
| **Decision under each outcome** | `pass`: register the confirmatory matrix at λ = 2.4 (burst seeds 2–5, each with a same-seed control; ≥ 3 of 5 with the one-sided Clopper–Pearson bound), this pair not counted. `drains`: record c_b and 1 − c_b / μ_warm in M2; if `overlap_share` > 0.5 propose closed-loop turns, otherwise the author chooses; `drains` does not by itself license writing up L1 as negative. `stuck_unconfirmed`: not rerun, the author decides. `control_not_healthy`: no retry. `inconclusive`: rerun the pair, at most twice. Burst run invalid on `burst_min_kv_max`: not rerun. |
| **Adversarial review** | Two separate rounds of adversarial review. On the λ = 2.25 draft: four blocking recommendations (which pair counts after a rerun, `inconclusive` split, the detectable band stated with revised predictions, `drains` not licensing a negative) and one choice for the author (λ = 2.25 or 2.4); all adopted, and the author chose 2.4. On the amendment: a guard refusing a run without `loadgen_force_close`, adopted before the runs. |
| **Result** | `drains`. Control `recovered`, burst run `recovered`; the burst filled the pool (KV max 0.9998). No span holds: c 2.183, 2.653, 2.377 against λp 2.150, 2.677, 2.350, hit share 0.841, 0.838, 0.838 against baseline 0.838, waiting ≤ 0.003. `overlap_share` 0.608 > 0.5. `secondary_cb` `sustainable`: c_b 3.397 req/s over one 393 s stretch (1335 completions) against λp 2.359; 1 − c_b / μ_warm = −0.132. |
| **Prediction held** | yes (likeliest outcome) |
| **Not separated** | Overlap protection (open-loop turns pin a session's prefix in use through the backlog) against a burst too short to evict the sessions' prefixes; the post-burst rate 1.4× λp says the state could not be stuck at 2.4 but does not separate these two. |
| **Corrections** | 2026-11-09: the `pass` branch's 'burst seeds 2–5 … ≥ 3 of 5' names four seeds, not five (not reached here). |
| **In the paper** | §4.4 (P10, the open-loop pair at λ = 2.4; text in Appendix C.13). Its prior is among those counted in Appendix D.2's calibration (P33). |
| **Notes** | Amended on 2026-11-09 (Amendment to REG-9 and REG-10): `loadgen_force_close: true` added to both configs before they ran. Registered decision taken: closed-loop turns proposed to the author (REG-11). |

[All registrations](README.md)
