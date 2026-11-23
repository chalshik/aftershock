# REG-24: the long-watch run

*Registered 2026-11-15 · GPU runs · claim C1 · result 2026-11-16*

| | |
|---|---|
| **Question** | C1's conditions 5 and 6 at λ = 2.0: after the burst, does the server stay stuck for a 4500 s watch, drain in a drop to λ = 1, and recover on the return to λ = 2.0, beside a no-burst control on the same schedule that stays warm? |
| **Runs** | Two runs on seed 43, ≈ 9.1 GPU h: the control `configs/stage2_control_l1pl350cl_r2.0_long_s43.yaml`, then the burst run `configs/confirm_l1pl350cl_r2.0_s43.yaml`. Baseline 300–900 s, burst 900–1080 s, watch to 5580 s, λ = 1 for 9000 s (5580–14580 s), then λ = 2.0 for 1500 s. |
| **Outcome rule** | `confirm.burst_outcome`, first match wins: `bad_baseline`, `not_persistent`, `inconclusive`, `reformed`, `emptied_in_watch`, `inconclusive`, `stuck_not_cold`, `drop_inconclusive`, `inconclusive`, `no_return`, `confirms`; the control by `confirm.control_outcome` (`control_warm` first). `confirms` requires (a)–(h): both valid on one build; `control_warm`; baseline pf ≥ 0.98; `stuck` at 4980–5580 s; REG-19's cold side over [1680, 5580) and [4980, 5580); no 300 s watch window recovered; no emptying in 1080–5580 s; in the drop an emptying and the total line back to its level at 900 s; the return's end recovered. A non-warm control gives `control_not_warm`. |
| **Prediction** | Given a warm control, a valid burst run and one build: `confirms` ≈ 0.62, `bad_baseline` ≈ 0.08, `no_return` ≈ 0.08, `drop_inconclusive` ≈ 0.06, the rest ≤ 0.04 each; unconditionally `confirms` ≈ 0.50; `control_warm` ≈ 0.85; K3 `inside` ≈ 0.7, implied r ≈ 0.95–1.05 (BELIEF). |
| **Decision under each outcome** | `confirms`: the registered sentence, and C1 at this cell then lacks only condition 4's order clause; `not_persistent`, `reformed`, `emptied_in_watch` or `stuck_not_cold`: condition 5 is not met at this cell; `drop_inconclusive`: condition 6 is inconclusive; `no_return`: condition 6 is not met; `control_not_warm`: the confirmation is void, and a tip of the control reopens the surge reading. |
| **Adversarial review** | Two rounds on the draft, applied in draft 4 before registration (the number of blocking changes is not stated in the log); no round on the registration entry itself. |
| **Result** | `confirms` with `control_warm` (1 of 1, one seed; upper bound 1.00, lower bound 0.05): stuck at 4980–5580 s (pf 0.000, hit 0.000, queue 299.4 rising, N 29.94, KV 0.950); first emptying 2333 s into the drop, total line drained at 2380 s; the return's end recovered. Readouts: K3 `inside` [2130, 4080]; implied r 1.031. |
| **Prediction held** | yes (likeliest outcome, ≈ 0.62 given a warm control) |
| **Not separated** | Persistence held by the clients' owed backlog against the pool size (1650 turns owed at the watch's end, more context than the 96,656-token pool) rather than by C2's loop; so the run does not test C2's chain. |
| **In the paper** | The long-watch run: §4.3 (P19, P20, P22, P36), Appendix D's table (P32), the controls count (P43), Appendix F (P51, run order). |
| **Notes** | The registered text is the cited draft 4, with later entries setting only the queue order (after D1) and the operations. The 9000 s drop departs from PROTOCOL's 1800 s by the author's ruling. Both runs were valid on one attempt. Replicated on four further seeds by REG-27. |

[All registrations](README.md)
