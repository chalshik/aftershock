# REG-18: the earlier matrix (per-span test)

*Registered 2026-11-11 · GPU runs · claim P0 · result 2026-11-11*

| | |
|---|---|
| **Question** | At λ = 2.0, does REG-14's 180 s burst leave the server stuck and cold through the 1500 s watch, while the same schedule without the burst stays healthy, in most of seeds 3–7? Registered as confirmatory for C1's matrix condition at one load. |
| **Runs** | REG-17's pair on seeds 3–7 (only `name` and `seed` changed), λ = 2.0, closed-loop turns, 2580 s, run in the order 4, 3, 5, 6, 7 with a futility stop. Seeds 4, 3, 5, 6 run (8 of 8 valid, no rerun); seed 7 not run after the stop. |
| **Outcome rule** | Per seed `gate.reg18_seed_outcome`: REG-17's order unchanged plus a late-surge-tip rule on the exposed seed 4, no pass on a missing control share, and the control judged on its own behind a burst-side `bad_baseline`. Matrix `gate.reg18_outcome`: exclusion of seed 4 only if `control_cold_surge`; `surge_reading_wrong` if any `control_cold_no_surge`; `controls_not_healthy` if 2 or more bad controls; `too_few`; then `supports` if k_clear ≥ 3, `supports_edge` if k ≥ 3 but k_clear < 3, `some` if k = 1–2, `none` if k = 0, with the one-sided 95 % Clopper–Pearson upper bound (none after a futility stop). Stop once k + seeds left < 3. |
| **Prediction** | Per seed P(`pass_clear` or `pass_warm_edge`): seed 3 ≈ 0.10, 4 ≈ 0.45, 5 ≈ 0.35, 6 ≈ 0.40, 7 ≈ 0.55. E[k] ≈ 1.85, P(k = 0) ≈ 0.09, P(k = 1–2) ≈ 0.66, P(k ≥ 3) ≈ 0.26; P(supports, k_clear ≥ 3) ≈ 0.2. Each control `recovered` ≈ 0.9 (seed 4 ≈ 0.75). |
| **Decision under each outcome** | `supports`: draft C1's remaining conditions at λ = 2.0 (a 4500 s watch, a drop to λ = 1, the return). `supports_edge`: the author decides. `some`, `none` or a futility stop: report k of n; the author chooses among a matrix at λ = 2.1, a revised per-seed test registered as a new claim, or reporting the zone as too narrow. `controls_not_healthy`: the author decides. `surge_reading_wrong`: stop and rethink. |
| **Adversarial review** | Two rounds. Round 1: five should-fix items (T1–T5, including a late tip read as `surge_reading_wrong`, a burst-side `bad_baseline` counted as a bad control, and the futility stop added); round 2: one blocking item (R2-T1: the stop could hide a control failure on seed 4, fixed by running seed 4 first) and three should-fix items. All were applied before the runs; no third round. |
| **Result** | `some`, 1 of 4 on seeds 4, 3, 5, 6 (k = 1, k_clear = 1, n = 4, 0 bad controls), stopped because ≥ 3 was out of reach, so no Clopper–Pearson bound. Outcome counts: `pass_clear` 1 (seed 4), `stuck_draining` 2 (seeds 3, 6, each failing the cold side in one span), `bad_baseline` 1 (seed 5, burst-run baseline pf 0.978 against 0.98). Every control `recovered`; every burst run ended with hit 0.000, KV 0.948–0.952 and the line never emptied. |
| **Prediction held** | yes (likeliest outcome; k = 1–2 ≈ 0.66) |
| **Not separated** | On seeds 3 and 6, a slowly draining backlog against a stuck state that failed a per-span test stricter than `stuck` at λ = 2.0; only a longer watch separates them. |
| **In the paper** | §4.2, P17; §4.1, P43; §4.6, P42 (seed 4's control as a counterexample); §6, P52 (the two near misses); App. F, P51 |
| **Notes** | The futility stop rested on a 0.002 margin: had seed 5's baseline been 0.98 or more and the seed passed, seed 7 would have run. The result is a count on the four seeds run in a fixed order, not a rate for λ = 2.0. Readouts: k_c1 (C1's test alone) 3 of 4; `backlog_persists` 2 of 2. |

[All registrations](README.md)
