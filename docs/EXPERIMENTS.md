# Experiment names and their ids in the log

The text names experiments by what they test. The study's record, an append-only log (`docs/log`), files each under an id; this table maps one to the other. Each entry there holds the pre-specification, its adversarial review and its result.

| Name in the text | Claim | Id(s) in the log | Runs |
|---|---|---|---|
| the earlier matrix (per-span test) | P0 | REG-17, REG-18 | 5 burst–control pairs at λ = 2.0, 1500 s window: REG-17's seed 2 and REG-18's seeds 3–6 (seed 7 not run after the futility stop) |
| the replication (whole-window test) | P0, C1 | REG-19 | 5 burst–control pairs, seeds 15–19, 3000 s window |
| the long-window run | C1 conditions 5–6 | REG-24 | 1 burst–control pair, seed 43, 4500 s window, drop to λ = 1 and return |
| the long-window replication | C1 conditions 5–6 | REG-27 | 4 burst–control pairs, seeds 61, 68, 72, 79, the long-window run's design |
| the wipe run | C1 condition 4 | REG-25 | 1 wipe–control pair, seed 44 |
| the turn-release test | K1 | REG-20 (arm A at λ = 2.0, arm B at λ = 2.5) | 5 + 3 open-loop burst runs beside the replication's closed-loop runs |
| the same-stack repeat, λ = 2.73, the default-release arm | K1 | TR (arms O, H, D) | 15 burst runs, seeds 15–19, each with a no-burst control; the closed-loop repeat is T1's arm U |
| the idle-age reading | K1 | M | no new runs: the replication's and the turn-release test's burst runs |
| the capacity sweeps | K2 | E1, E2, gate E0 | cold and warm sweeps at 8 and 16 turns, seed 0 |
| the token-work readings | K3 | REG-21 | retrospective, on runs already seen |
| the four-links reading | C2 | REG-22 | retrospective, on runs already seen |
| the served-mix test | K3 | K3b | the surge test's four runs that went cold |
| the session cap | C2 | D1 | 5 capped burst runs, seeds 15–19, cap at each seed's healthy peak |
| the cap inside the state | C2 | T1 (arm P; arm U uncapped) | 3 paused-cap runs, seeds 15, 17, 19, cap 22 from 1980 s; 5 uncapped runs, seeds 15–19 |
| the looser cap | C2 | D1b | 3 capped burst runs, seeds 15–17, cap 40 |
| the aging reading | C2 | B1 | retrospective, on the looser cap's runs |
| the surge test | K4 | K4 | 10 no-burst runs at λ = 2.2, 5 with a planned surge and 5 without |
| the band test | K4 | K4e | 10 no-burst runs at λ = 2.2 with 120 s peaks of 3.1–3.4, and K4's seeds 137 and 103 rerun as controls |
| the early-warning reading | K5 | REG-23 | retrospective, six collapses without a trigger |
| the 8192 burst test | P0 | P8192 | 5 burst–control pairs at λ = 2.2 and `--max-num-batched-tokens` 8192, seeds 1002, 1004, 1007–1009 |
| the cache-off burst control | C2 | T2 | 3 burst–control pairs with caching off at λ = 0.80, seeds 15–17 |
| the token-budget sweep | — | T4 | 24 capacity-sweep runs, caching off and on, at `--max-num-batched-tokens` 2048 and 8192, seed 0 |
| the second-schedule sweep; the development-mode twin | — | G1 and G2 (one registration) | 12 capacity-sweep runs on seed 1 (G1); 4 cache-off runs at λ = 3 and 4 with development mode on and off (G2) |
| the session-share test | K3 | REG-28 | the long-window replication's four burst runs |
| the re-warming reading | C2 | REG-30 | retrospective, the five long-window burst runs and their controls |
| the four post-review readouts | — | REG-31 | readouts on runs already seen and on schedules alone, never judged |
| the server-side batch bound | C2 | B3's gate | REG-19's five warm controls; no run (B3 closed) |
| the surge and keep-up reading | — | no registration (A28; P66) | a reading of seen runs, the twelve cache-on 2048-token runs of T4 and G1 |
| not named in the text | — | REG-1 | a reading of earlier runs; inconclusive (no qualifying span) |
| not named in the text | — | REG-2 | a reading of two earlier control runs; both control dips arrival-driven |
| not named in the text | — | REG-3 | dropped before computation |
| not named in the text | — | REG-4 | a verdict-code change; no valid verdict changed |
| not named in the text | — | REG-5 | not computed |
| not named in the text | — | REG-6 | not run; repeated at 350 W as REG-9 |
| not named in the text | — | REG-7 | a feasibility gate; not computed, superseded by REG-8 |
| not named in the text | — | REG-9 | no change |
| not named in the text | — | REG-26 | a reading of runs already seen; small |
| not named in the text | — | REG-29 | withdrawn |
| the cap as a rescue, first design | C2 | D1r | shelved, not pre-specified; no run (the rescue design that ran is the cap inside the state, T1) |
