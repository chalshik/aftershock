# M

*Registered 2026-11-23 · a reading of runs already made · claim K1 · result 2026-11-23*

| | |
|---|---|
| **Question** | After the burst, is later-turn prefix reuse set by the idle age in units of the free queue's turnover time (x = tokens allocated since the session's previous turn ended, over the free pool), the analogue of an LRU cache's characteristic time, in both turn-release arms? |
| **Runs** | 0 GPU h: REG-19's five closed-loop and REG-20 arm A's five open-loop burst runs at λ = 2.0, after the burst; REG-19's no-burst controls as a check on the free-pool measurement. |
| **Outcome rule** | Buckets of x: held, neg_A, [0, .5), [.5, 1), [1, 2), [2, 4), [4, 8), [8, 16), [16, ∞). A bucket is matched with ≥ 100 turns in each arm and ≥ 20 in each of ≥ 3 open runs. First match: `undefined`; `closed_not_outside` (< 0.9 of closed turns at x ≥ 1, or their reuse > 0.1); `arm_differs` (a matched bucket with reuse differing by ≥ 0.3); `separated` (no matched bucket); `common_curve` (matched buckets within 0.3 and the open [1, 2) bucket reusing ≥ 0.5 less than its [0, .5)); `agree_outside` (matches only at x ≥ 2); `mixed`. |
| **Prediction** | `separated` 0.46, `agree_outside` 0.28, `arm_differs` 0.12, `closed_not_outside` 0.05, `undefined` 0.03, `common_curve` 0.03, `mixed` 0.03. Registered with the statement that the closed side was known in advance from the runs already seen, so the reading tests the free-pool measurement, not the premise, and that matched against unmatched mostly measures the shock's size. |
| **Decision under each outcome** | `closed_not_outside`: the paper's characteristic-time sentence is withdrawn until resolved. `arm_differs`: it becomes 'consistent with', naming the other difference. `common_curve`: kept, with the figure. `agree_outside` or `separated`: kept, worded without a crossing claim. `mixed` or `undefined`: no change. |
| **Adversarial review** | One round and a confirmation. Round 1: four blocking items (negative ages filed as held, the crossing test pooled all x ≥ 1, the already-known closed side unstated, the prediction built on a wrong open-arm KV) and five minor ones, all applied; the confirmation found no new blocking item. |
| **Result** | `separated`: 21,695 closed later turns, all at x ≥ 8 (23 in [8, 16), 21,672 at ≥ 16), reuse 0.000 (5 of 5 runs, lower bound 0.549). No bucket matched: the open arm had 24 and 35 turns at x ≥ 8. Open profile (never judged): reuse 1.000 at x < 0.5 (13,267 turns), 0.934 in [.5, 1) (82), 0.467 in [1, 2) (18), 0.053 in [2, 4) (37), 0.022 in [4, 8) (41), 0.000 beyond. No-burst controls: 0.999–1.000 at x < 0.5, 0.54–0.63 in [.5, 1). |
| **Prediction held** | yes (likeliest outcome, 0.46) |
| **Not separated** | Idle age against the turnover time, against the working set exceeding the pool: not separated, as registered. |
| **In the paper** | §4.4 of the full paper; §5 of the workshop version (applied 2026-11-25) |
| **Notes** | The open profile's fall across x ≈ 1 rests on 18–82 turns per bucket, below the registered 100, so no crossing is claimed. |

[All registrations](README.md)
