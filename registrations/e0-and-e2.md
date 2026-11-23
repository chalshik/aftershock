# E0 and E2: the capacity sweeps

*Registered 2026-11-14 · GPU runs · claim K2 · result 2026-11-16*

| | |
|---|---|
| **Question** | Does the window from the cold floor to the warm capacity widen from 8 to 16 turns per session, with the warm side gaining in realized planned rate? It decides whether the 16-turn shock matrix (E3, 8–12 GPU h) is registered. |
| **Runs** | E2: two 16-turn closed-loop sweeps at L1, 350 W, seed 0, one run per rate: cache off (`sweep_l1t16pl350cl_pc_off`) at 1.0, 1.5, 2.0, 2.5 and 3.0 req/s, then cache on (`sweep_l1t16pl350cl_pc_on`) at 2.0, 2.2, 2.4, 2.6, 2.8, 3.0 and 3.2, with one flag (`k2.flag16`) between them; 12 runs, ≈ 4.2 GPU h. E0 (0 GPU h) reads them against REG-13's 8-turn cache-on sweep and E1's 8-turn cache-off sweep. |
| **Outcome rule** | `k2.decide` and `k2.observed`: R = (warm₁₆ / floor₁₆) ÷ (warm₈ / floor₈) against g = 1.083, with both floors by `e1.floor` over the saturated rates the two sweeps share (`k2.shared_floors`). "widens": R ≥ g and warm₁₆'s realized planned rate above that of the 8-turn grid rate just above warm₈ (on seed 0, 2.856 > 2.758); "floor_only": R ≥ g but the warm clause fails; "does_not_widen": R × next₁₆/warm₁₆ ≤ 1; "unresolved": any other case below g; "missing": a needed input is undefined (including a hole in the 16-turn chain). Only "widens" opens E3's registration. |
| **Prediction** | "widens" ≈ 0.35; "floor_only" ≈ 0.35; "does_not_widen" ≈ 0.05; "unresolved" ≈ 0.1; "missing" ≈ 0.15 (BELIEF). The token-cost models gave warm₁₆/warm₈ 1.070 (fixed cost) or 1.319 (proportional), floor₁₆/floor₈ 0.846 or 0.800, and R 1.281 / 1.380 or 1.666. |
| **Decision under each outcome** | Only "widens" opens the 16-turn shock matrix (E3); every other label stops K2 at ≈ 6.3 GPU h (E1 + E2), with "floor_only" and "does_not_widen" written up as K2 results. A "widens" resting on the named confounds goes to the author, who decides on an 8-turn cache-on 2.4/2.6 rerun (≈ 0.7 GPU h) before E3 is registered. |
| **Adversarial review** | Two rounds; round 2 (NOT READY, nine items) was applied before registration, items 1–3 by the author's choice, with no third round. The later tip-rule amendment had one round of its own (T1–T6), applied before E2 ran. |
| **Result** | "floor_only": warm₈ = 2.4, warm₁₆ = 2.2 (tip rule); floor₈ = 1.758, floor₁₆ = 1.400; warm ratio 0.917, floor ratio 0.797, R = 1.151 ≥ g = 1.083; the warm clause fails on both parts (planned 2.252 against 2.758, released 2.118 against 2.470). The window widened only because the cold floor fell; no E3 registration, and K2 stops at ≈ 6.3 GPU h (seed 0). |
| **Prediction held** | Tied: "floor_only" ≈ 0.35 was tied with "widens" ≈ 0.35 in the registration; under the competing forecast entered with the amendment before E2 ("floor_only" ≈ 0.2) it ranked third. The cost models' warm₁₆ of 2.6–3.2 missed (2.2). |
| **Not separated** | Why the warm side fell: KV room (peak KV 0.921 at 16 turns against 0.611 at 8, both at 2.0) against compute; one seed with a 900 s survival edge; code and day; unpaired schedules (weakened by the surge readout, not excluded). |
| **In the paper** | §4.1 and §6, K2 (P25, P26); Appendix C.9 (longer sessions); the 16-turn warm capacity row of Appendix D's table (P32). |
| **Notes** | Amended 2026-11-15, before E2 ran: warm capacities by the tip rule (a cache-on run passes iff valid, queue slope ≤ 0.01 and no `gate.kv_onset` anywhere in the run), the ratio rule reported only, and "widens" also needs warm₁₆'s released rate above 2.470. All twelve E2 runs were valid on their first attempt, and the flag printed {"floor16_defined": true}. One seed, one run per rate; no bound is computed for a capacity. |

[All registrations](README.md)
