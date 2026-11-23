# REG-23: the early-warning reading

*Registered 2026-11-14 · a reading of runs already made · claim K5 · result 2026-11-14*

| | |
|---|---|
| **Question** | When the warm server goes cold with no trigger, do the hit rate (H) and the recomputed prefill rate (U) move before latency (L) and the waiting line (Q), with few false alarms, and earlier than the arrival rate (A) moves? |
| **Runs** | Retrospective, 0 GPU h: 19 closed-loop L1 350 W runs without a trigger made before REG-19 (the controls of REG-11, REG-12, REG-14, REG-16, REG-17 and REG-18, and REG-13's cache-on sweep); 18 valid, of which 6 events (the KV pool fills, `gate.kv_onset`) and 12 quiet runs (5.87 h). |
| **Outcome rule** | `k5.judge`: onsets on a 5 s grid of 30 s trailing means, at levels fixed before computing; per signal and comparator `too_few` (< 5 events), `leads` (every event ahead by more than 30 s), `no_lead` (median ≤ 0), `within_window` (median above 0 and ≤ 30 s), else `mixed`. A signal is usable if it is ahead of both L (the 1.0 s TTFT SLO) and the queue, with a one-sided 95 % Poisson upper bound on false alarms below 1 per hour and an alarm share below 0.05. K5's label: `too_few_events` if both H and U have too few events; `cache_signal_leads` if H or U fails nothing; `leads_slo_only` if H or U fails only `slo_only`; `no_usable_cache_lead` otherwise. |
| **Prediction** | `no_usable_cache_lead` ≈ 0.8; `too_few_events` ≈ 0.12; `leads_slo_only` ≈ 0.04; `cache_signal_leads` ≈ 0.04 (BELIEF). |
| **Decision under each outcome** | `cache_signal_leads`: K5 may stay, worded as a retrospective measurement on the overload and surge population; `leads_slo_only`: K5 stays only reworded to "before the mean TTFT passes the 1.0 s SLO"; `too_few_events`: K5 leaves the claims as not testable on the runs on record; `no_usable_cache_lead`: K5 leaves the claims, with a limitations sentence fixed in advance for the first failing reason. |
| **Adversarial review** | One round on draft 4: NOT READY with six required changes (CA1–CA6) and eight further conditions (CB1–CB8), all answered in draft 5 before registration. Draft 6 added four fixes that were not reviewed again. |
| **Result** | `no_usable_cache_lead`; the first reason for both H and U is `latency:mixed`. Over the 1.0 s TTFT breach H was ahead by more than 30 s in 3 of 6 events (median 37.5 s) and U in 4 of 6 (median 42.5 s); over latency's first move (Lr) 0 of 6 for both (`no_lead`). False alarms over 5.87 h: H 8 (upper 2.46/h), U 13 (upper 3.52/h). The sent arrival rate crossed 3.2 req/s in 0 of 6 events. |
| **Prediction held** | yes (likeliest outcome, ≈ 0.8) |
| **Not separated** | Overload rather than a tip (every event followed planned surges of 3.35–3.71, five of six at λ ≥ 2.4; separated only by collapses below capacity without a surge, none on record); whether arrivals moved first (the sent rate is clipped by closed-loop release, while the planned peaks did cross); hit-rate counting in a backlog after the queue's onset. |
| **In the paper** | The early-warning reading: §4.6 (P45; the six collapses also in P42), Appendix C.17; the running batch on the same events is a later readout (P69). |
| **Notes** | Registered departures from the claim's stated test: burst runs and critical slowing down are not computed. The computation's freeze check printed the output's own new folder, and the label lines were read before that was noticed; the author ruled that the output stands. As registered, K5 left the claims. |

[All registrations](README.md)
