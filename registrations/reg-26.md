# REG-26

*Registered 2026-11-14 · a reading of runs already made · result 2026-11-14*

| | |
|---|---|
| **Question** | How much of the client's time to first token lies outside the server's own TTFT, and does it move the share of requests under the 1.0 s SLO that percent fast counts? |
| **Runs** | Retrospective, 0 GPU h: the 62 valid, unsealed `*l1pl350*` runs (`lag.A5["runs"]`); one invalid control and one run without `checks.json` left out. |
| **Outcome rule** | `lag.label` over each run's whole span: `gap` = client mean TTFT − the server's Δsum/Δcount; `share_shift` = the server's share at or under 1.0 s − the client's share at or under 1.0 s. `flag` if any run has `gap` > 0.10 s, `share_shift` > 0.01, or either NaN; `no_data` if no run; `small` otherwise. 60 s windows are a readout and never set the label. |
| **Prediction** | Largest whole-span `gap` ≤ 0.02 s, median ≤ 0.01 s; largest `share_shift` ≤ 0.005; P(`flag`) ≈ 0.15 (BELIEF). |
| **Decision under each outcome** | `small`: the question is answered for L1, and no verdict is touched; `flag`: it stays open, the runs over a line are listed as an open risk, and the author decides on a follow-up; `no_data`: open, with the reason. |
| **Adversarial review** | One round: NOT READY (items R1–R8); all handled afterwards and not reviewed again. |
| **Result** | `small`: largest whole-span `gap` 0.0172 s, median 0.0068 s; largest `share_shift` 0.0044, median 0.0; no NaN. One run had a single window gap of 0.1255 s (a readout, where the server's mean TTFT rose 4.98 s), within the line over its whole span. |
| **Prediction held** | yes (likeliest outcome; `small` ≈ 0.85, derived from P(`flag`) ≈ 0.15) |
| **Not separated** | Whether the time is server-side work after the output handler or the client: not separable, so the result is quoted as time outside the server's span, never as client lag. |
| **In the paper** | not in the paper (Appendix A lists REG-26 as not named in the text) |
| **Notes** | Approved under the author's standing instruction to close the open questions without asking, not by the author's own review of the text. It bounds neither per-request lag nor a single window's percent fast. |

[All registrations](README.md)
