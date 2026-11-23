# REG-4

*Registered 2026-11-04 · a verdict-code change · result 2026-11-04*

| | |
|---|---|
| **Question** | A verdict-code change: values a window cannot give become NaN, and a verdict branch that needs one gives `unclear` instead of falling through. Does it change any recorded verdict or level? |
| **Runs** | 0 GPU h. One recheck, before and after the change, of every run with a full verdict block (`tables.verdict_table`, invalid runs included), the 26 ramp steps (`tables.ramp_tables`) and the two REG-2 rows (`mechanisms.reg2_table`). |
| **Outcome rule** | Change to `load._level`: every server value is NaN when the scrapes do not cover the window (no scrape within `checks.max_scrape_gap_s`, 2.0 s, of each end, or a larger gap inside); hit is NaN when the query counter does not rise or a cache counter goes down. Change to `load.verdict`, in order: base pf NaN → `unclear`; base pf < 0.98 → `bad_baseline`; any of end pf, end hit, end q, base hit, base q NaN → `unclear`, else the unchanged `recovered` test; any sign of recovery true → `slow_drain`, a sign that cannot be computed → `unclear`, else the unchanged `stuck` test, else `unclear`. Every threshold and window unchanged. The recheck compares labels exactly and levels NaN-aware to 1e-9 with explicit NaN positions. |
| **Prediction** | 1. All 43 verdicts from `tables.verdict_table` are unchanged, and so are all their levels. 2. `20261102-093229_wipe_preempt_r3_s3` (invalid, stopped at about 1618 s) changes from raising to `unclear`. 3. The 26 ramp steps and the REG-2 rows are unchanged. |
| **Decision under each outcome** | A correctness fix, so the change stands under every outcome; the recheck is an impact log. No valid verdict changes: the audit items it addresses close. A valid verdict changes: a result names the run with its old and new verdict and every published statement that used it, and a correction is logged for each. A ramp or REG-2 value changes: logged the same way, REG-2's labels restated if they change. Known consequence: a total stall that admits nothing is `unclear`, not `stuck`. |
| **Adversarial review** | One round of adversarial review on the first draft and its diff. Its traces (a stall turning `unclear`, a NaN previous window blocking `recovered`, the NaN check hiding `bad_baseline`, a half-covered window judged without warning, interpolation across a gap) led to branch-aware NaN handling, a coverage rule, NaN on counter resets, gauges skipping NaN rows, a NaN-aware recheck and new tests, all before the recheck. |
| **Result** | No valid verdict changed. 9 differences, all in `20261102-093229_wipe_preempt_r3_s3` (invalid): before it raised; now its baseline is computed (pf 1.000, hit 0.652, q 0.0, 6 preemptions) and its verdict is `unclear`. Every verdict and level of the other 43 rows, all 26 ramp steps and both REG-2 rows are unchanged; the valid runs are 37 `recovered` and 2 `bad_baseline`. |
| **Prediction held** | yes (all three registered predictions hit; they are exact checks without probabilities) |
| **In the paper** | Not named in the text (Appendix A: a verdict-code change; no valid verdict changed). |

[All registrations](README.md)
