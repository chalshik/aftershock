# PW

*Registered 2026-11-25 · a reading of runs already made · claim P0 · result 2026-11-25*

| | |
|---|---|
| **Question** | Under REG-19's per-seed whole-window rule, applied unchanged, how many of the ten burst seeds run at λ = 2.0 with closed-loop turns after that rule was written count a pass: REG-19's five (15–19) and the five long-watch seeds (43, 61, 68, 72, 79), each long-watch pair judged as REG-19 judges its pairs, cut at 4080 s? |
| **Runs** | 0 GPU h: REG-19's ten runs and the ten long-watch runs of REG-24 and REG-27. Not blind: every run's registered outcome was known; the long-watch seeds were screened and seed 43 fixed their design. |
| **Outcome rule** | `pw.table`: REG-19's five seeds are first computed both through the cut view and through REG-19's own code and must agree (labels exact; rates within 1 %, waiting shares within 0.01, emptyings within 1 s), or the reading stops. Each pair is cut at 4080 s; requests planned in [2880, 4080) without a first token before the cut are set all fast and all missed in each run, and a seed whose label moves is `cut_sensitive` and not counted. k counted passes of n per group with one-sided 95 % bounds; the pooled ten only under an exchangeability assumption known not to hold. |
| **Prediction** | 10 of 10 counted with none `cut_sensitive` 0.80; a long-watch seed `cut_sensitive` 0.12; a long-watch seed failing outright 0.05; the check stopping the reading 0.03. |
| **Decision under each outcome** | 10 of 10: the papers may add a sentence giving the long-watch count beside the registered counts, as a retrospective count on screened seeds. Fewer, or any seed `cut_sensitive`: k of n reported as it is, naming each seed; the abstract keeps the registered counts. The check stops: no sentence in the papers. |
| **Adversarial review** | One round (a reading of runs already seen). It showed that the cut biases toward the predicted label (the draft had stated the opposite), that the REG-19 check was not enforced before the long-watch rows, that config, release and cap checks were missing, and that the pooled matrix label had no meaning at n = 10. All were applied, except two partial items, stated in the entry. |
| **Result** | The REG-19 check passed. As recorded, all ten seeds read `pass_clear`; counted 0 of 5 and 0 of 5 (upper bounds 0.451), because every seed, REG-19's own five included, is `cut_sensitive`: with the 1021–1554 late requests of each burst run set all fast, its label becomes `drains`. The bound proved uninformative, and no retrospective count is given. |
| **Prediction held** | no (10 of 10 counted was given 0.80) |
| **Not separated** | A pass does not separate a cold state held by lost reuse from one held by the backlog; the pool is not ten draws of one cell. |
| **In the paper** | §4.2 and Appendix C.4 of the full paper, §4.1 of the workshop version: the test as written passes 10 of 10; the pre-specified count is 0 of 10, its all-fast bound being empty behind a server queue that never empties (rewritten 2026-11-25 after a review). |

[All registrations](README.md)
