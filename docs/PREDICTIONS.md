# Pre-specified probabilities of the outcomes

The probability each pre-specification gave the outcome that occurred, for the cells of the paper's Tables 3, 5 and 6:

| Cell of Tables 3, 5 and 6 | λ | Pre-specified probability of the outcome that occurred |
|---|---|---|
| Burst, 3000 s window (the replication) | 2.0 | ≈ 0.033 |
| Burst, 4500 s window, drop to λ = 1 and return | 2.0 | ≈ 0.26 (seed 43: ≈ 0.62, given a warm control) |
| Turn release: closed loop | 2.0 | ≈ 0.40 (both arms) |
| Turn release: open loop | 2.0 | ≈ 0.40 (both arms) |
| Turn release: open loop | 2.5 | ≈ 0.6 |
| Turn release: open loop | 2.73 | ≈ 0.30 |
| Turn release on the later stack: closed / open | 2.0 | — / ≈ 0.86 |
| Turn release: held until the previous reply | 2.0 | ≈ 0.44 |
| Session cap at each seed's control peak (25–29) | 2.0 | ≈ 0.76 |
| Session cap of 40 | 2.0 | ≈ 0.49 |

The probability is not evidence. Of three outcomes not in Tables 3, 5 and 6: the aging reading ≈ 0.35, on a premise, narrow waits, that the queue-time readout did not bear out; the re-warming reading's prediction ≈ 0.6, which held in none; the band test ≈ 0.29, behind `mixed` ≈ 0.55.

The second table sets each pre-specified outcome beside what the text reports, for the main tests. Labels are the pre-specifications' own, in quotes.

| Test | Pre-specified outcome | What the text reports |
|---|---|---|
| The replication, whole-window test (§4.2) | "supports", 5 of 5 | the same, and beside it 3 of 5 under the earlier per-span test |
| The 8192 burst test (the paper's Appendix B.2) | none claimed: the pre-specified code and the pre-specification's sentence give different labels | counts only, 3 of 5, with no label (P67) |
| The long-window run and its replication (§4.2) | "confirms", 1 of 1; "replicates", 4 of 4 | the same |
| The order test (the paper's Appendix C.2) | "does not follow" | the same: condition 4's order clause is not met |
| The turn-release test (§4.3) | "supports" at λ = 2.0; "headroom weakened" at 2.5 | the same counts; the closed arm's outcomes were known before the open arm ran |
| Its repeat, λ = 2.73 and the third release rule (§4.3) | "replicates"; "headroom excluded"; "default sees" | the same counts; the repeat's closed arm was known when the repeat was pre-specified |
| The idle-age reading (§4.3) | "separated": no idle-age bucket held both arms | the same; no claim about where reuse falls off |
| The session cap (§4.4) | "supports", 5 of 5 cut | the same, against historical uncapped runs |
| The looser cap (§4.4) | "batch bound", 2 of 3 cold | the same |
| The cap inside the state (§4.4) | "mixed": 1 of 3 re-warmed within the pre-specified 240 s | the same, and beside it 3 of 3 before the backlog drained |
| The aging reading (the paper's Appendix B.3) | "supports aging", 3 of 3 | the same |
| The re-warming reading (the paper's Appendix B.3) | "drain bound"; the pre-specified prediction held in none | the same |
| The cache-off burst control (the paper's Appendix B.3) | "drains", 2 of 2 counted | the same; it says nothing about whether the cache is necessary |
| The surge test and the band test (the paper's Appendix B.4) | "supports"; "band warm" | the same |
| The early-warning reading (the paper's Appendix B.5) | "no usable cache lead" | the same |
| The four post-review readouts (§3, §4.3, §7; the paper's Appendix B.4) | readouts, no outcome | one pre-specified sentence was replaced because it did not describe the data, and one was made exact (P68, P70) |

Of the 30 outcome predictions pre-specified with a written prior, the outcome that occurred was the likeliest one in 21. It was the second or lower in 8, among them the turn-release test at λ = 2.0 (`supports` ≈ 0.40 against `inconclusive` ≈ 0.49), the long-window replication (`replicates` ≈ 0.26 against `mixed` ≈ 0.62) and the band test of the paper's Appendix B.4 (`band_warm` ≈ 0.29 against `mixed` ≈ 0.55). It was tied in one. The mean prior of the outcomes that occurred was 0.49 (P33). The replication's 5 of 5 had been given ≈ 0.033, so the pre-specified per-seed model (about 0.55 a seed) was miscalibrated; how often the whole-window test would fail a run that drains slowly was not measured.
