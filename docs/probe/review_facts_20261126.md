# Sourced facts for the paper corrections (reviewer items 1–9), 2026-11-26

Read-only pass. No file edited except this one; no pytest; `runs/` only read. Schedule- and run-only computations made in this pass are marked "computed 2026-11-26, not pre-specified" and use the repo's own functions (`.venv/bin/python`, read-only). Commit times are converted to UTC from git's +06:00.

## 1. Software stacks per arm

Fields come from each run's `env.json` (`packages.vllm`, `packages.torch`, `torch_cuda`, `cuda_driver`, `platform`). libc is **not** in `env.json`: `platform` reads `glibc2.39` in all 352 folders. The libc6 package build is DERIVED from the run's start time against the install time in decisions.md 2026-11-15 [CORR] "Automatic updates off" … (FILE(docs/decisions.md:20889-20895)): "On **2026-11-13 at 15:47:51–15:48:34 UTC**, `/usr/bin/unattended-upgrade` installed **linux-image 6.8.0-149** … and **libc6 2.39-0ubuntu8.9 → 8.12**. The driver (595.99.02) did not change"; "libc6 8.12: every process started after 15:48 on 2026-11-13". The kernel loaded at the 2026-11-15 boot (decisions.md 2026-11-15 [OPS] REG-20 arm A: session probe …, FILE(docs/decisions.md:20763)). The update timers were disabled on 2026-11-15 at about 07:44 UTC (FILE(docs/decisions.md:20912)). No later apt-history probe exists, so libc 8.12 after 2026-11-15 is BELIEF backed by the timers being off. No probe re-read the package version.

All arms below share vLLM 0.32.0, torch 2.13.0, CUDA (torch) 13.0, NVIDIA driver 595.99.02, an RTX 3090 at 350 W and uv.lock sha 8b1c04b4. The differences are the kernel, libc and the harness commit (RUN(each folder, env.json); tallied this session).

| Arm (runs) | Kernel | libc6 (derived) | Harness commit | Spans both? |
|---|---|---|---|---|
| (a) REG-19 closed main, s15–17 burst and control (`20261112-*`) | 6.8.0-147 | 8.9 | a7bb77a | — |
| (a) REG-19 closed main, s18–19 burst and control (`20261113-165516`, `-180511`, `-192110`, `-203108`) | 6.8.0-147 | **8.12** (started after 15:48 on 11-13) | a7bb77a | **the replication spans the two libc builds**; one kernel |
| (b) arm A open, `20261115-*_ol_r2.0_w3000_s15..19` (no suffix) | 6.8.0-149 | 8.12 | 1cface7 | — |
| arm B open at 2.5 (`20261116-0[456]*_ol_r2.5_*`) and the drift run `20261115-101850_…_s15_drift` | 149 | 8.12 | 1cface7 | — |
| (c) later-stack open repeat `_t1o`, 5 bursts and 5 controls (2026-11-23 to 11-25) | 149 | 8.12 | 4aae03d | — |
| (c) the matching closed repeat: T1's uncapped twin `_t1u`, 5 bursts on s15–19 (2026-11-23); also T1's capped `_t1p`, s15, 17, 19 | 149 | 8.12 | a108eb9 | — |
| (d) the held arm (TR arm D): `*_burst_l1pl350dr_r2.0_w3000_s15..19` with `stage2_control_l1pl350dr_*` (no `_trd` suffix) | 149 | 8.12 | 4aae03d | — |
| (d) open at 2.73 `_trh`, 5 bursts and 5 controls | 149 | 8.12 | 4aae03d | — |
| (e) cap `_cap` (5, 2026-11-16) and cap40 `_cap40` (3, 11-16 and 11-17) | 149 | 8.12 | e7503a7 / 9045d9f | — |
| (f) long window: controls `*_long_s{43,61,68,72,79}`, bursts `*_confirm_l1pl350cl_r2.0_s{43,61,68,72,79}` | 149 | 8.12 | 097017a (s43), 16f8cc7 (others) | — (E16: all ten "Linux-6.8.0-149…", FILE(docs/decisions.md:25184-25187)) |
| E1 cache-off sweep (`20261114-0*_sweep_l1pl350cl_pc_off_r*`, for item 3) | 147 | 8.12 | ce5f842 | — |

Notes:
- The "later software stack" of the turn-release repeat (kernel 149, libc 8.12) is the **same kernel and libc as arm A**. What changed between arm A (11-15) and `_t1o` (11-23) is the harness commit (1cface7 → 4aae03d) and the date. The closed arm's change from REG-19 to `_t1u` is kernel 147 → 149, with libc changing for seeds 15–17 only, and harness a7bb77a → a108eb9. decisions.md 2026-11-23 [REG] T1 says the same: "The libc change is checked for seeds 15–17 only" (FILE(docs/decisions.md:24908)).
- Two lost attempts at REG-19's seed-18 control (`20261112-151030`, `20261113-033842`) have no `checks.json` and ran on 147 with libc 8.9.
- **Possible error in the text.** §3 (FILE(docs/paper/sections/04-method.md:72)) says "Every open-loop and held burst run has its own no-burst control." That holds for `_t1o`, `_trh` and `dr` only. Arm A's five runs and arm B's three have no control. REG-20 made the controls conditional ("run only for an open seed whose burst run is cold-backlogged", FILE(docs/decisions.md:4054)), and K1 arm A's result says "No control ran: no open run was cold, so none was due" (FILE(docs/decisions.md:21067)). T1's `_t1u` closed runs also have no controls of their own.

## 2. Seed screening of the long-window seeds

- **Seed 43, REG-24's rule** (`confirm.seed_rule`, frozen 2026-11-13 23:23:39 UTC). Candidates are seeds 40–199 that no config uses, tried in seed order, each at drops 3600, 5400, 7200 and 9000 s. Seeds 40, 41 and 42 were examined and rejected; "Seeds 40, 41 and 42 fail W at every drop, on their baselines". Seed 43 failed W at 3600, 5400 and 7200 s and passed at 9000 s. "**W alone decides the pick.**" FILE(docs/drafts/reg24_confirm_draft1.md:105-153). Post-pick scan at 9000 s over 160 seeds: W passes 7, E passes 144 (so 16 fail the peak), F passes 160, all three 7 (FILE(docs/drafts/reg24_confirm_draft1.md:155-162)).
- **Seeds 61, 68, 72 and 79, REG-27's rule** (`reg27.picks`, committed at fd6a10e before any candidate schedule was computed). PROBE(docs/probe/reg27_draft/picks_fd6a10e_20261117T1954Z.txt), header: "candidates 149: 40..199; used_seeds in 40-199 now: [44, 102, …]", ending "PICKS (seed, tilt): [(61, 0.083), (68, 0.11), (72, 0.003), (79, -0.007)]".
  - **38 seeds examined** (40–79, without 43, which the rule leaves out, and 44, already used). **34 rejected, 4 taken** (DERIVED: count of the probe's 38 seed lines).
  - Of the 34 rejections, 33 fail **W only**. One, seed 74 (peak 3.367), fails W and E. F passes in all 38.
  - So the 3.2 req/s peak criterion alone rejected no seed that W would have admitted. The screen that bit was "planned load within 5 % of λ in each judged window". Appendix A.3's table lists W first, but §3/§4.2 describe the screen mainly as "no planned 120 s peak of 3.2 or more".
  - The picks' peaks are 2.725–3.000 (FILE(docs/decisions.md:23109)).
- **Seeds 15–19 against the same screen.** Computed 2026-11-26, schedule only and not pre-specified, with `harness.workload.build_schedule` on `burst_l1pl350cl_r2.0_w3000_s{15..19}`, session rows, `gate.rolling_rate` from 240 s. REG-19's runs have no drop or return, so only W's baseline (300–900 s) and end (3480–4080 s) windows exist, and F does not apply.

  | seed | peak 120 s (at) | E (< 3.2) | baseline λp | end λp | W (1.90–2.10) |
  |---|---|---|---|---|---|
  | 15 | 3.092 (2635 s) | pass | 1.958 | 1.992 | pass |
  | 16 | 2.875 (3465 s) | pass | 1.970 | 2.003 | pass |
  | 17 | 3.017 (2930 s) | pass | 1.972 | 1.902 | pass (at the edge) |
  | 18 | 2.908 (385 s) | pass | **2.217** | 1.915 | **fail** |
  | 19 | 2.833 (3085 s) | pass | **2.207** | **2.207** | **fail** |

  The peaks match REG-19's registered inputs: "No control reaches `surge_tip` 3.2 within 4080 s (highest 3.092)" (FILE(docs/decisions.md:3782-3788)). So the main runs pass the peak screen (highest 3.09, as in Appendix A.3), but seeds 18 and 19 would fail its load-window part. The peak criterion also covered a shorter span: 4080 s for REG-19 against about 15,000 s for the long-window runs.

## 3. The three all-miss values

| Value | What it is | Runs | Function | Source |
|---|---|---|---|---|
| **1.798** (`cap.floor_closed`, "1.80") | Closed-loop cache-off "cold floor": the mean throughput over the saturated rows | E1's six runs `20261114-021512 … -035448_sweep_l1pl350cl_pc_off_r{1.5,…,4}` (seed 0, 900 s each, one run per rate). Rows used: 2.5, 3.0, 3.5, 4.0 → 1.718, 1.797, 1.827, 1.848 | `e1.floor` → `gate.cold_floor` with REG-8's settings | FILE(docs/paper/numbers.toml:15-18); decisions.md 2026-11-14 [RESULT] E1 (FILE(docs/decisions.md:20230)); PROBE(docs/probe/20261114_e1_result/report_head.txt) |
| **1.706** ("1.71") | Throughput of E1's single λ = 2.0 run (planned 2.097 req/s, KV mean 0.755) | `20261114-023230_sweep_l1pl350cl_pc_off_r2` | `e1.observed` (`rows_cl`) | P47; numbers.toml `margin.matched`; [RESULT] E1: "E1's own 2.0 row completed 1.706 req/s at a mean KV of 0.755, below the 0.85 the rule needs" |
| **≈ 1.75** | Completion rate of the cold, backlogged burst runs: a cold-state rate, not a cache-off capacity | REG-19's 5 bursts: 1.740–1.758 (`gate.reg19_table`, `reg19.stuck_lo/hi`). REG-27's 4 long-window bursts: 1.744–1.765 (`reg27.table`, `lw.rate_lo/hi`, P53) | as named | numbers.toml:36-43, 260-266; §4.2 "it completed about 1.75 req/s where about 2.0 were planned" |

- **What "saturated" means in code.** `gate.cold_floor`: "mean KV ≥ 0.85, at least 2 runs, throughput < 0.95 × planned rate"; λ_sat is the lowest rate from which rows count, here 2.5 (decisions.md 2026-11-14 [REG] E1, "What is measured"). Throughput is completed requests with `t_end` in [240 s, end of arrivals) per second (`load.summarize`; docs/definitions.md μ_off/throughput). The floor is a mean over a rising row set (slope +0.084 req/s per req/s, SE 0.019).
- **Which value the 1.17 uses.** It uses 1.706: DERIVED 2.0 / 1.706 = 1.172 (numbers.toml `margin.matched`; P47). 1.11 is 2.0 / 1.798 (`margin.closed`, P1). The 1.17 puts the nominal λ (2.0) over a run whose planned rate was 2.097. Planned over completed for that run is 2.097 / 1.706 = 1.229 (DERIVED).
- **Which value the "56 %" uses.** The 1.798 floor: DERIVED 1.0 / 1.798 = 0.556. Against 1.706 it would be 59 %, and against the open-loop 1.681 it would be 59.5 %. §4.2's sentence (FILE(docs/paper/sections/05-results/02-cold-state.md:13)) carries no tag naming the denominator.
- Side observation: the cold burst runs (1.74–1.77) complete more than the cache-off engine's own λ = 2.0 run (1.706) and its 2.5 row (1.718). The floor rises with offered rate. Nothing measures why.

## 4. Warm capacity, closed loop

- **The rule** (`load.capacity`, monotone; `t4.warm_capacity` for T4/G1): the highest swept rate whose valid run, and every lower swept rate's run, has throughput ≥ 0.95 × planned rate and waiting-queue slope ≤ 0.01. Settings: FILE(configs/l1_base.yaml:10-14) (`min_throughput_frac: 0.95`, `max_q_slope: 0.01`, `against: planned_rate`, `monotone: true`); docs/definitions.md μ_warm. Each run is 900 s of arrivals, with a 240 s warm-up excluded and a 300 s drain (FILE(configs/base.yaml:29-30,43)). One run per rate, one seed per sweep.
- **Seed 0, 2.4 (REG-13).** Grid 2, 2.4, 2.6, 2.8, 3, with no 2.2 (FILE(configs/sweep_l1pl350cl_pc_on.yaml:13)). Result μ = 2.4 by the configs' rule, with 2.4 "a ratio of 0.9503 against 0.95, at the edge". The registered test, which added a due slope, gave `overload` (decisions.md 2026-11-10 [RESULT] REG-13, FILE(docs/decisions.md:2635-2650)); P7.
- **Seed 0, 2.2 (T4's 2048 arm, 2026-11-21).** Grid 2.0–3.0 in 0.2 steps (FILE(configs/sweep_t4_l1pl350cl_pc_on_b2048.yaml:12)). At 2.4 "throughput over planned is 0.9497 … 0.0003 under 0.95"; "one grid step lower … by the same edge row" (decisions.md 2026-11-21 [RESULT] T4, FILE(docs/decisions.md:23951-23953)); P62.
- **Seed 1, 2.8 (G1, 2026-11-22).** Same grid. "Seed 1 kept up at every rate from 2.0 to 2.8 with throughput 0.995–0.999 of planned, then failed at 3.0 (0.616 …)" (decisions.md 2026-11-22 [RESULT] G1, FILE(docs/decisions.md:24311)); P65.
- So the workshop's "2.2 to 2.8" is these three sweeps. The seed-0 values 2.2 and 2.4 differ by 0.0006 of throughput share at one row.
- A28, read afterwards (P66): 11 of 12 cache-on rows kept up exactly when the schedule had no 120 s planned peak ≥ 3.4.

## 5. "Requests outstanding at the burst's end"

- **Function.** `gate.shock_size` (FILE(analysis/gate.py:347-355)): "`outstanding`: session requests (no burst rows) due by the burst's end (t_released, else t_planned) and not ended by then".
  - It **excludes the burst's own requests.**
  - In closed loop it also **excludes turns not yet released**, because a later turn is due only once released.
  - It is called by `d1.py:676` (P38, the cap numbers 24–29 against 43–64; decisions.md 2026-11-16 [RESULT] D1, FILE(docs/decisions.md:22064)) and by `reg31.in_service_table` (P70).
- **Values.** PROBE(docs/probe/reg31/in_service_table_f41e8c8_20261122T150521Z.txt), P70:
  - closed s15–19: outstanding 64, 43, 52, 52, 45 (sessions in service 66, 44, 53, 55, 47);
  - **open arm A s15–19: outstanding 238, 152, 182, 199, 131** (sessions 38, 25, 33, 32, 25);
  - Appendix B gives 197–340 for the 2.5 and 2.73 open arms (P24, P75).
- **Why Figure 1 shows about 235.** Computed 2026-11-26, not pre-specified (`load.load_run`, `gate.shock_size`, metrics nearest 1080 s):

  | Runs (s15…s19) | `outstanding` | session requests planned ≤ end, not ended | burst requests not ended | server waiting / running |
  |---|---|---|---|---|
  | closed, uncapped | 64, 43, 52, 52, 45 | 366, 246, 302, 329, 246 | 211, 212, 205, 203, 192 | 244/31, 222/30, 224/33, 223/32, 204/31 |
  | closed, cap at control peak | 29, 24, 28, 26, 25 | 375, 246, 306, 341, 252 | 205, 215, 202, 196, 186 | 203/31, 207/31, 198/32, 191/31, 179/30 |
  | open, arm A | 238, 152, 182, 199, 131 | same as outstanding | 218, 228, 239, 238, 220 | 414/39, 343/34, 383/37, 393/43, 316/35 |

  - Check on seed 15 uncapped: 64 + 211 = 275 = 244 + 31 (DERIVED). The server queue of about 235 is mostly the burst's own requests (about 190–215 unfinished) plus the released session turns.
  - Consequence for §4.4's "smaller effective shock" sentence:
    - Counted as released session turns, the capped runs have fewer outstanding requests (24–29 against 43–64).
    - Counted as planned-and-unserved session turns, the two arms are about equal (246–375 against 246–366). The difference is the sessions held in the load generator's cap queue, which `shock_size` does not count.
    - The server's waiting line was 179–207 against 204–244.
    - These alternative counts are this session's reading, not registered.

## 6. `--max-num-batched-tokens` 2048

- **Not set in any config used by the 2048 runs.** `configs/base.yaml` passes only `--enable-prefix-caching`, `--kv-cache-metrics`, `--enable-prompt-tokens-details` and `--seed` (FILE(configs/base.yaml:15-19)). Only the 8192 configs set it (`--max-num-batched-tokens: 8192`, e.g. configs/sweep_t4_l1pl350cl_pc_on_b8192.yaml:11). Each server log shows it absent from the non-default args and `'compile_ranges_endpoints': [2048]`. For example, RUN(20261112-061951_burst_l1pl350cl_r2.0_w3000_s15, server.log:7,14): "non-default args: {… 'max_num_seqs': 256}".
- **It is the default.** There is no vLLM install on the Mac (`.venv` has no vllm). Two sources show it:
  - The installed wheel on the VM, saved as PROBE(docs/probe/20261105_vm_checks/scheduler_source.txt) (SOURCE(vLLM 0.32.0 engine/arg_utils.py:EngineArgs.get_batch_defaults)):
    ```
    else:
        # TODO(woosuk): Tune the default values for other hardware.
        default_max_num_batched_tokens = {
            UsageContext.LLM_CLASS: 8192,
            UsageContext.OPENAI_API_SERVER: 2048,
    ```
    Open question A10: "`max_num_batched_tokens` = 2048: the default for a device under 70 GiB and `vllm serve`, then min(256 × 40960, 2048) = 2048" (FILE(docs/open-questions.md:53)).
  - The branch conditions, which the probe does not contain, read from raw.githubusercontent.com/vllm-project/vllm/v0.32.0/vllm/engine/arg_utils.py (fetched 2026-11-26; the fetch service's copy is dated 2026-11-21):
    ```
    if device_memory >= 160 * GiB_bytes: … OPENAI_API_SERVER: 16384
    elif device_memory >= 70 * GiB_bytes and "a100" not in device_name: … OPENAI_API_SERVER: 8192
    else: … OPENAI_API_SERVER: 2048
    ```
- **Wording risk.** §4.1 says "8192, the default on larger cards". That is true for cards of ≥ 70 GiB that are not A100 (H100/H200). An A100-80GB keeps 2048, and cards of ≥ 160 GiB (B200) get 16384.

## 7. Open-loop follow-up turns whose prefix was still in use

- Source: PROBE(docs/probe/m_result/m_f31a69d_20261123T213612Z.txt), RUN(REG-20 arm A's five open burst runs, m.table) through `m.ages` and `m.profile` (decisions.md 2026-11-23 [RESULT] M, FILE(docs/decisions.md:25561)).
- Population: later turns with the previous turn ok, `t_end[k-1]` ≥ the burst's end and `t_first_token[k]` < the burst's end + watch (m.py docstring, lines 6–19). "held" means a < 0, i.e. the first token came before the previous turn ended.
- Open-arm buckets: held 12,887; neg_A 786; [0, .5) 13,267; [.5, 1) 82; [1, 2) 18; [2, 4) 37; [4, 8) 41; [8, 16) 24; [16, ∞) 35.
- **Total 27,177; held share 0.474** (DERIVED: sum of the buckets; 12,887 / 27,177). There are no censored open turns (same entry).
- P79's figure uses the same `m.M["arms"]`, `m.ages` and `m.profile`, so the same denominator applies (BELIEF that the figure's window equals m.table's; both call the same functions).
- Closed arm: 21,695 later turns, all at x ≥ 8.

## 8. References (web, read 2026-11-26)

- **(a) Schroeder, Wierman, Harchol-Balter, NSDI '06: pp. 239–252.**
  - The USENIX abstract page https://www.usenix.org/event/nsdi06/tech/schroeder.html reads "Pp. 239–252 of the Proceedings".
  - CaltechAUTHORS (https://authors.library.caltech.edu/records/fjvcg-e2n08) gives "Proceedings of the 3rd conference on Networked Systems Design & Implementation, 239-252".
  - The ACM DL guide record (10.5555/1267680.1267698) shows only "Page 18".
  - DBLP was not read.
  - references.toml:67 has no pages.
- **(b) Yao, Gao, Lyu, Grosof, Simchi-Levi, arXiv:2609.02027.**
  - It is **not shown as a NeurIPS 2026 main-track paper.** Co-author Chutong Gao's page (https://chutonggao.github.io/publications) lists it as "Short version accepted in the NeurIPS 2026 Second Workshop on ML×OR … Full version in preparation for submission to Operations Research".
  - The first author's page (https://heyuan-yao.github.io/about, "Last updated: Sep. 11, 2026") says "Workshop Version Submitted".
  - The arXiv record (v1, 2 Sep 2026) gives no venue.
  - So the venue is a short workshop version of the same paper, per a co-author's own page. The NeurIPS 2026 main-track or workshop programme page itself was not read.
- **(c) vLLM 0.32.0 prefix-caching design page.** https://docs.vllm.ai/en/v0.32.0/design/prefix_caching/ returned content. The fetch service labelled its copy "acquired 2026-11-08", so a live load today is not confirmed. Decisive text:
  - "Allocate new blocks by popping the heads of the free queue. If the head block is a cached block, this also “evicts” the block so that no other requests can reuse it anymore from now on."
  - "When a request is finished, we free all its blocks if no other requests are using them (reference count = 0) … the freed blocks are added to the tail of the free queue."
  - "Block 0 and 1 are not added to the free queue because they are being used by Request 1."
  - "Eviction (LRU): When the head block (least recently used block) of the free queue is cached, we have to evict the block."

  This supports "are evicted when those need space" and "the cache is not a fixed size". "Cached prefixes occupy **only** the pool blocks that running requests do not hold" is too strong. Blocks held by running requests are also cached and shareable: allocation "increases the reference count of the computed block by one, and removes the block from the free queue if the block wasn't used by other requests". The study's own "held" turns (item 7) reused prefixes still in use. A tighter wording: "idle cached prefixes occupy free pool blocks, and are evicted, least recently used first, when new blocks are allocated."
- **(d) Kwon et al. 2023.** Official title "Efficient Memory Management for Large Language Model Serving with PagedAttention", with **no "(vLLM)"**. SOSP '23 (29th Symposium on Operating Systems Principles), **pp. 611–626**, DOI 10.1145/3600006.3613165 (ACM DL, https://dl.acm.org/doi/10.1145/3600006.3613165: "Pages 611 - 626"). references.toml:165 has "(vLLM)" in the title and no pages.

## 9. When each test was pre-specified

| Test | Fixed (entry, commit UTC) | Requires, in one line | Before / after which data | Result |
|---|---|---|---|---|
| (a) Per-span test (earlier matrix, seeds 2–6) | Cold side `gate.reg17_cold_side`: decisions.md 2026-11-10 [REG] REG-17, 39f01a9 at 17:41 UTC, approved 17:44 UTC. Applied to seeds 3–7 by 2026-11-11 [REG] REG-18, 702bc26 at 08:44 UTC, approved 09:32 UTC | Burst `stuck`, control `recovered`; in every 300 s span after the burst (600–900, 900–1200, 1200–1500 s) completion rate < min(2.0, the span's λp), and end-window queue slope > 0 | Before seed 2's runs (17:44:32 UTC on 11-10) and before seeds 3–6 (from 09:33 UTC on 11-11). After the λ = 2.2/2.4 runs and the surge readouts were seen ("Seen before writing this", FILE(docs/decisions.md:3461)). REG-18 was written after REG-17's seed-2 result | Seed 2 `pass_clear` (REG-17). Seeds 4, 3, 5, 6: 1 of 4, `some`, futility stop; seed 7 not run (FILE(docs/decisions.md:3735)). On REG-19's runs: 3 of 5 |
| (b) REG-19 whole-window test (seeds 15–19) | 2026-11-11 [REG] REG-19, 4a6b33e at 17:47 UTC, approved a7bb77a at 17:55 UTC | Stuck through the 3000 s watch, line never empties (`reg19_drain_time`), completions below the load over the whole window W [1680, 4080) s, control `recovered`; ≥ 3 of 5 `pass_clear` | Before every REG-19 run (first 05:09 UTC on 11-12). After the earlier matrix's data: "registered after REG-18 and judged only on these seeds" | `supports`, 5 of 5, k_clear 5 (FILE(docs/decisions.md:16517)) |
| (c) Turn-release criterion (k1 `cold_backlogged` / `recovered`) | 2026-11-12 [REG] REG-20, registered and approved in 75e627e at 15:33 UTC | Cold-backlogged: reuse drop, mean waiting > baseline + 1.0, and KV level. Recovered: reuse ≥ baseline − 0.06 and waiting ≤ baseline + 1.0; exact one-sided McNemar test | **After REG-19 seeds 15–17's six runs existed** (05:09–about 15:05 UTC, sealed; the read deviations are FILE(docs/decisions.md:4274) and :4453, the latter at 17:59 UTC, after the registration). Before seeds 18–19 (from 16:55 UTC on 11-13) and before every open-loop run (11-15). Tried as a smoke test on reported REG-17/18 runs | Closed 5 of 5 cold, open 0 of 5 (5 recovered), p = 0.031 (K1 arm A, 2026-11-15). Arm B 0 of 3. TR on the later stack: open 5 of 5 recovered, D 5 of 5 stuck |
| (d) PW, the retrospective count over ten seeds | 2026-11-25 [REG] PW, approved by the author about 17:06 UTC, committed 3b5bc5c at 17:33 UTC, before any PW number | REG-19's `reg19_seed_outcome` on each pair cut at 4080 s; a seed counts only if it stays `pass_clear` both with unserved late requests counted all fast and all missed | **After all runs and their registered outcomes were known** ("Not blind", FILE(docs/decisions.md:25749)). Seeds **15, 16, 17, 18, 19, 43, 61, 68, 72, 79** | 0 of 10 counted (all `cut_sensitive`); 10 of 10 `pass_clear` as recorded (FILE(docs/decisions.md:25842); P82) |
| (e) Condition 4's order clause | Clause in PROTOCOL.md §3 since 47984d5 (2026-11-04 05:08 UTC; FILE(PROTOCOL.md:36)); the plan's order is FILE(docs/experiment-plan.md:285-293). Judged by 2026-11-14 [REG] "Condition 4's order clause …", 8f5a5bc at 10:30 UTC, run cutoff 20261114-102959 UTC | Onsets in the plan's order (reuse drops, then recomputed prefill rises, then N grows, then KV fills), and preemptions ≤ 0.01 of completions | After the 13 earlier and the REG-19 cold runs (not judged: "No run seen so far is judged"). Before REG-25's wipe run (`20261115-193119`) and REG-24's long-window run (`20261116-182704`) | 0 of 1 on each (P20, P21); the planned order held in 0 of 16 earlier cold bursts, post hoc (P36) |

Note on (a): the question's "REG-18's per-span test" was REG-17's test, which REG-18 extended with three rules (late surge tip, no pass on missing data, control judged on its own; FILE(docs/decisions.md:3590-3594)).
