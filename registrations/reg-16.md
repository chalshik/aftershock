# REG-16

*Registered 2026-11-10 · GPU runs · claim C1 · result 2026-11-10*

| | |
|---|---|
| **Question** | On the session schedule of a planned long burst run at λ = 2.2 (closed-loop turns, L1, 350 W), without a burst, does the server stay warm for the whole run: the watch window (1080–5580 s), a drop to λ = 1 (5580–14580 s) and the return to 2.2 (14580–16080 s)? |
| **Runs** | One no-burst control, seed 14, 16080 s, 9000 s drop, `configs/stage2_control_l1pl350cl_r2.2_long_s14.yaml`. Valid on the first attempt. |
| **Outcome rule** | `gate.reg16_control_outcome`, in order: `bad_baseline` (baseline pf < 0.98, or the watch verdict `bad_baseline`); `inconclusive` (baseline pf missing); `control_cold_persistent` (a cold window is the last of its phase, a cold stretch lasts ≥ 1500 s, or the watch verdict is `stuck`); `inconclusive` (a phase's last window missing after a cold window); `control_cold_transient`; `inconclusive` (other missing values, or the watch `unclear`); `control_unclear` (never cold, but a low hit with a queue); `control_not_recovered`; `control_warm`. A 300 s window is cold when hit ≤ base − 0.06 and pf by send time ≤ base − 0.10. |
| **Prediction** | Run valid p ≈ 0.7; baseline healthy p ≈ 0.9. Given a valid run with a healthy baseline: `control_warm` ≈ 0.5; `control_cold_persistent` ≈ 0.2; `control_cold_transient` ≈ 0.1; `control_unclear` or `control_not_recovered` ≈ 0.15; `inconclusive` ≈ 0.05. |
| **Decision under each outcome** | `control_warm`: draft the burst run on the same schedule (as REG-17), which together with the control separates a burst-made cold state from overload. `control_cold_persistent`: no burst run and no matrix at λ = 2.2; the author chooses a lower λ or reports λ = 2.2 as a load where the warm state is not stable. `control_cold_transient`: reported, does not stop the burst run by itself. Other labels: reported, the author decides; `inconclusive` rerun at most twice. |
| **Adversarial review** | Two rounds. Round 1 had three should-fix items (T1–T3, one bordering on blocking: a short dip would stop the plan; a collapse could be missed; the drain margin) and minor items; round 2 had nothing blocking and three should-fix items. All were applied before the run; the round-2 changes and the 9000 s drop were checked by tests only, not re-reviewed. |
| **Result** | `control_cold_persistent`: with no burst the control went cold at 4080–4380 s and stayed cold to the watch's end (watch verdict `stuck`; end window pf 0.000, hit 0.000, queue 96.9); 8 cold windows of 50, one unbroken stretch 4080–6480 s, `longest_cold_s` 2400 s. It warmed again in the drop (line empty by ≈ 6600 s) and stayed warm on the return (`return_recovered` true). |
| **Prediction held** | no (`control_cold_persistent` had ≈ 0.2 against `control_warm` ≈ 0.5) |
| **Not separated** | Overload during a surge followed by a backlog against two states at one load with the surge as trigger: the run cannot separate them, because the watch ended 1100–1400 s after the collapse with the backlog still growing. The correction adds that a surge of planned arrivals and a surge of new sessions are not separated either. |
| **Corrections** | 2026-11-10: the result read the collapse as following a sustained load above a closed-loop warm capacity ≈ 2.4 with a 120 s peak of 2.783; the run carried 300 s windows at λp 2.57 and 2.60 warm, and on a 5 s grid to the KV onset (4433 s) the highest 120 s planned rate was 3.45. No-burst collapses follow a 120 s surge of planned arrivals; the outcome stands. |
| **In the paper** | §4.1, P15; §3, P4 (the 9000 s drop emptied the line); §4.6, P42 |
| **Notes** | The author pre-approved the run before its final text and left the seed to the study's choice; seed 14 and the 9000 s drop were chosen on margin and load alone, before any run. The long burst run this registration prepared was never run; the id REG-17 went to a λ = 2.0 pair instead. |

[All registrations](README.md)
