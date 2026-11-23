"""T1 (docs/drafts/t1_draft5.md, with the author's draft 6 code decisions of 2026-11-23 on rule F round 2's B1 and M2;
not registered): a session cap engaged inside the established stuck state, beside a
concurrent uncapped twin. REG-19's stuck burst runs at λ = 2.0 on seeds 15-19, two arms per seed on one schedule, row
for row (`duration_s` 4080):

  U  uncapped twin: REG-19's burst config of the seed, renamed `..._t1u`; nothing else changes.
  P  pause cap: D1's capped config of the seed (D1r's parent) with `cap` 22 for every seed, `cap_from_s` the switch
     time the separation gate picks, and `cap_mode: pause` (harness/loadgen.py SessionCap): from the switch, a session
     whose turn ends while more than `cap` sessions hold a slot gives its slot up and joins a resume queue, served
     before new sessions. The owed turns wait at the clients; only their release times move. Named `..._t1p`
     (draft 4, gap 1: the old untracked `_rescue_pause` files of draft 2 are never used and are not touched).

Question (draft 3 §0, author 2026-11-22): once the cap has bounded the sessions holding a slot so that their working set
fits the pool (`t_fit`), with the line still owed, does the engine return to warm promptly (H0, `prompt_warm`), or stay
cold or warm only late (H1, `cold_fit` / `delayed`)? Draft 6 (author 2026-11-23, review B1): `t_fit` is the working-set
clause alone, so the clock starts at the cap (t_fit = t_cap whenever 22 x 3,968 fits, as it does at L1); draft 5's
waiting clause waited for the re-warm it was meant to time.

Per seed (`seed_label`, first match): bad_baseline, not_established, unseparable, control_failed (all out of n);
unclear; relapse; no t_w: cold_fit / ws_never_fit / unclear; drain_bound; unclear (t_w, no t_fit); warm_before_fit;
prompt_warm; delayed.
Across seeds (`outcome`, n = the counted labels): too_few, no_memory, memory, drain_bound, mixed.
Separation gate (rule G style, prediction only; pass condition in the draft, written before any computation):
`separation_gate(from_s)` on REG-19's uncapped burst runs; `choose_from_s` takes the first candidate that passes.
Never computed here: the gate and every readout of a real run wait for rule F round 2 and the author (rule G, rule 7).
`table` refuses while any T1 folder is sealed or while T1["from_s"] is unset. Nothing here reads the environment.
"""
from __future__ import annotations

import itertools
import math
from collections import Counter
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import yaml

from analysis import d1, d1r, withheld
from analysis.reg30 import REG30

REPO = Path(__file__).resolve().parent.parent
CONFIGS = REPO / "configs"

SEEDS = (15, 16, 17, 18, 19)   # REG-19's, fixed from the schedule alone (gate.reg19_order); U runs on all five
P_SEEDS = (15, 17, 19)         # P only where the gate at from_s 1980 separates (author 2026-11-23, after the gate;
                               # PROBE docs/probe/t1_gate/choose_20261123.json): 16 and 18 are unseparable, out of n


def _order() -> tuple:
    """(seed, arm) launch order, fixed before any run (draft 3 §1): per seed the two arms adjacent; P first at the 1st,
    3rd and 5th seed, U first at the 2nd and 4th; P only at P_SEEDS (author 2026-11-23). Queue: P15 U15 U16 P17 U17
    U18 P19 U19."""
    out = []
    for i, sd in enumerate(SEEDS):
        pair = ("P", "U") if i % 2 == 0 else ("U", "P")
        out += [(sd, arm) for arm in pair if arm == "U" or sd in P_SEEDS]
    return tuple(out)


T1 = {
    "arms": ("U", "P"),
    "suffix": {"U": "t1u", "P": "t1p"},
    "seeds": SEEDS,
    "p_seeds": P_SEEDS,
    "order": _order(),
    "seal": "[RESULT] T1 (rescue, two arms)",
    # draft 3 §2, the constants of `T1` as listed there
    "cap": 22,                                        # 96,656 / (3,968 + 256) = 22.9 -> 22 (draft 3 §1, DERIVED)
    "from_s_candidates": (1680.0, 1980.0, 2280.0, 2580.0, 2880.0),
    "span_s": 120.0,                                  # warm_spans: 120 s spans from from_s
    "hold_spans": 2,                                  # t_w: two consecutive warm spans; relapse: two cold ones
    "prompt_s": 240.0,                                # prompt_warm: 0 <= t_w - t_fit <= prompt_s (t_fit = t_cap when
                                                      # the working set fits at the cap; draft 6, B1); the gate's window
    "lead_min_s": REG30["lead_min_s"],                # 300 s, REG-30's (drain_bound; the gate's lead)
    # x_hi: the gate's drain rate after t_cap_est, req/s (value unchanged, author 2026-11-22). Not D1's capped spans
    # (2.65-3.01 req/s at caps 29, 29, 26; moved to 22 slots they give 2.65 x 22/29 = 2.01 to 3.01 x 22/26 = 2.55,
    # DERIVED, review 4a). Its robustness (DERIVED, review 4a, ignoring tau): the real drain starts at the re-warm,
    # so lead ~ (L(t_w) - level) / (x - lam); the gate guarantees L0 - level >~ 540 x (3.01 - 2.0) ~ 545, so
    # drain_bound (lead < 300) needs x > lam + 545 / 300 ~ 3.8 req/s, near the closed-loop ceiling 22 / 4.4 ~ 5. A
    # slower true drain leaves the label safe and costs only time or the run (a later from_s, or T1 not proposed).
    "x_hi": 3.01,
    # x_cold: the gate's rate before t_cap_est, req/s: d1.D1["x_cold"], REG-19's completion rate over W in its five
    # burst runs, 1.740-1.758 (mean 1.752, DERIVED; FILE decisions.md:16533-16537), rounded to 1.75. Not "the top of
    # the uncapped cold spans 1.70-1.76" (draft 5's wording, review §5); the value is unchanged.
    "x_cold": d1.D1["x_cold"],
    "gate_k": 3,                                      # the gate passes at a from_s when >= 3 of 5 seeds separate
    "most": 3,                                        # outcome: a label in >= 3 seeds
    "ws_margin_per_slot": 256,                        # tokens per slot: 128 decode + up to 128 of block rounding
    # draft 4's constants (Verdict), beside the list above
    "established_s": 300.0,                           # established / u_ok: the window (from_s - 300, from_s)
    "grid_s": d1.D1["grid_s"],                        # 1 s: t_cap, t_fit and the gate's fluid line
    # draft 5, Band (author 2026-11-22: from the switch time, on all counted seeds); draft 6 (author 2026-11-23, M2):
    # inside needs both D from from_s and t_d - t_cap from the cap inside the band. band_tol has no derivation: it is
    # arbitrary. When tau and the re-warm lag are small against D, D ~ 1 / (x - lam), so +-15 % on D is about a +-5 %
    # test on the mean post-cap drain rate: x in [2.88, 3.19] req/s against x_hi 3.01 (DERIVED, review §4b).
    "band_tol": 0.15,
    # the pause invariant (`pause_fault`), D1r's thresholds, unchanged from draft 2's code
    "max_wait_before_s": d1r.D1R["max_wait_before_s"],
    "max_idle_share": d1r.D1R["max_idle_share"],
    # set once, in the [REG], to the candidate `choose_from_s` gives (draft 3 §1); None until then
    "from_s": 1980.0,                                 # the gate, once (PROBE docs/probe/t1_gate/choose_20261123.json)
    # Prediction (BELIEF, draft 5 §Prediction and Appendix E); `predicted` and `predicted_band` reproduce the draft.
    # Draft 6 (B1): no per-seed input changes. Draft 5's beliefs are about what happens physically (E2: 16 of 16
    # parking-cap rescues, discounted for transfer) and already assumed t_fit ~ t_cap (Appendix C: t_cap 2653,
    # t_fit 2660), i.e. the clock B1 now codes; what B1 changes is that the code can reach the labels those beliefs
    # name (as coded at draft 5, a delayed re-warm read prompt_warm and a fitting cold seed cold_over_pool, review
    # Traces A, B). Moving any input would need a new BELIEF, so the old values are kept:
    #   BELIEF (not set): P(delayed) and P(cold_fit) per seed once waiting is coupled to reuse (review fact 2);
    #   BELIEF (not set): the mass of warm_before_fit (t_w < t_cap) and ws_never_fit, taken as 0 here as
    #     cold_over_pool's was in draft 5;
    #   BELIEF (not set): P(inside | label) under the t_cap band clause (M2), which can only lower it.
    "prediction": {
        "counted": {"prompt_warm": 0.74, "delayed": 0.07, "cold_fit": 0.02, "drain_bound": 0.07, "relapse": 0.02,
                    "unclear": 0.08},                 # draft 5 (draft 4: 0.65, 0.12, 0.04, 0.07, 0.04, 0.08)
        "out_of_n": {"bad_baseline": 0.04, "not_established": 0.03, "unseparable": 0.10, "control_failed": 0.05},
        "p_gate_passes": 0.8,
        # draft 5 Appendix E3: P(band inside | label), per counted label (labels it omits: 0)
        "band_inside_given": {"prompt_warm": 0.40, "delayed": 0.10, "drain_bound": 0.25, "cold_fit": 0.0,
                              "relapse": 0.15, "unclear": 0.25},
    },
}
OUT_OF_N = ("bad_baseline", "not_established", "unseparable", "control_failed")      # labels 1-4
COUNTED = ("unclear", "relapse", "cold_fit", "ws_never_fit", "drain_bound", "warm_before_fit", "prompt_warm",
           "delayed")                               # labels 5-12 (`label_from`'s rules 5-12)
LABELS = OUT_OF_N + COUNTED
MEMORY = ("cold_fit", "delayed")                    # H1's labels; ws_never_fit and warm_before_fit are neither H0's
                                                    # nor H1's: they count in n and toward no outcome's k
OUTCOMES = ("too_few", "no_memory", "memory", "drain_bound", "mixed")


def hold_s() -> float:
    """The two-span hold, 240 s: t_w's warm stretch, the relapse search's start after t_w, and u_ok's judged span."""
    return T1["hold_spans"] * T1["span_s"]


def cold_fit_by(t_end: float) -> float:
    """`cold_fit` needs t_fit <= this: duration_s - (prompt_s + the two-span hold) = 4080 - (240 + 240) = 3600 on T1's
    runs. A later fit leaves no room for a prompt warm and its hold, so a missing t_w cannot be read (draft 4, gap 3).
    With draft 6's t_fit (= t_cap when the working set fits) this asks that the cap bind by 3600 s."""
    return t_end - (T1["prompt_s"] + hold_s())


# ---------------------------------------------------------------- configs

def config_name(arm: str, seed: int) -> str:
    if arm not in T1["arms"]:
        raise ValueError(f"arm {arm!r} is not one of T1's")
    return f"burst_l1pl350cl_r2.0_w3000_s{seed}_{T1['suffix'][arm]}"


def run_pattern(arm: str, seed: int) -> str:
    return f"*_{config_name(arm, seed)}"


def _check_from_s(from_s) -> float:
    if from_s is None or float(from_s) not in T1["from_s_candidates"]:
        raise ValueError(f"arm P needs from_s, one of {T1['from_s_candidates']}, got {from_s!r}")
    return float(from_s)


def config_text(arm: str, seed: int, from_s: float | None = None) -> str:
    """U: REG-19's burst config of the seed, renamed. P: D1's capped config of the seed with cap, cap_from_s and
    cap_mode set explicitly (review R5); `from_s` is required for P and must be one of the candidates."""
    if arm not in T1["arms"]:
        raise ValueError(f"arm {arm!r} is not one of T1's")
    if seed not in T1["seeds"]:
        raise ValueError(f"seed {seed} is not one of T1's")
    head = (f"# T1 (docs/decisions.md [REG] T1 (rescue, two arms)), seed {seed}, arm {arm}.\n")
    if arm == "U":
        if from_s is not None:
            raise ValueError("arm U has no switch time")
        return (head +
                "# Uncapped twin: REG-19's burst config of the seed, renamed; nothing else changes.\n"
                "# Written by `python -m analysis.t1 configs U`. Runs only after `[DECISION] ... T1 starts`.\n"
                f"extends: burst_l1pl350cl_r2.0_w3000_s{seed}.yaml\n"
                f"name: {config_name(arm, seed)}\n")
    fs = _check_from_s(from_s)
    return (head +
            f"# Pause cap: D1's capped config of the seed (D1r's parent) with cap {T1['cap']} for every seed, switched\n"
            f"# on at {fs:g} s (the earliest candidate passing t1.separation_gate; [REG] T1), in pause mode.\n"
            f"# Written by `python -m analysis.t1 configs P {fs:g}`. Runs only after `[DECISION] ... T1 starts`.\n"
            f"extends: {d1.config_name(seed)}.yaml\n"
            f"name: {config_name(arm, seed)}\n"
            f"workload:\n"
            f"  sessions:\n"
            f"    cap: {T1['cap']}\n"
            f"    cap_from_s: {fs:g}\n"
            f"    cap_mode: pause\n")


def resolved(arm: str, seed: int, from_s: float | None = None) -> dict:
    from harness.config import _merge, load_config, with_defaults
    from harness.run import expand
    raw = yaml.safe_load(config_text(arm, seed, from_s))
    parent = raw.pop("extends")
    return expand(with_defaults(_merge(load_config(CONFIGS / parent), raw)))[0]


def write_configs(arm: str, from_s: float | None = None, configs_dir: Path | None = None) -> list[Path]:
    """Writes the arm's configs as new files (U: all five seeds; P: P_SEEDS); refuses if any exists (a used config is
    never edited). P only with the switch time the [REG] states."""
    d = configs_dir or CONFIGS
    seeds = T1["seeds"] if arm == "U" else T1["p_seeds"]
    texts = [config_text(arm, sd, from_s) for sd in seeds]
    paths = [d / f"{config_name(arm, sd)}.yaml" for sd in seeds]
    if any(p.exists() for p in paths):
        raise FileExistsError([str(p) for p in paths if p.exists()])
    for p, txt in zip(paths, texts):
        p.write_text(txt)
    return paths


# ---------------------------------------------------------------- slots held, from the pause fields

def _pause_fields() -> tuple:
    from harness.loadgen import PAUSE_FIELDS
    return PAUSE_FIELDS


def _sessions(run: dict):
    from analysis.margin import is_burst_request
    q = run["requests"]
    return q[~is_burst_request(run)]


def held_spans(run: dict) -> pd.DataFrame:
    """Per session (no burst row), the spans [start, end) in which it holds a slot: from admission (else first send) to
    its first pause, from each resume to the next pause, and from the last resume (or the admission, if it never
    paused) to its last reply (inf if any turn never ended). A session still paused at the end holds nothing after its
    last pause. Without `t_paused` in the records: one span per session, `d1r._held`'s."""
    q = _sessions(run)
    rows = []
    for sid, s in q.sort_values(["session_id", "turn"]).groupby("session_id"):
        t0 = s[s.turn == 0]
        if not len(t0):
            continue
        r0 = t0.iloc[0]
        start = r0["t_admitted"] if "t_admitted" in s else None
        if start is None or pd.isna(start):
            start = r0["t_sent"]
        if pd.isna(start):
            continue
        done = float(s.t_end.max()) if s.t_end.notna().all() else math.inf
        cur = float(start)
        if "t_paused" in s:
            p = s[s.t_paused.notna()]
            for tp, tr in zip(p.t_paused.astype(float), p.t_resumed.astype(float)):
                rows.append((int(sid), cur, float(tp)))
                cur = float(tr)
                if math.isnan(cur):
                    break
        if not math.isnan(cur):
            rows.append((int(sid), cur, done))
    return pd.DataFrame(rows, columns=["session_id", "start", "end"])


def _intervals(run: dict) -> tuple[np.ndarray, np.ndarray]:
    h = held_spans(run)
    return h.start.to_numpy(float), h.end.to_numpy(float)


def slots_held(run: dict, t: np.ndarray) -> np.ndarray:
    """Sessions holding a slot at each time t (right-continuous): in service minus paused, from the pause fields.
    A SECOND computation of "sessions in service" (docs/definitions.md, `slots_held` ⚠): `d1r.in_service` (and
    `k1.sessions_in_service`, `cap.max_sessions_in_service`) count a session from admission to its last reply, paused or
    not; this one leaves its pauses out, as harness/loadgen.py SessionCap's `used` does. Equal to `d1r.in_service` on a
    run without pauses. An input to `ws_slot` and so to `t_fit` (author, 2026-11-22)."""
    s, e = _intervals(run)
    t = np.asarray(t, float)
    return (np.searchsorted(np.sort(s), t, side="right") - np.searchsorted(np.sort(e), t, side="right")).astype(float)


def resume_queue(run: dict, t: np.ndarray) -> np.ndarray:
    """Sessions paused at each time t: t_paused <= t < t_resumed (never resumed: to the end of the record). 0 where the
    records carry no pause fields."""
    t = np.asarray(t, float)
    q = _sessions(run)
    if "t_paused" not in q:
        return np.zeros(len(t))
    p = q[q.t_paused.notna()]
    a = np.sort(p.t_paused.astype(float).to_numpy())
    b = np.sort(np.nan_to_num(p.t_resumed.astype(float).to_numpy(), nan=math.inf))
    return (np.searchsorted(a, t, side="right") - np.searchsorted(b, t, side="right")).astype(float)


def has_pause_fields(run: dict) -> bool:
    return any(k in run["requests"] for k in _pause_fields())


def ws_slot(run: dict, t: np.ndarray) -> np.ndarray:
    """The in-slot working set at each time t (tokens): Σ over the sessions holding a slot at t (by `held_spans`, the
    spans of `slots_held`) of the `prompt_tokens` of their latest turn sent by t (t_sent <= t); a session holding a
    slot that has sent nothing yet counts its turn 0's `prompt_tokens_planned`. That is the context a session needs
    cached for its next turn (k1.later_turns: a turn's reusable prefix is its previous prompt). A sent turn whose
    `prompt_tokens` is null (cut off before the server's usage chunk) counts its `prompt_tokens_planned` (draft 4, gap 9)."""
    t = np.asarray(t, float)
    spans = held_spans(run)
    if not len(spans):
        return np.zeros(len(t))
    q = _sessions(run)
    sent = q[q.t_sent.notna()].sort_values(["session_id", "t_sent"])
    lat = pd.DataFrame({"session_id": sent.session_id.astype(int).to_numpy(),
                        "a": sent.t_sent.astype(float).to_numpy(),
                        "b": sent.groupby("session_id").t_sent.shift(-1).astype(float).fillna(math.inf).to_numpy(),
                        "w": sent.prompt_tokens.astype(float).fillna(sent.prompt_tokens_planned.astype(float))
                        .to_numpy()})
    t0 = q[q.turn == 0].set_index("session_id")
    first = sent.groupby("session_id").t_sent.min()
    pre = pd.DataFrame({"session_id": t0.index.astype(int), "a": -math.inf,
                        "b": first.reindex(t0.index).astype(float).fillna(math.inf).to_numpy(),
                        "w": t0.prompt_tokens_planned.astype(float).to_numpy()})
    m = pd.concat([lat, pre], ignore_index=True).merge(spans, on="session_id")
    a, b = np.maximum(m.a, m.start).to_numpy(float), np.minimum(m.b, m.end).to_numpy(float)
    keep = a < b
    a, b, w = a[keep], b[keep], m.w.to_numpy(float)[keep]
    oa, ob = np.argsort(a), np.argsort(b)
    ca, cb = np.concatenate([[0.0], np.cumsum(w[oa])]), np.concatenate([[0.0], np.cumsum(w[ob])])
    return ca[np.searchsorted(a[oa], t, side="right")] - cb[np.searchsorted(b[ob], t, side="right")]


def slots_paused(run: dict, cap: int, from_s: float) -> dict:
    """Arm P's invariant, read from the records (harness/loadgen.py SessionCap, mode "pause"):
    - the records carry the pause fields (`fields`);
    - no pause before the switch (`pauses_before`);
    - every pause found `slots_at_pause` >= cap, i.e. more than cap held a slot before it gave its own up
      (`pauses_under`, the count that did not);
    - no hand-over at or after the switch, a resume or the admission of a session arriving at or after it, found
      cap or more sessions already holding a slot (`handovers_over`, counted on `slots_held`'s spans just before the
      hand-over, the incoming session excluded; `resumes_reported_over`, loadgen's own `slots_at_resume` above cap);
    - as D1r: no session arriving before the switch waited (`max_wait_before_s`), and the share of grid seconds after
      the switch with a session waiting (admission or resume queue) while fewer than cap held a slot (`idle_share`).
    Reported: `t_at_cap`, the first grid time after the switch with slots held <= cap (= `t_cap`), and the peak after."""
    q = _sessions(run)
    fields = all(k in q for k in _pause_fields())
    out = {"cap": cap, "from_s": from_s, "fields": fields}
    s0 = q[q.turn == 0]
    arr = s0.t_arrived.astype(float).to_numpy() if "t_arrived" in s0 else np.full(len(s0), np.nan)
    adm = s0.t_admitted.astype(float).to_numpy() if "t_admitted" in s0 else np.full(len(s0), np.nan)
    before = (arr < from_s) & np.isfinite(adm)
    out["max_wait_before_s"] = float(np.nanmax(adm[before] - arr[before])) if before.any() else 0.0
    if fields:
        p = q[q.t_paused.notna()]
        tp = p.t_paused.astype(float).to_numpy()
        sap = p.slots_at_pause.astype(float).to_numpy()
        tr = p.t_resumed.astype(float).to_numpy()
        sar = p.slots_at_resume.astype(float).to_numpy()
    else:
        tp = sap = tr = sar = np.array([])
    out["pauses"] = int(len(tp))
    out["pauses_before"] = int((tp < from_s).sum())
    out["pauses_under"] = int((sap < cap).sum()) + int(np.isnan(sap).sum())
    out["resumes"] = int(np.isfinite(tr).sum())
    out["resumes_reported_over"] = int((sar[np.isfinite(tr)] > cap).sum())
    late_adm = adm[np.isfinite(adm) & (arr >= from_s)]
    hand = np.sort(np.concatenate([late_adm, tr[np.isfinite(tr) & (tr >= from_s)]]))
    s, e = _intervals(run)
    held_before = (np.searchsorted(np.sort(s), hand, side="left") - np.searchsorted(np.sort(e), hand, side="right"))
    out["handovers"] = int(len(hand))
    out["handovers_over"] = int((held_before >= cap).sum())
    t_end = float(run["cfg"]["workload"]["duration_s"])
    grid = np.arange(from_s, t_end, T1["grid_s"])
    held = slots_held(run, grid)
    cq = np.nan_to_num(d1.cap_queue(run, grid), nan=0.0)
    waiting = (cq + resume_queue(run, grid)) > 0
    out["idle_share"] = float(((held < cap) & waiting).sum() / waiting.sum()) if waiting.any() else 0.0
    at = np.nonzero(held <= cap)[0]
    out["t_at_cap"] = float(grid[at[0]]) if len(at) else None
    out["max_held_after_cap"] = float(held[at[0]:].max()) if len(at) else None
    return out


def pause_fault(sl: dict) -> bool:
    """Arm P's switched cap did not do what harness/loadgen.py says (`slots_paused`). A flag, not a label: the seed
    counts in n as unclear (D1's DM2)."""
    return (not sl["fields"] or sl["pauses_before"] > 0 or sl["pauses_under"] > 0 or sl["handovers_over"] > 0
            or sl["resumes_reported_over"] > 0 or sl["max_wait_before_s"] > T1["max_wait_before_s"]
            or sl["idle_share"] > T1["max_idle_share"])


def u_fault(run: dict) -> bool:
    """Arm U's validity item (draft 3 §1): no cap field and no pause field in its records."""
    return has_pause_fields(run) or "t_arrived" in run["requests"]


# ---------------------------------------------------------------- the judged quantities

def reuse_unpaused(run: dict, a: float, b: float) -> float:
    """`c2.reuse_in` (with c2's min_turns) over `k1.later_turns` whose first token falls in [a, b), leaving out the turn
    right after a pause (records with `t_paused` set). A second computation of reuse (docs/definitions.md ⚠). On a run
    without pause fields it equals `k1.reuse` only on windows with at least C2["min_turns"] later turns; with fewer it
    is NaN and `k1.reuse` is not (review A7)."""
    from analysis.c2 import C2, reuse_in
    from analysis.k1 import later_turns
    turns = later_turns(run)
    if "t_paused" in turns:
        turns = turns[turns.t_paused.isna()]
    return reuse_in(turns, a, b, C2["min_turns"])


def baseline(run: dict) -> dict:
    """The run's own baseline window (k1.windows: [at - baseline_s, at)) and its reuse_unpaused, mean waiting and pf
    (judged inputs); beside them, never judged (draft 6, review M3), `k1.reuse`, the server's own prefix hit rate
    (`load._level` hit) and its mean KV usage (`load._level` kv) over the same window."""
    from analysis.k1 import reuse, windows
    from analysis.load import _level
    w = windows(run["cfg"])["base"]
    lv = _level(run, *w)
    return {"window": w, "reuse": reuse_unpaused(run, *w), "q": lv["q"], "pf": lv["pf"],
            "reuse_k1": reuse(run, *w), "hit_server": lv["hit"], "kv": lv["kv"]}


def established(run: dict, from_s: float) -> dict:
    """`k1.cold_backlogged` on (from_s - established_s, from_s) against the run's own baseline window (review R1)."""
    from analysis.k1 import cold_backlogged, windows
    return cold_backlogged(run, {"base": windows(run["cfg"])["base"],
                                 "end": (from_s - T1["established_s"], from_s)})


def t_cap(run: dict, from_s: float) -> float | None:
    """The first grid time >= from_s with `slots_held` <= cap (= `slots_paused(...)["t_at_cap"]`); None if never."""
    grid = np.arange(from_s, float(run["cfg"]["workload"]["duration_s"]), T1["grid_s"])
    at = np.nonzero(slots_held(run, grid) <= T1["cap"])[0]
    return float(grid[at[0]]) if len(at) else None


def pool_tokens(run: dict) -> float:
    """C, the KV pool in tokens, from the run's checked config value `checks.expected_kv_tokens` (review A6). NaN if
    the config does not state it."""
    c = (run["cfg"].get("checks") or {}).get("expected_kv_tokens")
    return math.nan if c is None else float(c)


def t_fit(run: dict, t_cap_: float | None) -> float | None:
    """The working-set clause alone (draft 6, author 2026-11-23, review B1): t_cap itself if `ws_slot` <= C - cap x
    ws_margin_per_slot at t_cap, else the first grid time >= t_cap at which it holds; None if no such time before
    duration_s, or if t_cap or C is missing. At L1 22 x 3,968 = 87,296 <= 96,656 - 22 x 256 = 91,024 (DERIVED), so
    t_fit = t_cap whenever the cap holds; the check stays for a run whose recorded prompts exceed the plan. Draft 5's
    second clause (the server's mean waiting back at baseline) is gone: a cold capped server keeps a queue, so that
    clause waited for the re-warm it was meant to time (review fact 2)."""
    c = pool_tokens(run)
    if t_cap_ is None or math.isnan(c):
        return None
    grid = np.arange(t_cap_, float(run["cfg"]["workload"]["duration_s"]), T1["grid_s"])
    at = np.nonzero(ws_slot(run, grid) <= c - T1["cap"] * T1["ws_margin_per_slot"])[0]
    return float(grid[at[0]]) if len(at) else None


def warm_spans(run: dict, from_s: float, base: dict) -> dict:
    """span_s spans from from_s while a span ends by duration_s. A span is warm when reuse_unpaused >= base reuse -
    recovered_hit_drop and its mean waiting <= base waiting + recovered_q_excess (K1's recovered); cold when
    reuse_unpaused <= base reuse - stuck_hit_drop. Either is None when its inputs are NaN. `t_w`: the start of the first
    of hold_spans consecutive warm spans. `relapse`: hold_spans consecutive cold spans, all starting at or after
    t_w + the hold (`t_relapse` the first's start). Beside each span's reuse_unpaused, never judged (draft 6, review
    M3): `k1.reuse` (resumed turns included), the server's own prefix hit rate and its mean KV usage (`load._level`)."""
    from analysis.k1 import reuse
    from analysis.load import _level
    v = run["cfg"]["verdict"]
    t_end, step = float(run["cfg"]["workload"]["duration_s"]), T1["span_s"]
    spans, a = [], float(from_s)
    while a + step <= t_end:
        b = a + step
        lv = _level(run, a, b)
        ru, q = reuse_unpaused(run, a, b), lv["q"]
        nan = math.isnan(ru) or math.isnan(q) or math.isnan(base["reuse"]) or math.isnan(base["q"])
        warm = None if nan else bool(ru >= base["reuse"] - v["recovered_hit_drop"]
                                     and q <= base["q"] + v["recovered_q_excess"])
        cold = (None if math.isnan(ru) or math.isnan(base["reuse"])
                else bool(ru <= base["reuse"] - v["stuck_hit_drop"]))
        spans.append({"span": (a, b), "reuse_unpaused": ru, "reuse_k1": reuse(run, a, b), "hit_server": lv["hit"],
                      "kv": lv["kv"], "q": q, "warm": warm, "cold": cold})
        a = b
    k = T1["hold_spans"]
    t_w = next((spans[i]["span"][0] for i in range(len(spans) - k + 1)
                if all(s["warm"] is True for s in spans[i:i + k])), None)
    t_rel = None
    if t_w is not None:
        later = [s for s in spans if s["span"][0] >= t_w + hold_s()]
        t_rel = next((later[i]["span"][0] for i in range(len(later) - k + 1)
                      if all(s["cold"] is True for s in later[i:i + k])), None)
    return {"spans": spans, "t_w": t_w, "relapse": t_rel is not None, "t_relapse": t_rel,
            "any_nan": any(s["warm"] is None or s["cold"] is None for s in spans)}


def drained(run: dict, from_s: float) -> float:
    """t_d: `confirm.drained_at` on [from_s, duration_s) at `gate.owed_backlog` at the trigger's start (the total line
    back at its level at the trigger; REG-30's notion, review B2); inf if not reached."""
    from analysis.confirm import drained_at
    from analysis.gate import owed_backlog
    t = drained_at(run, from_s, float(run["cfg"]["workload"]["duration_s"]),
                   owed_backlog(run, float(run["cfg"]["trigger"]["at_s"])))
    return math.inf if math.isnan(t) else t


def judged_span(t_w: float | None, t_fit_: float | None, from_s: float) -> tuple[float, float]:
    """P's judged span for u_ok: [t_w, + hold) if t_w, else [t_fit, + hold) if t_fit, else [from_s, + hold)."""
    a = t_w if t_w is not None else (t_fit_ if t_fit_ is not None else from_s)
    return (float(a), float(a) + hold_s())


def u_ok(run_u: dict, span: tuple[float, float], from_s: float) -> dict:
    """U is `k1.cold_backlogged` (against its own baseline) on three windows (review R2): (from_s - established_s,
    from_s), its end window [3480, 4080) (`k1.windows`), and P's judged span. `ok` only if all three are True (a None,
    from a NaN input, is not ok)."""
    from analysis.k1 import cold_backlogged, windows
    w = windows(run_u["cfg"])
    wins = {"established": (from_s - T1["established_s"], from_s), "end": w["end"], "judged": tuple(span)}
    got = {k: cold_backlogged(run_u, {"base": w["base"], "end": x})["cold_backlogged"] for k, x in wins.items()}
    return {"windows": wins, "cold_backlogged": got, "ok": all(x is True for x in got.values())}


# ---------------------------------------------------------------- the label

def evidence(run_p: dict, run_u: dict, from_s: float) -> dict:
    """Every input of `label_from` for one seed, from its two runs."""
    bp, bu = baseline(run_p), baseline(run_u)
    sl = slots_paused(run_p, T1["cap"], from_s)
    tc = t_cap(run_p, from_s)
    tf = t_fit(run_p, tc)
    ws = warm_spans(run_p, from_s, bp)
    td = drained(run_p, from_s)
    t_w = ws["t_w"]
    span = judged_span(t_w, tf, from_s)
    c = pool_tokens(run_p)
    nan_input = (any(math.isnan(x) for x in (bp["reuse"], bp["q"], bp["pf"], bu["pf"], c)) or ws["any_nan"])
    from analysis.gate import owed_backlog
    return {"from_s": from_s, "owed_at_from_s": owed_backlog(run_p, from_s),     # draft 5, Band readout
            "base_p": bp, "base_u": bu, "min_baseline_pf": run_p["cfg"]["verdict"]["min_baseline_pf"],
            "established": established(run_p, from_s), "slots": sl, "pause_fault": pause_fault(sl), "t_cap": tc,
            "t_fit": tf, "pool": c, "warm": ws, "t_w": t_w, "relapse": ws["relapse"], "t_d": td,
            "lead": (td - t_w) if t_w is not None else None, "judged_span": span,
            "u_ok": u_ok(run_u, span, from_s), "u_fault": u_fault(run_u), "nan_input": nan_input,
            "t_end": float(run_p["cfg"]["workload"]["duration_s"])}


def label_from(ev: dict, separable: bool) -> str:
    """draft 4's `seed_label`, first match, with draft 6's changes (author 2026-11-23, review B1): rule 7 labels a seed
    with no t_w whose working set never fit (t_fit None) `ws_never_fit` (draft 5's `cold_over_pool`, which could only
    mean "waiting never back at baseline"), and a seed with no t_w whose working set fit by `cold_fit_by` `cold_fit`;
    rule 10 labels a re-warm before the fit (t_w < t_fit) `warm_before_fit`, never prompt evidence. Prompt is
    t_w - t_fit, which is t_w - t_cap when the working set fits at the cap. `separable`: the seed's own row of the
    gate at T1["from_s"]."""
    lo = ev["min_baseline_pf"]
    if any((not math.isnan(b["pf"])) and b["pf"] < lo for b in (ev["base_p"], ev["base_u"])):
        return "bad_baseline"                                           # 1, out of n
    if ev["established"]["cold_backlogged"] is not True:
        return "not_established"                                        # 2, out of n (None too: d1r's rule)
    if not separable:
        return "unseparable"                                            # 3, out of n
    if not ev["u_ok"]["ok"] or ev["u_fault"]:
        return "control_failed"                                         # 4, out of n (gaps 6, 10)
    if ev["pause_fault"] or ev["nan_input"]:
        return "unclear"                                                # 5
    t_w, t_f = ev["t_w"], ev["t_fit"]
    if t_w is not None and ev["relapse"]:
        return "relapse"                                                # 6
    if t_w is None:                                                     # 7
        if t_f is None:
            return "ws_never_fit"
        return "cold_fit" if t_f <= cold_fit_by(ev["t_end"]) else "unclear"
    if ev["lead"] < T1["lead_min_s"]:
        return "drain_bound"                                            # 8
    if t_f is None:
        return "unclear"                                                # 9: a t_w with no t_fit, in n
    if t_w < t_f:
        return "warm_before_fit"                                        # 10: a negative lead is no prompt evidence
    return "prompt_warm" if t_w - t_f <= T1["prompt_s"] else "delayed"  # 11, 12


def seed_label(run_p: dict, run_u: dict, separable: bool, from_s: float) -> tuple[str, dict]:
    ev = evidence(run_p, run_u, from_s)
    return label_from(ev, separable), ev


def decide(labels: list[str]) -> str:
    """`outcome`'s decision alone (draft 3 §2, first match)."""
    counted = [x for x in labels if x in COUNTED]
    n, k, m = len(counted), Counter(counted), T1["most"]
    if n < m:
        return "too_few"
    if k["prompt_warm"] >= m:
        return "no_memory"
    if sum(k[x] for x in MEMORY) >= m:
        return "memory"
    if k["drain_bound"] >= m:
        return "drain_bound"
    return "mixed"


def outcome(labels: list[str]) -> dict:
    """draft 3 §2: n = the seeds with a counted label (5-12); first match: too_few (n < most), no_memory
    (k_prompt_warm >= most), memory (k_cold_fit + k_delayed >= most), drain_bound (k_drain_bound >= most), mixed. Each
    label's k of n with its one-sided 95 % Clopper-Pearson upper bound (rule H)."""
    from analysis.stats import cp_upper
    bad = [x for x in labels if x not in LABELS]
    if bad:
        raise ValueError(f"not T1 labels: {bad}")
    counted = [x for x in labels if x in COUNTED]
    n, k = len(counted), Counter(counted)
    k_mem = sum(k[x] for x in MEMORY)
    dec = decide(labels)
    cells = {x: {"k": k[x], "n": n, "upper": cp_upper(k[x], n) if n else None} for x in COUNTED}
    cells["memory"] = {"k": k_mem, "n": n, "upper": cp_upper(k_mem, n) if n else None}
    return {"decision": dec, "n": n, "counts": dict(Counter(labels)), "cells": cells}


def predicted(n_seeds: int = len(SEEDS)) -> dict:
    """Exact over n_seeds seeds drawn independently (draft 3 §5, BELIEF inputs): out of n with the summed out_of_n
    probability, else a counted label from `counted`. Returns the outcome distribution given the gate passes, the
    unconditional no_memory (x p_gate_passes), E[k_prompt_warm], and P(control_failed in >= most seeds) at its own
    per-seed probability alone (the draft's 0.0012)."""
    p = T1["prediction"]
    if abs(sum(p["counted"].values()) - 1) > 1e-9:
        raise ValueError("the counted prediction does not sum to 1")
    p_out = sum(p["out_of_n"].values())
    cats = {"bad_baseline": p_out, **{x: (1 - p_out) * v for x, v in p["counted"].items()}}  # out of n as one label
    dist = dict.fromkeys(OUTCOMES, 0.0)
    for combo in itertools.product(cats, repeat=n_seeds):
        dist[decide(list(combo))] += math.prod(cats[c] for c in combo)
    q = p["out_of_n"]["control_failed"]
    m = T1["most"]
    p_cf = sum(math.comb(n_seeds, j) * q ** j * (1 - q) ** (n_seeds - j) for j in range(m, n_seeds + 1))
    return {"given_gate": dist, "no_memory_unconditional": dist["no_memory"] * p["p_gate_passes"],
            "e_k_prompt_warm": n_seeds * (1 - p_out) * p["counted"]["prompt_warm"], "p_control_failed_most": p_cf}


def predicted_band(n_seeds: int = len(SEEDS), p_inside: float | None = None) -> dict:
    """The band's prediction (draft 5 Appendix E3, review §4b's independent-seed arithmetic; BELIEF inputs): each seed
    out of n with the summed out_of_n probability, else counted and `inside` with p_inside, the mean of
    `band_inside_given` over `counted` (labels it omits: 0); exact over n_seeds independent seeds, by `band_outcome`'s
    rule (unjudged seeds are not modelled, as in the draft). `p_inside` overrides the mean: the draft's stated,
    rounded 0.34 gives the review's 0.120 / 0.805; the unrounded mean 0.3435 gives the draft's 0.123 / 0.802."""
    p = T1["prediction"]
    p_out = sum(p["out_of_n"].values())
    if p_inside is None:
        p_inside = sum(v * p["band_inside_given"].get(x, 0.0) for x, v in p["counted"].items())
    m = T1["most"]
    dist = dict.fromkeys(BAND_OUTCOMES, 0.0)
    for n_in in range(n_seeds + 1):
        for n_miss in range(n_seeds - n_in + 1):
            pr = (math.comb(n_seeds, n_in) * math.comb(n_seeds - n_in, n_miss) * ((1 - p_out) * p_inside) ** n_in
                  * ((1 - p_out) * (1 - p_inside)) ** n_miss * p_out ** (n_seeds - n_in - n_miss))
            dec = ("band_inside" if n_in >= m else "band_miss" if n_in + n_miss >= m else "band_too_few")
            dist[dec] += pr
    return {"p_inside": p_inside, "given_gate": dist}



def predicted_after_gate() -> dict:
    """The prediction once the gate is known (author 2026-11-23): the [REG]'s numbers. Only the P_SEEDS seeds can be
    counted (the others are unseparable, out of n, and have no P run), so the outcome and the band are taken exact over
    len(P_SEEDS) seeds, each out of n with the summed out_of_n probability less `unseparable` (the gate has decided it),
    else as `predicted` / `predicted_band`. Same BELIEF inputs; DERIVED."""
    p = T1["prediction"]
    o = {k: v for k, v in p["out_of_n"].items() if k != "unseparable"}
    p_out, n = sum(o.values()), len(T1["p_seeds"])
    cats = {"bad_baseline": p_out, **{x: (1 - p_out) * v for x, v in p["counted"].items()}}
    dist = dict.fromkeys(OUTCOMES, 0.0)
    for combo in itertools.product(cats, repeat=n):
        dist[decide(list(combo))] += math.prod(cats[c] for c in combo)
    p_in = sum(v * p["band_inside_given"].get(x, 0.0) for x, v in p["counted"].items())
    m = T1["most"]
    band = dict.fromkeys(BAND_OUTCOMES, 0.0)
    for n_in in range(n + 1):
        for n_miss in range(n - n_in + 1):
            pr = (math.comb(n, n_in) * math.comb(n - n_in, n_miss) * ((1 - p_out) * p_in) ** n_in
                  * ((1 - p_out) * (1 - p_in)) ** n_miss * p_out ** (n - n_in - n_miss))
            band["band_inside" if n_in >= m else "band_miss" if n_in + n_miss >= m else "band_too_few"] += pr
    return {"n_judgeable": n, "p_out_other": p_out, "outcome": dist, "band": band,
            "e_k_prompt_warm": n * (1 - p_out) * p["counted"]["prompt_warm"]}

# ---------------------------------------------------------------- the separation gate (rule G style; never run here)

def pause_inputs(un: dict, from_s: float) -> dict:
    """From an uncapped run identical to P up to the switch (REG-19's run of the seed): n_in, the sessions in service at
    the switch (turn 0 planned before from_s and some turn not ended by it), and work_a, their turns not ended by it."""
    q = _sessions(un)
    first = q[q.turn == 0].set_index("session_id").t_planned
    mine = q[q.session_id.isin(first[first < from_s].index)]
    open_ = ~(mine.t_end < from_s)
    return {"work_a": float(open_.sum()), "n_in": int(mine[open_].session_id.nunique())}


def t_cap_est(n_in: int, from_s: float) -> float:
    """draft 3 §4 item 1, first_order_pause's phase A (review A1, one formula): from_s + max(0, n_in - cap) / x_cold."""
    return from_s + max(0.0, float(n_in) - T1["cap"]) / T1["x_cold"]


def fluid_drain(l0: float, level: float, arrivals: np.ndarray, from_s: float, t_c: float) -> float:
    """t_d_est: a fluid total line on a grid_s grid from from_s, starting at l0, gaining the planned requests at their
    t_planned (`arrivals`), served at x_cold per second before t_c and at x_hi from it (by each grid second's start);
    the first grid time at which it is <= level. The grid runs until the line must have drained."""
    g = T1["grid_s"]
    if l0 <= level:
        return float(from_s)
    arrivals = np.sort(np.asarray(arrivals, float))
    last = max(t_c, float(arrivals.max()) if len(arrivals) else from_s)
    end = last + (l0 + len(arrivals)) / min(T1["x_cold"], T1["x_hi"]) + 2 * g
    edges = np.arange(from_s, end + g, g)
    arrived = np.diff(np.searchsorted(arrivals, edges, side="left")).astype(float)
    serve = np.where(edges[:-1] >= t_c, T1["x_hi"], T1["x_cold"]) * g
    b = float(l0)
    for i in range(len(arrived)):
        b = max(0.0, b + arrived[i] - serve[i])
        if b <= level:
            return float(edges[i + 1])
    return math.inf


def gate_seed(un: dict, from_s: float, control: dict | None = None) -> dict:
    """One seed's row of the gate, on its REG-19 uncapped burst run `un` (and, as a readout, its control): the seed
    separates when t_d_est - t_cap_est >= prompt_s + lead_min_s (540 s)."""
    from analysis.confirm import owed_series
    from analysis.gate import owed_backlog
    from analysis.k1 import windows
    x = pause_inputs(un, from_s)
    tc = t_cap_est(x["n_in"], from_s)
    l0 = owed_backlog(un, from_s)
    level = owed_backlog(un, float(un["cfg"]["trigger"]["at_s"]))
    q = un["requests"]
    arrivals = q.t_planned[q.t_planned >= from_s].to_numpy(float)
    td = fluid_drain(l0, level, arrivals, from_s, tc)
    out = {"run": un["dir"].name if "dir" in un else None, "from_s": from_s, **x, "t_cap_est": tc, "owed_at_from_s": l0,
           "healthy_level": level, "arrivals_after": int(len(arrivals)), "t_d_est": td,
           "margin": td - tc, "separates": bool(td - tc >= T1["prompt_s"] + T1["lead_min_s"])}
    if control is not None:   # readout: REG-19's control's mean total line over the baseline window (grid_s grid)
        base = windows(un["cfg"])["base"]
        out["control_mean_owed_base"] = float(np.mean(owed_series(control, np.arange(*base, T1["grid_s"]))))
        out["control"] = control["dir"].name if "dir" in control else None
    return out


def separation_gate(from_s: float) -> dict:
    """draft 3 §4: per seed, `gate_seed` on REG-19's uncapped burst run (d1.D1["uncapped"]) with REG-19's control
    beside it; passes at from_s when >= gate_k seeds separate. NEVER computed before rule F round 2 and the author
    (rule G); the pass condition is the draft's."""
    from analysis.load import RUNS, load_run
    fs = float(from_s)
    if fs not in T1["from_s_candidates"]:
        raise ValueError(f"from_s {from_s!r} is not one of {T1['from_s_candidates']}")
    rows = {sd: gate_seed(load_run(RUNS / d1.D1["uncapped"][sd]), fs, load_run(RUNS / d1.D1["controls"][sd]))
            for sd in T1["seeds"]}
    k = sum(r["separates"] for r in rows.values())
    return {"from_s": fs, "rows": rows, "k_separate": k, "passes": bool(k >= T1["gate_k"])}


def choose_from_s(gate=separation_gate) -> dict:
    """The first candidate, in the registered order, at which the gate passes; from_s None if none does (T1 is then
    not proposed). Stops at the first pass."""
    tried = []
    for fs in T1["from_s_candidates"]:
        g = gate(fs)
        tried.append(g)
        if g["passes"]:
            return {"from_s": fs, "gate": g, "tried": tried}
    return {"from_s": None, "gate": None, "tried": tried}


# ---------------------------------------------------------------- readouts (never judged)

def d1_label_with_resume(run: dict) -> dict:
    """`d1.label` on its end window with the resume queue added to the cap's admission queue (draft 3 §8.2): d1's
    cap_queue is replaced by cap_queue + resume_queue for this one call only. A readout, never judged."""
    orig = d1.cap_queue

    def both(r, t):
        return orig(r, t) + resume_queue(r, t)

    with mock.patch.object(d1, "cap_queue", both):
        return d1.label(run)


def readouts(run_p: dict, run_u: dict, ev: dict) -> dict:
    """Reported beside the label, never judged (draft 3 §2): t_w - t_cap, the lead, the total line at t_w, U's reuse on
    P's judged span, ws_slot / slots held / both queues over time, the queues and latency from planned time over P's
    warm hold, d1.label with the resume queue, and the completion rate from t_w to the end against λ.
    Draft 6 (review M3): over [from_s, t_w) (to duration_s without a t_w), the server's mean KV usage and its
    preemptions (`d1.preemptions`), its own prefix hit rate (`load._level`), `k1.reuse` and `reuse_unpaused`, beside
    the baseline's (`baseline`); the per-span values are in `evidence`'s warm spans."""
    from analysis.gate import completion_rate, owed_backlog
    from analysis.k1 import reuse
    from analysis.load import _level
    fs, t_w, t_end = ev["from_s"], ev["t_w"], ev["t_end"]
    grid = np.arange(fs, t_end, T1["grid_s"])
    span = ev["judged_span"]
    b_w = t_end if t_w is None else t_w
    lv = _level(run_p, fs, b_w) if b_w > fs else {"hit": math.nan, "kv": math.nan}
    bp = ev["base_p"]
    out = {"t_w_minus_t_cap": None if t_w is None or ev["t_cap"] is None else t_w - ev["t_cap"],
           "before_t_w": {"span": (fs, b_w), "kv": lv["kv"],
                          "preemptions": d1.preemptions(run_p, fs, b_w) if b_w > fs else math.nan,
                          "hit_server": lv["hit"], "reuse_k1": reuse(run_p, fs, b_w),
                          "reuse_unpaused": reuse_unpaused(run_p, fs, b_w)},
           "baseline": {k: bp.get(k) for k in ("reuse", "reuse_k1", "hit_server", "kv", "q")},
           "lead": ev["lead"], "owed_at_t_w": None if t_w is None else owed_backlog(run_p, t_w),
           "u_on_judged_span": {"span": span, "reuse_unpaused": reuse_unpaused(run_u, *span),
                                "reuse_k1": reuse(run_u, *span)},
           "series": {"t": grid.tolist(), "ws_slot": ws_slot(run_p, grid).tolist(),
                      "slots_held": slots_held(run_p, grid).tolist(), "in_service": d1r.in_service(run_p, grid).tolist(),
                      "resume_queue": resume_queue(run_p, grid).tolist(),
                      "cap_queue": d1.cap_queue(run_p, grid).tolist()},
           "d1_label_with_resume": d1_label_with_resume(run_p)["label"],
           "completion_after": None if t_w is None else completion_rate(run_p, t_w, t_end), "lam": d1.D1["lam"]}
    if t_w is not None:
        a, b = t_w, t_w + hold_s()
        g = np.arange(a, b, T1["grid_s"])
        s = _sessions(run_p)
        w = s[(s.t_planned >= a) & (s.t_planned < b) & s.t_first_token.notna()]
        lat = (w.t_first_token - w.t_planned).astype(float)
        out["warm_hold"] = {"span": (a, b), "resume_queue_mean": float(resume_queue(run_p, g).mean()),
                            "cap_queue_mean": float(np.nanmean(d1.cap_queue(run_p, g))),
                            "latency_from_planned_median": float(lat.median()) if len(lat) else math.nan,
                            "latency_from_planned_mean": float(lat.mean()) if len(lat) else math.nan}
    return out


# ---------------------------------------------------------------- the band (judged) and U's end window (readout)

BAND = ("inside", "early", "late", "unjudged")
BAND_OUTCOMES = ("band_inside", "band_miss", "band_too_few")


def drain_band(ev: dict, gate_row: dict) -> dict:
    """draft 5, Band, one seed. Predicted D_est = t_d_est - from_s, t_d_est from the committed probe's gate row of the
    seed (the gate's fluid line, `fluid_drain`); measured D = t_d - from_s on P (`drained`, the t_d the lead uses).
    `inside` (draft 6, author 2026-11-23, review M2) needs both |D / D_est - 1| <= band_tol (from the switch) and
    |(t_d - t_cap) / (t_d_est - t_cap_est) - 1| <= band_tol (from the cap), so a late cap offset by a fast drain
    (review Trace C) is not a pass. Otherwise `early` or `late` by the sign of a deviation outside the band (the
    switch's first). With t_d = inf: `late` if from_s + (1 + band_tol) D_est <= duration_s, else `unjudged`. A D_est
    <= 0 or a t_d of None is `unjudged` (no ratio exists; neither case is in the draft); so is a seed inside on the
    switch's clause whose deviation from the cap cannot be formed (no t_cap, or t_d_est <= t_cap_est), since its
    second clause cannot be judged.
    Readouts: both signed deviations, and P's total line at from_s against the gate's L0."""
    fs, tol = float(ev["from_s"]), T1["band_tol"]
    d_est = float(gate_row["t_d_est"]) - fs
    t_d = None if ev["t_d"] is None else float(ev["t_d"])
    out = {"d_est": d_est, "d": None if t_d is None else t_d - fs, "deviation": None, "deviation_from_t_cap": None,
           "owed_at_from_s": ev.get("owed_at_from_s"), "gate_l0": gate_row.get("owed_at_from_s")}
    if t_d is None or not d_est > 0:
        return {**out, "band": "unjudged"}
    if math.isinf(t_d):
        return {**out, "band": "late" if fs + (1 + tol) * d_est <= float(ev["t_end"]) else "unjudged"}
    d = t_d - fs
    out["deviation"] = d / d_est - 1
    e_cap = float(gate_row["t_d_est"]) - float(gate_row["t_cap_est"])
    cap_ok = ev.get("t_cap") is not None and e_cap > 0
    if cap_ok:
        d_cap = t_d - float(ev["t_cap"])
        out["deviation_from_t_cap"] = d_cap / e_cap - 1
    # |x / e - 1| <= tol as |x - e| <= tol e (e > 0): the ratio form rounds 0.85 out
    if abs(d - d_est) > tol * d_est:
        return {**out, "band": "early" if d < d_est else "late"}
    if not cap_ok:
        return {**out, "band": "unjudged"}
    if abs(d_cap - e_cap) > tol * e_cap:
        return {**out, "band": "early" if d_cap < e_cap else "late"}
    return {**out, "band": "inside"}


def band_outcome(rows: list[dict]) -> dict:
    """draft 5, Band, across seeds. `rows`: per seed {"label", "band"} (band: `drain_band`'s output, or None). Judged:
    the counted seeds (labels 5-12) less any `unjudged`, n_band of them. band_inside if k_inside >= most;
    band_miss if n_band >= most and k_inside < most; band_too_few if n_band < most. k_inside of n_band with its
    one-sided 95 % Clopper-Pearson upper bound; k_inside among the prompt_warm seeds is a readout."""
    from analysis.stats import cp_upper
    m = T1["most"]
    judged = [r for r in rows if r["label"] in COUNTED and r["band"] is not None and r["band"]["band"] != "unjudged"]
    n = len(judged)
    k = sum(r["band"]["band"] == "inside" for r in judged)
    if k >= m:
        dec = "band_inside"
    elif n >= m:
        dec = "band_miss"
    else:
        dec = "band_too_few"
    pw = [r for r in judged if r["label"] == "prompt_warm"]
    return {"decision": dec, "k_inside": k, "n_band": n, "upper": cp_upper(k, n) if n else None,
            "counts": dict(Counter(r["band"]["band"] for r in judged)),
            "prompt_warm": {"k_inside": sum(r["band"]["band"] == "inside" for r in pw), "n": len(pw)},
            "deviations": [r["band"]["deviation"] for r in judged]}


def drained_split(rows: list[dict]) -> dict:
    """A readout beside `outcome`, never judged (draft 6, review 4a): per counted label, the seeds whose line drained
    (t_d finite) and those whose line never drained (t_d = inf, `drained`), so that `no_memory`'s "re-warmed >= 300 s
    before the owed line drained" is not read off a line that never drained. `rows`: per seed {"seed", "label",
    "t_d"}. `prompt_warm` is no_memory's count."""
    out = {}
    for r in rows:
        if r["label"] not in COUNTED:
            continue
        t_d = r["t_d"]
        c = out.setdefault(r["label"], {"drained": [], "never_drained": [], "unknown": []})
        key = ("unknown" if t_d is None or math.isnan(t_d) else "never_drained" if math.isinf(t_d) else "drained")
        c[key].append(r["seed"])
    pw = out.get("prompt_warm", {"drained": [], "never_drained": []})
    return {"by_label": out, "prompt_warm_drained": len(pw["drained"]),
            "prompt_warm_never_drained": len(pw["never_drained"])}


def _platform(run: dict) -> str | None:
    return (run.get("env") or {}).get("platform")


def reference_platforms() -> dict:
    """draft 5, U's end window: the `platform` strings (env.json) of K1 arm A's open runs (`k1.open_pattern(2.0, s)`
    on REG-20's seeds, every valid run) and of D1's capped runs (`d1.capped_run` at `d1r.d1_cap`, the counted run of
    each D1 seed), by run name. Sealed folders are skipped by `load.find_runs`."""
    from analysis.k1 import REG20, open_pattern
    from analysis.load import find_runs
    arm_a = {r["dir"].name: _platform(r) for sd in REG20["closed_seeds"]
             for r in find_runs(open_pattern(REG20["a_lam"], sd))}
    caps = {}
    for sd in d1.D1["seeds"]:
        r, _ = d1.capped_run(sd, d1r.d1_cap(sd))
        if r is not None:
            caps[r["dir"].name] = _platform(r)
    return {"k1_arm_a": arm_a, "d1": caps}


def u_end_readout(u_runs: dict, refs: dict | None = None) -> dict:
    """draft 5, U's end window (a readout, never judged). Seeds: every seed with a valid U run (`u_runs`: seed -> the
    counted U run, or None), no `u_fault` and baseline pf >= min_baseline_pf, whatever P's label. k: those U runs
    `k1.cold_backlogged` on their end window [3480, 4080) against their own baseline (REG-20's computation). A None
    is listed and left out of n. k of n with its one-sided 95 % Clopper-Pearson upper bound and, as a lower bound,
    1 - cp_upper(n - k, n). Beside it: each U run's env.json `platform`, and `refs` (`reference_platforms`).
    `platforms_match` (draft 6, review 4c): True when K1 arm A's runs share one platform string, D1's capped runs share
    the same one, and every counted U run's string equals it; False otherwise (a missing string included); None
    without `refs` or with no counted U run. `blocks_appendix_f_rewrite` = not platforms_match: the registration can
    say that a mismatch (or no comparison) blocks Appendix F's rewrite. What U checks: REG-19's seeds 18-19 already
    ran on libc6 8.12 (FILE docs/status.md:69), so U checks the kernel change (147 -> 149) on seeds 15-19 and the libc
    change on seeds 15-17 only."""
    from analysis.k1 import cold_backlogged, windows
    from analysis.stats import cp_upper
    seeds, excluded, none = {}, {}, []
    for sd, u in sorted(u_runs.items()):
        if u is None:
            excluded[sd] = "no valid U run"
            continue
        if u_fault(u):
            excluded[sd] = "u_fault"
            continue
        pf = baseline(u)["pf"]
        if not pf >= u["cfg"]["verdict"]["min_baseline_pf"]:
            excluded[sd] = f"baseline pf {pf}"
            continue
        w = windows(u["cfg"])
        cb = cold_backlogged(u, {"base": w["base"], "end": w["end"]})["cold_backlogged"]
        if cb is None:
            none.append(sd)
            continue
        seeds[sd] = {"run": u["dir"].name if "dir" in u else None, "cold_backlogged": bool(cb),
                     "platform": _platform(u)}
    n, k = len(seeds), sum(r["cold_backlogged"] for r in seeds.values())
    match = None
    if refs is not None and n:
        arm_a, caps = set(refs.get("k1_arm_a", {}).values()), set(refs.get("d1", {}).values())
        u = {r["platform"] for r in seeds.values()}
        match = bool(len(arm_a) == 1 and arm_a == caps == u and None not in u)
    return {"k": k, "n": n, "upper": cp_upper(k, n) if n else None,
            "lower": 1 - cp_upper(n - k, n) if n else None, "seeds": seeds, "none": none, "excluded": excluded,
            "platforms": sorted({r["platform"] for r in seeds.values()}, key=str), "reference": refs,
            "platforms_match": match, "blocks_appendix_f_rewrite": match is not True}


# ---------------------------------------------------------------- the registered computation

def arm_run(arm: str, seed: int, from_s: float | None) -> tuple[dict | None, list[str]]:
    """The arm's counted run at the seed (the earliest valid one; `load.find_runs`, valid only) and any later valid
    ones. Its resolved config must equal `resolved(arm, seed, from_s)` (run id and source path aside)."""
    from analysis.load import find_runs
    runs = sorted(find_runs(run_pattern(arm, seed)), key=lambda r: r["dir"].name)
    if not runs:
        return None, []
    r = runs[0]
    got = {k: v for k, v in r["cfg"].items() if k not in ("run_id", "source_config")}
    diff = d1._diff_keys(got, resolved(arm, seed, from_s if arm == "P" else None))
    if diff:
        raise ValueError(f"{r['dir'].name}: config differs from T1's arm {arm} at {diff}")
    return r, [x["dir"].name for x in runs[1:]]


def sealed() -> list[str]:
    import analysis.load as L
    if not L.RUNS.exists():
        return []
    tags = tuple(f"_{s}" for s in T1["suffix"].values())
    return sorted(d.name for d in L.RUNS.iterdir() if d.name.endswith(tags) and withheld.sealed_by(d.name) == "T1")


PROBE_DIR = REPO / "docs" / "probe" / "t1_gate"


def committed_gate(probe_dir: Path | None = None) -> dict:
    """The gate's committed output (draft 4, Gate): the one JSON file in docs/probe/t1_gate/, as
    `python -m analysis.t1 choose` prints it (`choose_from_s`: from_s, gate, tried). Refuses if there is not exactly
    one, or if it names no passing from_s."""
    import json
    d = probe_dir or PROBE_DIR
    files = sorted(d.glob("*.json")) if d.exists() else []
    if len(files) != 1:
        raise ValueError(f"{d}: need exactly one committed gate output (*.json), found {len(files)}")
    out = json.loads(files[0].read_text())
    if out.get("from_s") is None or not (out.get("gate") or {}).get("passes"):
        raise ValueError(f"{files[0].name}: the committed gate names no passing from_s")
    return out


def check_gate(gate: dict, probe: dict, from_s: float) -> None:
    """draft 4, gap 16: `table` refuses if the recomputed gate at T1["from_s"] does not pass, if the probe was computed
    at another from_s, or if any seed's `separates` differs from the committed probe's."""
    if not gate["passes"]:
        raise ValueError(f"the recomputed separation gate at {from_s:g} s does not pass")
    if float(probe["from_s"]) != float(from_s):
        raise ValueError(f"the committed gate chose {probe['from_s']}, T1['from_s'] is {from_s}")
    got = {int(sd): bool(r["separates"]) for sd, r in gate["rows"].items()}
    want = {int(sd): bool(r["separates"]) for sd, r in probe["gate"]["rows"].items()}
    if got != want:
        raise ValueError(f"the recomputed gate's per-seed `separates` {got} differs from the committed {want}")


def table(probe_dir: Path | None = None) -> dict:
    """T1's registered computation, once after the queue: `AFTERSHOCK_UNSEAL=T1 uv run python -m analysis.t1 table`.
    Refuses while any T1 folder is sealed or T1["from_s"] is unset, and if the recomputed gate does not agree with the
    committed one (`check_gate`). Per arm and seed the earliest valid run counts (later ones listed); a seed with no
    valid run of either arm is `not_run`, out of n (so 16 and 18, which have no P run, author 2026-11-23; their U runs
    still count in `u_end`). A U fault labels its seed `control_failed` (draft 4, gap 10).
    Beside the outcome, `drained_split` (draft 6): the counted seeds whose line drained and those whose never did."""
    fs = T1["from_s"]
    if fs is None:
        raise ValueError("T1['from_s'] is set only in the [REG], after the separation gate")
    s = sealed()
    if s:
        raise withheld.WithheldError(f"{len(s)} T1 folder(s) sealed; set AFTERSHOCK_UNSEAL=T1")
    probe = committed_gate(probe_dir)
    gate = separation_gate(fs)
    check_gate(gate, probe, fs)
    rows, not_run, later, u_runs = {}, [], {}, {}
    probe_rows = {int(sd): r for sd, r in probe["gate"]["rows"].items()}
    for sd in T1["seeds"]:
        (p, lp), (u, lu) = arm_run("P", sd, fs), arm_run("U", sd, None)
        u_runs[sd] = u
        for arm, lst in (("P", lp), ("U", lu)):
            if lst:
                later[config_name(arm, sd)] = lst
        if p is None or u is None:
            not_run.append(sd)
            continue
        lab, ev = seed_label(p, u, gate["rows"][sd]["separates"], fs)
        rows[sd] = {"seed": sd, "P": p["dir"].name, "U": u["dir"].name, "label": lab, "u_fault": ev["u_fault"],
                    "band": drain_band(ev, probe_rows[sd]), "evidence": ev, "readouts": readouts(p, u, ev)}
    return {"from_s": fs, "gate": gate, "outcome": outcome([r["label"] for r in rows.values()]),
            "drained_split": drained_split([{"seed": sd, "label": r["label"], "t_d": r["evidence"]["t_d"]}
                                            for sd, r in rows.items()]),
            "band": band_outcome(list(rows.values())), "u_end": u_end_readout(u_runs, reference_platforms()),
            "rows": rows, "not_run": not_run, "u_faults": [sd for sd, r in rows.items() if r["u_fault"]],
            "later_valid_not_counted": later}


if __name__ == "__main__":
    import json
    import sys
    cmd = sys.argv[1:]
    if cmd == ["table"]:
        print(json.dumps(table(), indent=1, default=str))
    elif cmd == ["choose"]:     # the gate, once, after its pass condition is committed; output to docs/probe/t1_gate/
        print(json.dumps(choose_from_s(), indent=1, default=str))
    elif cmd == ["predicted"]:
        print(json.dumps(predicted(), indent=1))
    elif cmd == ["predicted_after_gate"]:
        print(json.dumps(predicted_after_gate(), indent=1))
    elif cmd == ["predicted_band"]:
        print(json.dumps(predicted_band(), indent=1))
    elif cmd == ["configs", "U"]:
        print("\n".join(str(p) for p in write_configs("U")))
    elif len(cmd) == 3 and cmd[:2] == ["configs", "P"]:
        print("\n".join(str(p) for p in write_configs("P", float(cmd[2]))))
    else:
        sys.exit("usage: python -m analysis.t1 table | choose | predicted | predicted_after_gate | predicted_band | configs U"
                 " | configs P <from_s>")
