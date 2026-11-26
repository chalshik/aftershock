# NVIDIA AIPerf

- Repo: github.com/ai-dynamo/aiperf
- Commit read: `d3ffc40d5488181ef5d4f11e1d4ce9fe4a2a83a8` (HEAD of default branch, read 2026-11-25 via shallow clone). `pyproject.toml:14` gives `version = "0.14.0"`.
- Latest release at time of reading: `v0.14.0` (published 2026-11-20, GitHub releases API)
- The docs quoted below are the repo's `docs/` tree at that commit, which is the source of docs.nvidia.com/aiperf. I did not fetch the rendered site separately.

## Mode selection

`docs/benchmark-modes/timing-modes-reference.md:23-28`:

```
When multiple options are specified, AIPerf uses this priority:

1. `--fixed-schedule` or mooncake_trace dataset → Timestamp-based scheduling
2. `--user-centric-rate` → Per-user turn gap scheduling
3. `--request-rate` → Rate-based scheduling with arrival patterns
4. `--concurrency` only → Burst mode (as fast as possible within limits)
```

## 1. Synthetic multi-turn (`--conversation-turn-mean > 1`) under `--request-rate` or `--concurrency`

`docs/tutorials/multi-turn.md:62-64`:

```
- **`--conversation-turn-delay-mean <MS>`**: Average delay between turns in milliseconds
  - Default: 0ms
  - Simulates realistic user "think time" between messages
```

`docs/tutorials/multi-turn.md:446-448`:

```
   - Execute turn 1 (first turn, no delay), wait for response
   - Append assistant's response to conversation history
   - Apply turn delay (simulating user reading/thinking time)
```

`docs/tutorials/multi-turn.md:246-255`:

```
By default, `--request-rate` counts **all root-session turns**, including continuations.
After a turn completes and its think time elapses, the continuation waits for
the next arrival slot and takes priority over new sessions. ...

Add `--continuation-dispatch direct` to pace **session starts** instead.
Continuations become eligible after the previous completion plus the configured
think time, without consuming another rate slot.
```

`docs/tutorials/multi-turn.md:264`: "The default is `rate_slot`."

**Classification: closed.** The next turn is sent after the previous reply completes, plus a think time measured from that completion (default 0 ms). Under `--request-rate` with the default `rate_slot` it also waits for the next global arrival slot. With `--continuation-dispatch direct` it is sent right after completion plus think time.

## 2. Trace replay (`--fixed-schedule`, auto-enabled for `mooncake_trace` files whose first record has `timestamp`)

`docs/benchmark-modes/trace-replay.md:97`:

```
When you supply a trace dataset (`--custom-dataset-type mooncake_trace`, `bailian_trace`, `burst_gpt_trace`, ...) and the file's first record carries a `timestamp` field, AIPerf automatically switches the profiling phase to fixed-schedule mode
```

Source, `src/aiperf/timing/strategies/fixed_schedule.py:5-7`:

```
Replays conversation traces at precise timestamps from dataset metadata.
First turns sent by main loop at absolute timestamps, subsequent turns
dispatched using delay_ms or timestamp_ms from metadata.
```

`src/aiperf/timing/strategies/fixed_schedule.py:171` (`handle_credit_return`, which runs when the previous turn's credit comes back), `:194-207`:

```
        if next_meta.timestamp_ms is not None:
            self._scheduler.schedule_at_perf_sec(
                self.schedule_target_perf_sec(next_meta.timestamp_ms),
                self._credit_issuer.issue_credit(turn),
            )
        elif next_meta.delay_ms is not None:
            self._scheduler.schedule_later(
                next_meta.delay_ms / MILLIS_PER_SECOND,
                self._credit_issuer.issue_credit(turn),
            )
        else:
            self._scheduler.execute_async(
                self._credit_issuer.issue_credit(turn),
            )
```

`src/aiperf/common/loop_scheduler.py:270` (docstring of `schedule_at_perf_sec`): "If in the past, delegates to execute_async()." The code is at `:283`, `if offset_sec <= 0:`.

`src/aiperf/workers/worker.py:1724` (`_process_credit` docstring): "Process a credit (1 credit = 1 request)."

Mooncake row fields, `src/aiperf/dataset/loader/models.py:346-352`: `timestamp` is "The timestamp of a request in milliseconds" and `delay` is "Amount of milliseconds to wait before sending the turn".

The Baseten loader documents the same rule in prose. `docs/cli-options.md:873` (`--open-loop-replay`):

```
Open-loop replay (the default): each session starts at its absolute, speedup-scaled recorded timestamp; continuation turns fire at max(recorded timestamp, prior-turn completion). Pass `--no-open-loop-replay` for closed-loop back-pressure: continuation turns fire a think-time (recorded start-to-start gap minus recorded e2e duration) after the prior turn completes ... Only honored by the baseten_trace loader.
```

`docs/tutorials/baseten-trace.md:58`:

```
**Known limitation:** without `--open-loop-strict`, a session's continuation turns fire at max(scheduled time, prior turn completion) — an open-loop timestamp inside a session cannot preempt an in-flight prior turn.
```

`docs/tutorials/baseten-trace.md:56` (`--open-loop-strict`): "...treats every trace row as an independent single-turn session so ALL requests ... fire at their absolute recorded timestamps, even if earlier turns of the same recorded session have not completed."

**Classification: held (default for timestamped multi-turn trace replay).** A continuation turn is scheduled only when the previous turn's credit returns, and then at its recorded timestamp. A timestamp already in the past fires at once, so the turn goes out at max(recorded time, prior completion). Rows that carry `delay` instead of `timestamp` are closed with a think time measured from the prior completion. Rows with neither are closed with zero delay. Baseten traces have two non-default variants: `--no-open-loop-replay` is closed, and `--open-loop-strict` is open (per-turn session grouping is dropped).

## 3. User-centric rate (`--user-centric-rate`)

`docs/cli-options.md:1467`: "Each user has a gap = num_users / qps between turns. Users block on their previous turn (no interleaving within a user)."

`docs/tutorials/user-centric-timing.md:81`: "Each user's turns are spaced at least `num_users / QPS` seconds apart (exactly this interval when responses complete before the scheduled time)."

**Classification: held.** Each turn is scheduled a fixed gap after the user's previous turn and blocks on the previous reply, so it is sent at max(schedule, previous completion).

## 4. Exgentic v2 public dataset (agentic traces)

`docs/benchmark-datasets.md:111`:

```
Fixed-schedule mode emits each recorded call as an independently scheduled one-turn request using its start offset from the source session. ... Without `--fixed-schedule`, each source session remains a closed-loop multi-turn conversation: AIPerf waits for one live response before applying the recorded residual delay and sending the next request.
```

## Overall classification: configurable

Synthetic multi-turn is closed (think time after the reply, default 0, plus a rate slot under `--request-rate`). Timestamped trace replay defaults to held. Other modes: closed (`delay` rows, `--no-open-loop-replay`, non-fixed-schedule Exgentic), held (`--user-centric-rate`) and open (`--open-loop-strict` for Baseten, fixed-schedule Exgentic, where each call becomes an independent one-turn request). The agentic-replay and adaptive strategies (`src/aiperf/timing/strategies/agentic_replay.py`, `adaptive_scale*.py`) were not read.
