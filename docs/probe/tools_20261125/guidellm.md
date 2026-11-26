# GuideLLM

- Repo: github.com/vllm-project/guidellm
- Commit read: `20abd43d0c30625d998ed3dad839ec754a9c24fc` (HEAD of default branch, read 2026-11-25 via shallow clone)
- Latest release at time of reading: `v0.8.1` (published 2026-10-08, GitHub releases API). HEAD may be ahead of v0.8.1, and I did not diff the two.
- `--profile` has no default. `src/guidellm/schemas/benchmark/entrypoints.py:306-310` declares `profile: ProfileArgs = Field(...)` with `examples=[{"kind": "sweep", ...}]` and no `default`, so the user must pick a profile.

## Multi-turn exists

`docs/en/guides/multiturn.md:39-47`:

```
### How Multiturn Orchestration Works

When executing a multiturn benchmark, GuideLLM:

1. **Sends a turn** (prefix + prompt_0) to the model and captures the response
2. **Records the exchange** Stores the request and response in the aggregator as a single request
3. **Builds conversation history** by combining the requests and the model's responses
4. **Sends the next turn** (prompt_i) along with the conversation history
```

## Source: a turn becomes eligible only after its parent completes

`src/guidellm/scheduler/dag.py:172-173` (docstring of `mark_completed`):

```
        When a child's last parent completes, think time starts: the child's
        ``available_after`` is set to ``now + unlocking_parent.requeue_delay``.
```

`:191-193`:

```
        delay = self._graph.nodes[node_id].settings.requeue_delay or 0.0
        parent_completed_at = time.time()
        unlock_at = parent_completed_at + delay
```

The worker loop (`src/guidellm/scheduler/worker.py:363-376`) takes the strategy's next slot time (`request_time = await self.strategy.next_request_time(...)`, `:366`) and starts a task that first waits for a ready node (`state, node_id = await self._get_next_ready_node()`, `:493`).

## Per profile

**users** (`docs/en/getting-started/benchmark.md:178`):

```
The next turn is scheduled `delay` seconds after the previous response finishes, like a person who reads the reply and then thinks. This is closed-loop pacing: as latency rises, each user slows down, so the offered request rate falls near capacity.
```

The result is closed, with a think time measured from the end of the reply.

**replay** (trace) profile. `docs/en/getting-started/benchmark.md:269`:

```
`schedule_turn=idle_gap` (the default) keeps the gap that followed the recorded duration: if a request was recorded as 1 second and the next timestamp is 5 seconds later, the next request starts 4 seconds after this one actually finishes. ... `schedule_turn=timestamp` runs each request at its trace time and holds it only while a prior turn is still in progress.
```

`src/guidellm/scheduler/strategies.py:910-911` (`TraceReplayStrategy`): `schedule_turn: Literal["timestamp", "idle_gap"] = Field(default="idle_gap",`. `:974-975`:

```
            gap = child_timestamp - parent_timestamp - parent_duration
            targets.append(parent_end + self.time_scale * max(gap, 0.0))
```

The default `idle_gap` is closed: the recorded idle gap is measured from the parent's actual end. `schedule_turn=timestamp` is held: the trace time, or later if the prior turn is still running. `docs/en/guides/trace_replay.md:91` adds that `trace_synthetic` and `mooncake` replay each row as an independent, single-request conversation. Those formats are therefore open single-turn: each row is sent at its timestamp.

**constant / poisson / concurrent / synchronous / throughput** with multi-turn data. `docs/en/guides/multiturn.md:71`: "All standard benchmarking arguments apply to multiturn tasks, such as `--profile`, profile rate parameters ... Any options that operate on "requests" will treat each turn as a separate request". From the code above (DERIVED from reading, not stated in the docs): a follow-up turn becomes eligible at parent completion + `requeue_delay`, and is then sent by the next free slot of the worker's rate or concurrency schedule. That is closed in dependency, with dispatch paced by the profile.

**Classification: configurable.** No profile is the default (the user must choose one). Every multi-turn path gates a follow-up turn on the previous turn's completion. The users profile and replay's default `idle_gap` are closed, with the delay measured from the reply's end. `replay,schedule_turn=timestamp` is held. Rate and concurrency profiles are closed in dependency and take the profile's next slot (my inference from the source). Mooncake and trace_synthetic traces are open single-turn. I did not verify whether a rate-profile slot that falls before a turn is eligible is consumed or carried forward.
