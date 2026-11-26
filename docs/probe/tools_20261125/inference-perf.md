# kubernetes-sigs/inference-perf

- Repo: github.com/kubernetes-sigs/inference-perf
- Commit read: `083ff6daf4a9899601623f69569103a0a1621628` (HEAD of default branch, read 2026-11-25 via shallow clone). `pyproject.toml:3` gives `version = "0.7.0"`.
- Latest release at time of reading: `v0.7.0` (published 2026-09-15, GitHub releases API)

## Load types and default

`inference_perf/config/loadgen/config.py:28-34`: `LoadType` has the values `constant`, `poisson`, `trace_replay`, `concurrent`, `trace_session_replay` and `batch`. `:285`: `type: LoadType = Field(default=LoadType.CONSTANT, ...)`. `docs/loadgen.md:14`: "Adds Requests at Constant / Poisson distribution for desired QPS".

## A. `conversation_replay` data with the `concurrent` load (synthetic multi-turn agentic)

`docs/conversation_replay.md:3-4`: "N conversations run concurrently and each conversation's turns run sequentially as fast as the server will serve them."

`docs/conversation_replay.md:92-97`:

```
The `concurrent` load type plus `preferred_worker_id` ensures each conversation is pinned to a
worker so its turns run strictly in order.
...
A conversation keeps its slot while simulated tool work runs, and
its next turn is dispatched as soon as that delay finishes.
```

`docs/conversation_replay.md:175-176`:

```
After each assistant response, the conversation sleeps for the sampled number of seconds
before releasing its session lock and starting the next turn.
```

**Classification: closed.** The next turn goes out after the previous response, plus an optional `tool_call_latency_sec` delay measured from that response. The number of conversations is set by `concurrency_level`.

## B. `shared_prefix` with `enable_multi_turn_chat: true` under `constant` / `poisson` (rate-based) load

`inference_perf/config/datagen/config.py:95-97`:

```
    enable_multi_turn_chat: bool = Field(
        default=False,
        description="Send each group's prompts as consecutive turns of one chat conversation. Requires api.type completion.",
```

The dispatcher sleeps until the request's scheduled time, `inference_perf/loadgen/load_generator.py:416-419`:

```
                        current_time = time.perf_counter()
                        sleep_time = request_time - current_time
                        if sleep_time > 0:
                            try:
```

then calls `await self.client.process_request(request_data, stage_id, request_time, lora_adapter)` (`:482`). Building the request body (`inference_perf/apis/user_session.py:176`, `self._session_context = await self.user_session.get_context(self.target_round)`) blocks on the session's in-flight lock, `user_session.py:95-99`:

```
        if not self._waiting_rounds.empty() or self._in_flight.locked():
            future: asyncio.Future[bool] = asyncio.Future()
            self._waiting_rounds.put_nowait(future)
            await future
        await self._in_flight.acquire()
```

The lock is released in `update_context` after the response (`:150-157`), which is called from `process_response` (`:300-307`).

**Classification: held (DERIVED from code, not stated in the docs).** Each turn is dispatched at its constant or Poisson schedule time, and if the session's previous turn is still in flight it waits for that turn's response. It is therefore sent at max(scheduled time, previous completion). Waiting rounds are released in FIFO order. I did not check whether the client stamps the request start time before or after this wait.

## C. OTel trace replay (`load.type: trace_session_replay`)

`docs/otel_trace_replay.md:629-631`:

```
4. **Preserve timing** — The delay between when predecessors finish and when each call starts is recorded as `wait_ms`
...
Root events (no predecessors) start immediately. All others wait for their predecessors, then observe `wait_ms` before dispatching.
```

`docs/otel_trace_replay.md:129`: `max_wait_ms` (default 15000) "Caps the delay between predecessor completion and event dispatch".

**Classification: closed.** The recorded wait is measured from the predecessor's completion.

## D. Weka trace replay

`docs/weka_trace_replay.md:12-13`:

```
4. **Session-based Execution**: A thread pool runs sessions concurrently. Within each session, nodes are executed as soon as their parents complete.
5. **Think-Time Simulation**: Think times and idle gaps between turns are simulated and capped using `trace_idle_gap_cap_seconds`.
```

**Classification: closed.** A node runs after its parent completes, with a capped think time.

## Overall: configurable

There is no single multi-turn default. The default load type is `constant`, and `enable_multi_turn_chat` defaults to false, so the out-of-the-box workload is single-turn and open (constant rate). The documented multi-turn paths (conversation_replay, OTel, Weka) are closed. shared_prefix multi-turn under a rate load is held, by my reading of the code. I did not read the plain `trace_replay` load type (`docs/loadgen.md:273`) for multi-turn behavior.
