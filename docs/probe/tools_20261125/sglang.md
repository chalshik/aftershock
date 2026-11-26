# SGLang: `python -m sglang.benchmark.serving` (bench_serving) and `benchmark/hicache/bench_multiturn.py`

- Repo: github.com/sgl-project/sglang
- Commit read: `1a6164afdaa7a5738527475897c2391ec3453c7e` (HEAD of default branch, read 2026-11-25 via shallow clone)
- Latest release at time of reading: `v0.5.23` (published 2026-11-02, GitHub releases API)
- Path note: at this commit there is no `python/sglang/bench_serving.py`. The serving benchmark is `python/sglang/benchmark/serving.py`, which the repo docs invoke as `python3 -m sglang.benchmark.serving` (e.g. `docs/cookbook/autoregressive/OpenBMB/MiniCPM-V-4_6.mdx:378`). `benchmark/` has no `multi_turn_chat` directory. The multi-turn benchmark is `benchmark/hicache/bench_multiturn.py`.

## A. bench_serving (`python/sglang/benchmark/serving.py`)

### A1. Single-turn arrivals (default)

`:2472-2476`:

```
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
```

`get_request`, `:1125-1132`:

```
            if request_rate == float("inf"):
                # If the request rate is infinity, then we don't need to wait.
                continue

            # Sample the request interval from the exponential distribution.
            interval = np.random.exponential(1.0 / request_rate)
            # The next request will be sent after the interval.
            await asyncio.sleep(interval)
```

`--use-trace-timestamps` (`:1097-1119`) sleeps to each row's recorded timestamp. `--max-concurrency` adds a semaphore (`:1435`).

### A2. Multi-turn rows (agentic-trace dataset, `--gsp-num-turns > 1`, any row whose prompt is a list of rounds)

Detection and wrapping, `:1431`:

```
        request_func = wrap_multi_turn_request_func(request_func, backend=backend)
```

`wrap_multi_turn_request_func`, `:1350-1379`, condensed (the elided lines `:1351-1371` normalize the round's messages and build the request input):

```
        for round_index in range(len(prompts)):
            ...
            output = await request_func(
                inner_input, pbar=pbar if round_index == len(prompts) - 1 else None
            )
            outputs.append(output)

            prev_messages.append(
                {"role": "assistant", "content": output.generated_text}
            )
```

`python/sglang/benchmark/datasets/agentic_trace.py:38-40`:

```
    the list of per-turn message deltas; ``bench_serving`` detects this shape as
    multi-turn and replays each conversation round by round, feeding the
    server's real assistant reply back into the next round's history.
```

The whole conversation is one task from `get_request`, so the conversation start follows the request-rate schedule (A1), but each round inside it is awaited.

**Classification (multi-turn rows): closed, with zero think time.** Each round is sent only after the previous round's reply has finished (`await request_func`). There is no delay between rounds. Conversation starts are open: Poisson at `--request-rate`, or all at t=0 by default (`inf`).

### A3. Mooncake trace replay with `--mooncake-num-rounds > 1`

`:1565`: `print("Using time-based Mooncake request scheduler, ignoring --request-rate.")`

`python/sglang/benchmark/datasets/mooncake.py:78-79`:

```
        # Once the session starts, generate all rounds for it as a burst
        # This simulates a user engaging in a multi-turn conversation
```

`mooncake.py:120-122`:

```
            # Add a placeholder assistant response for the next round's context
            # We use a placeholder because we don't know the real response
            placeholder_response = " ".join(["story"] * output_len_per_round)
```

**Classification (Mooncake multi-round): open (burst).** All rounds of a session are yielded at the session's trace timestamp, without waiting for any reply. Later rounds carry placeholder reply text, not the real reply.

**Overall bench_serving: configurable.** Single-turn is the default (open: `inf`, so all at t=0, or Poisson). The multi-turn dataset rows are closed within a conversation. Mooncake multi-round is an open burst.

## B. `benchmark/hicache/bench_multiturn.py`

Defaults, `:30-71`: `--num-clients` 256, `--max-parallel` 128, `--num-rounds` 5, `--distribution` "poisson", `--request-rate` 1.0. `:113-117`:

```
        "--enable-round-barrier",
        action="store_true",
        help="If set, only send i-th turn requests after all (i-1)-th turn requests finished.",
```

A follow-up turn is created only when the previous response has been handled, `:489` and `:519-520`:

```
                if self.client_records[client_id]["round"] < client_total:
...
                    else:
                        self.ready_queue.append(new_req)
```

The sender pops one ready request per Poisson interval, subject to the in-flight cap, `:407-408` and `:419-429`:

```
                if self.sent_requests - self.completed_requests < self.max_parallel:
                    new_request = self.ready_queue.pop()
...
                if self.distribution == "poisson":
                    sleep_time = random.expovariate(self.request_rate)
...
                await asyncio.sleep(sleep_time)  # Wait before sending the next request
```

The ready-queue pop policy defaults to "random" (`:124-128`). Without `--disable-auto-run`, the script sweeps the rates `[16, 14, ..., 1]` (`:755-760`).

**Classification: closed (dependency), with release paced by a global dispatcher.** A follow-up turn becomes eligible only after the previous reply has been received. It is then sent when the global Poisson-paced sender (default 1 req/s, capped at 128 in flight) randomly picks it from the shared ready queue. The delay is therefore set by a global rate, not by a per-session think time or a per-turn schedule. `--enable-round-barrier` adds a per-round barrier, so round i is sent only after all round i-1 replies have arrived, followed by a 10 s sleep (`:537`).
