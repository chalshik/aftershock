# vLLM benchmarks: `vllm bench serve` and `benchmarks/multi_turn/`

- Repo: github.com/vllm-project/vllm
- Commit read: `a588d3974e08fa0a614152ee6b62762f62777ed4` (HEAD of default branch, read 2026-11-25 via shallow clone)
- Latest release at time of reading: `v0.34.0` (published 2026-11-16, GitHub releases API)
- Note: this is the tool's HEAD, not the study's pinned vLLM 0.32.0.

## A. `vllm bench serve` (vllm/benchmarks/serve.py)

`benchmarks/benchmark_serving.py:6-9` is a stub that exits:

```
    print("""DEPRECATED: This script has been moved to the vLLM CLI.

Please use the following command instead:
    vllm bench serve
```

Arrival schedule, `vllm/benchmarks/serve.py:1821-1828`:

```
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, "
        "then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process or gamma distribution "
        "to synthesize the request arrival times.",
```

`vllm/benchmarks/serve.py:528-532` (gamma/Poisson inter-arrival):

```
                theta = 1.0 / (current_request_rate * burstiness)

                # Sample the request interval from the gamma distribution.
                # If burstiness is 1, it follows exponential distribution.
                delay_ts.append(np.random.gamma(shape=burstiness, scale=theta))
```

Optional concurrency cap, `vllm/benchmarks/serve.py:1063-1074`:

```
    semaphore = (
        asyncio.Semaphore(max_concurrency)
        if max_concurrency
        else contextlib.nullcontext()
    )

    async def limited_request_func(
        request_func_input, session, pbar, request_arrival_time
    ):
        async with semaphore:
            output = await request_func(
                request_func_input=request_func_input, session=session, pbar=pbar
            )
```

Trace replay with timestamps (`--dataset-name timed_trace`), `vllm/benchmarks/serve.py:1919-1923`:

```
        help="Use timing information from the traces instead of the configuration. "
        "This is useful when replaying traces faithfully based on their timestamps. "
        "When unset, defaults to False, except for --dataset-name=timed_trace where "
        "it defaults to True. Use --no-self-timed to force off. When off, user "
        "defined generation rates are used and in trace timing info is ignored.",
```

and `vllm/benchmarks/serve.py:555-569` sleeps until `start_ts + delay_ts[i]` for each request in turn, with no dependence on earlier replies.

Multi-turn data is reduced to single requests. ShareGPT, `vllm/benchmarks/datasets/datasets.py:1388-1389`, uses only the first two messages:

```
                entry["conversations"][0]["value"],
                entry["conversations"][1]["value"],
```

and BFCL, `vllm/benchmarks/datasets/datasets.py:4552-4553`:

```
            # BFCL question is list[list[dict]] — outer is turns. Use the
            # first turn only; skip multi-turn categories in this loader.
```

**Classification: no multi-turn.** No code path in serve.py sends a turn that depends on an earlier reply. Arrivals are open by default (`--request-rate inf`, so everything is sent at t=0; a finite rate gives Poisson or gamma arrivals; `timed_trace` replays the recorded timestamps). `--max-concurrency` adds a client-side semaphore, which makes the run concurrency-capped (closed per slot) when it is set. I did not exhaustively check every dataset loader in datasets.py for multi-turn chat_messages histories. Only ShareGPT and BFCL were checked, and both use the first turn only.

## B. `benchmarks/multi_turn/benchmark_serving_multi_turn.py`

Per-client loop, `client_main` (`:587`). The client awaits each turn before choosing the next one, `:722-723`:

```
                    result = await send_turn(
                        session,
```

After a successful turn the conversation goes back into the client's queue, `:795-796`:

```
                    # Conversation is not finished, insert it at the back of the queue
                    conv_id_queue.appendleft(conv_id)
```

The optional delay comes after the reply, `:798-800`:

```
            # Sleep between requests (if lambda is positive)
            if args.request_rate > 0:
                await poisson_sleep(args.request_rate, args.verbose)
```

`:558-565`:

```
async def poisson_sleep(request_rate: float, verbose: bool = False) -> None:
    # Generate a random time interval from the Poisson distribution
    assert request_rate > 0

    interval = np.random.exponential(1.0 / request_rate)
```

Default, `:1443-1448`:

```
        "--request-rate",
        type=float,
        default=0,
        help="Expected request rate (Poisson process) per client in requests/sec."
        "Set to 0 for no delay between requests.",
```

Other knobs: `--num-clients` (default 1, `:1379-1383`), `--max-active-conversations` (default None, `:1386-1390`) and `--conversation-sampling` (default ROUND_ROBIN, `:1460-1463`).

**Classification: closed.** Each client process has at most one request in flight (`await send_turn`). A conversation's next turn is sent only after its previous reply has finished, and possibly after other conversations' turns on the same client. The optional `--request-rate` delay is an exponential sleep that starts when the reply finishes. It is not a schedule. Default: no delay.
