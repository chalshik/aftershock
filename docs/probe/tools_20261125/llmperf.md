# LLMPerf (ray-project/llmperf)

- Repo: github.com/ray-project/llmperf
- Commit read: `f1d6bed47e4501b0e371082b41601b59ab55269f` (HEAD of default branch, last commit 2024-12-08, read 2026-11-25 via shallow clone)
- Latest release at time of reading: `v2.0` (published 2023-12-06, GitHub releases API)

## Evidence

`README.md:18`: "The load test spawns a number of concurrent requests to the LLM API and measures the inter-token latency and generation throughput per request and across concurrent requests."

`token_benchmark_ray.py:51-52`: "num_concurrent_requests: The number of concurrent requests to make. Increase this to increase the amount of load and vice versa."

Per-thread loop, `token_benchmark_ray.py:92-113` (excerpt):

```
    def launch_request(thread_index):
        nonlocal num_completed_requests
        clients = construct_clients(llm_api=llm_api, num_clients=1)
        req_launcher = RequestsLauncher(clients)
...
            req_launcher.launch_requests(request_config)

            outs = req_launcher.get_next_ready()
```

`token_benchmark_ray.py:134-137`:

```
    for i in range(num_concurrent_requests):
        thread = threading.Thread(target=launch_request, args=(i,))
        threads.append(thread)
        thread.start()
```

`src/llmperf/requests_launcher.py:21-22`: `if self._llm_client_pool.has_free():` / `self._llm_client_pool.submit(`. The pool holds one client per thread, so each thread has at most one request in flight.

Message construction, `src/llmperf/ray_clients/openai_chat_completions_client.py:22-25`:

```
        message = [
            {"role": "system", "content": ""},
            {"role": "user", "content": prompt},
        ]
```

Each prompt is a freshly sampled set of sonnet lines (`token_benchmark_ray.py:83-88`). No conversation history is carried between requests.

**Classification: no multi-turn; fixed concurrency (closed per worker).** Each of the `--num-concurrent-requests` threads sends one independent single-turn request and submits the next only once a slot is free. There is no request-rate or open-loop mode in this script. Caveat: `get_next_ready()` defaults to `block=False` (`requests_launcher.py:29`), so a thread may spin rather than block. The one-in-flight-per-thread bound still holds through `has_free()`.
