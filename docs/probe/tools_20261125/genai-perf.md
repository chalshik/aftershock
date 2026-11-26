# NVIDIA GenAI-Perf (with Perf Analyzer)

- Repo: github.com/triton-inference-server/perf_analyzer (GenAI-Perf lives in `genai-perf/`)
- Commit read: `5ec2b2063853944376871352a23f53bbd3deb582` (HEAD of default branch, commit date 2026-11-17, read 2026-11-25 via shallow clone)
- Version: `genai-perf/genai_perf/__init__.py:27` gives `__version__ = "0.0.17dev"`. The latest PyPI release is `genai-perf 0.0.16` (uploaded 2025-08-26, from the PyPI JSON API). The repo has no GitHub releases.
- Status, `genai-perf/README.md:31`: "⚠️ **GenAI-Perf is being phased out.** We are no longer actively developing new features for GenAI-Perf. For new performance benchmarking needs, please use [AIPerf](https://github.com/ai-dynamo/aiperf) instead."

## Docs: `genai-perf/docs/multi_turn.md`

`:81-82`:

```
- `--session-concurrency 5`: Enables session mode and runs up to 5 sessions in
parallel.
```

`:86-87`:

```
- `--session-turn-delay-mean 1000`: Introduces a 1-second delay between user
turns (simulating real-world interaction).
```

`:124-126` (custom dataset field):

```
- `delay`: Sets the delay in milliseconds to wait after receiving a response
before sending the next request. This field is required except for the
last turn in a session.
```

## Source: `src/session_concurrency/session_concurrency_manager.cc:138-160`

```
SessionConcurrencyManager::SendSequentialRequestsForOneSession(
...
  for (size_t i{0}; i < one_session_payloads.size(); ++i) {
...
    request_handler_->SendRequestAndWaitForResponse(
        payload_dataset_index, chat_history, request_record);
...
    GetAndWaitForDelay(payload_dataset_index);
  }
```

`:174`: `std::this_thread::sleep_for(*delay);`

**Classification: closed.** In session mode (`--session-concurrency`), each session runs its turns one after another. Each request waits for its response (`SendRequestAndWaitForResponse`), then sleeps for the `delay` measured from the response, then sends the next turn. Sessions are concurrency-bounded (up to N sessions in parallel). There is no timestamp-scheduled multi-turn mode in these sources. I did not read the non-session single-turn modes (`--concurrency`, `--request-rate`) for this row.
