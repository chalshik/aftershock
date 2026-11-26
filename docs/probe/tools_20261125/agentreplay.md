# AgentReplay (research tool, paper text only)

- Source: Pan et al., "AgentReplay: Token-wise Trace Replay is Essential for Fair Serving System Performance Benchmarking", arXiv:2609.32283v1 (26 Sep 2026, preprint under review), read 2026-11-25 through alphaXiv (`answer_pdf_queries`, extracted PDF text).
- Version: paper v1. I did not locate or read any code, and the extracted pages give no repository URL.

## Quoted passages

p. 4 (§3.1 Trace recording):

> "The experiment separately specifies an arrival policy A, using either task release times or a fixed number of concurrent tasks."

p. 5 (§3.2, Tool-duration replay):

> "For the recorded graph, the replay driver releases each request after its required predecessors complete. For a tool invoked at time t with recorded invocation-to-result duration d, the emulator makes its result available after a nonblocking wait until t + d, subject to timer and scheduling error. Independent tools wait asynchronously, while a join waits for all required predecessors. Faster LLM execution can therefore release later requests earlier, rather than preserving their original absolute timestamps."

p. 2:

> "The replay driver follows recorded request dependencies, allowing batching, scheduling, and completion times to change with the serving system."

## Classification

**Follow-up turns: closed, with the delay measured from the predecessor's completion.** A request is released only after its predecessors complete. The recorded tool duration d is waited from the tool's invocation time t during replay, and that invocation follows the preceding LLM reply. The paper says explicitly that original absolute timestamps are not preserved. Task (session) starts are configurable: open (recorded task release times) or a fixed number of concurrent tasks.

Caveat: the paper does not state the gap rule when tool-duration replay is disabled (the tools run live). It is presumably still dependency-gated, but that is not stated.
