# LLMRunner (research tool, paper text only)

- Source: Upadhyaya and Mishra, "LLMRunner: Closed Workload Model Performance Testing Framework for LLM Servings", arXiv:2611.15372v1 (11 Nov 2026), read 2026-11-25 through alphaXiv (`answer_pdf_queries`, extracted PDF text).
- Version: paper v1. I did not locate or read any code. The paper (p. 3) says "to provide a open source performance tool for LLM workloads", but the extracted pages I read give no repository URL.

## Quoted passages

p. 4 (workload block, multi-turn mode):

> "In multi_turn mode (line 24) the harness maintains a pool of active conversations, where each worker appends the next user turn to the accumulated history, sends the full context to the server, receives the assistant reply, and appends it before issuing the next turn"

p. 4:

> "The think_time parameter (lines 35–37), which can be a constant, a sampled distribution, or timestamps replayed from a dataset, controls the inter-turn delay per worker."

p. 5 (Load generation):

> "1) Rate Mode (Open-Loop): In rate mode the client issues requests as a Poisson process at a target rate λ requests per second, with a configurable maximum in-flight cap. ... Because the client does not wait for prior requests to complete, rate mode is open-loop"

> "2) Threads Mode (Closed-Loop): In threads mode, N concurrent worker threads each execute the loop: issue a request, wait for the reply, sleep for a sampled think time, and repeat."

p. 7 (multi-turn experiments, Figs. 8 and 9) report results against "Load Generator Threads", i.e. threads mode.

## Classification

**Multi-turn: closed (threads mode), with a think time after the reply.** In multi-turn mode a worker receives the reply before issuing the next turn, and in threads mode it then sleeps for the think time. Single-turn traffic can be open (rate mode, Poisson) or closed (threads mode).

**Unclear:** (1) The paper does not say how multi-turn follow-ups are released in rate mode. (2) For "timestamps replayed from a dataset", it does not say whether the delay is applied after the reply or against a schedule. With no code read, the row should be cited as "paper states closed-loop multi-turn in threads mode" only.
