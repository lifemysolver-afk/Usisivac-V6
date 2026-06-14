
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-06-14 - LLM Client Memoization
**Learning:** SDK client instantiation (Groq, OpenAI, Google GenAI) has significant overhead (~30ms-100ms) which accumulates in multi-agent loops or parallel persona evaluations. Using @functools.lru_cache to reuse client instances reduces this to <1µs.
**Action:** Always memoize LLM SDK clients at the provider level, ensuring API keys are part of the cache key if they change.
