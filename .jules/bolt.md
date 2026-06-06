
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-06-06 - LLM Client Memoization
**Learning:** Instantiating LLM SDK clients (Groq, OpenAI, Google GenAI) on every request introduces significant overhead (up to ~100ms for Gemini), which accumulates in multi-agent loops. SDK clients are designed to be reused to leverage internal connection pooling and avoid redundant config parsing.
**Action:** Always memoize LLM client instances using `@functools.lru_cache` keyed by the API configuration (key, base_url) to reduce per-call latency to near-zero.
