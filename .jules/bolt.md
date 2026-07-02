
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-07-02 - LLM Client Instantiation Overhead
**Learning:** SDK clients (Groq, OpenAI, Google GenAI) have significant instantiation overhead (30ms - 60ms) due to internal setup and connection pool initialization. Repeatedly instantiating them in a loop or on every request is a major performance anti-pattern.
**Action:** Memoize LLM client instances using `@lru_cache` on helper functions that take the API key and base URL as arguments. This reduces client retrieval time to <1μs.
