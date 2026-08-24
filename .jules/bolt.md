
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-03-30 - Memoizing LLM SDK Client Factory Instantiations
**Learning:** Instantiating LLM SDK clients (Groq, OpenAI, Gemini) and HTTP session objects on every request introduces ~36ms per-call overhead from parsing environment variables, setting up HTTP client configurations, and connection setup. Using `@functools.lru_cache` on client factory functions reduces client retrieval to ~0.37ms (~96x speedup).
**Action:** Always memoize SDK client and HTTP session factory helpers with `@functools.lru_cache`, and provide autouse fixture cache clearing in unit tests to prevent test pollution.
