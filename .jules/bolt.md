
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-06-09 - Memoizing LLM Clients to Eliminate Instantiation Latency
**Learning:** Instantiating LLM SDK clients (Groq, OpenAI, Google GenAI) inside function calls adds ~35ms-250ms of pure overhead per request due to configuration parsing and connection pool setup. Memoizing these clients based on API keys and base URLs reduces this overhead to <1µs.
**Action:** Always wrap LLM client instantiation in `@functools.lru_cache` and ensure the cache key includes all configuration parameters (API key, base URL, etc.) to allow safe reuse across the application.
