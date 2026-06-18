
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-06-18 - Model Singleton and LLM Client Memoization
**Learning:** Initializing heavy models (like SentenceTransformer) across multiple independent modules consumes massive RAM (~700MB each) and compounds startup latency. Similarly, instantiating LLM clients on every request adds ~35-100ms of unnecessary overhead.
**Action:** Consolidate heavy model loading into a shared, memoized singleton. Use `@functools.lru_cache` to reuse LLM client instances, reducing retrieval overhead to <1µs.
