
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-25 - SDK Client Memoization and Shared Resources
**Learning:** Re-instantiating LLM SDK clients (Groq, OpenAI, Gemini) adds ~35-60ms of overhead per call. Memoizing these clients using `@functools.lru_cache` reduces this to near-zero. Furthermore, reusing a shared embedding function across different ChromaDB collections/clients (even with different persistent paths) saves ~700MB RAM and ~18s of load time per instance.
**Action:** Always use memoized helper functions for resource-heavy client instantiations. Unify embedding model loading across all modules using a shared `_ef()` factory.
