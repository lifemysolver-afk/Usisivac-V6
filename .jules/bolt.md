
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-06-10 - Memoizing LLM Client Instantiation
**Learning:** Instantiating LLM clients (Groq, OpenAI, Gemini) on every call adds ~35-100ms of overhead due to SDK initialization and auth parsing. While small for single calls, this overhead aggregates significantly in multi-agent workflows or parallel evaluations (like VetoBoard).
**Action:** Use @functools.lru_cache to memoize client instances by API key and base URL to reduce retrieval latency to sub-microsecond levels.
