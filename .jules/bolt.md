
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-03-08 - Memoizing LLM SDK Client Instantiation
**Learning:** Re-instantiating provider SDK clients (e.g., Groq, OpenAI) on every single LLM call adds ~34.5ms setup overhead per invocation. Memoizing client instances with `@functools.lru_cache(maxsize=10)` reduces retrieval overhead to ~0.00044ms per call.
**Action:** Always wrap SDK client instantiations and HTTP sessions in `@functools.lru_cache` to eliminate setup overhead across agent iterations.
