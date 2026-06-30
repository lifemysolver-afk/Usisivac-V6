
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-06-30 - LLM Client Memoization and PR Hygiene
**Learning:** Instantiating LLM SDK clients (Groq, Gemini) on every call adds ~100ms of overhead. Memoizing these clients with `@lru_cache` reduces retrieval to <1μs. Additionally, environment-specific artifacts (logs, databases) are easily accidentally staged and must be manually purged before PR submission.
**Action:** Use memoized client getters for all LLM providers. Always verify `git status` to ensure no untracked binaries or logs are staged.
