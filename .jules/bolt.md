
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-22 - PR Hygiene and Resource Unification
**Learning:** Shared heavy resources (like embedding models) must be unified to avoid memory bloat and slow startup. Additionally, PRs must be strictly cleaned of ephemeral artifacts like `chroma_db/`, `logs/`, and `.agent/` state files, which are auto-generated during verification but should never be committed.
**Action:** Use a standardized cleanup routine (`rm -rf chroma_db logs .agent`) before final submission. Use `@functools.lru_cache` for both heavy models and API clients to ensure singleton behavior across the application.
