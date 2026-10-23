
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-10-23 - Batch Embedding and Vectorized Cosine Distance for Multi-Agent QA Audit
**Learning:** Computing semantic drift sequentially per agent output calls the embedding model N times and re-embeds reference goals redundantly. Batch embedding all agent outputs via `embed_batch` and computing cosine similarity with single matrix-vector multiplication (`action_embs @ essence_emb`) yields ~3x speedup on audit benchmarks.
**Action:** Always batch embedding calls when evaluating multiple text items against a single reference embedding in QA/auditing pipelines.
