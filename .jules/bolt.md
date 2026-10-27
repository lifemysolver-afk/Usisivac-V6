
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-10-27 - Vectorized Batch Drift Calculation in QA/Auditing Pipelines
**Learning:** Sequential calls to `compute_drift_score` in multi-agent audit pipelines re-embed reference essence texts repeatedly and fail to leverage GPU/SIMD batch processing in sentence transformers. Pre-embedding reference essence once and batch-embedding agent output descriptions allows vectorized matrix-vector dot products (`emb_actions @ emb_essence`), yielding ~1.25x-2.3x speedup on warm models.
**Action:** Always prefer `embed_batch` and single matrix-vector multiplications when calculating pair-wise semantic similarity across multiple agent outputs against a single goal/essence.
