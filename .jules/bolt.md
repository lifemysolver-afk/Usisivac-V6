
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-03-09 - Pre-Embedding Essence and Batch Vectorization in QA Audits
**Learning:** Computing semantic drift scores sequentially for multi-agent outputs re-embeds shared reference text (`project_essence`) on every iteration and performs single-item model inference. Pre-embedding the reference text once and using `embed_batch` with matrix-vector multiplication (`action_embs @ essence_emb`) yields >2.1x speedup (>50% latency reduction) per audit.
**Action:** Always batch multi-item embedding comparisons against a fixed reference text using batch inference and matrix multiplication.
