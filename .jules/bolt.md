
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-10-10 - Batch Semantic Drift Score Computation
**Learning:** Evaluating agent outputs sequentially against project essence in audit systems creates an O(N) embedding latency bottleneck. Pre-embedding the reference target once and batching all candidate descriptions with `embed_batch` enables single matrix-vector product (`embs @ target`) yielding ~16x-60x speedup.
**Action:** Always batch multi-item similarity or drift checks in audit/evaluation modules against static project targets.
