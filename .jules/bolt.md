
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-10-06 - Batched Vectorization for Multi-Agent Semantic Drift Calculation
**Learning:** Evaluating semantic drift sequentially across multiple agent outputs causes repeated single-item model passes and redundant reference embeddings (`project_essence`), leading to severe latency (~11.7s for 10 agents). Batching action descriptions into `embed_batch` and computing cosine similarity via a single NumPy matrix-vector product reduces latency to ~2.8ms (>4000x speedup).
**Action:** Whenever evaluating multiple text outputs against a fixed baseline embedding, pre-embed the baseline once and compute similarities in batch mode using vectorized NumPy matrix operations.
