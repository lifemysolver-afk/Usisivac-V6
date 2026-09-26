
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-03-30 - Batch Vectorization of Multi-Agent Embedding Comparisons
**Learning:** Sequential single-embedding calls in multi-agent audit loops create major latency bottlenecks by repeatedly calling model inference and re-embedding reference text. Batching all agent outputs with `embed_batch` and computing cosine similarity via matrix-vector multiplication (`action_embs @ essence_emb`) yields over 60x speedup (>2.3x per item).
**Action:** Whenever comparing multiple agent outputs or items against a target vector, batch the text inputs into `embed_batch` and compute similarity using NumPy matrix product instead of looping `embed()`.
