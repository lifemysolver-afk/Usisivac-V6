
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-09-24 - Batch Vectorization of Multi-Agent Semantic Drift
**Learning:** In multi-agent audits (`guardian/guardian.py`), invoking `compute_drift_score` sequentially causes redundant embedder calls. Pre-embedding `project_essence` once and batch-embedding all agent outputs with `embed_batch` allows cosine similarity to be computed in a single matrix-vector product (`action_embs @ essence_emb`), reducing audit drift computation latency by >50x (~2.78s to ~0.05s for 8 agents).
**Action:** Whenever evaluating multiple items against a single reference embedding, pre-embed the reference once and use `embed_batch` + matrix multiplication instead of single-item loops.
