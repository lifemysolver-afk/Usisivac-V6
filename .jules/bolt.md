
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-10-17 - Vectorizing Multi-Agent Semantic Drift Scoring in Guardian Audit
**Learning:** Computing semantic drift scores sequentially for multi-agent output audit calls `embed` individually per agent and repeatedly re-embeds identical `project_essence` strings. Pre-embedding `project_essence` once and batching agent descriptions using `embed_batch` allows a single matrix-vector dot product (`action_embs @ essence_emb`), reducing drift scoring latency by ~2x across 6 agents.
**Action:** When evaluating semantic similarity between a single anchor prompt/essence and multiple agent outputs, batch all agent descriptions into a single embedding call and use NumPy matrix multiplication for similarity calculation.
