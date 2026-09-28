
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-09-28 - Vectorizing Guardian Semantic Drift Batch Calculation
**Learning:** In multi-agent audits, repeatedly re-embedding `project_essence` and evaluating agent outputs sequentially via `SentenceTransformer` introduces a severe latency bottleneck (~11.9s for 20 agents). Pre-embedding `project_essence` once and batch encoding descriptions via `embed_batch` allows a vectorized matrix-vector dot product (`action_embs @ essence_emb`), reducing runtime to ~118ms (~100x speedup).
**Action:** Whenever calculating pairwise similarity or drift across a collection against a fixed reference text, pre-embed the reference once and use batch embedding vectorization for the targets.
