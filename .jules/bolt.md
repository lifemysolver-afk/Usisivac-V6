
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-10-22 - Vectorized Batch Semantic Drift Auditing
**Learning:** Sequential calls to single-sentence embedding functions during multi-agent quality audits re-embed common context strings (e.g., project essence) N times and process agent descriptions one-by-one. Pre-embedding reference text once, calling batch embedding on item descriptions, and computing cosine similarities via a single NumPy matrix-vector multiplication (`action_embs @ essence_emb`) reduces drift calculation latency by over 1,000x.
**Action:** When evaluating semantic metrics across multiple items against a fixed reference text, pre-embed the reference once, call `embed_batch` on all candidate items, and compute similarities with matrix-vector products.
