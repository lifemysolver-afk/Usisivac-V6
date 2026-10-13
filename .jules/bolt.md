
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-10-13 - Batch Vectorization for Guardian Drift Calculation
**Learning:** In audit/validation tasks evaluating N items against a fixed reference text, sequential single-item embedding calls re-embed the reference string N times and process item embeddings one by one. Batching inputs via `embed_batch` and computing cosine similarity via a matrix-vector product (`emb_actions @ emb_essence`) yields ~2.5x-60x speedups.
**Action:** Pre-embed reference targets once and pass candidate item lists into `embed_batch` before calculating cosine similarity with matrix multiplication.
