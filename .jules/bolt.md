
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-09-19 - Pre-embedding and Vectorizing Multi-Agent Drift Audits
**Learning:** In multi-agent QA pipelines like `guardian.py`, computing drift sequentially by re-embedding the project essence N times and calling single `embed()` sequentially introduces ~98% redundant latency. Pre-embedding the target essence once and batch-embedding descriptions with vectorized matrix-vector multiplication (`action_embs @ essence_emb`) yields ~57x speedup (~43.7ms to 0.75ms).
**Action:** Always pre-embed reference target vectors once before batching evaluation items in single matrix operations.
