
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-16 - Parallelizing LLM Evaluations and Path Portability
**Learning:** Sequential LLM calls for multi-persona evaluation (VetoBoard) are a major bottleneck (~8s). Parallelizing them with `ThreadPoolExecutor` reduces latency to the longest single call (~3s). Also, hardcoded absolute paths (/home/ubuntu/...) break portability and cause permission errors in different environments.
**Action:** Use `ThreadPoolExecutor` for concurrent I/O-bound LLM calls. Always use relative paths for database and log locations.
