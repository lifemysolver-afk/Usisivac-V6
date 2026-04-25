
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.
## 2025-05-16 - Parallelizing LLM-based Quorum Voting
**Learning:** Sequential LLM calls for multi-persona voting (e.g., VetoBoard) create a significant latency bottleneck proportional to the number of personas. Parallelizing these calls using ThreadPoolExecutor reduces wall-clock time to the latency of the slowest single call.
**Action:** Use ThreadPoolExecutor for independent LLM calls in quorum or multi-agent evaluation patterns. Always ensure the environment is cleaned of ephemeral side-effects (logs, DBs) before submission to avoid PR clutter.
