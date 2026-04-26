
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-16 - Parallelizing Multi-Persona LLM Evaluations
**Learning:** Parallelizing independent LLM calls in a committee-based validator (like VetoBoard) reduces latency from sum(latencies) to max(latencies), yielding ~61% speedup. However, it sacrifices sequential short-circuiting (e.g., stopping at the first Veto), potentially increasing token usage in veto cases.
**Action:** Use `ThreadPoolExecutor` for concurrent I/O-bound LLM tasks. Ensure API consistency by carefully mapping results back to the expected return format, especially for early exit paths.
