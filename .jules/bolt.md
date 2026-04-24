
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-22 - VetoBoard Parallelization
**Learning:** Sequential LLM calls for multi-persona evaluation (e.g., VetoBoard) create a major latency bottleneck (~7-8s for 5 personas). Parallelizing these I/O-bound tasks using `ThreadPoolExecutor` reduces latency to the duration of the single slowest call (~2.5-3.5s), achieving a ~50-60% speedup.
**Action:** Use `ThreadPoolExecutor` for concurrent LLM evaluations or any independent network-bound tasks to improve responsiveness.
