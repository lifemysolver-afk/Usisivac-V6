
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-22 - Parallelization of Multi-Persona LLM Evaluations
**Learning:** Sequential LLM calls for multi-persona validations (like VetoBoard) create a major latency bottleneck, as the total time is the sum of all response times. Using ThreadPoolExecutor reduces latency to the duration of the slowest single response.
**Action:** Use concurrent.futures.ThreadPoolExecutor for any multi-agent or multi-persona LLM evaluation workflows to ensure responsiveness.
