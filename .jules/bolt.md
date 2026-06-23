
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-06-23 - Centralized Embedding Model Lifecycle
**Learning:** Independent instantiation of SentenceTransformers models (e.g., 'all-MiniLM-L6-v2') in multiple modules leads to massive redundant memory usage (~800MB per instance) and startup latency (~19s per instance). ChromaDB collections created with the 'default' EF will conflict with 'sentence_transformer' EF on retrieval.
**Action:** Use a centralized, memoized provider in 'core/rag_engine.py' for both the ChromaDB EmbeddingFunction and the underlying SentenceTransformer model. Use a non-destructive fallback (try-except) when accessing existing ChromaDB collections to handle EF mismatches without data loss.
