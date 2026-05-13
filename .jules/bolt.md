
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-25 - Unified Shared Resources for Multi-Component RAG
**Learning:** Embedding model initialization (SentenceTransformer) is extremely slow (~18s). Initializing separate standard ChromaDB `SentenceTransformerEmbeddingFunction` instances in different modules (RAG, Discussion, Ingest) causes massive redundant delays and RAM usage.
**Action:** Use a centralized, memoized `_get_model()` and a custom `FastSharedEF` in `core/rag_engine.py` that all other modules must reuse. This reduces component initialization from ~10s to ~0.2s.
