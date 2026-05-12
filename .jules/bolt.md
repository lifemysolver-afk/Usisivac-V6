
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-05-12 - Unified Embedding Model Initialization
**Learning:** Initializing SentenceTransformer('all-MiniLM-L6-v2') takes ~18s and consumes ~700MB RAM. Redundant initializations across different modules (RAG, Neural Filter, Discussion Engine) multiply this cost significantly. ChromaDB's default EmbeddingFunction also adds overhead by potentially re-checking models.
**Action:** Centralize model initialization in a memoized helper (e.g., core.rag_engine._get_model) and use a lightweight custom EmbeddingFunction wrapper to share the instance across all components.
