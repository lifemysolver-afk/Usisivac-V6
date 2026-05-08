
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-22 - Unified Resource Initialization and Client Memoization
**Learning:** Initializing heavy models like `SentenceTransformer` or recreating SDK clients (Groq, OpenAI) across different modules (`rag_engine`, `neural_filter`, `brain_mass_ingest`) wastes ~12s per module and consumes hundreds of megabytes of redundant RAM.
**Action:** Use a centralized, memoized entry point for heavy models and shared clients. A custom `FastSharedEF` for ChromaDB allows direct reuse of a pre-loaded model, bypassing the overhead of standard wrapper re-initialization.
