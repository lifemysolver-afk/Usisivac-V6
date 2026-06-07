
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-06-07 - Cross-Component Model Sharing
**Learning:** Initializing the same heavy ML model (like SentenceTransformer) in multiple independent components leads to massive RAM bloat and multi-minute startup delays in agents. Standard ChromaDB EFs re-initialize the model every time.
**Action:** Implement a singleton pattern for heavy models at the core level and create custom Embedding Function wrappers that reuse this shared instance. This reduced startup time by ~75% and saved ~2GB of RAM in multi-agent flows.
