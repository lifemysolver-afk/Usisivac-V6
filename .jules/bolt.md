
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-25 - Sharing Centralized Embedding Models Across Vector DB Engines
**Learning:** Omission of an explicit embedding function in ChromaDB collection initialization causes ChromaDB to instantiate a default ONNX embedding model (~800MB RAM, 6.5s load time). Sharing a memoized SentenceTransformer embedding instance across engines via `get_embedding_function()` drops initialization latency to <0.5s (~13x speedup).
**Action:** Always expose and reuse memoized embedding functions across modules utilizing vector databases rather than relying on default collection initializers.
