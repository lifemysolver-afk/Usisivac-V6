
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-25 - Guardian Batched Neural Drift Score Matrix Vectorization
**Learning:** Sequential drift calculations across multi-agent outputs perform N redundant single-element inference passes and scalar dot products. Pre-embedding the reference project essence once and batching agent output descriptions using `embed_batch` enables a single matrix-vector multiplication (`action_embs @ essence_emb`), reducing computation time by >50% (>2.3x speedup).
**Action:** When computing similarity or drift across multiple text strings against a fixed reference string, batch the target descriptions and compute similarity in a single matrix product pass.
