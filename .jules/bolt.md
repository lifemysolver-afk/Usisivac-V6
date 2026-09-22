
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-22 - Tri-Way Relay History Reverse Byte Chunking
**Learning:** Reading and line-splitting large JSONL log files sequentially to retrieve a small tail (`limit=50`) scales linearly $O(N)$ with file size in both CPU time and memory allocation.
**Action:** Use reverse byte chunking (`os.SEEK_END`) with buffer boundary splitting to read log files backwards. This reduces history retrieval time from ~650ms to ~0.5ms on 100k lines (~1000x speedup).
