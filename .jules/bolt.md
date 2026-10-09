
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-25 - Reverse Byte Chunking for Log Tail History
**Learning:** Reading append-only log files sequentially from the beginning to retrieve recent history (`get_history`) scales linearly with log size ($O(N)$). Seeking to the end (`SEEK_END`) and reading byte chunks backwards reduces complexity to $O(\text{limit})$, yielding ~800x speedup (0.5s down to 0.5ms) on 100k line logs without memory overhead.
**Action:** Always use reverse byte chunking (`seek(0, SEEK_END)`) for tail operations on append-only log files.
