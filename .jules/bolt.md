
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-05-01 - Reverse Byte Chunk Scanning for Append-Only Logs
**Learning:** Loading and splitting large JSONL log files into memory (`read_text().split("\n")`) causes O(N) memory and runtime degradation when retrieving recent entries. Reading backwards from EOF using file seeks (`os.SEEK_END`) in chunks yields an ~800x speedup and O(limit) memory usage.
**Action:** Use reverse byte chunk scanning with raw bytes buffering for tail retrieval from append-only log files.
