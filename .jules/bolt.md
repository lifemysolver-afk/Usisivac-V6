
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-05-20 - Reverse Chunk Byte Reading for Log Tailing
**Learning:** Tailing growing JSONL log files (like agent conversation histories) by reading the full file into memory scales linearly in execution time and memory allocation O(N) as log size grows. Using binary file seeking from the end (`os.SEEK_END`) with chunked backwards scanning reduces complexity to O(limit) and yields ~350x speedups (246ms -> 0.7ms for 50k line log files).
**Action:** Always use reverse byte chunk seeking (`f.seek(0, os.SEEK_END)`) for tailing or retrieving recent entries from append-only JSONL / text log files.
