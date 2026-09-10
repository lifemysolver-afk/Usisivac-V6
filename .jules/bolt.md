
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-09-10 - Tailing Log Files via Reverse Byte Seeking
**Learning:** Reading and parsing full JSONL conversation logs to fetch recent agent context creates an O(N_total) memory and CPU bottleneck as logs grow. Reverse byte-chunk reading with `os.SEEK_END` reduces complexity to O(limit) time and memory, yielding ~800x speedup.
**Action:** Use `f.seek(0, os.SEEK_END)` and reverse buffer scanning when reading recent entries from append-only log files.
