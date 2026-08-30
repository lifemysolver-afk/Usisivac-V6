## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-22 - Reverse Byte Chunking for Tail Reads on Growing Log Files
**Learning:** Reading and decoding entire `.jsonl` files to retrieve the last $N$ records causes $O(\text{file\_size})$ time and memory scaling, which degrades latency as conversation logs grow. Using `f.seek(0, os.SEEK_END)` and backward chunk reading cuts retrieval time to $O(\text{limit})$, achieving an ~800x speedup.
**Action:** Always use reverse byte chunking for `get_history` or tail retrieval functions on append-only log files.
