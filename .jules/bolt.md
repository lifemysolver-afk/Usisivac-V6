
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-10-30 - Reverse Byte-Chunk Reading for Log History Retrieval
**Learning:** Polling log history by reading full files in memory (`read_text().split("\n")`) creates $O(N)$ time and memory overhead scaling linearly with log file size. Using reverse byte-chunk seeking (`f.seek(pos)` from end of file) reduces complexity to $O(\text{limit})$, achieving >20,000x speedups on large JSONL logs (~0.4 ms vs ~8.8s on 1M entries) and preventing RAM bloat.
**Action:** Use reverse byte-chunk seeking for tailing or retrieving fixed $N$ recent entries from appended log files. Ensure `errors="replace"` is used when decoding string chunks to safely handle byte splits across multi-byte UTF-8 boundaries.
