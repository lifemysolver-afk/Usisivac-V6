
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-08-29 - O(limit) Reverse Byte Reading for Append-Only JSONL Chat Logs
**Learning:** Reading and parsing entire append-only log files with `read_text().split('\n')` introduces linear O(N) I/O and memory overhead as logs grow. Seeking to EOF (`os.SEEK_END`) and reading backward in fixed chunks until `limit` matching lines are parsed reduces memory and time complexity to O(limit), yielding >800x speedup for log retrieval.
**Action:** Use reverse byte chunk streaming with early exit for any tail or recent-history queries on append-only `.jsonl` or `.log` files.
