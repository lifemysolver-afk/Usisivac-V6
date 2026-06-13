
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-06-15 - LLM Client Instantiation Memoization
**Learning:** Instantiating LLM SDK clients (Groq, OpenAI, Gemini) on every call adds significant overhead (~70-250ms per call in real environments). Using `@functools.lru_cache` to memoize client instances keyed by API configuration reduces this overhead to <1µs.
**Action:** Always memoize I/O clients and heavy SDK objects using LRU caches to prevent redundant initialization in tight loops.
