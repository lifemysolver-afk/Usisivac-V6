
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-06-25 - LLM Client Instantiation and Connection Pooling
**Learning:** Instantiating LLM SDK clients (Groq, OpenAI, Google GenAI) inside calling functions is a major latency anti-pattern. Each instantiation incurs ~37-105ms overhead due to environment parsing and connection pool setup. Caching these clients using `functools.lru_cache` enables TCP/TLS connection reuse, reducing retrieval overhead to <1µs and significantly speeding up subsequent API calls.
**Action:** Always memoize LLM client instantiation using helper functions decorated with `@lru_cache(maxsize=10)`, using the API key as the cache key to ensure safety during key rotation.
