
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-07-08 - Memoizing LLM Clients and Reusing Sessions
**Learning:** Frequent instantiation of LLM SDK clients (Groq, OpenAI, Google GenAI) adds measurable latency (~37ms-110ms) to every I/O-bound call. Using a shared requests.Session for REST-based providers like Hugging Face enables connection pooling, further reducing TLS/TCP handshake overhead.
**Action:** Always memoize SDK clients at the module level using @lru_cache and prefer requests.Session for direct API calls to the same host.
