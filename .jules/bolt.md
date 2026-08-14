
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-25 - Memoizing LLM SDK Client Instantiation
**Learning:** Repeatedly instantiating LLM SDK clients (Groq, OpenAI, Google GenAI) inside function calls creates significant overhead (~32-111ms per instantiation) due to repeated environment parsing, transport initialization, and connection pool allocation.
**Action:** Always memoize SDK client creation using `@functools.lru_cache(maxsize=10)` keyed on API key and endpoint configuration to ensure instant (~0.0001ms) client reuse across repeated LLM requests.
