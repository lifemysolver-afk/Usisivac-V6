
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-06-20 - LLM Client Memoization and Connection Reuse
**Learning:** SDK client instantiation (Groq, OpenAI, Gemini) adds significant overhead (~35-100ms) on every call. Memoizing these clients with `functools.lru_cache` reduces this to <1µs and enables internal connection pooling, further reducing network latency. Accidental staging of environment side-effects (logs, DBs) is a common risk during benchmarking.
**Action:** Always memoize SDK clients using `@functools.lru_cache`. Ensure that ephemeral files like `chroma_db/`, `logs/`, and `.agent/` are removed before submission to prevent repository pollution.
