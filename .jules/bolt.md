
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.
## 2025-05-25 - LLM Client SDK Memoization
**Learning:** LLM SDK client instantiation (Groq, OpenAI, Gemini) carries significant overhead (~35-100ms for Groq/OpenAI, ~300ms for Gemini) because it often involves loading configuration, validating keys, and setting up transport layers. In multi-agent systems where many short calls are made, this overhead can dominate total execution time.
**Action:** Always memoize SDK client instances using `@lru_cache` with API keys/base URLs as cache keys to ensure a single singleton client is reused across the process.
