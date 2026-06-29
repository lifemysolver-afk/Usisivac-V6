
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-06-10 - LLM Client Instantiation Overhead
**Learning:** Instantiating SDK clients (Groq, OpenAI, Gemini) takes ~30ms to ~110ms per call. In high-frequency pipelines or multi-persona evaluations, this overhead accumulates quickly.
**Action:** Memoize LLM client instantiation using @lru_cache, with API keys and base URLs as cache keys to ensure proper reuse and configuration isolation.
