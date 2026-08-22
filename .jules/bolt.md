
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-25 - LLM Client Instantiation Overhead Memoization
**Learning:** Instantiating LLM SDK clients (e.g. `Groq`, `OpenAI`, `genai.Client`) inside function calls on every invocation incurs significant setup latency (~34ms to ~103ms per call) due to repetitive env parsing and transport config. Caching clients using `@functools.lru_cache(maxsize=10)` keyed by API credentials reduces retrieval overhead to <0.001ms (a ~34,000x-120,000x speedup).
**Action:** Always memoize SDK client and HTTP session instantiation across LLM provider functions using LRU caching.
