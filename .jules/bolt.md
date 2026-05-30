
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-25 - Memoizing LLM SDK Clients
**Learning:** LLM SDK clients (Groq, OpenAI, Gemini) have a measurable instantiation overhead (up to ~40ms-1s) when created inside a hot function call. Memoizing these clients reduces this to microseconds and enables internal TCP connection pooling, which is critical for reducing latency in multi-agent systems like DiscussionEngine and VetoBoard.
**Action:** Always memoize LLM SDK clients at the module level or using `lru_cache` when they are used in high-frequency loops or multi-agent orchestrations.
