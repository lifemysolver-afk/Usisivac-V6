
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2026-07-23 - Memoizing LLM Clients and Reusing TCP Connections
**Learning:** Instantiating LLM SDK Clients (Groq, OpenAI, Gemini genai) has a massive hidden latency penalty (up to 2000ms for Gemini, 900ms for OpenAI, 490ms for Groq) due to package import and internal configuration overhead. Connection setup handshake overhead for REST-based Inference APIs (like Hugging Face) adds 10-50ms per call.
**Action:** Always cache client SDK instances using `functools.lru_cache`. Reuse HTTP sessions with `requests.Session` connection pools to enable connection-keepalive and eliminate handshake overhead.
