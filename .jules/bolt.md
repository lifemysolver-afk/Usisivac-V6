
## 2025-05-15 - RAG Pipeline Embedding Reuse and Vectorization
**Learning:** Redundant neural network passes in RAG pipelines (embedding query multiple times, re-embedding docs already in DB) are the primary latency bottleneck. Vectorizing MMR with NumPy yields significant speedups over Python loops for candidate sets >10.
**Action:** Always check if embeddings can be retrieved from the vector store and passed through the pipeline before calling inference. Use NumPy matrix operations for diversity selection algorithms.

## 2025-05-20 - Parallelizing Multi-Agent/Persona LLM Evaluations
**Learning:** Sequential LLM calls for persona-based validation (like VetoBoard) create a major latency bottleneck that scales linearly with the number of personas. Threading is highly effective here since the tasks are purely I/O bound.
**Action:** Use ThreadPoolExecutor for any multi-agent/persona consensus or validation step to keep latency close to the response time of the slowest single agent.

## 2025-05-25 - LLM Client Memoization and Fallback Optimization
**Learning:** SDK clients for LLM providers (Groq, OpenAI, Google GenAI) have a measurable instantiation overhead (~35ms). Repeatedly creating these in multi-agent loops or high-frequency calling patterns adds up. Additionally, fixed sleep delays in key-rotation loops significantly slow down recovery during rate-limiting.
**Action:** Use memoized helper functions for SDK client instantiation. Remove unnecessary delays in rotation/fallback loops to minimize latency during provider switches.
