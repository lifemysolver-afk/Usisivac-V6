## 2025-05-19 - RAG Pipeline & Neural Filter Optimization
**Learning:** Significant performance gains can be achieved by reusing embeddings between the retrieval (ChromaDB) and filtering (Neural Filter) layers. Requesting embeddings directly from ChromaDB avoids redundant, expensive neural network passes. Additionally, vectorizing inner loops in MMR (Maximal Marginal Relevance) and MLP scoring using NumPy matrix operations drastically reduces Python overhead.
**Action:** Always check if document embeddings can be retrieved from the vector store before re-embedding. Prioritize NumPy matrix operations over Python 'for' loops for similarity and scoring logic.

## 2025-05-19 - Environment Hygiene for Agent Workflows
**Learning:** Autonomous agents can easily pollute a repository with ephemeral side-effects like binary database files (`chroma_db/`), internal logs, and temporary solution files. These artifacts should not be committed as they bloat the PR and version history.
**Action:** Explicitly clean up all generated artifacts (`chroma_db/`, `logs/`, `src/generated/`, `.agent/`) and benchmark scripts before final submission.
