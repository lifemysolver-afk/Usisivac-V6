## 2026-04-17 - [RAG Pipeline Embedding Redundancy]
**Learning:** Redundant neural network passes for embeddings (querying ChromaDB then re-embedding retrieved docs for filtering) was the primary bottleneck in the RAG pipeline. Reusing vectors between stages and vectorizing the MMR diversity filter reduced warm query latency by ~90% (from 92ms to 9ms).
**Action:** Always check if retrieval backends (like ChromaDB) can return stored vectors and if they accept pre-computed embeddings to avoid redundant inference in the downstream filtering/reranking pipeline.
