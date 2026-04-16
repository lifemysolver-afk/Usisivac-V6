## 2026-04-16 - [RAG Pipeline Embedding Reuse]
**Learning:** Redundant embedding generation in RAG pipelines (embedding query then embedding results for reranking) is a major bottleneck. Reusing embeddings from the vector store (ChromaDB) and caching query embeddings can reduce latency by >95% for repeat queries and significantly for new queries.
**Action:** Always check if retrieval results can return embeddings and pass them forward to any reranking/filtering steps.
