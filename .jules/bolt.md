## 2024-04-14 - Vectorized Neural Filter and Embedding Reuse

**Learning:** Vectorizing the scoring loop and reusing embeddings significantly reduces latency in the knowledge retrieval pipeline. Specifically, replacing individual `np.dot` calls and `MLPScorer.forward` passes with matrix operations, and avoiding redundant `embed_batch` calls for MMR, yielded a ~47% performance improvement.

**Action:** Always look for opportunities to vectorize scoring logic and reuse expensive artifacts like embeddings when multiple filtering/selection stages are present. Also, be careful to keep the environment clean of agent-run side effects (like binary database updates) when preparing a PR.
