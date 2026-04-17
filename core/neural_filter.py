"""
╔══════════════════════════════════════════════════════════════════════╗
║  Neural Knowledge Filter — "Veliki Filter"                          ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Neuronska mreža koja filtrira i rangira znanje iz ChromaDB-a.
Cilj: izvući MAKSIMUM relevantnog znanja za dati problem.

Arhitektura:
  1. SentenceTransformer embedding (384-dim)
  2. 3-slojni MLP scorer (384 → 128 → 64 → 1)
  3. Relevance score [0.0 – 1.0]
  4. Diversity filter (MMR — Maximal Marginal Relevance)
  5. Quality gate (odbacuje score < 0.3)
"""

import json, functools
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR   = Path(__file__).parent.parent
MODEL_PATH = BASE_DIR / "models" / "neural_filter_weights.npz"


# ─── Lightweight MLP (pure numpy, no GPU needed) ─────────────────────────────
class MLPScorer:
    """
    3-slojni MLP za scoring relevantnosti.
    Trenira se online na feedback-u agenata.
    Inicijalizovan sa Xavier inicijalizacijom.
    """
    def __init__(self, input_dim: int = 384):
        self.input_dim = input_dim
        self._init_weights()

    def _init_weights(self):
        if MODEL_PATH.exists():
            data = np.load(MODEL_PATH)
            self.W1 = data["W1"]; self.b1 = data["b1"]
            self.W2 = data["W2"]; self.b2 = data["b2"]
            self.W3 = data["W3"]; self.b3 = data["b3"]
        else:
            # Xavier initialization
            self.W1 = np.random.randn(self.input_dim, 128) * np.sqrt(2.0/self.input_dim)
            self.b1 = np.zeros(128)
            self.W2 = np.random.randn(128, 64) * np.sqrt(2.0/128)
            self.b2 = np.zeros(64)
            self.W3 = np.random.randn(64, 1) * np.sqrt(2.0/64)
            self.b3 = np.zeros(1)

    def _relu(self, x): return np.maximum(0, x)
    def _sigmoid(self, x): return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, x: np.ndarray) -> float:
        h1 = self._relu(x @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        out = self._sigmoid(h2 @ self.W3 + self.b3)
        return float(out[0])

    def update(self, x: np.ndarray, target: float, lr: float = 0.001):
        """Online learning — ažurira težine na osnovu feedback-a."""
        h1 = self._relu(x @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        out = self._sigmoid(h2 @ self.W3 + self.b3)
        err = out[0] - target
        # Backprop (simplified)
        dW3 = np.outer(h2, [err * out[0] * (1 - out[0])])
        self.W3 -= lr * dW3
        self.b3 -= lr * err * out[0] * (1 - out[0])
        self.save()

    def save(self):
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        np.savez(MODEL_PATH, W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2, W3=self.W3, b3=self.b3)


# ─── Embedding Engine ─────────────────────────────────────────────────────────
_embedder = None

def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


@functools.lru_cache(maxsize=128)
def embed(text: str) -> np.ndarray:
    return _get_embedder().encode(text, normalize_embeddings=True)


def embed_batch(texts: List[str]) -> np.ndarray:
    return _get_embedder().encode(texts, normalize_embeddings=True, batch_size=32)


# ─── MMR Diversity Filter ─────────────────────────────────────────────────────
def mmr_select(query_emb: np.ndarray,
               doc_embs: np.ndarray,
               docs: List[dict],
               top_k: int = 5,
               lambda_mmr: float = 0.7) -> List[dict]:
    """
    Maximal Marginal Relevance — balansira relevantnost i raznovrsnost.
    Vectorized implementation for speed.
    """
    if len(docs) == 0: return []

    selected_idx = []
    remaining = list(range(len(docs)))

    # Pre-calculate relevance to query for all docs
    relevance = doc_embs @ query_emb

    for _ in range(min(top_k, len(docs))):
        if not selected_idx:
            best_idx = int(np.argmax(relevance))
        else:
            # Vectorized similarity between remaining and selected
            rem_embs = doc_embs[remaining]
            sel_embs = doc_embs[selected_idx]
            # Max similarity to ANY selected doc for each remaining doc
            max_sim = np.max(rem_embs @ sel_embs.T, axis=1)

            scores = lambda_mmr * relevance[remaining] - (1 - lambda_mmr) * max_sim
            best_idx = remaining[np.argmax(scores)]

        selected_idx.append(best_idx)
        remaining.remove(best_idx)

    return [docs[i] for i in selected_idx]


# ─── Main Filter API ──────────────────────────────────────────────────────────
_scorer = None

def get_scorer() -> MLPScorer:
    global _scorer
    if _scorer is None:
        _scorer = MLPScorer()
    return _scorer


def filter_knowledge(query: str,
                     raw_docs: List[Dict],
                     top_k: int = 5,
                     quality_threshold: float = 0.25,
                     use_mmr: bool = True,
                     q_emb: Optional[np.ndarray] = None) -> List[Dict]:
    """
    Glavni filter — prima sirove ChromaDB rezultate,
    vraća top_k najrelevantnijih i najraznovrsnijih dokumenata.
    """
    if not raw_docs: return []

    scorer = get_scorer()
    if q_emb is None:
        q_emb = embed(query)

    # Re-use embeddings if provided in raw_docs (prevents redundant inference)
    if all("_embedding" in d for d in raw_docs):
        d_embs = np.array([d["_embedding"] for d in raw_docs])
    else:
        texts  = [d.get("content", "") for d in raw_docs]
        d_embs = embed_batch(texts)

    scored_with_embs = []
    for i, (doc, d_emb) in enumerate(zip(raw_docs, d_embs)):
        # Cosine similarity (embeddings su normalized)
        cos_sim = float(np.dot(q_emb, d_emb))
        # MLP relevance score
        mlp_score = scorer.forward(d_emb)
        # Combined score
        combined = 0.6 * cos_sim + 0.4 * mlp_score
        if combined >= quality_threshold:
            doc_copy = dict(doc)
            doc_copy["_score"]     = round(combined, 4)
            doc_copy["_cos_sim"]   = round(cos_sim, 4)
            doc_copy["_mlp_score"] = round(mlp_score, 4)
            scored_with_embs.append((doc_copy, d_emb))

    if not scored_with_embs:
        return []

    # Sort by score
    scored_with_embs.sort(key=lambda x: x[0]["_score"], reverse=True)

    if use_mmr and len(scored_with_embs) > top_k:
        # Re-use already computed embeddings for MMR
        docs_only = [x[0] for x in scored_with_embs]
        embs_only = np.array([x[1] for x in scored_with_embs])
        scored = mmr_select(q_emb, embs_only, docs_only, top_k=top_k)
    else:
        scored = [x[0] for x in scored_with_embs[:top_k]]

    return scored


def feedback_update(query: str, doc_content: str, was_useful: bool):
    """
    Online learning — agent daje feedback da li je dokument bio koristan.
    Ažurira MLP težine.
    """
    scorer = get_scorer()
    d_emb  = embed(doc_content)
    target = 1.0 if was_useful else 0.0
    scorer.update(d_emb, target)
