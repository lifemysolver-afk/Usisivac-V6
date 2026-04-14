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

import json
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

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass supports both single vectors and batch processing.
        Returns an array of scores (or a single float if input is 1D).
        """
        h1 = self._relu(x @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        out = self._sigmoid(h2 @ self.W3 + self.b3)
        scores = out.flatten()
        if x.ndim == 1:
            return float(scores[0])
        return scores

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
    lambda_mmr=1.0 → samo relevantnost, 0.0 → samo raznovrsnost.
    """
    if len(docs) == 0:
        return []

    selected_idx = []
    remaining = list(range(len(docs)))

    # ⚡ Bolt: Pre-calculate relevance to query (vectorized)
    relevances = doc_embs @ query_emb

    for _ in range(min(top_k, len(docs))):
        best_idx, best_score = None, -np.inf
        for i in remaining:
            rel = float(relevances[i])
            if selected_idx:
                sel_embs = doc_embs[selected_idx]
                max_sim  = float(np.max(sel_embs @ doc_embs[i]))
            else:
                max_sim = 0.0
            score = lambda_mmr * rel - (1 - lambda_mmr) * max_sim
            if score > best_score:
                best_score = score
                best_idx   = i
        if best_idx is not None:
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
                     use_mmr: bool = True) -> List[Dict]:
    """
    Glavni filter — prima sirove ChromaDB rezultate,
    vraća top_k najrelevantnijih i najraznovrsnijih dokumenata.

    Pipeline:
      1. Embed query + docs
      2. MLP scorer → relevance score (vectorized)
      3. Quality gate (score < threshold → odbaci)
      4. MMR diversity filter (reusing embeddings)
      5. Vrati rangirane dokumente sa score-ovima
    """
    if not raw_docs:
        return []

    scorer = get_scorer()
    q_emb  = embed(query)

    texts   = [d.get("content", "") for d in raw_docs]
    d_embs  = embed_batch(texts)

    # ⚡ Bolt: Vectorized scoring
    cos_sims = d_embs @ q_emb
    mlp_scores = scorer.forward(d_embs)
    combined_scores = 0.6 * cos_sims + 0.4 * mlp_scores

    scored = []
    keep_indices = []
    for i, score in enumerate(combined_scores):
        if score >= quality_threshold:
            doc_copy = dict(raw_docs[i])
            doc_copy["_score"]     = round(float(score), 4)
            doc_copy["_cos_sim"]   = round(float(cos_sims[i]), 4)
            doc_copy["_mlp_score"] = round(float(mlp_scores[i]), 4)
            scored.append(doc_copy)
            keep_indices.append(i)

    if not scored:
        return []

    # Sort by score descending
    sorted_pairs = sorted(zip(scored, keep_indices), key=lambda x: x[0]["_score"], reverse=True)
    scored = [p[0] for p in sorted_pairs]
    keep_indices = [p[1] for p in sorted_pairs]

    if use_mmr and len(scored) > top_k:
        # ⚡ Bolt: Reuse embeddings to avoid redundant embed_batch
        scored_embs = d_embs[keep_indices]
        scored = mmr_select(q_emb, scored_embs, scored, top_k=top_k)
    else:
        scored = scored[:top_k]

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
