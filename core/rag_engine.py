"""
╔══════════════════════════════════════════════════════════════════════╗
║  RAG Engine — ChromaDB + Neural Filter                              ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Kolekcije:
  knowledge_base   — naučni radovi, best practices, data science znanje
  kaggle_insights  — Kaggle kernels i competition strategies
  agent_history    — prethodne agent akcije i rezultati
  golden_recipes   — verifikovani top-tier paterni
  domain_specific  — znanje specifično za trenutni problem
"""

import json, datetime, functools
from pathlib import Path
from typing import List, Dict, Optional

BASE_DIR     = Path(__file__).parent.parent
CHROMA_PATH  = BASE_DIR / "chroma_db"
FALLBACK_DIR = BASE_DIR / "knowledge_base"
EMBED_MODEL  = "all-MiniLM-L6-v2"

COLLECTIONS = [
    "knowledge_base", "kaggle_insights",
    "agent_history",  "golden_recipes", "domain_specific"
]


# ─── ChromaDB Client ──────────────────────────────────────────────────────────
@functools.lru_cache(maxsize=1)
def _client():
    import chromadb
    return chromadb.PersistentClient(path=str(CHROMA_PATH))

@functools.lru_cache(maxsize=1)
def _ef():
    from chromadb.utils import embedding_functions
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL)


# ─── Ingest ───────────────────────────────────────────────────────────────────
def ingest(documents: List[str], metadatas: List[dict],
           ids: List[str], collection: str) -> dict:
    """Stvarni ChromaDB upsert. Nikad ne simulira."""
    try:
        col = _client().get_or_create_collection(name=collection, embedding_function=_ef())
        col.upsert(documents=documents, metadatas=metadatas, ids=ids)
        return {"ok":True, "upserted":len(documents),
                "total":col.count(), "collection":collection, "backend":"chromadb"}
    except Exception as e:
        return _json_ingest(documents, metadatas, ids, collection, str(e))


def _json_ingest(documents, metadatas, ids, collection, err) -> dict:
    FALLBACK_DIR.mkdir(parents=True, exist_ok=True)
    fp = FALLBACK_DIR / f"{collection}.json"
    existing = json.loads(fp.read_text("utf-8")) if fp.exists() else []
    ex_ids   = {d["id"] for d in existing}
    new = [{"id":i,"content":d,"metadata":m}
           for d,m,i in zip(documents,metadatas,ids) if i not in ex_ids]
    existing.extend(new)
    fp.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    return {"ok":True,"upserted":len(new),"total":len(existing),
            "collection":collection,"backend":"json","chroma_err":err}


# ─── Query (raw, before neural filter) ───────────────────────────────────────
def query_raw(text: str, collection: str, n: int = 20,
              query_embeddings: Optional[List[float]] = None) -> List[dict]:
    """Vraća sirove rezultate — neural_filter ih dalje obrađuje."""
    try:
        col = _client().get_collection(name=collection, embedding_function=_ef())

        kwargs = {"n_results": min(n, col.count() or 1), "include": ["documents", "metadatas", "embeddings"]}
        if query_embeddings is not None:
            kwargs["query_embeddings"] = [query_embeddings]
        else:
            kwargs["query_texts"] = [text]

        r = col.query(**kwargs)
        docs  = r.get("documents",[[]])[0]
        metas = r.get("metadatas",[[]])[0]
        embs  = r.get("embeddings",[[]])[0]

        return [{"content":d,"metadata":m, "_embedding": e} for d,m,e in zip(docs,metas,embs)]
    except Exception:
        return _json_query(text, collection, n)


def _json_query(text: str, collection: str, n: int) -> List[dict]:
    fp = FALLBACK_DIR / f"{collection}.json"
    if not fp.exists(): return []
    try:
        docs = json.loads(fp.read_text("utf-8"))
        tl   = text.lower()
        scored = sorted(
            [(sum(1 for w in tl.split() if w in d.get("content","").lower()), d)
             for d in docs],
            key=lambda x: x[0], reverse=True)
        return [{"content":d["content"],"metadata":d.get("metadata",{})}
                for _,d in scored[:n] if _>0]
    except Exception:
        return []


# ─── Smart Query (raw + neural filter) ───────────────────────────────────────
def query_smart(text: str, collection: str,
                top_k: int = 5, threshold: float = 0.25) -> List[dict]:
    """Puni pipeline: ChromaDB → Neural Filter → MMR → top_k rezultata."""
    from core.neural_filter import filter_knowledge, embed
    # Compute query embedding once and reuse it
    q_emb = embed(text)
    # ChromaDB expects a list, but we keep it as numpy for filter_knowledge
    raw = query_raw(text, collection, n=30, query_embeddings=q_emb.tolist())
    return filter_knowledge(text, raw, top_k=top_k, quality_threshold=threshold, query_embedding=q_emb)


# ─── Stats ────────────────────────────────────────────────────────────────────
def stats() -> dict:
    out = {}
    try:
        cli = _client()
        ef  = _ef()
        for c in COLLECTIONS:
            try:
                col = cli.get_collection(name=c, embedding_function=ef)
                out[c] = {"count":col.count(),"backend":"chromadb"}
            except Exception:
                fp = FALLBACK_DIR / f"{c}.json"
                if fp.exists():
                    try:
                        out[c] = {"count":len(json.loads(fp.read_text())),"backend":"json"}
                    except Exception:
                        out[c] = {"count":0,"backend":"error"}
                else:
                    out[c] = {"count":0,"backend":"empty"}
    except Exception as e:
        out["error"] = str(e)
    return out
