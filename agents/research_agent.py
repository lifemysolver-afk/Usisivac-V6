"""
+----------------------------------------------------------------------+
|  ResearchAgent - Univerzalni Usisivac Znanja                        |
|  Usisivac V6 | Trinity Protocol                                     |
+----------------------------------------------------------------------+

Pretrazuje ChromaDB, usisava znanje iz bilo kog domena,
izvlaci "Golden Recipe" (best practices) za dati problem.
Radi za SVE - ne samo Kaggle.
"""

import sys, json, datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, register_proof, log_work
from core.rag_engine import query_smart, query_raw, ingest, stats
from core.llm_client import call as llm_call
from core.neural_filter import filter_knowledge, feedback_update
from core import state_manager as SM

AGENT = "ResearchAgent"


# -- Univerzalna baza znanja za bilo koji domen ------------------------------
UNIVERSAL_KNOWLEDGE = {
    "data_science_fundamentals": [
        {"id":"ds_eda_001","content":"EDA best practices: (1) Check shape, dtypes, nulls. (2) Distribution plots for all numeric. (3) Correlation heatmap. (4) Target variable analysis. (5) Outlier detection via IQR/z-score. (6) Feature-target relationships. (7) Cardinality check for categoricals. (8) Time-based patterns if temporal.","metadata":{"domain":"universal","topic":"EDA","source":"data_science_handbook"}},
        {"id":"ds_feat_001","content":"Feature Engineering patterns: (1) Polynomial features for non-linear relationships. (2) Binning continuous variables. (3) Target encoding for high-cardinality categoricals. (4) Interaction features (A*B, A/B). (5) Aggregation features (groupby stats). (6) Time features (hour, day_of_week, month). (7) Text features (TF-IDF, embeddings). (8) Lag features for time series.","metadata":{"domain":"universal","topic":"feature_engineering","source":"kaggle_grandmasters"}},
        {"id":"ds_val_001","content":"Validation strategies: (1) Stratified K-Fold for classification. (2) TimeSeriesSplit for temporal data. (3) GroupKFold when groups must not leak. (4) Nested CV for hyperparameter tuning. (5) Adversarial validation to detect train/test drift. (6) NEVER use test set for any decision.","metadata":{"domain":"universal","topic":"validation","source":"cross_validation_guide"}},
    ],
    "ml_algorithms": [
        {"id":"ml_grad_001","content":"Gradient Boosting mastery: (1) XGBoost: fast, handles missing values. (2) LightGBM: faster, leaf-wise growth, better for large data. (3) CatBoost: native categorical support, ordered boosting. Tuning priority: learning_rate  n_estimators  max_depth  min_child_weight  subsample  colsample_bytree. Always use early_stopping.","metadata":{"domain":"universal","topic":"gradient_boosting","source":"ml_mastery"}},
        {"id":"ml_nn_001","content":"Neural Network patterns: (1) TabNet for tabular data. (2) 1D-CNN for sequence features. (3) Transformer for attention-based feature interaction. (4) Embedding layers for categoricals. (5) BatchNorm + Dropout for regularization. (6) Learning rate scheduling: cosine annealing or one-cycle. (7) Mixed precision for speed.","metadata":{"domain":"universal","topic":"neural_networks","source":"deep_learning_guide"}},
        {"id":"ml_ens_001","content":"Ensemble strategies: (1) Stacking: train meta-learner on OOF predictions. (2) Blending: weighted average of diverse models. (3) Bagging: reduce variance with random subsets. (4) Diversity is key: mix tree-based + linear + NN. (5) Rank averaging for robust ensembles. (6) Optuna for weight optimization.","metadata":{"domain":"universal","topic":"ensembles","source":"competition_winners"}},
    ],
    "nlp_techniques": [
        {"id":"nlp_emb_001","content":"NLP embeddings: (1) SentenceTransformers for semantic similarity. (2) ByT5 for byte-level processing (no tokenizer needed). (3) mT5/mBART for multilingual tasks. (4) TF-IDF + SVD as strong baseline. (5) Fine-tuning: LoRA/QLoRA for efficient adaptation. (6) Contrastive learning for better embeddings.","metadata":{"domain":"nlp","topic":"embeddings","source":"nlp_survey"}},
        {"id":"nlp_mt_001","content":"Machine Translation techniques: (1) Transfer learning from related languages. (2) Back-translation for data augmentation. (3) Byte-level models for morphologically rich languages. (4) Translation memory for consistency. (5) Hierarchical attention for document-level MT. (6) chrF metric better than BLEU for morphological languages.","metadata":{"domain":"nlp","topic":"machine_translation","source":"arxiv_survey"}},
    ],
    "anti_patterns": [
        {"id":"ap_leak_001","content":"Data leakage anti-patterns: (1) NEVER fit scaler/encoder on full data before split. (2) NEVER use future information in features. (3) NEVER include target-derived features. (4) NEVER use test set for feature selection. (5) ALWAYS check if feature is available at prediction time. (6) GroupKFold when entities repeat across rows.","metadata":{"domain":"universal","topic":"data_leakage","source":"anti_pattern_guide"}},
        {"id":"ap_over_001","content":"Overfitting anti-patterns: (1) Too many features relative to samples. (2) Too deep trees without regularization. (3) No early stopping. (4) Validation score >> test score = overfit to validation. (5) Feature selection on full dataset. (6) Hyperparameter tuning without nested CV.","metadata":{"domain":"universal","topic":"overfitting","source":"anti_pattern_guide"}},
    ],
}


def ingest_universal_knowledge() -> dict:
    """Ingestuje svu univerzalnu bazu znanja u ChromaDB."""
    log_work(AGENT, "INGEST_START", "Universal knowledge base")
    total = 0
    results = {}

    for category, docs in UNIVERSAL_KNOWLEDGE.items():
        doc_texts = [d["content"] for d in docs]
        doc_metas = [d["metadata"] for d in docs]
        doc_ids   = [d["id"] for d in docs]

        r = ingest(doc_texts, doc_metas, doc_ids, "knowledge_base")
        results[category] = r
        total += r.get("upserted", 0)

    proof = register_proof(AGENT, "Universal knowledge ingested", ingest_count=total)
    log_work(AGENT, "INGEST_DONE", f"categories={len(results)}, total_docs={total}")

    SM.set_agent_output(AGENT, {"ingest": results, "total": total})
    return {"status": "INGESTED", "total": total, "categories": results, "proof": proof}


def research(query: str, domain: str = "universal") -> dict:
    """
    Pretrazuje bazu znanja za dati query.
    Koristi Neural Filter za ekstrakciju maksimuma.
    """
    log_work(AGENT, "RESEARCH_START", f"query='{query}' domain='{domain}'")

    # Pretrazi sve relevantne kolekcije
    all_results = []
    for col in ["knowledge_base", "kaggle_insights", "golden_recipes", "domain_specific"]:
        try:
            docs = query_smart(query, col, top_k=5, threshold=0.2)
            for d in docs:
                d["_source_collection"] = col
            all_results.extend(docs)
        except Exception:
            continue

    if not all_results:
        # Ako nema rezultata, prvo ingestuj univerzalno znanje
        log_work(AGENT, "EMPTY_KB", "Knowledge base prazna, ingestujem...")
        ingest_universal_knowledge()
        # Ponovi pretragu
        for col in ["knowledge_base"]:
            try:
                docs = query_smart(query, col, top_k=5, threshold=0.2)
                all_results.extend(docs)
            except Exception:
                continue

    # Sortiraj po score-u
    all_results.sort(key=lambda x: x.get("_score", 0), reverse=True)
    top = all_results[:10]

    log_work(AGENT, "RESEARCH_DONE", f"found={len(all_results)}, top={len(top)}")
    SM.set_agent_output(AGENT, {"query": query, "results_count": len(top)})

    return {
        "status": "RESEARCH_DONE",
        "query": query,
        "results": top,
        "total_found": len(all_results),
    }


def extract_golden_recipe(problem_description: str) -> dict:
    """
    Izvlaci "Golden Recipe" - optimalni pristup za dati problem.
    Koristi LLM + RAG za sintezu.
    """
    log_work(AGENT, "GOLDEN_RECIPE_START", problem_description[:100])

    # Prikupi znanje
    research_results = research(problem_description)
    context_docs = research_results.get("results", [])

    context = "\n\n".join([
        f"[{d.get('_source_collection','?')}] (score:{d.get('_score',0):.3f})\n{d.get('content','')[:500]}"
        for d in context_docs
    ])

    system_prompt = (
        "Ti si ResearchAgent u Usisivac V6 sistemu. "
        "Na osnovu prikupljenog znanja iz baze, sintetisi GOLDEN RECIPE - "
        "optimalni pristup za dati problem. "
        "Format: JSON sa kljucevima: approach, steps, models, features, validation, risks."
    )

    prompt = f"PROBLEM:\n{problem_description}\n\nKNOWLEDGE BASE:\n{context}\n\nGenerisi Golden Recipe:"

    llm_response = llm_call(prompt, system=system_prompt)

    # Ingestuj recipe u golden_recipes kolekciju
    recipe_id = f"recipe_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    ingest(
        [llm_response], [{"problem": problem_description[:200], "agent": AGENT}],
        [recipe_id], "golden_recipes"
    )

    proof = register_proof(AGENT, "Golden recipe extracted",
                           ingest_count=1, artifact_path=None)

    log_work(AGENT, "GOLDEN_RECIPE_DONE", f"recipe_id={recipe_id}")

    return {
        "status": "RECIPE_EXTRACTED",
        "recipe_id": recipe_id,
        "recipe": llm_response,
        "sources_used": len(context_docs),
        "proof": proof,
    }


def run(task: dict) -> dict:
    """Glavni entry point za Orchestrator."""
    action = task.get("action", "research")

    if action == "ingest":
        return ingest_universal_knowledge()
    elif action == "research":
        return research(task.get("query", ""), task.get("domain", "universal"))
    elif action == "golden_recipe":
        return extract_golden_recipe(task.get("problem", ""))
    else:
        return {"status": "UNKNOWN_ACTION", "action": action}
