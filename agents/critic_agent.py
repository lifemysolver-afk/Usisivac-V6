"""
╔══════════════════════════════════════════════════════════════════════╗
║  CriticAgent — Čuvar Kvaliteta                                      ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Identifikuje zablude, anti-patterne, zamke.
Sprečava overfitting, data leakage, loše prakse.
Radi za BILO KOJI domen.
"""

import sys, json, datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, register_proof, log_work
from core.rag_engine import query_smart
from core.llm_client import call as llm_call
from core import state_manager as SM

AGENT = "CriticAgent"

# ─── Kritični anti-paterni za detekciju ───────────────────────────────────────
CRITICAL_CHECKS = {
    "data_leakage": [
        "fit_transform on full data before split",
        "using test data for feature selection",
        "target encoding without proper CV",
        "future information in features",
        "using ID columns as features",
    ],
    "overfitting": [
        "no early stopping",
        "too many features relative to samples",
        "validation score much higher than test",
        "no regularization on deep models",
        "hyperparameter tuning on test set",
    ],
    "bad_practices": [
        "ignoring class imbalance",
        "not checking for duplicates",
        "dropping NaN without analysis",
        "using accuracy for imbalanced data",
        "not setting random seeds",
    ],
}


def critique_plan(plan: dict) -> dict:
    """Kritikuje plan koji je Strategist napravio."""
    log_work(AGENT, "CRITIQUE_PLAN_START", json.dumps(plan)[:200])

    issues = []
    warnings = []

    # Provjeri da li plan ima validaciju
    plan_text = json.dumps(plan).lower()
    if "validation" not in plan_text and "cross" not in plan_text:
        issues.append("CRITICAL: Plan nema strategiju validacije")
    if "baseline" not in plan_text:
        warnings.append("WARNING: Plan nema baseline model")
    if "leakage" not in plan_text:
        warnings.append("WARNING: Plan ne pominje provjeru data leakage-a")

    result = {
        "status": "CRITIQUE_DONE",
        "issues": issues,
        "warnings": warnings,
        "pass": len(issues) == 0,
    }

    log_work(AGENT, "CRITIQUE_PLAN_DONE",
             f"issues={len(issues)}, warnings={len(warnings)}")
    SM.set_agent_output(AGENT, result)
    return result


def critique_code(code: str, context: dict = None) -> dict:
    """
    Kritikuje kod koji je CoderAgent napisao.
    Koristi RAG za anti-pattern detekciju + LLM za duboku analizu.
    """
    log_work(AGENT, "CRITIQUE_CODE_START", f"code_len={len(code)}")

    # 1. Statička provjera anti-paterna
    static_issues = []
    code_lower = code.lower()

    for category, patterns in CRITICAL_CHECKS.items():
        for pattern in patterns:
            keywords = pattern.lower().split()
            if all(kw in code_lower for kw in keywords[:3]):
                static_issues.append(f"[{category}] Possible: {pattern}")

    # Specifične provjere
    if "fit_transform" in code and "train_test_split" in code:
        idx_fit = code.index("fit_transform")
        idx_split = code.index("train_test_split")
        if idx_fit < idx_split:
            static_issues.append("[data_leakage] CRITICAL: fit_transform BEFORE train_test_split")

    if ".fit(" in code and "X_test" in code:
        static_issues.append("[data_leakage] WARNING: .fit() possibly called on test data")

    # 2. RAG-based provjera
    rag_results = query_smart(
        "common data science anti-patterns and pitfalls",
        "knowledge_base", top_k=3
    )
    rag_context = "\n".join([r.get("content","")[:300] for r in rag_results])

    # 3. LLM deep analysis
    system = (
        "Ti si CriticAgent — strogi reviewer koda. "
        "Pronađi SVE probleme: data leakage, overfitting rizike, loše prakse. "
        "Format: JSON lista objekata sa severity (CRITICAL/WARNING/INFO) i description."
    )
    prompt = f"CODE:\n```python\n{code[:3000]}\n```\n\nKNOWN ANTI-PATTERNS:\n{rag_context}\n\nKritikuj:"

    llm_response = llm_call(prompt, system=system)

    result = {
        "status": "CRITIQUE_DONE",
        "static_issues": static_issues,
        "llm_analysis": llm_response,
        "rag_sources": len(rag_results),
        "pass": len(static_issues) == 0,
        "severity": "CRITICAL" if any("CRITICAL" in i for i in static_issues) else "OK",
    }

    proof = register_proof(AGENT, "Code critique completed",
                           ingest_count=len(static_issues) + 1)
    result["proof"] = proof

    log_work(AGENT, "CRITIQUE_CODE_DONE",
             f"static={len(static_issues)}, severity={result['severity']}")
    SM.set_agent_output(AGENT, result)
    return result


def critique_features(features: list, target_info: dict = None) -> dict:
    """Kritikuje feature engineering output."""
    log_work(AGENT, "CRITIQUE_FEATURES_START", f"n_features={len(features)}")

    issues = []
    if len(features) > 500:
        issues.append("WARNING: Previše feature-a (>500), rizik od overfitting-a")
    if len(features) < 3:
        issues.append("WARNING: Premalo feature-a (<3), možda nedovoljno informacija")

    result = {
        "status": "CRITIQUE_DONE",
        "n_features": len(features),
        "issues": issues,
        "pass": len([i for i in issues if "CRITICAL" in i]) == 0,
    }

    log_work(AGENT, "CRITIQUE_FEATURES_DONE", f"issues={len(issues)}")
    return result


def run(task: dict) -> dict:
    action = task.get("action", "critique_code")
    if action == "critique_plan":
        return critique_plan(task.get("plan", {}))
    elif action == "critique_code":
        return critique_code(task.get("code", ""), task.get("context"))
    elif action == "critique_features":
        return critique_features(task.get("features", []), task.get("target_info"))
    else:
        return {"status": "UNKNOWN_ACTION", "action": action}
