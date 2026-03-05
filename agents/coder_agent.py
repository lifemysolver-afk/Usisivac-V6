"""
╔══════════════════════════════════════════════════════════════════════╗
║  CoderAgent V6 — Mozak Operacije                                    ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Na osnovu izveštaja ResearchAgent-a i CriticAgent-a,
samostalno piše Python kod.
Generiše: feature engineering, model training, pipeline skripte.
ANTI-SIM: Kod se STVARNO piše na disk, ne simulira.
"""

import sys, json, datetime, os
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, register_proof, log_work, file_hash
from core.rag_engine import query_smart
from core.llm_client import call as llm_call
from core import state_manager as SM

AGENT = "CoderAgent"
OUTPUT_DIR = BASE / "src" / "generated"


def generate_code(problem: str, research_output: dict, critic_output: dict = None) -> dict:
    """
    Generiše Python kod na osnovu research-a i kritike.
    ANTI-SIM: Kod se STVARNO piše na disk.
    """
    log_work(AGENT, "CODE_GEN_START", problem[:100])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Pripremi kontekst iz research-a
    research_context = ""
    if research_output and "results" in research_output:
        for r in research_output["results"][:5]:
            research_context += f"\n---\n{r.get('content','')[:400]}\n"

    # Pripremi kritiku
    critic_context = ""
    if critic_output:
        issues = critic_output.get("static_issues", [])
        if issues:
            critic_context = f"\nIZBEGAVAJ OVE PROBLEME:\n" + "\n".join(f"- {i}" for i in issues)

    # RAG kontekst
    rag_docs = query_smart(problem, "knowledge_base", top_k=3)
    rag_context = "\n".join([d.get("content","")[:300] for d in rag_docs])

    system = (
        "Ti si CoderAgent V6 — autonomni Python programer. "
        "Piši čist, produkcioni Python kod. "
        "PRAVILA:\n"
        "1. Uvek koristi proper train/test split PRIJE fit_transform\n"
        "2. Uvek dodaj early_stopping\n"
        "3. Uvek postavi random_state/seed\n"
        "4. Koristi StratifiedKFold za klasifikaciju\n"
        "5. Dodaj komentare na srpskom i engleskom\n"
        "6. Kod mora biti IZVRŠIV — ne pseudo-kod\n"
        "Vrati SAMO Python kod, bez markdown blokova."
    )

    prompt = (
        f"PROBLEM:\n{problem}\n\n"
        f"RESEARCH INSIGHTS:\n{research_context}\n\n"
        f"RAG KNOWLEDGE:\n{rag_context}\n\n"
        f"{critic_context}\n\n"
        "Napiši kompletan Python kod koji rešava ovaj problem:"
    )

    code = llm_call(prompt, system=system)

    # Očisti kod od markdown blokova ako ih ima
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    # STVARNO piši na disk
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_solution_{ts}.py"
    filepath = OUTPUT_DIR / filename
    filepath.write_text(code, encoding="utf-8")

    # Verifikuj da je fajl napisan
    fh = file_hash(str(filepath))
    if fh == "FILE_MISSING":
        log_work(AGENT, "CODE_GEN_FAILED", "Fajl nije napisan na disk!")
        return {"status": "FAILED", "reason": "File not written to disk"}

    proof = register_proof(AGENT, "Code generated and written to disk",
                           file_edited=str(filepath))

    log_work(AGENT, "CODE_GEN_DONE",
             f"file={filename}, hash={fh}, lines={len(code.splitlines())}")

    SM.set_agent_output(AGENT, {
        "file": str(filepath), "hash": fh,
        "lines": len(code.splitlines())
    })

    return {
        "status": "CODE_GENERATED",
        "file": str(filepath),
        "filename": filename,
        "hash": fh,
        "lines": len(code.splitlines()),
        "code_preview": code[:500],
        "proof": proof,
    }


def generate_features(data_description: str, research_output: dict) -> dict:
    """Generiše feature engineering skriptu."""
    log_work(AGENT, "FEAT_GEN_START", data_description[:100])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    system = (
        "Ti si FeatureEngineer. Piši Python kod za feature engineering. "
        "Koristi pandas, numpy, sklearn. "
        "Svaki feature mora imati komentar zašto je koristan. "
        "Vrati SAMO Python kod."
    )

    research_context = ""
    if research_output and "results" in research_output:
        for r in research_output["results"][:3]:
            research_context += f"\n{r.get('content','')[:300]}\n"

    prompt = (
        f"DATA:\n{data_description}\n\n"
        f"RESEARCH:\n{research_context}\n\n"
        "Generiši feature engineering pipeline:"
    )

    code = llm_call(prompt, system=system)
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_features_{ts}.py"
    filepath = OUTPUT_DIR / filename
    filepath.write_text(code, encoding="utf-8")

    fh = file_hash(str(filepath))
    proof = register_proof(AGENT, "Features generated", file_edited=str(filepath))

    log_work(AGENT, "FEAT_GEN_DONE", f"file={filename}, hash={fh}")

    return {
        "status": "FEATURES_GENERATED",
        "file": str(filepath),
        "hash": fh,
        "lines": len(code.splitlines()),
        "proof": proof,
    }


def run(task: dict) -> dict:
    action = task.get("action", "generate_code")
    if action == "generate_code":
        return generate_code(
            task.get("problem", ""),
            task.get("research_output", {}),
            task.get("critic_output"),
        )
    elif action == "generate_features":
        return generate_features(
            task.get("data_description", ""),
            task.get("research_output", {}),
        )
    else:
        return {"status": "UNKNOWN_ACTION", "action": action}
