"""
+----------------------------------------------------------------------+
|  Guardian - QA Auditor & Drift Detector                             |
|  Usisivac V6 | Trinity Protocol | JudgeGuard v2.1                   |
+----------------------------------------------------------------------+

Audit pipeline:
  1. Semantic drift scoring (0.0  1.0, limit: 0.4)
  2. Artifact integrity verification (SHA-256)
  3. Anti-simulation enforcement
  4. Proof registry validation
  5. Work log audit trail
  6. Self-healing: ako drift > 0.4, salje feedback za korekciju
"""

import sys, json, datetime, hashlib
from pathlib import Path
from typing import Dict, List

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import (
    enforce, register_proof, log_work, file_hash, PROOF_REG
)
from core.rag_engine import query_smart
from core.llm_client import call as llm_call
from core import state_manager as SM

AGENT = "Guardian"
DRIFT_THRESHOLD = 0.4
AUDIT_LOG = BASE / "logs" / "guardian_audit.jsonl"


def compute_drift_score(action_description: str, project_essence: str) -> float:
    """
    Izracunava semanticki drift score.
    Koristi embedding cosine similarity + LLM procenu.
    Score: 0.0 (potpuno uskladen)  1.0 (potpuno devijiran)
    """
    try:
        from core.neural_filter import embed
        import numpy as np

        emb_action  = embed(action_description)
        emb_essence = embed(project_essence)
        cos_sim = float(np.dot(emb_action, emb_essence))
        drift = 1.0 - max(0.0, min(1.0, cos_sim))
        return round(drift, 4)
    except Exception:
        # Fallback: keyword overlap
        a_words = set(action_description.lower().split())
        e_words = set(project_essence.lower().split())
        if not e_words:
            return 0.5
        overlap = len(a_words & e_words) / max(len(e_words), 1)
        return round(1.0 - min(1.0, overlap), 4)


def verify_proof_registry() -> dict:
    """
    Verifikuje sve proof-ove u registru.
    Provjerava da li su fajlovi jos uvek na disku i hash-evi validni.
    """
    log_work(AGENT, "PROOF_VERIFY_START", str(PROOF_REG))

    if not PROOF_REG.exists():
        return {"status": "NO_PROOFS", "total": 0, "valid": 0, "invalid": 0}

    total = valid = invalid = 0
    invalid_details = []

    for line in PROOF_REG.read_text("utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            proof = json.loads(line)
            total += 1

            if not proof.get("proof_valid"):
                invalid += 1
                invalid_details.append({
                    "agent": proof.get("agent"),
                    "claim": proof.get("claim"),
                    "reason": proof.get("blocked_reason"),
                })
                continue

            # Re-verify file hashes
            proofs = proof.get("proofs", {})
            fp = proofs.get("file_path")
            if fp:
                current_hash = file_hash(fp)
                stored_hash  = proofs.get("file_hash", "")
                if current_hash != stored_hash:
                    invalid += 1
                    invalid_details.append({
                        "agent": proof.get("agent"),
                        "claim": proof.get("claim"),
                        "reason": f"Hash mismatch: {stored_hash}  {current_hash}",
                    })
                    continue

            valid += 1

        except json.JSONDecodeError:
            invalid += 1

    result = {
        "status": "VERIFIED",
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "invalid_details": invalid_details[:10],
    }

    log_work(AGENT, "PROOF_VERIFY_DONE",
             f"total={total}, valid={valid}, invalid={invalid}")
    return result


def verify_artifacts(pipeline_results: dict) -> dict:
    """Verifikuje da svi generisani fajlovi postoje na disku."""
    log_work(AGENT, "ARTIFACT_VERIFY_START", "")

    found = []
    missing = []

    for agent_name, result in pipeline_results.items():
        if isinstance(result, dict):
            fp = result.get("file")
            if fp and Path(fp).exists():
                found.append({"agent": agent_name, "file": fp,
                              "hash": file_hash(fp)})
            elif fp:
                missing.append({"agent": agent_name, "file": fp})

    result = {
        "status": "VERIFIED",
        "found": found,
        "missing": missing,
        "all_present": len(missing) == 0,
    }

    log_work(AGENT, "ARTIFACT_VERIFY_DONE",
             f"found={len(found)}, missing={len(missing)}")
    return result


def audit_work_log() -> dict:
    """Analizira work_log.md za anomalije."""
    wl = BASE / "logs" / "work_log.md"
    if not wl.exists():
        return {"status": "NO_LOG", "entries": 0}

    content = wl.read_text("utf-8")
    entries = content.count("### ")
    blocks  = content.count("ANTI_SIM_BLOCK")
    errors  = content.count("FAILED") + content.count("ERROR")

    return {
        "status": "AUDITED",
        "entries": entries,
        "anti_sim_blocks": blocks,
        "errors": errors,
        "health": "GOOD" if blocks == 0 and errors < 3 else "NEEDS_ATTENTION",
    }


def full_audit(pipeline_results: dict) -> dict:
    """
    Kompletan Guardian audit.
    ANTI-SIM: Stvarno izracunava drift, verifikuje proof-ove, pise audit log.
    """
    log_work(AGENT, "FULL_AUDIT_START", "")
    SM.set_status("GUARDIAN_AUDITING", AGENT)

    state = SM.read()
    project_essence = state.get("goal", "") or state.get("project", "")

    # 1. Drift score za svaki agent output
    drift_scores = {}
    for agent_name, result in pipeline_results.items():
        if isinstance(result, dict):
            desc = json.dumps(result, default=str)[:500]
            score = compute_drift_score(desc, project_essence)
            drift_scores[agent_name] = score
            SM.set_drift(agent_name, score)

    avg_drift = sum(drift_scores.values()) / max(len(drift_scores), 1)

    # 2. Proof verification
    proof_result = verify_proof_registry()

    # 3. Artifact verification
    artifact_result = verify_artifacts(pipeline_results)

    # 4. Work log audit
    log_result = audit_work_log()

    # 5. Overall verdict
    passed = (
        avg_drift < DRIFT_THRESHOLD
        and proof_result.get("invalid", 0) == 0
        and artifact_result.get("all_present", False)
    )

    audit_record = {
        "timestamp": datetime.datetime.now().isoformat(),
        "drift_score": round(avg_drift, 4),
        "drift_per_agent": drift_scores,
        "drift_threshold": DRIFT_THRESHOLD,
        "drift_passed": avg_drift < DRIFT_THRESHOLD,
        "proof_verification": proof_result,
        "artifact_verification": artifact_result,
        "work_log_audit": log_result,
        "overall_passed": passed,
        "verdict": "APPROVED" if passed else "REJECTED",
    }

    # Pisi audit log
    AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(AUDIT_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps(audit_record, ensure_ascii=False, default=str) + "\n")

    proof = register_proof(
        AGENT, f"Full audit: {'APPROVED' if passed else 'REJECTED'}",
        artifact_path=str(AUDIT_LOG)
    )

    log_work(AGENT, "FULL_AUDIT_DONE",
             f"drift={avg_drift:.4f}, proofs_ok={proof_result.get('valid',0)}, "
             f"artifacts_ok={artifact_result.get('all_present')}, verdict={'APPROVED' if passed else 'REJECTED'}")

    audit_record["proof"] = proof
    SM.set_agent_output(AGENT, audit_record)

    return audit_record


def self_heal(audit_result: dict) -> dict:
    """
    Self-healing: ako audit ne prode, generise feedback za korekciju.
    """
    if audit_result.get("overall_passed"):
        return {"status": "NO_HEALING_NEEDED"}

    issues = []
    if audit_result.get("drift_score", 0) >= DRIFT_THRESHOLD:
        issues.append(f"Drift score {audit_result['drift_score']:.4f} > {DRIFT_THRESHOLD}")
    if audit_result.get("proof_verification", {}).get("invalid", 0) > 0:
        issues.append("Invalid proofs detected")
    if not audit_result.get("artifact_verification", {}).get("all_present", True):
        issues.append("Missing artifacts")

    feedback = {
        "status": "HEALING_REQUIRED",
        "issues": issues,
        "recommendation": "Re-run pipeline with corrections",
        "corrections": [],
    }

    for issue in issues:
        if "drift" in issue.lower():
            feedback["corrections"].append("Refocus agents on original problem statement")
        if "proof" in issue.lower():
            feedback["corrections"].append("Re-execute failed steps with proper proof")
        if "artifact" in issue.lower():
            feedback["corrections"].append("Re-generate missing files")

    log_work(AGENT, "SELF_HEAL", f"issues={len(issues)}, corrections={len(feedback['corrections'])}")
    return feedback


def run(task: dict) -> dict:
    action = task.get("action", "full_audit")
    if action == "full_audit":
        result = full_audit(task.get("pipeline_results", {}))
        if not result.get("overall_passed"):
            result["healing"] = self_heal(result)
        return result
    elif action == "drift_score":
        return {
            "drift_score": compute_drift_score(
                task.get("description", ""),
                task.get("essence", "")
            )
        }
    elif action == "verify_proofs":
        return verify_proof_registry()
    elif action == "verify_artifacts":
        return verify_artifacts(task.get("pipeline_results", {}))
    else:
        return {"status": "UNKNOWN_ACTION", "action": action}
