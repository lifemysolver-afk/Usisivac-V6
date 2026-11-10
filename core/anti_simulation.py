"""
╔══════════════════════════════════════════════════════════════════════╗
║  ANTI_SIMULATION_v3 — Core Enforcement Layer                        ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

STROGE ZABRANE — Nijedan agent NE SME izjaviti ništa od sledećeg
bez stvarnog PROOF_HASH-a koji dokazuje izvršenu akciju:

  ✗  "trening završen" / "training complete" / "training finished"
  ✗  "BLEU 19.1" ili bilo koji score bez proof-a
  ✗  "audit pass" / "audit passed"
  ✗  "model spreman" / "model ready"
  ✗  "ingest complete" / "ingest završen"
  ✗  "task complete" / "zadatak završen"
  ✗  "features generated" / "cleaning complete"
  ✗  "golden recipe found" bez stvarnog RAG query-a
"""

import hashlib, json, datetime, os
from pathlib import Path
from typing import Optional

BASE_DIR   = Path(__file__).parent.parent
PROOF_REG  = BASE_DIR / "logs" / "proof_registry.jsonl"
WORK_LOG   = BASE_DIR / "logs" / "work_log.md"

FORBIDDEN = [
    "trening završen","training complete","training finished","training done",
    "successfully trained","model trained","model ready","model spreman",
    "bleu 19","bleu 20","bleu 21","bleu 22","bleu 23","bleu 24","bleu 25",
    "score achieved","accuracy: 0.9","accuracy: 1.0","f1: 0.9",
    "audit pass","audit passed","audit complete","audit ok",
    "ingest complete","ingest završen","data ingested","ingestion done",
    "task complete","zadatak završen","task done","completed successfully",
    "features generated","features ready","feature engineering done",
    "cleaning complete","data cleaned","preprocessing done",
    "golden recipe found","recipe extracted","knowledge extracted",
]


def file_hash(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path,"rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()[:16]
    except FileNotFoundError:
        return "FILE_MISSING"


def register_proof(agent: str, claim: str,
                   file_edited: str = None,
                   script_run: str  = None,
                   script_output: str = None,
                   ingest_count: int  = None,
                   artifact_path: str = None) -> dict:
    ts = datetime.datetime.now().isoformat()
    proof = {"timestamp":ts,"agent":agent,"claim":claim,
             "proof_valid":False,"proofs":{},"blocked_reason":None}

    if file_edited:
        fh = file_hash(file_edited)
        if fh == "FILE_MISSING":
            proof["blocked_reason"] = f"FILE_NOT_FOUND:{file_edited}"
            _wp(proof); return proof
        proof["proofs"]["file_hash"] = fh
        proof["proofs"]["file_path"] = str(file_edited)

    if script_run:
        if not os.path.exists(script_run):
            proof["blocked_reason"] = f"SCRIPT_NOT_FOUND:{script_run}"
            _wp(proof); return proof
        proof["proofs"]["script_path"] = str(script_run)
        if script_output:
            proof["proofs"]["out_hash"] = hashlib.sha256(
                script_output.encode()).hexdigest()[:16]

    if ingest_count is not None:
        if ingest_count == 0:
            proof["blocked_reason"] = "INGEST_COUNT_ZERO"
            _wp(proof); return proof
        proof["proofs"]["ingest_count"] = ingest_count

    if artifact_path:
        ah = file_hash(artifact_path)
        if ah == "FILE_MISSING":
            proof["blocked_reason"] = f"ARTIFACT_MISSING:{artifact_path}"
            _wp(proof); return proof
        proof["proofs"]["artifact_hash"] = ah

    if not proof["blocked_reason"]:
        proof["proof_valid"] = True
    _wp(proof)
    return proof


def _wp(proof):
    PROOF_REG.parent.mkdir(parents=True, exist_ok=True)
    with open(PROOF_REG,"a",encoding="utf-8") as f:
        f.write(json.dumps(proof, ensure_ascii=False)+"\n")


def enforce(agent: str, text: str) -> dict:
    tl = text.lower()
    hits = [p for p in FORBIDDEN if p.lower() in tl]
    if hits:
        log_work(agent,"ANTI_SIM_BLOCK",f"violations={hits} | text={text[:120]}")
        return {"BLOCKED":True,"agent":agent,"violations":hits,
                "message":f"[ANTI_SIM_v3] {agent} blokirano. Fraze: {hits}"}
    return {"BLOCKED":False,"agent":agent,"output":text}


def log_work(agent: str, action: str, details: str = ""):
    ts  = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    entry = f"\n### {ts} | {agent.upper()}\n- **Action**: {action}\n"
    if details:
        entry += f"- **Details**: {details}\n"
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(WORK_LOG,"a",encoding="utf-8") as f:
        f.write(entry)


def log_proof(claim: str, details: dict = None, agent: str = "SYSTEM"):
    """Convenience function to log a proof to the registry."""
    if details is None:
        details = {}
    # Extract only valid arguments for register_proof
    valid_keys = {'file_edited', 'script_run', 'script_output', 'ingest_count', 'artifact_path'}
    filtered_details = {k: v for k, v in details.items() if k in valid_keys}
    register_proof(agent=agent, claim=claim, **filtered_details)
