"""
╔══════════════════════════════════════════════════════════════════════╗
║  CleanerAgent — Automatsko Čišćenje Podataka                        ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Statističko čišćenje, outlier detekcija (z-score/IQR),
normalizacija, handling missing values.
Generiše cleaning skriptu i STVARNO je izvršava.
"""

import sys, json, datetime, subprocess
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, register_proof, log_work, file_hash
from core.llm_client import call as llm_call
from core import state_manager as SM

AGENT = "CleanerAgent"
OUTPUT_DIR = BASE / "src" / "generated"


def generate_cleaning_script(data_description: str, data_path: str = None) -> dict:
    """
    Generiše i piše cleaning skriptu na disk.
    ANTI-SIM: Fajl se STVARNO kreira.
    """
    log_work(AGENT, "CLEAN_GEN_START", data_description[:100])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    system = (
        "Ti si CleanerAgent — ekspert za čišćenje podataka. "
        "Piši Python kod koji:\n"
        "1. Učitava CSV podatke\n"
        "2. Detektuje i loguje missing values\n"
        "3. Detektuje outliere (z-score > 3 i IQR metod)\n"
        "4. Primenjuje odgovarajuće strategije (imputation, capping, removal)\n"
        "5. Normalizuje numeričke kolone\n"
        "6. Enkodira kategoričke kolone\n"
        "7. Čuva čist dataset\n"
        "8. Generiše cleaning report\n"
        "Vrati SAMO Python kod. Mora biti IZVRŠIV."
    )

    prompt = (
        f"DATA DESCRIPTION:\n{data_description}\n\n"
        f"DATA PATH: {data_path or 'data/input.csv'}\n\n"
        "Generiši kompletnu cleaning skriptu:"
    )

    code = llm_call(prompt, system=system)
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0]
    elif "```" in code:
        code = code.split("```")[1].split("```")[0]

    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generated_cleaning_{ts}.py"
    filepath = OUTPUT_DIR / filename
    filepath.write_text(code, encoding="utf-8")

    fh = file_hash(str(filepath))
    if fh == "FILE_MISSING":
        log_work(AGENT, "CLEAN_GEN_FAILED", "Fajl nije napisan!")
        return {"status": "FAILED", "reason": "File not written"}

    proof = register_proof(AGENT, "Cleaning script generated",
                           file_edited=str(filepath))

    log_work(AGENT, "CLEAN_GEN_DONE", f"file={filename}, hash={fh}")

    return {
        "status": "CLEANING_SCRIPT_GENERATED",
        "file": str(filepath),
        "hash": fh,
        "lines": len(code.splitlines()),
        "proof": proof,
    }


def run_cleaning(script_path: str) -> dict:
    """
    STVARNO pokreće cleaning skriptu.
    ANTI-SIM: Subprocess sa capture_output.
    """
    log_work(AGENT, "CLEAN_RUN_START", script_path)

    if not Path(script_path).exists():
        return {"status": "FAILED", "reason": f"Script not found: {script_path}"}

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=120,
            cwd=str(BASE)
        )

        proof = register_proof(
            AGENT, "Cleaning script executed",
            script_run=script_path,
            script_output=result.stdout
        )

        log_work(AGENT, "CLEAN_RUN_DONE",
                 f"exit={result.returncode}, stdout_lines={len(result.stdout.splitlines())}")

        return {
            "status": "CLEANING_EXECUTED" if result.returncode == 0 else "CLEANING_FAILED",
            "exit_code": result.returncode,
            "stdout": result.stdout[:500],
            "stderr": result.stderr[:500] if result.returncode != 0 else "",
            "proof": proof,
        }
    except subprocess.TimeoutExpired:
        log_work(AGENT, "CLEAN_TIMEOUT", "120s timeout")
        return {"status": "TIMEOUT"}
    except Exception as e:
        log_work(AGENT, "CLEAN_ERROR", str(e))
        return {"status": "ERROR", "error": str(e)}


def run(task: dict) -> dict:
    action = task.get("action", "generate")
    if action == "generate":
        return generate_cleaning_script(
            task.get("data_description", ""),
            task.get("data_path")
        )
    elif action == "execute":
        return run_cleaning(task.get("script_path", ""))
    else:
        return {"status": "UNKNOWN_ACTION", "action": action}
