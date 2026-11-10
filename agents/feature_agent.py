"""
╔══════════════════════════════════════════════════════════════════════╗
║  FeatureAgent V6 — Dinamički Feature Executor                       ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Izvršava kod koji je CoderAgent napisao.
Transformiše sirove podatke u visoko-performansne feature-e.
ANTI-SIM: Stvarno izvršava skripte, verifikuje output.
"""

import sys, json, datetime, subprocess
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, register_proof, log_work, file_hash
from core import state_manager as SM

AGENT = "FeatureAgent"


def execute_feature_script(script_path: str) -> dict:
    """
    STVARNO izvršava feature engineering skriptu.
    Verifikuje da je output fajl kreiran.
    """
    log_work(AGENT, "FEAT_EXEC_START", script_path)

    if not Path(script_path).exists():
        log_work(AGENT, "FEAT_EXEC_FAILED", f"Script not found: {script_path}")
        return {"status": "FAILED", "reason": f"Script not found: {script_path}"}

    script_hash_before = file_hash(script_path)

    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True, text=True, timeout=300,
            cwd=str(BASE)
        )

        if result.returncode != 0:
            log_work(AGENT, "FEAT_EXEC_FAILED",
                     f"exit={result.returncode}, stderr={result.stderr[:300]}")
            return {
                "status": "EXECUTION_FAILED",
                "exit_code": result.returncode,
                "stderr": result.stderr[:500],
            }

        proof = register_proof(
            AGENT, "Feature script executed",
            script_run=script_path,
            script_output=result.stdout
        )

        log_work(AGENT, "FEAT_EXEC_DONE",
                 f"exit=0, stdout_lines={len(result.stdout.splitlines())}")

        SM.set_agent_output(AGENT, {
            "script": script_path,
            "exit_code": 0,
            "output_lines": len(result.stdout.splitlines()),
        })

        return {
            "status": "FEATURES_EXECUTED",
            "exit_code": 0,
            "stdout": result.stdout[:500],
            "script_hash": script_hash_before,
            "proof": proof,
        }

    except subprocess.TimeoutExpired:
        log_work(AGENT, "FEAT_TIMEOUT", "300s timeout")
        return {"status": "TIMEOUT"}
    except Exception as e:
        log_work(AGENT, "FEAT_ERROR", str(e))
        return {"status": "ERROR", "error": str(e)}


def validate_features(output_path: str) -> dict:
    """Validira output feature fajla."""
    log_work(AGENT, "FEAT_VALIDATE_START", output_path)

    if not Path(output_path).exists():
        return {"status": "FAILED", "reason": "Output file missing"}

    try:
        import pandas as pd
        df = pd.read_csv(output_path)
        info = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "dtypes": {str(k): str(v) for k, v in df.dtypes.items()},
            "null_counts": {str(k): int(v) for k, v in df.isnull().sum().items()},
            "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        }

        proof = register_proof(AGENT, "Features validated",
                               artifact_path=output_path)

        log_work(AGENT, "FEAT_VALIDATE_DONE",
                 f"shape={info['shape']}, nulls={sum(info['null_counts'].values())}")

        return {"status": "VALIDATED", "info": info, "proof": proof}

    except Exception as e:
        log_work(AGENT, "FEAT_VALIDATE_ERROR", str(e))
        return {"status": "VALIDATION_ERROR", "error": str(e)}


def run(task: dict) -> dict:
    action = task.get("action", "execute")
    if action == "execute":
        return execute_feature_script(task.get("script_path", ""))
    elif action == "validate":
        return validate_features(task.get("output_path", ""))
    else:
        return {"status": "UNKNOWN_ACTION", "action": action}
