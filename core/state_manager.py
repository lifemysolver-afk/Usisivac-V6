"""
╔══════════════════════════════════════════════════════════════════════╗
║  State Manager — Shared Agent State                                 ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import json, threading, datetime
from pathlib import Path

BASE_DIR   = Path(__file__).parent.parent
STATE_FILE = BASE_DIR / ".agent" / "work_share_state.json"
_lock      = threading.Lock()

VALID_STATUSES = {
    "INITIALIZED","STRATEGIST_PLANNING","RESEARCHER_INGESTING",
    "EXECUTOR_RUNNING","GUARDIAN_AUDITING","PENDING_REVIEW",
    "LOOP_RUNNING","COMPLETED","FAILED",
    "BLOCKED_BY_ANTI_SIM","DRIFT_EXCEEDED","WAITING_FOR_INPUT",
}

DEFAULT = {
    "version":"V6","protocol":"Trinity Protocol",
    "global_status":"INITIALIZED",
    "project":None,"goal":None,"domain":None,
    "current_phase":None,"current_agent":None,
    "loop_iteration":0,"max_iterations":100,
    "created_at":None,"updated_at":None,
    "phases":{},"checklist":[],
    "agent_outputs":{},"proof_hashes":{},
    "drift_scores":{},"relay_messages":[],
    "knowledge_stats":{},
}


def read() -> dict:
    with _lock:
        try:
            return json.loads(STATE_FILE.read_text("utf-8"))
        except Exception:
            return dict(DEFAULT)


def write(state: dict):
    with _lock:
        state["updated_at"] = datetime.datetime.now().isoformat()
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2, ensure_ascii=False))


def init(project: str, goal: str, domain: str = "universal") -> dict:
    s = dict(DEFAULT)
    s.update({"project":project,"goal":goal,"domain":domain,
              "global_status":"STRATEGIST_PLANNING",
              "created_at":datetime.datetime.now().isoformat()})
    write(s)
    return s


def set_status(status: str, agent: str = None):
    if status not in VALID_STATUSES:
        raise ValueError(f"Invalid status: {status}")
    s = read()
    s["global_status"] = status
    if agent: s["current_agent"] = agent
    write(s)


def set_agent_output(agent: str, output: dict):
    s = read()
    s.setdefault("agent_outputs",{})[agent] = {
        "data":output,"ts":datetime.datetime.now().isoformat()}
    write(s)


def set_drift(agent: str, score: float):
    s = read()
    s.setdefault("drift_scores",{})[agent] = {
        "score":score,"passed":score<0.4,
        "ts":datetime.datetime.now().isoformat()}
    write(s)


def inc_loop():
    s = read()
    s["loop_iteration"] = s.get("loop_iteration",0) + 1
    write(s)
    return s["loop_iteration"]


def add_relay(from_a: str, to_a: str, msg: str):
    s = read()
    s.setdefault("relay_messages",[]).append({
        "from":from_a,"to":to_a,"message":msg,
        "ts":datetime.datetime.now().isoformat()})
    write(s)
