"""
╔══════════════════════════════════════════════════════════════════════╗
║  Tri-Way Relay — Claude ↔ Gemini ↔ Cline                           ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Relay sistem za komunikaciju između tri AI agenta:
  - Claude (Anthropic) — via Cline extension u VS Code
  - Gemini (Google)    — via Gemini CLI u terminalu
  - Cline (VS Code)    — via Cline extension

Poruke se čuvaju u logs/agent_conversation.jsonl
i u .agent/work_share_state.json → relay_messages[]
"""

import sys, json, datetime, threading
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import log_work
from core import state_manager as SM

AGENT = "Relay"
CHAT_LOG = BASE / "logs" / "agent_conversation.jsonl"
RELAY_STATE = BASE / "logs" / "relay_state.json"

PARTICIPANTS = {
    "claude": {"name": "Claude", "provider": "Anthropic", "interface": "Cline/VS Code"},
    "gemini": {"name": "Gemini", "provider": "Google", "interface": "Gemini CLI/Terminal"},
    "cline":  {"name": "Cline",  "provider": "VS Code Extension", "interface": "VS Code"},
}

_lock = threading.Lock()


def send(from_agent: str, to_agent: str, message: str,
         msg_type: str = "text") -> dict:
    """
    Šalje poruku između agenata.
    Loguje u JSONL i state.
    """
    ts = datetime.datetime.now().isoformat()
    entry = {
        "timestamp": ts,
        "from": from_agent,
        "to": to_agent,
        "message": message,
        "type": msg_type,
        "protocol": "triway_relay_v1",
    }

    with _lock:
        CHAT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CHAT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    SM.add_relay(from_agent, to_agent, message)
    log_work(AGENT, "RELAY_MSG", f"{from_agent}→{to_agent}: {message[:80]}")

    return {"status": "SENT", "entry": entry}


def broadcast(from_agent: str, message: str) -> dict:
    """Šalje poruku svim ostalim agentima."""
    results = []
    for pid in PARTICIPANTS:
        if pid != from_agent.lower():
            r = send(from_agent, pid, message)
            results.append(r)
    return {"status": "BROADCAST", "sent_to": len(results)}


def get_history(limit: int = 50, participant: str = None) -> list:
    """Vraća istoriju poruka."""
    if not CHAT_LOG.exists():
        return []

    messages = []
    for line in CHAT_LOG.read_text("utf-8").strip().split("\n"):
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
            if participant:
                if msg.get("from") == participant or msg.get("to") == participant:
                    messages.append(msg)
            else:
                messages.append(msg)
        except Exception:
            continue

    return messages[-limit:]


def get_context_for_agent(agent_name: str, max_messages: int = 20) -> str:
    """
    Generiše kontekst string za agenta — poslednje poruke iz relay-a.
    Koristi se za injektovanje u prompt agenta.
    """
    history = get_history(limit=max_messages, participant=agent_name.lower())
    if not history:
        return "[No relay messages yet]"

    lines = []
    for msg in history:
        ts = msg.get("timestamp", "?")[:19]
        fr = msg.get("from", "?")
        to = msg.get("to", "?")
        m  = msg.get("message", "")[:200]
        lines.append(f"[{ts}] {fr}→{to}: {m}")

    return "\n".join(lines)


def relay_task_handoff(from_agent: str, to_agent: str,
                       task: dict, context: str = "") -> dict:
    """
    Predaje task od jednog agenta drugom kroz relay.
    Uključuje kontekst iz prethodnih poruka.
    """
    msg = json.dumps({
        "type": "task_handoff",
        "task": task,
        "context": context,
        "relay_history": get_context_for_agent(to_agent, 10),
    }, ensure_ascii=False)

    return send(from_agent, to_agent, msg, msg_type="task_handoff")


# ─── Gemini CLI Integration ──────────────────────────────────────────────────
def format_for_gemini_cli(message: str) -> str:
    """Formatira poruku za Gemini CLI terminal."""
    return f"@relay:{message}"


def parse_gemini_cli_response(response: str) -> dict:
    """Parsira odgovor iz Gemini CLI."""
    return {
        "source": "gemini_cli",
        "response": response,
        "timestamp": datetime.datetime.now().isoformat(),
    }
