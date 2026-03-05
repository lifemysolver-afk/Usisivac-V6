"""
╔══════════════════════════════════════════════════════════════════════╗
║  A2A Servers — Agent-to-Agent HTTP Protocol                         ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Ports:
  8081 → Orchestrator (Strategist)
  8082 → CoderAgent (Executor)
  8083 → Guardian (QA)
  8084 → ResearchAgent
  8085 → CriticAgent
  8086 → CleanerAgent
  8087 → FeatureAgent
  8088 → Relay (tri-way chat)
"""

import http.server, socketserver, json, threading, sys, os, datetime
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, log_work
from core import state_manager as SM
from core.rag_engine import query_smart, ingest, stats as rag_stats

# ─── Chat relay ───────────────────────────────────────────────────────────────
CHAT_MESSAGES = []
CHAT_LOCK = threading.Lock()
CHAT_LOG = BASE / "logs" / "agent_conversation.jsonl"

AGENTS_CFG = {
    "Orchestrator":  {"port":8081, "role":"Strategist & Coordinator"},
    "CoderAgent":    {"port":8082, "role":"ML Engineer & Code Generator"},
    "Guardian":      {"port":8083, "role":"QA Auditor & Drift Detector"},
    "ResearchAgent": {"port":8084, "role":"Knowledge Vacuum & RAG Enricher"},
    "CriticAgent":   {"port":8085, "role":"Quality Guard & Anti-Pattern Detector"},
    "CleanerAgent":  {"port":8086, "role":"Data Cleaner & Normalizer"},
    "FeatureAgent":  {"port":8087, "role":"Feature Executor & Validator"},
    "Relay":         {"port":8088, "role":"Tri-Way Chat (Claude↔Gemini↔Cline)"},
}


def append_chat(from_a, to_a, msg):
    ts = datetime.datetime.now().isoformat()
    entry = {"timestamp":ts,"from":from_a,"to":to_a,"message":msg}
    with CHAT_LOCK:
        CHAT_MESSAGES.append(entry)
        CHAT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(CHAT_LOG,"a") as f:
            f.write(json.dumps(entry)+"\n")
    SM.add_relay(from_a, to_a, msg)
    return entry


class AgentHandler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_GET(self):
        name = self.server.agent_name
        cfg  = AGENTS_CFG[name]

        if self.path == "/agent-card":
            self._r(200, {"name":name,"role":cfg["role"],
                          "protocol":"Usisivac V6","status":"ONLINE"})
        elif self.path == "/status":
            self._r(200, {"agent":name,"state":SM.read()})
        elif self.path == "/live-chat":
            with CHAT_LOCK:
                self._r(200, {"messages":CHAT_MESSAGES[-50:]})
        elif self.path == "/rag-stats":
            self._r(200, rag_stats())
        elif self.path.startswith("/rag/query"):
            from urllib.parse import urlparse, parse_qs
            p = parse_qs(urlparse(self.path).query)
            q = p.get("q",[""])[0]
            c = p.get("collection",["knowledge_base"])[0]
            self._r(200, {"results":query_smart(q,c)})
        else:
            self.send_error(404)

    def do_POST(self):
        name = self.server.agent_name
        cl   = int(self.headers.get("Content-Length",0))
        body = json.loads(self.rfile.read(cl)) if cl else {}

        if self.path == "/live-chat":
            e = append_chat(name, body.get("to","all"), body.get("message",""))
            self._r(200, {"status":"SENT","entry":e})
            return

        if self.path == "/relay":
            # Tri-way relay: Claude ↔ Gemini ↔ Cline
            from_a = body.get("from","unknown")
            to_a   = body.get("to","all")
            msg    = body.get("message","")
            e = append_chat(from_a, to_a, msg)
            self._r(200, {"status":"RELAYED","entry":e})
            return

        if self.path == "/execute":
            # Anti-sim check
            task = body.get("task","")
            chk = enforce(name, task)
            if chk["BLOCKED"]:
                self._r(403, chk)
                return

            # Route to appropriate agent
            action = body.get("action","")
            if name == "ResearchAgent":
                from agents.research_agent import run as r
                self._r(200, r(body))
            elif name == "CoderAgent":
                from agents.coder_agent import run as r
                self._r(200, r(body))
            elif name == "CriticAgent":
                from agents.critic_agent import run as r
                self._r(200, r(body))
            elif name == "CleanerAgent":
                from agents.cleaner_agent import run as r
                self._r(200, r(body))
            elif name == "FeatureAgent":
                from agents.feature_agent import run as r
                self._r(200, r(body))
            elif name == "Guardian":
                from guardian.guardian import run as r
                self._r(200, r(body))
            else:
                self._r(200, {"status":"RECEIVED","agent":name})
            return

        self.send_error(404)

    def _r(self, code, data):
        self.send_response(code)
        self.send_header("Content-type","application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data,indent=2,default=str).encode())


class AgentServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    def __init__(self, addr, handler, name):
        self.agent_name = name
        super().__init__(addr, handler)


def start_all():
    threads = []
    for name, cfg in AGENTS_CFG.items():
        port = cfg["port"]
        t = threading.Thread(
            target=lambda n=name, p=port: AgentServer(("",p), AgentHandler, n).serve_forever(),
            daemon=True
        )
        t.start()
        threads.append(t)
        print(f"  [{name}] → http://localhost:{port}")

    log_work("A2A_SERVER", "ALL_ONLINE",
             f"agents={list(AGENTS_CFG.keys())}, ports={[c['port'] for c in AGENTS_CFG.values()]}")
    print(f"\n  All {len(AGENTS_CFG)} agents online. Press Ctrl+C to stop.\n")
    return threads


if __name__ == "__main__":
    print("╔══════════════════════════════════════════╗")
    print("║  Usisivac V6 — A2A Servers Starting...   ║")
    print("╚══════════════════════════════════════════╝\n")
    threads = start_all()
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        print("\nShutting down...")
