import sys, os, uuid
from pathlib import Path
import numpy as np
import pytest

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

def test_anti_sim():
    from core.anti_simulation import enforce
    r = enforce("TestAgent", "normal text")
    assert not r["BLOCKED"]
    r2 = enforce("TestAgent", "trening završen")
    assert r2["BLOCKED"]

def test_proof():
    from core.anti_simulation import register_proof
    p = register_proof("TestAgent", "test claim", ingest_count=5)
    assert p["proof_valid"]

def test_rag():
    from core.rag_engine import ingest, stats
    uid = str(uuid.uuid4())[:8]
    r = ingest(["doc "+uid], [{"s": "t"}], ["id_" + uid], "knowledge_base")
    assert r["ok"]
    s = stats()
    assert "knowledge_base" in s

def test_neural_filter():
    from core.neural_filter import MLPScorer, filter_knowledge
    scorer = MLPScorer(input_dim=384)
    v = np.random.randn(384).astype(np.float32)
    v /= np.linalg.norm(v)
    assert 0.0 <= scorer.forward(v) <= 1.0

    docs = [{"content": "text 1"}, {"content": "text 2"}]
    results = filter_knowledge("query", docs, quality_threshold=-2.0)
    assert len(results) > 0

def test_state():
    from core import state_manager as SM
    s = SM.init("test_project_"+str(uuid.uuid4())[:4], "goal", "universal")
    assert "project" in s

def test_llm(monkeypatch):
    from core.llm_client import call
    # Unset all keys to force mock response
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    monkeypatch.setenv("GEMINI_KEY_1", "")
    r = call("Hello")
    assert "MOCK_RESPONSE" in r

def test_research_agent(monkeypatch):
    from agents.research_agent import run
    # Mocking SM to avoid JSON issues
    import core.state_manager as SM
    monkeypatch.setattr(SM, "set_agent_output", lambda *args: None)

    r = run({"action": "ingest"})
    assert r["status"] == "INGESTED"

def test_critic_agent():
    from agents.critic_agent import run
    r = run({"action": "critique_plan", "plan": {"steps": ["step"]}})
    assert r["status"] == "CRITIQUE_DONE"

def test_coder_agent(monkeypatch):
    from agents.coder_agent import run
    import core.state_manager as SM
    monkeypatch.setattr(SM, "set_agent_output", lambda *args: None)

    r = run({"action": "generate_code", "problem": "p", "research_output": {"results": []}})
    assert r["status"] == "CODE_GENERATED"

def test_guardian(monkeypatch):
    from guardian.guardian import run
    import core.state_manager as SM
    monkeypatch.setattr(SM, "set_agent_output", lambda *args: None)

    r = run({"action": "full_audit", "pipeline_results": {}})
    assert "drift_score" in r

def test_relay():
    from relay.triway_relay import send
    r = send("gemini", "claude", "msg")
    assert r["status"] == "SENT"

def test_structure():
    required = ["README.md", "requirements.txt", "judge_guard.py"]
    for f in required:
        assert (BASE / f).exists()
