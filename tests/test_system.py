"""
+----------------------------------------------------------------------+
|  End-to-End Test - Usisivac V6                                      |
|  Verifikuje da svi moduli rade ispravno                             |
+----------------------------------------------------------------------+
"""

import sys, json, os
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))
os.chdir(BASE)

PASS = 0
FAIL = 0

def test(name, fn):
    global PASS, FAIL
    try:
        result = fn()
        if result:
            PASS += 1
            print(f"  [PASS] {name}")
        else:
            FAIL += 1
            print(f"  [FAIL] {name} - returned False")
    except Exception as e:
        FAIL += 1
        print(f"  [FAIL] {name} - {e}")


# -- Test 1: Anti-Simulation ------------------------------------------------
def test_anti_sim():
    from core.anti_simulation import enforce, register_proof, log_work
    r = enforce("TestAgent", "ovo je normalan tekst")
    assert not r["BLOCKED"], "Should not block normal text"
    r2 = enforce("TestAgent", "trening zavrsen bez dokaza")
    assert r2["BLOCKED"], "Should block forbidden phrase"
    log_work("TestAgent", "TEST", "test entry")
    return True

# -- Test 2: Proof Registry --------------------------------------------------
def test_proof():
    from core.anti_simulation import register_proof
    p = register_proof("TestAgent", "test claim", ingest_count=5)
    assert p["proof_valid"], "Proof should be valid with ingest_count=5"
    p2 = register_proof("TestAgent", "test claim", ingest_count=0)
    assert not p2["proof_valid"], "Proof should be invalid with ingest_count=0"
    return True

# -- Test 3: RAG Engine ------------------------------------------------------
def test_rag():
    from core.rag_engine import ingest, query_raw, stats
    r = ingest(
        ["Test document about machine learning"],
        [{"source": "test"}],
        ["test_001"],
        "knowledge_base"
    )
    assert r["ok"], f"Ingest should succeed: {r}"
    assert r["upserted"] >= 1, "Should upsert at least 1 doc"
    s = stats()
    assert "knowledge_base" in s, "Stats should include knowledge_base"
    return True

# -- Test 4: Neural Filter --------------------------------------------------
def test_neural_filter():
    from core.neural_filter import MLPScorer, embed, filter_knowledge
    scorer = MLPScorer(input_dim=384)
    import numpy as np
    x = np.random.randn(384)
    score = scorer.forward(x)
    assert 0.0 <= score <= 1.0, f"Score should be [0,1]: {score}"
    # Test filter
    docs = [
        {"content": "Machine learning is great for prediction"},
        {"content": "Cooking recipes for pasta"},
    ]
    results = filter_knowledge("machine learning prediction", docs, top_k=2, quality_threshold=0.0)
    assert len(results) > 0, "Should return at least 1 result"
    return True

# -- Test 5: State Manager --------------------------------------------------
def test_state():
    from core import state_manager as SM
    s = SM.init("test_project", "test goal", "universal")
    assert s["project"] == "test_project"
    SM.set_status("RESEARCHER_INGESTING", "ResearchAgent")
    s2 = SM.read()
    assert s2["global_status"] == "RESEARCHER_INGESTING"
    SM.set_drift("TestAgent", 0.25)
    s3 = SM.read()
    assert s3["drift_scores"]["TestAgent"]["passed"] == True
    return True

# -- Test 6: LLM Client (mock mode) ----------------------------------------
def test_llm():
    from core.llm_client import call
    r = call("Hello world", provider="groq")
    assert r is not None, "LLM should return something (even mock)"
    assert len(r) > 0
    return True

# -- Test 7: ResearchAgent --------------------------------------------------
def test_research_agent():
    from agents.research_agent import run
    r = run({"action": "ingest"})
    assert r["status"] == "INGESTED", f"Should ingest: {r}"
    assert r["total"] > 0, "Should ingest > 0 docs"
    r2 = run({"action": "research", "query": "feature engineering best practices"})
    assert r2["status"] == "RESEARCH_DONE"
    return True

# -- Test 8: CriticAgent ----------------------------------------------------
def test_critic_agent():
    from agents.critic_agent import run
    r = run({"action": "critique_plan", "plan": {"steps": ["train model"]}})
    assert r["status"] == "CRITIQUE_DONE"
    return True

# -- Test 9: CoderAgent ------------------------------------------------------
def test_coder_agent():
    from agents.coder_agent import run
    r = run({
        "action": "generate_code",
        "problem": "Simple linear regression",
        "research_output": {"results": []},
    })
    assert r["status"] == "CODE_GENERATED", f"Should generate: {r}"
    assert Path(r["file"]).exists(), "Generated file should exist on disk"
    return True

# -- Test 10: Guardian ------------------------------------------------------
def test_guardian():
    from guardian.guardian import run
    r = run({"action": "full_audit", "pipeline_results": {}})
    assert "drift_score" in r, "Should have drift_score"
    assert "verdict" in r, "Should have verdict"
    return True

# -- Test 11: Relay ----------------------------------------------------------
def test_relay():
    from relay.triway_relay import send, get_history, broadcast
    r = send("gemini", "claude", "Test message from test suite")
    assert r["status"] == "SENT"
    h = get_history(limit=5)
    assert len(h) > 0, "Should have at least 1 message"
    return True

# -- Test 12: File Structure ------------------------------------------------
def test_structure():
    required = [
        "core/anti_simulation.py", "core/rag_engine.py", "core/neural_filter.py",
        "core/llm_client.py", "core/state_manager.py",
        "agents/research_agent.py", "agents/critic_agent.py",
        "agents/coder_agent.py", "agents/cleaner_agent.py", "agents/feature_agent.py",
        "orchestrator/orchestrator.py", "orchestrator/a2a_servers.py",
        "guardian/guardian.py", "relay/triway_relay.py",
        "config/antigravity_setup.py", ".vscode/settings.json", ".clinerules",
        "MASTER_PROMPT.md", "README.md", "requirements.txt",
    ]
    missing = [f for f in required if not (BASE / f).exists()]
    assert len(missing) == 0, f"Missing files: {missing}"
    return True


# -- Run All ------------------------------------------------------------------
if __name__ == "__main__":
    print("+------------------------------------------+")
    print("|  Usisivac V6 - End-to-End Tests          |")
    print("+------------------------------------------+\n")

    test("Anti-Simulation Enforcement", test_anti_sim)
    test("Proof Registry", test_proof)
    test("RAG Engine (ChromaDB)", test_rag)
    test("Neural Filter (MLP + MMR)", test_neural_filter)
    test("State Manager", test_state)
    test("LLM Client (mock/real)", test_llm)
    test("ResearchAgent (ingest + research)", test_research_agent)
    test("CriticAgent (critique)", test_critic_agent)
    test("CoderAgent (code generation)", test_coder_agent)
    test("Guardian (audit + drift)", test_guardian)
    test("Tri-Way Relay", test_relay)
    test("File Structure Integrity", test_structure)

    print(f"\n{'='*50}")
    print(f"  Results: {PASS} passed, {FAIL} failed, {PASS+FAIL} total")
    print(f"{'='*50}")

    sys.exit(0 if FAIL == 0 else 1)
