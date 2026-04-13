import sys, os, unittest.mock as mock, uuid, tempfile
from pathlib import Path
import numpy as np
import pytest

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

# ─── Mocks ──────────────────────────────────────────────────────────────────
@pytest.fixture(autouse=True)
def mock_external_deps():
    # Mock SentenceTransformer
    with mock.patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_instance = mock.Mock()
        def mock_encode(texts, **kwargs):
            if isinstance(texts, str):
                v = np.random.randn(384).astype(np.float32)
                return v / (np.linalg.norm(v) + 1e-9)
            v = np.random.randn(len(texts), 384).astype(np.float32)
            return v / (np.linalg.norm(v, axis=1, keepdims=True) + 1e-9)
        mock_instance.encode.side_effect = mock_encode
        mock_st.return_value = mock_instance

        # Mock ChromaDB
        with mock.patch("chromadb.PersistentClient") as mock_chroma:
            mock_col = mock.Mock()
            mock_col.count.return_value = 1
            mock_col.query.return_value = {
                "documents": [["mock content"]],
                "metadatas": [[{"source": "mock"}]]
            }
            mock_chroma.return_value.get_or_create_collection.return_value = mock_col
            mock_chroma.return_value.get_collection.return_value = mock_col

            # Mock Paths for RAG to avoid conflict in CI
            with tempfile.TemporaryDirectory() as tmpdir:
                tmp_path = Path(tmpdir)
                with mock.patch("core.rag_engine.CHROMA_PATH", tmp_path / "chroma"):
                    with mock.patch("core.rag_engine.FALLBACK_DIR", tmp_path / "kb"):
                        yield

# ─── Tests ───────────────────────────────────────────────────────────────────

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
    assert r["upserted"] >= 1
    s = stats()
    assert "knowledge_base" in s

def test_neural_filter():
    from core.neural_filter import MLPScorer, filter_knowledge
    scorer = MLPScorer(input_dim=384)
    v = np.random.randn(384).astype(np.float32)
    v /= np.linalg.norm(v)
    assert 0.0 <= scorer.forward(v) <= 1.0

    docs = [{"content": "text 1"}, {"content": "text 2"}]
    # Force low threshold to ensure match even with random vectors
    results = filter_knowledge("query", docs, quality_threshold=-2.0)
    assert len(results) > 0

def test_state():
    from core import state_manager as SM
    s = SM.init("test_project_"+str(uuid.uuid4())[:4], "goal", "universal")
    assert "project" in s

def test_llm():
    from core.llm_client import call
    with mock.patch.dict(os.environ, {}, clear=True):
        r = call("Hello")
        assert "MOCK_RESPONSE" in r

def test_research_agent():
    from agents.research_agent import run
    # Mocking SM.set_agent_output to avoid JSON serialization of Mocks
    with mock.patch("core.state_manager.set_agent_output"):
        r = run({"action": "ingest"})
        assert r["status"] == "INGESTED"

def test_critic_agent():
    from agents.critic_agent import run
    r = run({"action": "critique_plan", "plan": {"steps": ["step"]}})
    assert r["status"] == "CRITIQUE_DONE"

def test_coder_agent():
    from agents.coder_agent import run
    with mock.patch("core.state_manager.set_agent_output"):
        r = run({"action": "generate_code", "problem": "p", "research_output": {"results": []}})
        assert r["status"] == "CODE_GENERATED"

def test_guardian():
    from guardian.guardian import run
    with mock.patch("guardian.guardian.compute_drift_score", return_value=0.1):
        with mock.patch("core.state_manager.set_agent_output"):
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
