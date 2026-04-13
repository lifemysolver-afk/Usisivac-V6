import sys, os, json, tempfile, unittest.mock as mock
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

@pytest.fixture(autouse=True)
def mock_external():
    with mock.patch("sentence_transformers.SentenceTransformer") as mock_st:
        mock_st.return_value.encode.return_value = np.random.randn(384).astype(np.float32)
        with mock.patch("chromadb.PersistentClient") as mock_chroma:
            mock_col = mock.Mock()
            mock_col.count.return_value = 1
            mock_chroma.return_value.get_or_create_collection.return_value = mock_col
            mock_chroma.return_value.get_collection.return_value = mock_col
            yield

def test_loptica_engine():
    from loptica.loptica_engine import LopticaEngine
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = LopticaEngine("test", state_dir=tmpdir)
        assert engine.get_current_phase() == "RESEARCH"

def test_knowledge_base():
    from loptica.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(":memory:")
    assert kb.add_solution("test", 1, "me") > 0

def test_veto_board():
    from loptica.veto_board import VetoBoard
    board = VetoBoard(use_llm=False)
    assert board.evaluate_action("Safe")["verdict"] == "PASS"

def test_brain_mass_ingest():
    from loptica.brain_mass_ingest import BrainMassIngest
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a file long enough to pass filters
        content = "# Test Knowledge\n\n" + "X" * 100
        (Path(tmpdir) / "t.md").write_text(content)

        ingestor = BrainMassIngest(db_path=tmpdir)
        ingestor.collection = mock.Mock()
        ingestor.collection.count.side_effect = [0, 1]

        res = ingestor.ingest(tmpdir)
        assert res["status"] == "OK"

if __name__ == "__main__":
    pytest.main([__file__])
