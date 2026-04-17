import sys, os, tempfile
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

def test_loptica_engine():
    from loptica.loptica_engine import LopticaEngine
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = LopticaEngine("test_loptica", state_dir=tmpdir)
        assert engine.get_current_phase() == "RESEARCH"

def test_knowledge_base():
    from loptica.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(":memory:")
    assert kb.add_solution("test", 1, "me") > 0

def test_veto_board():
    from loptica.veto_board import VetoBoard
    board = VetoBoard(use_llm=False)
    assert board.evaluate_action("Safe Action")["verdict"] == "PASS"

def test_brain_mass_ingest(monkeypatch):
    from loptica.brain_mass_ingest import BrainMassIngest
    with tempfile.TemporaryDirectory() as tmpdir:
        content = "# Knowledge\n\n" + "X" * 100
        (Path(tmpdir) / "data.md").write_text(content)
        ingestor = BrainMassIngest(db_path=tmpdir)
        # ingestor.collection is a Mock from conftest.py
        ingestor.collection.count.side_effect = [0, 1]
        assert ingestor.ingest(tmpdir)["status"] == "OK"
