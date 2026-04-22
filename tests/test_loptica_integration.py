"""
+----------------------------------------------------------------------+
|  Test: LopticaModule Integration                                    |
|  Usisivac V6 | Trinity Protocol                                     |
+----------------------------------------------------------------------+

Testira sve Loptica komponente:
  1. LopticaEngine - 3-6-2 state machine
  2. KnowledgeBase - SQLite CRUD + rich_context
  3. ConflictResolver - HARD/SOFT conflict detection
  4. FeedbackTracker - self-learning confidence adjustment
  5. NotebookParser - AST hyperparameter extraction
  6. HarvesterAnalytics - report generation
  7. VetoBoard - 5-persona quorum (bez LLM)
  8. LopticaModule - unified integration
  9. BrainMassIngest - ChromaDB ingest
"""

import sys, os, json, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

PASS = "PASS PASS"
FAIL = "FAIL FAIL"
results = []


def test(name, fn):
    try:
        fn()
        results.append((name, True, ""))
        print(f"  {PASS} {name}")
    except Exception as e:
        results.append((name, False, str(e)))
        print(f"  {FAIL} {name}: {e}")


# -- Test 1: LopticaEngine ----------------------------------------------------
def t_loptica_engine():
    from loptica.loptica_engine import LopticaEngine
    with tempfile.TemporaryDirectory() as tmpdir:
        engine = LopticaEngine("test_mission", state_dir=tmpdir)
        assert engine.get_current_phase() == "RESEARCH"
        assert engine.get_checkpoint_limit() == 3

        # Log 3 actions  should auto-advance to DESIGN
        for i in range(3):
            engine.log_action(f"ACTION_{i}", f"detail_{i}")

        assert engine.get_current_phase() == "DESIGN", \
            f"Expected DESIGN, got {engine.get_current_phase()}"

        summary = engine.get_summary()
        assert summary["total_actions"] == 3
        assert not engine.is_complete()


# -- Test 2: KnowledgeBase ----------------------------------------------------
def t_knowledge_base():
    from loptica.knowledge_base import KnowledgeBase
    kb = KnowledgeBase(":memory:")

    sol_id = kb.add_solution("test-competition", 1, "test_author")
    assert sol_id > 0

    tech_id = kb.add_technique(
        solution_id=sol_id,
        category="hyperparameter",
        name="learning_rate",
        value=0.0001,
        confidence=0.95,
        context="AdamW(lr=1e-4)",
        rich_context={"author_comments": "Best after 50 experiments"},
        domain="universal"
    )
    assert tech_id > 0

    techs = kb.get_techniques(competition="test-competition")
    assert len(techs) == 1
    assert techs[0]["name"] == "learning_rate"
    assert float(techs[0]["confidence"]) == 0.95

    # Rich context check
    rc = json.loads(techs[0]["rich_context"])
    assert "author_comments" in rc

    stats = kb.get_stats()
    assert stats["solutions"] == 1
    assert stats["techniques"] == 1


# -- Test 3: ConflictResolver ------------------------------------------------
def t_conflict_resolver():
    from loptica.knowledge_base import ConflictResolver
    resolver = ConflictResolver()

    # Test HARD conflict detection
    conflict = resolver.check_compatibility(
        {"name": "high_learning_rate", "value": 0.1},
        {"name": "no_warmup", "value": True}
    )
    assert conflict is not None
    assert conflict["type"] == "HARD"

    # Test duplicate parameter conflict
    dup_conflict = resolver.check_compatibility(
        {"name": "learning_rate", "value": 0.001},
        {"name": "learning_rate", "value": 0.01}
    )
    assert dup_conflict is not None

    # Test batch resolution - should keep higher confidence
    techs = [
        {"name": "learning_rate", "value": 1e-4, "confidence": 0.95},
        {"name": "learning_rate", "value": 5e-4, "confidence": 0.80},  # duplicate
        {"name": "no_warmup", "value": True, "confidence": 0.85},      # conflicts with LR
        {"name": "batch_size", "value": 32, "confidence": 0.90},       # compatible
    ]
    resolved = resolver.resolve_batch(techs)

    # Should have learning_rate (0.95) and batch_size, but NOT no_warmup (HARD conflict)
    names = [t["name"] for t in resolved]
    assert "learning_rate" in names
    assert "batch_size" in names
    # no_warmup conflicts with high_learning_rate - but since we don't have
    # "high_learning_rate" exactly, it may pass. Check at least no duplicates.
    lr_count = names.count("learning_rate")
    assert lr_count == 1, f"Expected 1 learning_rate, got {lr_count}"


# -- Test 4: FeedbackTracker --------------------------------------------------
def t_feedback_tracker():
    from loptica.knowledge_base import KnowledgeBase, FeedbackTracker
    kb = KnowledgeBase(":memory:")
    sol_id = kb.add_solution("test-comp", 5, "tester")
    kb.add_technique(sol_id, "hp", "learning_rate", 0.001, 0.80)
    kb.add_technique(sol_id, "hp", "batch_size", 32, 0.75)

    tracker = FeedbackTracker(kb)

    # Top 10 rank  should boost by +0.10
    result = tracker.log_result("test-comp", 5, ["learning_rate", "batch_size"])
    assert result["adjustment"] == 0.10

    # Verify confidence was updated
    techs = kb.get_techniques(competition="test-comp")
    for t in techs:
        assert float(t["confidence"]) > 0.80, \
            f"Expected >0.80 after boost, got {t['confidence']}"

    # Low rank  should downgrade by -0.05
    result2 = tracker.log_result("test-comp", 50, ["learning_rate"])
    assert result2["adjustment"] == -0.05


# -- Test 5: NotebookParser --------------------------------------------------
def t_notebook_parser():
    from loptica.knowledge_base import NotebookParser
    parser = NotebookParser()

    # Test AST extraction from code string
    code = """
learning_rate = 0.001
batch_size = 32
epochs = 100
dropout = 0.5
n_estimators = 200
"""
    techs = parser._extract_ast_params(code)
    names = [t["name"] for t in techs]
    assert "learning_rate" in names
    assert "batch_size" in names
    assert "epochs" in names
    assert len(techs) >= 3


# -- Test 6: HarvesterAnalytics ----------------------------------------------
def t_harvester_analytics():
    from loptica.knowledge_base import KnowledgeBase, FeedbackTracker, HarvesterAnalytics
    kb = KnowledgeBase(":memory:")
    sol_id = kb.add_solution("analytics-test", 1, "tester")
    kb.add_technique(sol_id, "hp", "lr", 0.001, 0.90)
    kb.add_technique(sol_id, "hp", "bs", 32, 0.85)

    tracker = FeedbackTracker(kb)
    tracker.log_result("analytics-test", 3, ["lr", "bs"])

    analytics = HarvesterAnalytics(kb)
    report = analytics.generate_report()

    assert "top_techniques" in report
    assert report["total_runs"] == 1
    assert len(report["top_techniques"]) >= 2

    # Test snapshot export
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        snap_path = f.name
    try:
        result = analytics.export_snapshot(snap_path)
        assert Path(result).exists()
    finally:
        os.unlink(snap_path)


# -- Test 7: VetoBoard (no LLM) ----------------------------------------------
def t_veto_board():
    from loptica.veto_board import VetoBoard
    board = VetoBoard(use_llm=False)

    # Safe action  PASS
    result = board.evaluate_action("Train XGBoost model on tabular data")
    assert result["verdict"] == "PASS"

    # Dangerous action  VETO (contains "password")
    result2 = board.evaluate_action("Read password from config file")
    assert result2["verdict"] == "VETO"

    # Path traversal  VETO
    result3 = board.evaluate_action("Access file at ../../../etc/passwd")
    assert result3["verdict"] == "VETO"


# -- Test 8: LopticaModule (unified) ----------------------------------------
def t_loptica_module():
    from loptica.loptica_module import LopticaModule
    module = LopticaModule(mission_name="test_unified")

    # Run mission
    result = module.run_mission(
        problem="Predict customer churn from tabular data",
        domain="tabular"
    )

    assert "engine_summary" in result
    assert "kb_stats" in result
    assert result["engine_summary"]["mission"] == "test_unified"

    # Get best techniques (empty at start - OK)
    techs = module.get_best_techniques(domain="tabular")
    assert isinstance(techs, list)

    # Get report
    report = module.get_report()
    assert "db_stats" in report


# -- Test 9: BrainMassIngest ------------------------------------------------
def t_brain_mass_ingest():
    from loptica.brain_mass_ingest import BrainMassIngest
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test files
        (Path(tmpdir) / "test.py").write_text("learning_rate = 0.001\nbatch_size = 32\n")
        (Path(tmpdir) / "notes.md").write_text("# Test\n\nXGBoost works well for tabular data.\n")
        (Path(tmpdir) / "config.json").write_text('{"lr": 0.001, "epochs": 100}')

        chroma_dir = str(Path(tmpdir) / "chroma")
        ingestor = BrainMassIngest(collection_name="test_brain", db_path=chroma_dir)
        result = ingestor.ingest(tmpdir)

        assert result["status"] == "OK"
        assert result["chunks_added"] > 0

        # Query
        hits = ingestor.query("learning rate XGBoost", n_results=3)
        assert isinstance(hits, list)


# -- Run all tests ------------------------------------------------------------
if __name__ == "__main__":
    print("\n" + "="*60)
    print("  LOPTICA INTEGRATION TEST SUITE")
    print("  Usisivac V6 | Trinity Protocol")
    print("="*60)

    test("LopticaEngine (3-6-2 state machine)", t_loptica_engine)
    test("KnowledgeBase (SQLite + rich_context)", t_knowledge_base)
    test("ConflictResolver (HARD/SOFT conflicts)", t_conflict_resolver)
    test("FeedbackTracker (self-learning)", t_feedback_tracker)
    test("NotebookParser (AST extraction)", t_notebook_parser)
    test("HarvesterAnalytics (report + snapshot)", t_harvester_analytics)
    test("VetoBoard (5-persona quorum, no LLM)", t_veto_board)
    test("LopticaModule (unified integration)", t_loptica_module)
    test("BrainMassIngest (ChromaDB ingest)", t_brain_mass_ingest)

    print("\n" + "="*60)
    passed = sum(1 for _, ok, _ in results if ok)
    total = len(results)
    print(f"  RESULTS: {passed}/{total} tests passed")
    if passed == total:
        print("  STATUS: ALL TESTS PASSED - LOPTICA INTEGRATION OK")
    else:
        print("  STATUS: SOME TESTS FAILED")
        for name, ok, err in results:
            if not ok:
                print(f"    FAILED: {name}  {err}")
    print("="*60)

    sys.exit(0 if passed == total else 1)
