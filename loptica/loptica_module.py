"""
╔══════════════════════════════════════════════════════════════════════╗
║  LopticaModule — Unified Integration Layer                          ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Spaja sve Loptica komponente u jedan pozivni interfejs koji koristi
Usisivac V6 Orchestrator:

  loptica.run_mission(problem, domain) → dict sa svim rezultatima

Interna arhitektura:
  LopticaEngine     → 3-6-2 state machine
  VetoBoard         → 5-persona quorum (CEO/CTO/CFO/LEGAL/CRITIC)
  KnowledgeBase     → SQLite tehnika sa rich_context
  ConflictResolver  → Winner-takes-all za HARD konflikte
  FeedbackTracker   → Self-learning confidence adjustment
  BrainMassIngest   → ChromaDB masovni ingest
  NotebookParser    → AST ekstrakcija iz .ipynb
  HarvesterAnalytics→ Izveštaji i snapshoti
"""

import json, logging
from datetime import datetime
from pathlib import Path

from loptica.loptica_engine import LopticaEngine
from loptica.veto_board import VetoBoard
from loptica.knowledge_base import (
    KnowledgeBase, ConflictResolver, FeedbackTracker,
    NotebookParser, HarvesterAnalytics
)
from loptica.brain_mass_ingest import BrainMassIngest
from core.anti_simulation import log_work

logger = logging.getLogger(__name__)
AGENT = "LopticaModule"


class LopticaModule:
    """
    Centralna klasa za Loptica integraciju u Usisivac V6.
    Koristi se iz Orchestrator-a kao pre-pipeline korak.
    """

    def __init__(self, mission_name: str = "usisivac_v6"):
        self.engine = LopticaEngine(mission_name)
        self.veto = VetoBoard(use_llm=True)
        self.kb = KnowledgeBase()
        self.resolver = ConflictResolver()
        self.tracker = FeedbackTracker(self.kb)
        self.parser = NotebookParser()
        self.analytics = HarvesterAnalytics(self.kb)
        self._brain_ingest_done = False

    # ── Public API ────────────────────────────────────────────────────────────

    def run_mission(self, problem: str, domain: str = "universal",
                    notebook_paths: list = None) -> dict:
        """
        Pokreće kompletan Loptica mission za dati problem.
        Vraća dict sa svim rezultatima svih faza.
        """
        log_work(AGENT, "MISSION_START",
                 f"phase={self.engine.get_current_phase()} problem='{problem[:60]}'")

        results = {
            "mission": self.engine.mission_name,
            "problem": problem,
            "domain": domain,
            "timestamp": datetime.now().isoformat(),
        }

        # ── Faza 1: RESEARCH ─────────────────────────────────────────────────
        if self.engine.get_current_phase() == "RESEARCH":
            results["research"] = self._phase_research(problem, domain, notebook_paths)

        # ── Faza 2: DESIGN ───────────────────────────────────────────────────
        if self.engine.get_current_phase() == "DESIGN":
            results["design"] = self._phase_design(problem, domain)

        # ── Faza 3: IMPLEMENTATION ───────────────────────────────────────────
        if self.engine.get_current_phase() == "IMPLEMENTATION":
            results["implementation"] = self._phase_implementation(problem, domain)

        # ── Faza 4: VALIDATION ───────────────────────────────────────────────
        if self.engine.get_current_phase() == "VALIDATION":
            results["validation"] = self._phase_validation()

        results["engine_summary"] = self.engine.get_summary()
        results["kb_stats"] = self.kb.get_stats()

        log_work(AGENT, "MISSION_STEP",
                 f"phase={self.engine.get_current_phase()} kb={self.kb.get_stats()}")
        return results

    def ingest_notebook(self, notebook_path: str, competition: str,
                        rank: int = 999, author: str = "unknown") -> dict:
        """
        Parsira .ipynb notebook i ubacuje tehnike u KnowledgeBase.
        Prolazi kroz VetoBoard pre ingesta.
        """
        # VetoBoard check
        veto_result = self.veto.evaluate_action(
            f"Ingest notebook: {notebook_path}",
            context=f"competition={competition}"
        )
        if veto_result["verdict"] == "VETO":
            log_work(AGENT, "VETO_BLOCK", f"notebook={notebook_path} reason={veto_result['reason']}")
            return {"status": "VETOED", "reason": veto_result["reason"]}

        # Parse
        parsed = self.parser.extract_from_notebook(notebook_path)
        techs = parsed.get("hyperparameters", [])

        if not techs:
            return {"status": "EMPTY", "notebook": notebook_path}

        # Resolve conflicts
        resolved = self.resolver.resolve_batch(techs)

        # Store in KB
        sol_id = self.kb.add_solution(competition, rank, author)
        stored = []
        for tech in resolved:
            tid = self.kb.add_technique(
                solution_id=sol_id,
                category="hyperparameter",
                name=tech["name"],
                value=tech["value"],
                confidence=tech.get("confidence", 0.8),
                domain="universal"
            )
            stored.append({"id": tid, "name": tech["name"], "value": tech["value"]})

        entry = self.engine.log_action("NOTEBOOK_INGESTED", {
            "notebook": notebook_path,
            "techniques_stored": len(stored)
        })

        log_work(AGENT, "NOTEBOOK_INGEST_OK",
                 f"notebook={Path(notebook_path).name} stored={len(stored)}")
        return {"status": "OK", "stored": stored, "engine_entry": entry}

    def brain_ingest(self, root_dir: str = None) -> dict:
        """Masovni ChromaDB ingest iz direktorijuma."""
        root = root_dir or "/home/ubuntu/Usisivac-V6"
        ingestor = BrainMassIngest()
        result = ingestor.ingest(root)
        self._brain_ingest_done = True
        log_work(AGENT, "BRAIN_INGEST", json.dumps(result))
        return result

    def log_competition_result(self, competition: str, rank: int,
                               techniques_used: list) -> dict:
        """Loguje rezultat takmičenja i prilagođava confidence."""
        result = self.tracker.log_result(competition, rank, techniques_used)
        log_work(AGENT, "FEEDBACK_LOGGED", json.dumps(result))
        return result

    def get_best_techniques(self, domain: str = None,
                            min_confidence: float = 0.7) -> list:
        """Vraća tehnike sa visokim confidence-om iz KB."""
        return self.kb.get_techniques(domain=domain, min_confidence=min_confidence)

    def get_report(self) -> dict:
        """Generiše analitički izveštaj iz KB."""
        return self.analytics.generate_report()

    # ── Private Phase Methods ─────────────────────────────────────────────────

    def _phase_research(self, problem: str, domain: str,
                        notebook_paths: list = None) -> dict:
        """RESEARCH faza: ingest notebooka i pretraga KB."""
        results = {"phase": "RESEARCH", "ingested": [], "kb_query": []}

        # Ingest notebooka ako su prosleđeni
        if notebook_paths:
            for nb_path in notebook_paths:
                r = self.ingest_notebook(nb_path, competition=domain)
                results["ingested"].append(r)

        # Pretraga KB za relevantne tehnike
        techs = self.kb.get_techniques(domain=domain, min_confidence=0.6)
        results["kb_query"] = techs[:10]
        results["kb_count"] = len(techs)

        entry = self.engine.log_action("RESEARCH_COMPLETE", {
            "problem": problem[:60],
            "techniques_found": len(techs)
        })
        results["engine_entry"] = entry
        return results

    def _phase_design(self, problem: str, domain: str) -> dict:
        """DESIGN faza: conflict resolution i strategija."""
        techs = self.kb.get_techniques(domain=domain, min_confidence=0.5)
        resolved = self.resolver.resolve_batch(techs)

        entry = self.engine.log_action("DESIGN_COMPLETE", {
            "total_techs": len(techs),
            "after_resolution": len(resolved)
        })
        return {
            "phase": "DESIGN",
            "resolved_techniques": resolved[:10],
            "conflicts_removed": len(techs) - len(resolved),
            "engine_entry": entry
        }

    def _phase_implementation(self, problem: str, domain: str) -> dict:
        """IMPLEMENTATION faza: generisanje konfiguracije."""
        techs = self.kb.get_techniques(domain=domain, min_confidence=0.7)
        resolved = self.resolver.resolve_batch(techs)

        config = {
            "problem": problem,
            "domain": domain,
            "recommended_techniques": [
                {"name": t["name"], "value": t["value"], "confidence": t["confidence"]}
                for t in resolved[:15]
            ]
        }

        entry = self.engine.log_action("IMPLEMENTATION_COMPLETE", {
            "config_techniques": len(config["recommended_techniques"])
        })
        return {"phase": "IMPLEMENTATION", "config": config, "engine_entry": entry}

    def _phase_validation(self) -> dict:
        """VALIDATION faza: finalni izveštaj i snapshot."""
        report = self.analytics.generate_report()
        snapshot = self.analytics.export_snapshot()

        entry = self.engine.log_action("FINAL_SYNTHESIS", {
            "report_techniques": len(report.get("top_techniques", [])),
            "snapshot": snapshot
        })
        return {
            "phase": "VALIDATION",
            "report": report,
            "snapshot_path": snapshot,
            "engine_entry": entry
        }
