"""
╔══════════════════════════════════════════════════════════════════════╗
║  LopticaKnowledgeBase — SQLite + Rich Context + Self-Learning       ║
║  Usisivac V6 | Trinity Protocol                                     ║
║  Integrisano iz: Trinity_AIMO_Loptica_Final / Copy_of_Loptica.ipynb ║
╚══════════════════════════════════════════════════════════════════════╝

Komponente:
  KnowledgeBase     → SQLite baza tehnika sa rich_context JSON
  ConflictResolver  → Winner-takes-all po confidence score-u
  FeedbackTracker   → Self-learning: boost/downgrade confidence
  NotebookParser    → AST ekstrakcija hiperparametara iz .ipynb
  HarvesterAnalytics→ Agregacija i izveštaji
"""

import sqlite3, json, ast, logging
from datetime import datetime
from pathlib import Path
from difflib import SequenceMatcher

logger = logging.getLogger(__name__)


# ─── KnowledgeBase ───────────────────────────────────────────────────────────

class KnowledgeBase:
    """
    SQLite baza za tehnike sa punim semantičkim kontekstom.
    Podržava: solutions, techniques, board_reviews, competition_results.
    """

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path("db/loptica_kb.db"))
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self):
        c = self.conn.cursor()
        c.executescript("""
            CREATE TABLE IF NOT EXISTS solutions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition TEXT NOT NULL,
                rank INTEGER,
                author TEXT,
                code_hash TEXT UNIQUE,
                parsed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE TABLE IF NOT EXISTS techniques (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                solution_id INTEGER,
                category TEXT NOT NULL,
                name TEXT NOT NULL,
                value TEXT,
                context TEXT,
                rich_context JSON,
                confidence REAL DEFAULT 0.5,
                domain TEXT DEFAULT 'universal',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(solution_id) REFERENCES solutions(id)
            );

            CREATE TABLE IF NOT EXISTS board_reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                technique_id INTEGER,
                entity TEXT NOT NULL,
                verdict TEXT NOT NULL,
                reasoning TEXT,
                score REAL,
                FOREIGN KEY(technique_id) REFERENCES techniques(id)
            );

            CREATE TABLE IF NOT EXISTS competition_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                competition TEXT,
                final_rank INTEGER,
                techniques_used TEXT,
                submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );

            CREATE INDEX IF NOT EXISTS idx_tech_name ON techniques(name);
            CREATE INDEX IF NOT EXISTS idx_tech_conf ON techniques(confidence DESC);
            CREATE INDEX IF NOT EXISTS idx_tech_domain ON techniques(domain);
        """)
        self.conn.commit()

    def add_solution(self, competition: str, rank: int, author: str) -> int:
        c = self.conn.cursor()
        c.execute("INSERT OR IGNORE INTO solutions (competition, rank, author) VALUES (?, ?, ?)",
                  (competition, rank, author))
        self.conn.commit()
        return c.lastrowid

    def add_technique(self, solution_id: int, category: str, name: str,
                      value, confidence: float, context: str = "",
                      rich_context: dict = None, domain: str = "universal") -> int:
        c = self.conn.cursor()
        rich_json = json.dumps(rich_context) if rich_context else None
        c.execute("""INSERT INTO techniques
                     (solution_id, category, name, value, confidence, context, rich_context, domain)
                     VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                  (solution_id, category, name, str(value), confidence, context, rich_json, domain))
        self.conn.commit()
        return c.lastrowid

    def get_techniques(self, competition: str = None, domain: str = None,
                       min_confidence: float = 0.0) -> list:
        c = self.conn.cursor()
        if competition:
            c.execute("""SELECT t.* FROM techniques t
                         JOIN solutions s ON t.solution_id = s.id
                         WHERE s.competition = ? AND t.confidence >= ?
                         ORDER BY t.confidence DESC""", (competition, min_confidence))
        elif domain:
            c.execute("SELECT * FROM techniques WHERE domain = ? AND confidence >= ? ORDER BY confidence DESC",
                      (domain, min_confidence))
        else:
            c.execute("SELECT * FROM techniques WHERE confidence >= ? ORDER BY confidence DESC",
                      (min_confidence,))
        return [dict(row) for row in c.fetchall()]

    def get_stats(self) -> dict:
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM solutions")
        n_sol = c.fetchone()[0]
        c.execute("SELECT COUNT(*) FROM techniques")
        n_tech = c.fetchone()[0]
        c.execute("SELECT AVG(confidence) FROM techniques")
        avg_conf = c.fetchone()[0] or 0.0
        return {"solutions": n_sol, "techniques": n_tech, "avg_confidence": round(avg_conf, 3)}


# ─── ConflictResolver ────────────────────────────────────────────────────────

class ConflictResolver:
    """
    Rešava sukobe između nekompatibilnih tehnika.
    Winner-takes-all po confidence score-u za HARD konflikte.
    """

    KNOWN_CONFLICTS = {
        ("high_learning_rate", "no_warmup"): {
            "reason": "High LR requires warmup to stabilize early training",
            "type": "HARD"
        },
        ("dropout_0.5", "batch_norm"): {
            "reason": "Batch Norm and high Dropout interfere during inference",
            "type": "HARD"
        },
        ("heavy_augmentation", "small_dataset"): {
            "reason": "May destroy valid signal in limited data",
            "type": "SOFT"
        },
        ("oversampling", "test_set_augmentation"): {
            "reason": "Data leakage risk if applied before split",
            "type": "HARD"
        },
    }

    def check_compatibility(self, tech_a: dict, tech_b: dict) -> dict | None:
        pair = (tech_a["name"], tech_b["name"])
        if pair in self.KNOWN_CONFLICTS:
            return self.KNOWN_CONFLICTS[pair]
        if pair[::-1] in self.KNOWN_CONFLICTS:
            return self.KNOWN_CONFLICTS[pair[::-1]]
        # Duplikat istog parametra
        if tech_a["name"] == tech_b["name"] and tech_a.get("value") != tech_b.get("value"):
            return {"reason": f"Duplicate parameter: {tech_a['name']}", "type": "HARD"}
        return None

    def resolve_batch(self, techniques: list) -> list:
        """Filtrira listu tehnika, uklanja HARD konflikte. Zadržava veći confidence."""
        resolved = []
        sorted_techs = sorted(techniques, key=lambda x: x.get("confidence", 0), reverse=True)

        for tech in sorted_techs:
            compatible = True
            for accepted in resolved:
                conflict = self.check_compatibility(tech, accepted)
                if conflict and conflict["type"] == "HARD":
                    logger.info(f"CONFLICT: {tech['name']} vs {accepted['name']} → {conflict['reason']}")
                    compatible = False
                    break
            if compatible:
                resolved.append(tech)

        return resolved


# ─── FeedbackTracker ─────────────────────────────────────────────────────────

class FeedbackTracker:
    """
    Self-learning: prilagođava confidence tehnika na osnovu stvarnih rezultata.
    Top 10% → +0.10 | Ostalo → -0.05
    """

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def log_result(self, competition: str, rank: int, techniques_used: list):
        c = self.kb.conn.cursor()
        c.execute("INSERT INTO competition_results (competition, final_rank, techniques_used) VALUES (?, ?, ?)",
                  (competition, rank, json.dumps(techniques_used)))

        adjustment = 0.10 if rank <= 10 else -0.05
        for tech_name in techniques_used:
            c.execute("""UPDATE techniques
                         SET confidence = MIN(1.0, MAX(0.1, confidence + ?))
                         WHERE name = ?""", (adjustment, tech_name))

        self.kb.conn.commit()
        logger.info(f"Feedback: rank={rank}, adj={adjustment:+.2f}, techs={techniques_used}")
        return {"competition": competition, "rank": rank, "adjustment": adjustment}


# ─── NotebookParser ──────────────────────────────────────────────────────────

class NotebookParser:
    """
    Ekstrahuje hiperparametre i tehnike iz .ipynb fajlova koristeći AST.
    """

    KNOWN_PARAMS = {
        "lr", "learning_rate", "batch_size", "epochs", "optimizer",
        "backbone", "dropout", "weight_decay", "num_layers", "hidden_size",
        "n_estimators", "max_depth", "min_samples_split", "subsample",
        "colsample_bytree", "reg_alpha", "reg_lambda", "gamma",
    }

    def extract_from_notebook(self, path: str) -> dict:
        import nbformat
        try:
            with open(path, encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)
        except Exception as e:
            return {"error": str(e), "hyperparameters": []}

        all_techs = []
        for cell in nb.cells:
            if cell.cell_type == "code":
                all_techs.extend(self._extract_ast_params(cell.source))

        return {"hyperparameters": all_techs, "total": len(all_techs)}

    def _extract_ast_params(self, code: str) -> list:
        techs = []
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name) and target.id.lower() in self.KNOWN_PARAMS:
                            value = self._get_value(node.value)
                            if value is not None:
                                techs.append({
                                    "name": target.id,
                                    "value": value,
                                    "confidence": 0.8
                                })
        except SyntaxError:
            pass
        return techs

    def _get_value(self, node):
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.operand, ast.Constant):
            return -node.operand.value
        return None


# ─── HarvesterAnalytics ──────────────────────────────────────────────────────

class HarvesterAnalytics:
    """Agregira i prikazuje statistike iz KnowledgeBase."""

    def __init__(self, kb: KnowledgeBase):
        self.kb = kb

    def generate_report(self) -> dict:
        c = self.kb.conn.cursor()
        c.execute("""SELECT name, MAX(confidence) as max_conf,
                            COUNT(*) as freq, AVG(confidence) as avg_conf
                     FROM techniques GROUP BY name ORDER BY max_conf DESC""")
        rows = [dict(r) for r in c.fetchall()]

        c.execute("SELECT COUNT(*) FROM competition_results")
        total_runs = c.fetchone()[0]

        report = {
            "total_runs": total_runs,
            "top_techniques": rows[:20],
            "db_stats": self.kb.get_stats()
        }
        return report

    def export_snapshot(self, output_path: str = None) -> str:
        if output_path is None:
            output_path = "db/loptica_snapshot.db"
        import shutil
        with sqlite3.connect(output_path) as backup:
            self.kb.conn.backup(backup)
        return output_path
