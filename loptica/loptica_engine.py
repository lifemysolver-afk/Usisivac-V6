"""
╔══════════════════════════════════════════════════════════════════════╗
║  LopticaEngine — 3-6-2 Phase State Machine                          ║
║  Usisivac V6 | Trinity Protocol                                     ║
║  Integrisano iz: Trinity_AIMO_Loptica_Final / loptica_engine.py     ║
╚══════════════════════════════════════════════════════════════════════╝

3-6-2 Checkpoint logika:
  Faza 1 (RESEARCH)        → 3 koraka pre prelaska
  Faza 2 (DESIGN)          → 6 koraka pre prelaska
  Faza 3 (IMPLEMENTATION)  → 2 koraka pre prelaska
  Faza 4 (VALIDATION)      → finalna sinteza

Svaki prelaz se loguje u work_log.md i state JSON fajlu.
"""

import json, os
from datetime import datetime
from pathlib import Path


class LopticaEngine:
    """
    Centralni state machine za Trinity Protocol misije.
    Prati faze, checkpoint-e i istoriju akcija.
    """

    PHASES = ["RESEARCH", "DESIGN", "IMPLEMENTATION", "VALIDATION"]
    CHECKPOINT_SEQUENCE = [3, 6, 2, 1]  # koraci po fazi

    def __init__(self, mission_name: str, state_dir: str = None):
        self.mission_name = mission_name
        self.state_dir = Path(state_dir or "logs/loptica_states")
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.state_dir / f"{mission_name}_state.json"

        self.current_phase_idx = 0
        self.step_count = 0
        self.history = []

        self._load_state()

    # ── State Persistence ────────────────────────────────────────────────────

    def _load_state(self):
        if self.state_file.exists():
            try:
                with open(self.state_file) as f:
                    s = json.load(f)
                self.current_phase_idx = s.get("current_phase_idx", 0)
                self.step_count = s.get("step_count", 0)
                self.history = s.get("history", [])
            except Exception:
                pass
        else:
            self._save_state()

    def _save_state(self):
        state = {
            "mission_name": self.mission_name,
            "current_phase_idx": self.current_phase_idx,
            "current_phase": self.get_current_phase(),
            "step_count": self.step_count,
            "checkpoint_limit": self.CHECKPOINT_SEQUENCE[min(self.current_phase_idx, len(self.CHECKPOINT_SEQUENCE)-1)],
            "history": self.history[-50:],  # čuvamo poslednjih 50 unosa
            "updated_at": datetime.now().isoformat()
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    # ── Phase Control ────────────────────────────────────────────────────────

    def get_current_phase(self) -> str:
        return self.PHASES[min(self.current_phase_idx, len(self.PHASES) - 1)]

    def get_checkpoint_limit(self) -> int:
        idx = min(self.current_phase_idx, len(self.CHECKPOINT_SEQUENCE) - 1)
        return self.CHECKPOINT_SEQUENCE[idx]

    def log_action(self, action: str, details) -> dict:
        """Loguje akciju i inkrementira step_count. Vraća entry dict."""
        self.step_count += 1
        limit = self.get_checkpoint_limit()

        entry = {
            "timestamp": datetime.now().isoformat(),
            "phase": self.get_current_phase(),
            "action": action,
            "details": details,
            "step": f"{self.step_count}/{limit}"
        }
        self.history.append(entry)

        # Auto-advance ako smo dostigli checkpoint limit
        if self.step_count >= limit:
            old_phase = self.get_current_phase()
            self.advance_phase()
            entry["phase_advanced"] = f"{old_phase} → {self.get_current_phase()}"

        self._save_state()
        return entry

    def advance_phase(self) -> bool:
        """Ručno prelazi na sledeću fazu. Vraća True ako je uspelo."""
        if self.current_phase_idx < len(self.PHASES) - 1:
            self.current_phase_idx += 1
            self.step_count = 0
            self._save_state()
            return True
        return False

    def is_complete(self) -> bool:
        return (self.current_phase_idx == len(self.PHASES) - 1 and
                any(e["action"] == "FINAL_SYNTHESIS" for e in self.history))

    def get_summary(self) -> dict:
        return {
            "mission": self.mission_name,
            "phase": self.get_current_phase(),
            "phase_idx": self.current_phase_idx,
            "step": f"{self.step_count}/{self.get_checkpoint_limit()}",
            "total_actions": len(self.history),
            "complete": self.is_complete()
        }
