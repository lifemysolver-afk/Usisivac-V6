"""
╔══════════════════════════════════════════════════════════════════════╗
║  VetoBoard — 5-Persona Quorum Validator                             ║
║  Usisivac V6 | Trinity Protocol                                     ║
║  Integrisano iz: Trinity_AIMO_Loptica_Final / boardroom.py          ║
╚══════════════════════════════════════════════════════════════════════╝

5 persona ocenjuju svaku akciju pre izvršavanja:
  CEO    → Poslovna vrednost
  CTO    → Tehnička izvodljivost
  CFO    → Troškovi resursa
  LEGAL  → Bezbednost (ima pravo veta)
  CRITIC → Rizik od overfitting/greške

Quorum: 3/5 glasova "PASS" = akcija prolazi.
LEGAL veto = trenutno odbijanje.
"""

import os
from core.llm_client import call as llm_call


class VetoBoard:
    """
    5-persona glasanje za svaku kritičnu akciju u pipeline-u.
    Koristi LLM za realne ocene umesto hardkodovane logike.
    """

    PERSONAS = {
        "CEO": "You evaluate the BUSINESS VALUE of this action. Is it worth the time and resources?",
        "CTO": "You evaluate the TECHNICAL FEASIBILITY. Is this implementable without breaking the system?",
        "CFO": "You evaluate the RESOURCE COST. Is the API/compute cost justified?",
        "LEGAL": "You are the SAFETY GUARDIAN. Detect credential leaks, path traversal, data privacy violations. VETO immediately if found.",
        "CRITIC": "You are the ML CRITIC. Detect overfitting risks, data leakage, invalid assumptions.",
    }

    RISK_KEYWORDS = ["password", "secret", "token", "/etc/", "../", "rm -rf", "drop table", "delete from"]
    QUORUM = 3

    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm

    def evaluate_action(self, action_description: str, context: str = "") -> dict:
        """
        Evaluira akciju kroz 5 persona.
        Vraća: {"verdict": "PASS"|"VETO", "reason": str, "votes": dict}
        """
        # 1. LEGAL instant veto check (bez LLM — deterministički)
        if self._legal_veto(action_description):
            return {
                "verdict": "VETO",
                "reason": "LEGAL VETO: Safety violation detected in action description.",
                "votes": {"LEGAL": "VETO"},
                "quorum_reached": False
            }

        # 2. Glasanje svih persona
        votes = {}
        reasonings = {}

        for persona, persona_prompt in self.PERSONAS.items():
            vote, reasoning = self._get_vote(persona, persona_prompt, action_description, context)
            votes[persona] = vote
            reasonings[persona] = reasoning

            # LEGAL veto iz LLM odgovora
            if persona == "LEGAL" and vote == "VETO":
                return {
                    "verdict": "VETO",
                    "reason": f"LEGAL VETO: {reasoning}",
                    "votes": votes,
                    "quorum_reached": False
                }

        # 3. Quorum check
        pass_count = sum(1 for v in votes.values() if v == "PASS")
        quorum_reached = pass_count >= self.QUORUM

        return {
            "verdict": "PASS" if quorum_reached else "VETO",
            "reason": f"Quorum: {pass_count}/{len(self.PERSONAS)} PASS votes.",
            "votes": votes,
            "reasonings": reasonings,
            "quorum_reached": quorum_reached
        }

    def _legal_veto(self, action: str) -> bool:
        """Deterministička provera bezbednosnih rizika."""
        action_lower = action.lower()
        return any(kw in action_lower for kw in self.RISK_KEYWORDS)

    def _get_vote(self, persona: str, persona_prompt: str,
                  action: str, context: str) -> tuple:
        """Dobija glas od jedne persone. Vraća (vote, reasoning)."""
        if not self.use_llm:
            # Fallback: uvek PASS osim za CRITIC koji bude WARN
            return ("WARN" if persona == "CRITIC" else "PASS"), "Fallback vote (LLM disabled)"

        prompt = f"""{persona_prompt}

ACTION TO EVALUATE: {action}
CONTEXT: {context}

Respond with EXACTLY this format:
VOTE: PASS or VETO
REASON: One sentence explanation.

No other text."""

        try:
            response = llm_call(prompt, provider="groq")
            lines = response.strip().split("\n")
            vote = "PASS"
            reason = "No reason provided"
            for line in lines:
                if line.startswith("VOTE:"):
                    v = line.replace("VOTE:", "").strip().upper()
                    vote = "VETO" if "VETO" in v else "PASS"
                elif line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
            return vote, reason
        except Exception as e:
            return "PASS", f"LLM error (defaulting PASS): {e}"
