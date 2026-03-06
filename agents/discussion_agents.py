"""
╔══════════════════════════════════════════════════════════════════════╗
║  Discussion Agents — Neural Debate Engine                            ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Tri agenta debatuju o relevantnosti znanja pre ingesta u RAG:
  - Proponent (Groq/Llama)  → Brani relevantnost
  - Opponent  (Mistral)     → Napada relevantnost
  - Moderator (Groq/Llama)  → Presuda + JSON verdict
"""

import json, re, traceback
from core.llm_client import call as llm_call


class Proponent:
    """Brani relevantnost znanja. Koristi Groq (Llama 3.3 70B)."""

    def argue(self, topic: str, context: str) -> str:
        prompt = f"""You are the PROPONENT in a knowledge relevance debate.

TOPIC: {topic}

CONTEXT: {context}

YOUR TASK:
Argue WHY this knowledge is RELEVANT and CRITICAL for the current data science objective.
- Provide precise numbers where possible (e.g., expected AUC improvement, training time reduction)
- Reference specific techniques, papers, or Kaggle competition results
- Explain the technical mechanism of why this helps
- NO SIMULATION: If you lack specific data, state "I lack data on X" instead of fabricating

Keep your argument under 300 words. Be precise and technical."""

        try:
            return llm_call(prompt, provider="groq")
        except Exception as e:
            return f"[Proponent ERROR: {e}]"


class Opponent:
    """Napada relevantnost znanja. Koristi Mistral."""

    def argue(self, topic: str, proponent_argument: str) -> str:
        prompt = f"""You are the OPPONENT in a knowledge relevance debate.

TOPIC: {topic}

PROPONENT'S ARGUMENT:
{proponent_argument}

YOUR TASK:
Argue WHY this knowledge might be IRRELEVANT, REDUNDANT, or MISLEADING.
- Identify potential pitfalls: overfitting, data leakage, implementation complexity
- Point out if the proponent made unsupported claims
- Suggest what BETTER alternatives exist
- NO SIMULATION: Be critical but honest. If the proponent is right, say so.

Keep your argument under 300 words. Be precise and critical."""

        try:
            return llm_call(prompt, provider="mistral")
        except Exception as e:
            return f"[Opponent ERROR: {e}]"


class Moderator:
    """Presuda. Koristi Groq (Llama 3.3 70B) za objektivnost."""

    def decide(self, topic: str, proponent_arg: str, opponent_arg: str) -> dict:
        prompt = f"""You are the MODERATOR (JudgeGuard v2.2) in a knowledge relevance debate.

TOPIC: {topic}

PROPONENT'S ARGUMENT:
{proponent_arg}

OPPONENT'S ARGUMENT:
{opponent_arg}

YOUR TASK:
Evaluate both arguments objectively. Decide if this knowledge should be ingested into the RAG knowledge base.

You MUST respond with ONLY a valid JSON object, nothing else:
{{"decision": "INGEST", "relevance_score": 0.85, "confidence": 0.9, "reasoning": "The proponent's technical justification is sound because...", "key_metrics": {{"impact_estimate": "+0.005 AUC", "risk_level": "Low"}}}}

Rules:
- "decision" must be exactly "INGEST" or "REJECT"
- "relevance_score" must be 0.0 to 1.0
- "confidence" must be 0.0 to 1.0
- "reasoning" must be a clear explanation that other AI agents can understand
- "key_metrics" must include "impact_estimate" and "risk_level" (Low/Med/High)

Respond with ONLY the JSON object. No markdown, no explanation."""

        try:
            raw = llm_call(prompt, provider="groq")
            return self._parse_verdict(raw)
        except Exception as e:
            return self._default_verdict(f"Moderator call failed: {e}")

    def _parse_verdict(self, raw: str) -> dict:
        """Robusno parsiranje JSON-a iz LLM odgovora."""
        # Pokušaj 1: direktan JSON parse
        try:
            return json.loads(raw.strip())
        except json.JSONDecodeError:
            pass

        # Pokušaj 2: izvuci JSON iz markdown code block-a
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                pass

        # Pokušaj 3: nađi prvi { ... } blok
        match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Pokušaj 4: proveri da li tekst sadrži ključne reči
        raw_lower = raw.lower()
        if "reject" in raw_lower:
            return self._default_verdict("Parsed from text: REJECT signal detected", decision="REJECT")
        
        return self._default_verdict(f"Could not parse JSON from: {raw[:200]}")

    def _default_verdict(self, reason: str, decision: str = "INGEST") -> dict:
        """Podrazumevani verdict kada parsing ne uspe."""
        return {
            "decision": decision,
            "relevance_score": 0.5,
            "confidence": 0.3,
            "reasoning": reason,
            "key_metrics": {
                "impact_estimate": "unknown",
                "risk_level": "Med"
            }
        }
