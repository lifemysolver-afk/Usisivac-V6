"""
╔══════════════════════════════════════════════════════════════════════╗
║  Orchestrator — Non-Stop Agent Loop                                 ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Pipeline:
  0. DiscussionEngine → Pre-ingest debate (Proponent vs Opponent vs Moderator)
  1. ResearchAgent    → usisava znanje, izvlači Golden Recipe
  2. CriticAgent      → kritikuje plan i nalaze
  3. CoderAgent       → piše kod na osnovu research-a + kritike
  4. CriticAgent      → kritikuje kod (second pass)
  5. CleanerAgent     → generiše i izvršava cleaning
  6. FeatureAgent     → izvršava feature engineering
  7. Guardian         → audit drift_score + verifikacija
  8. → LOOP
"""

import sys, json, time, datetime, signal, os, traceback
from pathlib import Path

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, log_work
from core import state_manager as SM
from core.rag_engine import stats as rag_stats
from core.discussion_engine import DiscussionEngine
from agents.discussion_agents import Proponent, Opponent, Moderator

from agents.research_agent import run as research_run
from agents.critic_agent import run as critic_run
from agents.coder_agent import run as coder_run
from agents.cleaner_agent import run as cleaner_run
from agents.feature_agent import run as feature_run

AGENT = "Orchestrator"
RUNNING = True

# Inicijalizacija Discussion Engine-a
discussion_engine = DiscussionEngine()
proponent = Proponent()
opponent = Opponent()
moderator = Moderator()


def signal_handler(sig, frame):
    global RUNNING
    log_work(AGENT, "SHUTDOWN_SIGNAL", f"Signal {sig} received")
    RUNNING = False


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def run_discussion(problem: str) -> dict:
    """
    Pokreće Neural Discussion Engine.
    Proponent (Groq) brani, Opponent (Mistral) napada, Moderator (Groq) presuda.
    Sve se čuva u discussion_db (ChromaDB) i discussion_log.jsonl.
    """
    log_work(AGENT, "STEP_0", "Neural Discussion Engine → Pre-ingest debate")
    print(f"  [Discussion] Starting debate on: {problem[:60]}...")

    # 1. Proponent argues FOR
    print(f"  [Discussion] Proponent (Groq/Llama) arguing FOR...")
    prop_arg = proponent.argue(problem, "Initial problem statement")
    print(f"  [Discussion] Proponent done ({len(prop_arg)} chars)")

    # 2. Opponent argues AGAINST
    print(f"  [Discussion] Opponent (Mistral) arguing AGAINST...")
    opp_arg = opponent.argue(problem, prop_arg)
    print(f"  [Discussion] Opponent done ({len(opp_arg)} chars)")

    # 3. Moderator decides
    print(f"  [Discussion] Moderator (Groq/Llama) deciding...")
    verdict = moderator.decide(problem, prop_arg, opp_arg)
    print(f"  [Discussion] Verdict: {verdict.get('decision', 'N/A')} "
          f"(relevance: {verdict.get('relevance_score', 'N/A')}, "
          f"confidence: {verdict.get('confidence', 'N/A')})")

    # 4. Save to shared memory (ChromaDB + JSONL)
    transcript = (
        f"=== PROPONENT (Groq/Llama 3.3 70B) ===\n{prop_arg}\n\n"
        f"=== OPPONENT (Mistral Small) ===\n{opp_arg}\n\n"
        f"=== MODERATOR VERDICT ===\n{json.dumps(verdict, indent=2)}"
    )
    disc_id = discussion_engine.save_discussion(
        topic=problem,
        participants=["Proponent:Groq", "Opponent:Mistral", "Moderator:Groq"],
        transcript=transcript,
        verdict=verdict.get("decision", "INGEST")
    )
    log_work(AGENT, "DISCUSSION_SAVED", f"id={disc_id} verdict={verdict.get('decision')}")

    return {
        "proponent": prop_arg,
        "opponent": opp_arg,
        "verdict": verdict,
        "discussion_id": disc_id,
    }


def run_pipeline(problem: str, domain: str = "universal",
                 data_description: str = "", data_path: str = None) -> dict:
    """Jedan prolaz kroz kompletan pipeline."""
    log_work(AGENT, "PIPELINE_START", f"problem='{problem[:80]}' domain='{domain}'")
    results = {}

    # ── STEP 0: Neural Discussion ─────────────────────────────────────────────
    try:
        disc_result = run_discussion(problem)
        results["discussion"] = disc_result
        verdict = disc_result.get("verdict", {})

        if verdict.get("decision") == "REJECT":
            print(f"  [Pipeline] Topic REJECTED by discussion. Reason: {verdict.get('reasoning', 'N/A')[:100]}")
            log_work(AGENT, "DISCUSSION_REJECT", verdict.get("reasoning", "")[:200])
            return results
    except Exception as e:
        print(f"  [Discussion] Error (non-fatal): {e}")
        log_work(AGENT, "DISCUSSION_ERROR", str(e))
        results["discussion"] = {"verdict": {"decision": "INGEST", "reasoning": f"Discussion failed: {e}"}}

    # ── STEP 1: Research ──────────────────────────────────────────────────────
    log_work(AGENT, "STEP_1", "ResearchAgent → knowledge ingest + research")
    SM.set_status("RESEARCHER_INGESTING", "ResearchAgent")

    kb_stats = rag_stats()
    if kb_stats.get("knowledge_base", {}).get("count", 0) == 0:
        results["ingest"] = research_run({"action": "ingest"})

    results["research"] = research_run({
        "action": "research",
        "query": problem,
        "domain": domain,
    })

    results["recipe"] = research_run({
        "action": "golden_recipe",
        "problem": problem,
    })

    # ── STEP 2: Critic (plan review) ─────────────────────────────────────────
    log_work(AGENT, "STEP_2", "CriticAgent → plan critique")
    results["critique_plan"] = critic_run({
        "action": "critique_plan",
        "plan": {
            "problem": problem,
            "research": results.get("research", {}),
            "recipe": results.get("recipe", {}),
        }
    })

    # ── STEP 3: Coder ────────────────────────────────────────────────────────
    log_work(AGENT, "STEP_3", "CoderAgent → code generation")
    SM.set_status("EXECUTOR_RUNNING", "CoderAgent")

    results["code"] = coder_run({
        "action": "generate_code",
        "problem": problem,
        "research_output": results.get("research", {}),
        "critic_output": results.get("critique_plan", {}),
    })

    if data_description:
        results["features_code"] = coder_run({
            "action": "generate_features",
            "data_description": data_description,
            "research_output": results.get("research", {}),
        })

    # ── STEP 4: Critic (code review) ─────────────────────────────────────────
    log_work(AGENT, "STEP_4", "CriticAgent → code critique")
    code_preview = results.get("code", {}).get("code_preview", "")
    results["critique_code"] = critic_run({
        "action": "critique_code",
        "code": code_preview,
    })

    # ── STEP 5: Cleaner ──────────────────────────────────────────────────────
    if data_description:
        log_work(AGENT, "STEP_5", "CleanerAgent → data cleaning")
        results["cleaning"] = cleaner_run({
            "action": "generate",
            "data_description": data_description,
            "data_path": data_path,
        })

    # ── STEP 6: Feature Execution ─────────────────────────────────────────────
    feat_script = results.get("features_code", {}).get("file")
    if feat_script:
        log_work(AGENT, "STEP_6", "FeatureAgent → execute features")
        results["features_exec"] = feature_run({
            "action": "execute",
            "script_path": feat_script,
        })

    # ── STEP 7: Guardian Audit ────────────────────────────────────────────────
    log_work(AGENT, "STEP_7", "Guardian → drift audit")
    SM.set_status("GUARDIAN_AUDITING", "Guardian")

    from guardian.guardian import run as guardian_run
    results["audit"] = guardian_run({
        "action": "full_audit",
        "pipeline_results": results,
    })

    # ── Finalizacija ──────────────────────────────────────────────────────────
    drift = results.get("audit", {}).get("drift_score", 1.0)
    if drift > 0.4:
        SM.set_status("DRIFT_EXCEEDED")
        log_work(AGENT, "DRIFT_EXCEEDED", f"drift={drift:.3f} > 0.4")
    else:
        SM.set_status("PENDING_REVIEW")
        log_work(AGENT, "PIPELINE_DONE", f"drift={drift:.3f}")

    SM.inc_loop()
    return results


def run_non_stop(problem: str, domain: str = "universal",
                 data_description: str = "", data_path: str = None,
                 max_iterations: int = None, delay: int = 30):
    """Non-stop petlja."""
    global RUNNING

    max_iter = max_iterations or int(os.getenv("MAX_LOOP_ITERATIONS", "100"))
    iteration = 0

    SM.init(problem, f"Non-stop pipeline for: {problem}", domain)
    SM.set_status("LOOP_RUNNING", AGENT)
    log_work(AGENT, "NON_STOP_START",
             f"max_iter={max_iter}, delay={delay}s, domain={domain}")

    print(f"\n{'='*60}")
    print(f"  USISIVAC V6 — NON-STOP MODE + DISCUSSION ENGINE")
    print(f"  Problem: {problem[:50]}")
    print(f"  Domain:  {domain}")
    print(f"  Max iterations: {max_iter}")
    print(f"  Providers: Groq (Proponent+Moderator) + Mistral (Opponent)")
    print(f"  Press Ctrl+C to stop gracefully")
    print(f"{'='*60}\n")

    while RUNNING and iteration < max_iter:
        iteration += 1
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        print(f"\n[{ts}] ── Iteration {iteration}/{max_iter} ──")

        try:
            results = run_pipeline(problem, domain, data_description, data_path)

            drift = results.get("audit", {}).get("drift_score", "N/A")
            code_file = results.get("code", {}).get("filename", "N/A")
            research_n = results.get("research", {}).get("total_found", 0)
            disc_verdict = results.get("discussion", {}).get("verdict", {}).get("decision", "N/A")

            print(f"  Discussion: {disc_verdict}")
            print(f"  Research docs: {research_n}")
            print(f"  Code: {code_file}")
            print(f"  Drift score: {drift}")
            print(f"  KB stats: {rag_stats()}")

            if isinstance(drift, (int, float)) and drift > 0.4:
                print(f"  ⚠ DRIFT EXCEEDED — will retry with corrections")

        except KeyboardInterrupt:
            RUNNING = False
            break
        except Exception as e:
            log_work(AGENT, "ITERATION_ERROR", str(e))
            print(f"  ERROR: {e}")
            traceback.print_exc()

        if RUNNING and iteration < max_iter:
            print(f"  Sleeping {delay}s before next iteration...")
            try:
                time.sleep(delay)
            except KeyboardInterrupt:
                RUNNING = False
                break

    log_work(AGENT, "NON_STOP_END", f"iterations={iteration}")
    SM.set_status("COMPLETED")
    print(f"\n{'='*60}")
    print(f"  USISIVAC V6 — STOPPED after {iteration} iterations")
    print(f"{'='*60}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Usisivac V6 Orchestrator")
    parser.add_argument("problem", help="Problem description")
    parser.add_argument("--domain", default="universal")
    parser.add_argument("--data", default="")
    parser.add_argument("--data-path", default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--delay", type=int, default=30)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()

    if args.once:
        SM.init(args.problem, f"Single run: {args.problem}", args.domain)
        results = run_pipeline(args.problem, args.domain, args.data, args.data_path)
        print(json.dumps(results, indent=2, default=str)[:3000])
    else:
        run_non_stop(args.problem, args.domain, args.data, args.data_path,
                     args.max_iter, args.delay)
