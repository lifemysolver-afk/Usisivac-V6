"""
╔══════════════════════════════════════════════════════════════════════╗
║  Autonomous Coding Loop Manager                                     ║
║  Usisivac V6 | Gemini Orchestrator Extension                        ║
║  Orchestrates: Plan → Implement → Test → Debug → Review → Commit   ║
║                      → Reflect → Improve → Loop                     ║
║  Anti-Simulation Enforced: All actions logged to proof_registry     ║
╚══════════════════════════════════════════════════════════════════════╝

The Autonomous Loop Manager coordinates all agents in the coding loop:
  - Initializes and manages the Loop Controller
  - Triggers agents in sequence based on loop state
  - Handles state transitions and error recovery
  - Persists state across sessions
"""

import sys, json, time, datetime, traceback
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, log_work, log_proof
from core import state_manager as SM
from orchestrator.loop_controller import LoopController, LoopState
from agents.reflection_agent import run as reflection_run


class AutonomousLoopManager:
    """
    Manages the autonomous coding loop, coordinating all agents and state transitions.
    
    Responsibilities:
    - Initialize and manage Loop Controller
    - Trigger agents in correct sequence
    - Handle state transitions
    - Manage error recovery and debugging
    - Persist state across sessions
    - Generate reports and metrics
    """
    
    def __init__(
        self,
        task_id: str,
        task_objective: str,
        max_iterations: int = 10,
        agent_callbacks: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize the Autonomous Loop Manager.
        
        Args:
            task_id: Unique identifier for the task
            task_objective: High-level description of the task
            max_iterations: Maximum number of loop iterations
            agent_callbacks: Dict of agent names to callback functions
        """
        self.task_id = task_id
        self.task_objective = task_objective
        self.max_iterations = max_iterations
        
        # Initialize Loop Controller
        self.controller = LoopController(task_id, task_objective, max_iterations)
        
        # Agent callbacks
        self.agent_callbacks = agent_callbacks or {}
        
        # Metrics
        self.metrics = {
            "total_iterations": 0,
            "successful_iterations": 0,
            "failed_iterations": 0,
            "total_time_seconds": 0.0,
            "average_iteration_time": 0.0,
            "test_success_rate": 0.0
        }
        
        self.log_path = BASE / "logs" / f"autonomous_loop_{task_id}.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._log("INFO", f"Autonomous Loop Manager initialized for task: {task_id}")
    
    def _log(self, level: str, message: str, data: Optional[Dict] = None):
        """Log manager activity"""
        timestamp = datetime.datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "task_id": self.task_id,
            "message": message,
            "data": data or {}
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def run_loop(self) -> Dict[str, Any]:
        """
        Run the autonomous coding loop until completion.
        
        Returns:
            Dict with final results and metrics
        """
        self._log("INFO", "Starting autonomous coding loop")
        loop_start_time = time.time()
        
        try:
            while not self.controller.is_complete():
                iteration_start_time = time.time()
                iteration_id = self.controller.start_iteration()
                
                if iteration_id is None:
                    self._log("INFO", "Max iterations reached, terminating loop")
                    break
                
                self._log("INFO", f"Starting iteration {self.controller.current_iteration}", {
                    "iteration_id": iteration_id
                })
                
                # Run the iteration
                success = self._run_iteration(iteration_id)
                
                iteration_duration = time.time() - iteration_start_time
                self.controller.end_iteration(iteration_id, iteration_duration)
                
                if success:
                    self.metrics["successful_iterations"] += 1
                else:
                    self.metrics["failed_iterations"] += 1
                
                self.metrics["total_iterations"] += 1
                self.metrics["total_time_seconds"] += iteration_duration
                
                self._log("INFO", f"Iteration {self.controller.current_iteration} complete", {
                    "iteration_id": iteration_id,
                    "duration": iteration_duration,
                    "success": success
                })
            
            loop_duration = time.time() - loop_start_time
            self.metrics["total_time_seconds"] = loop_duration
            
            if self.metrics["total_iterations"] > 0:
                self.metrics["average_iteration_time"] = (
                    loop_duration / self.metrics["total_iterations"]
                )
            
            # Save final checkpoint
            self.controller.save_checkpoint()
            
            result = {
                "status": "complete",
                "task_id": self.task_id,
                "total_iterations": self.metrics["total_iterations"],
                "successful_iterations": self.metrics["successful_iterations"],
                "failed_iterations": self.metrics["failed_iterations"],
                "total_time_seconds": loop_duration,
                "average_iteration_time": self.metrics["average_iteration_time"],
                "controller_status": self.controller.get_status()
            }
            
            self._log("INFO", "Autonomous loop completed successfully", result)
            
            return result
        
        except Exception as e:
            self._log("ERROR", f"Autonomous loop failed: {str(e)}", {
                "traceback": traceback.format_exc(),
                "iteration": self.controller.current_iteration
            })
            
            # Save checkpoint before exiting
            self.controller.save_checkpoint()
            
            return {
                "status": "failed",
                "task_id": self.task_id,
                "error": str(e),
                "iteration": self.controller.current_iteration,
                "metrics": self.metrics
            }
    
    def _run_iteration(self, iteration_id: str) -> bool:
        """
        Run a single iteration of the loop.
        
        Returns:
            bool: True if iteration succeeded, False otherwise
        """
        try:
            # Step 1: Plan
            plan = self._call_agent("plan_agent", iteration_id)
            if not plan:
                self._log("ERROR", "Plan agent failed")
                return False
            
            self.controller.record_plan(iteration_id, plan)
            
            # Step 2: Implement
            implementation = self._call_agent("implement_agent", iteration_id, plan)
            if not implementation:
                self._log("ERROR", "Implement agent failed")
                return False
            
            self.controller.record_implementation(iteration_id, implementation)
            
            # Step 3: Test
            test_results = self._call_agent("test_agent", iteration_id, implementation)
            if not test_results:
                self._log("ERROR", "Test agent failed")
                return False
            
            tests_passed = self.controller.record_test_results(iteration_id, test_results)
            
            # Step 4: Debug (if tests failed)
            if not tests_passed:
                debug_info = self._call_agent("debug_agent", iteration_id, test_results)
                if debug_info:
                    self.controller.record_debug_info(iteration_id, debug_info)
                    # Go back to Step 2: Implement
                    return self._run_iteration(iteration_id)
                else:
                    self._log("ERROR", "Debug agent failed")
                    return False
            
            # Step 5: Review
            review_feedback = self._call_agent("review_agent", iteration_id, implementation)
            if review_feedback:
                self.controller.record_review(iteration_id, review_feedback)
            
            # Step 6: Commit
            commit_hash = self._call_agent("commit_agent", iteration_id)
            if commit_hash:
                self.controller.record_commit(iteration_id, commit_hash)
            
            # Step 7: Reflect
            iteration_summary = {
                "plan": plan,
                "implementation": implementation,
                "test_results": test_results,
                "review_feedback": review_feedback
            }
            
            reflection = self._call_agent(
                "reflection_agent",
                iteration_id,
                iteration_summary,
                self.controller.get_history()
            )
            
            if reflection:
                should_continue = self.controller.record_reflection(iteration_id, reflection)
                
                if not should_continue:
                    self._log("INFO", "Reflection agent indicated task completion")
                    return True
            
            return True
        
        except Exception as e:
            self._log("ERROR", f"Iteration failed: {str(e)}", {
                "traceback": traceback.format_exc()
            })
            return False
    
    def _call_agent(self, agent_name: str, *args, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Call an agent via callback.
        
        Args:
            agent_name: Name of the agent to call
            *args: Positional arguments for the agent
            **kwargs: Keyword arguments for the agent
            
        Returns:
            Agent result or None if callback not defined
        """
        if agent_name not in self.agent_callbacks:
            self._log("WARN", f"Agent callback not defined: {agent_name}")
            return None
        
        try:
            callback = self.agent_callbacks[agent_name]
            result = callback(*args, **kwargs)
            
            self._log("DEBUG", f"Agent {agent_name} executed", {
                "result_keys": list(result.keys()) if isinstance(result, dict) else "N/A"
            })
            
            return result
        
        except Exception as e:
            self._log("ERROR", f"Agent {agent_name} failed: {str(e)}", {
                "traceback": traceback.format_exc()
            })
            return None
    
    def resume_from_checkpoint(self) -> bool:
        """
        Resume the loop from a saved checkpoint.
        
        Returns:
            bool: True if successfully resumed
        """
        self._log("INFO", "Attempting to resume from checkpoint")
        
        loaded_controller = LoopController.load_checkpoint(self.task_id)
        
        if loaded_controller is None:
            self._log("WARN", "No checkpoint found, starting fresh")
            return False
        
        self.controller = loaded_controller
        
        self._log("INFO", "Successfully resumed from checkpoint", {
            "iteration": self.controller.current_iteration,
            "state": self.controller.current_state.value
        })
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the loop"""
        return {
            "task_id": self.task_id,
            "objective": self.task_objective,
            "controller_status": self.controller.get_status(),
            "metrics": self.metrics,
            "log_path": str(self.log_path)
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        return {
            "total_iterations": self.metrics["total_iterations"],
            "successful_iterations": self.metrics["successful_iterations"],
            "failed_iterations": self.metrics["failed_iterations"],
            "total_time_seconds": self.metrics["total_time_seconds"],
            "average_iteration_time": self.metrics["average_iteration_time"],
            "success_rate": (
                self.metrics["successful_iterations"] / self.metrics["total_iterations"] * 100
                if self.metrics["total_iterations"] > 0 else 0.0
            )
        }
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate a comprehensive report of the autonomous loop execution"""
        report = {
            "task_id": self.task_id,
            "objective": self.task_objective,
            "report_time": datetime.datetime.utcnow().isoformat(),
            "status": self.controller.get_status(),
            "metrics": self.get_metrics(),
            "iterations": [
                {
                    "iteration_id": record.iteration_id,
                    "timestamp": record.timestamp,
                    "state": record.state,
                    "duration_seconds": record.duration_seconds,
                    "proof_hash": record.proof_hash
                }
                for record in self.controller.get_history()
            ],
            "log_path": str(self.log_path)
        }
        
        # Log report generation to proof registry
        log_proof("AUTONOMOUS_LOOP_REPORT_GENERATED", {
            "task_id": self.task_id,
            "total_iterations": self.metrics["total_iterations"],
            "success_rate": report["metrics"]["success_rate"]
        })
        
        return report


if __name__ == "__main__":
    # Example usage
    manager = AutonomousLoopManager(
        task_id="example_task",
        task_objective="Implement and test a new feature",
        max_iterations=5
    )
    
    print("Autonomous Loop Manager initialized")
    print(f"Status: {json.dumps(manager.get_status(), indent=2)}")
