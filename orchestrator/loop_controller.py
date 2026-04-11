"""
╔══════════════════════════════════════════════════════════════════════╗
║  Loop Controller — Autonomous Coding Loop Orchestrator              ║
║  Usisivac V6 | Gemini Orchestrator Extension                        ║
║  Anti-Simulation Enforced: All iterations logged to proof_registry  ║
╚══════════════════════════════════════════════════════════════════════╝

The Loop Controller manages the persistent autonomous coding loop:
  plan → implement → test → debug → review → commit → reflect → improve

Each iteration is logged to proof_registry.jsonl with cryptographic hash.
"""

import sys, json, time, datetime, hashlib, uuid
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, log_work, log_proof
from core import state_manager as SM


class LoopState(Enum):
    """State machine for the autonomous coding loop"""
    INIT = "init"
    PLANNING = "planning"
    IMPLEMENTING = "implementing"
    TESTING = "testing"
    DEBUGGING = "debugging"
    REVIEWING = "reviewing"
    COMMITTING = "committing"
    REFLECTING = "reflecting"
    COMPLETE = "complete"
    FAILED = "failed"


@dataclass
class IterationRecord:
    """Record of a single loop iteration"""
    iteration_id: str
    timestamp: str
    state: str
    plan: Optional[Dict[str, Any]] = None
    implementation: Optional[Dict[str, Any]] = None
    test_results: Optional[Dict[str, Any]] = None
    debug_info: Optional[Dict[str, Any]] = None
    review_feedback: Optional[Dict[str, Any]] = None
    commit_hash: Optional[str] = None
    reflection: Optional[Dict[str, Any]] = None
    duration_seconds: float = 0.0
    proof_hash: Optional[str] = None


class LoopController:
    """
    Central orchestrator of the autonomous coding loop.
    
    Responsibilities:
    - Iteration management and history
    - Termination logic
    - Task graph modification
    - Dynamic agent triggering
    - State persistence
    - Anti-simulation proof generation
    """
    
    def __init__(self, task_id: str, task_objective: str, max_iterations: int = 10):
        """
        Initialize the Loop Controller.
        
        Args:
            task_id: Unique identifier for the task
            task_objective: High-level description of the task
            max_iterations: Maximum number of loop iterations before termination
        """
        self.task_id = task_id
        self.task_objective = task_objective
        self.max_iterations = max_iterations
        
        self.current_iteration = 0
        self.current_state = LoopState.INIT
        self.iteration_history: List[IterationRecord] = []
        
        self.task_graph: Dict[str, Any] = {
            "task_id": task_id,
            "objective": task_objective,
            "created_at": datetime.datetime.utcnow().isoformat(),
            "status": "active",
            "dependencies": {}
        }
        
        self.log_path = BASE / "logs" / f"loop_controller_{task_id}.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._log_event("LOOP_INIT", {
            "task_id": task_id,
            "objective": task_objective,
            "max_iterations": max_iterations
        })
    
    def _log_event(self, event_type: str, data: Dict[str, Any]):
        """Log an event to the loop controller log"""
        event = {
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "event_type": event_type,
            "task_id": self.task_id,
            "iteration": self.current_iteration,
            "state": self.current_state.value,
            "data": data
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
    
    def _compute_proof_hash(self, record: IterationRecord) -> str:
        """Compute cryptographic hash for anti-simulation proof"""
        content = json.dumps(asdict(record), sort_keys=True, default=str)
        return hashlib.sha256(content.encode()).hexdigest()
    
    def start_iteration(self) -> str:
        """
        Start a new iteration of the loop.
        
        Returns:
            iteration_id: Unique identifier for this iteration
        """
        if self.current_iteration >= self.max_iterations:
            self.current_state = LoopState.COMPLETE
            self._log_event("MAX_ITERATIONS_REACHED", {
                "current": self.current_iteration,
                "max": self.max_iterations
            })
            return None
        
        self.current_iteration += 1
        self.current_state = LoopState.PLANNING
        
        iteration_id = f"{self.task_id}_iter_{self.current_iteration}_{uuid.uuid4().hex[:8]}"
        
        self._log_event("ITERATION_START", {
            "iteration_id": iteration_id,
            "iteration_number": self.current_iteration
        })
        
        return iteration_id
    
    def record_plan(self, iteration_id: str, plan: Dict[str, Any]) -> bool:
        """
        Record the plan generated by the Plan Agent.
        
        Args:
            iteration_id: The iteration ID
            plan: Plan dictionary from Plan Agent
            
        Returns:
            bool: True if recorded successfully
        """
        if self.current_state != LoopState.PLANNING:
            self._log_event("STATE_ERROR", {
                "expected": LoopState.PLANNING.value,
                "actual": self.current_state.value
            })
            return False
        
        self.current_state = LoopState.IMPLEMENTING
        
        self._log_event("PLAN_RECORDED", {
            "iteration_id": iteration_id,
            "plan_steps": len(plan.get("steps", [])),
            "strategy": plan.get("strategy", "unknown")
        })
        
        return True
    
    def record_implementation(self, iteration_id: str, implementation: Dict[str, Any]) -> bool:
        """
        Record code changes made by the Implement Agent.
        
        Args:
            iteration_id: The iteration ID
            implementation: Implementation details (files changed, lines added, etc.)
            
        Returns:
            bool: True if recorded successfully
        """
        if self.current_state != LoopState.IMPLEMENTING:
            self._log_event("STATE_ERROR", {
                "expected": LoopState.IMPLEMENTING.value,
                "actual": self.current_state.value
            })
            return False
        
        self.current_state = LoopState.TESTING
        
        self._log_event("IMPLEMENTATION_RECORDED", {
            "iteration_id": iteration_id,
            "files_modified": implementation.get("files_modified", 0),
            "lines_added": implementation.get("lines_added", 0),
            "lines_removed": implementation.get("lines_removed", 0)
        })
        
        return True
    
    def record_test_results(self, iteration_id: str, test_results: Dict[str, Any]) -> bool:
        """
        Record test execution results.
        
        Args:
            iteration_id: The iteration ID
            test_results: Test results (passed, failed, errors, etc.)
            
        Returns:
            bool: True if tests passed; False if debugging needed
        """
        if self.current_state != LoopState.TESTING:
            self._log_event("STATE_ERROR", {
                "expected": LoopState.TESTING.value,
                "actual": self.current_state.value
            })
            return False
        
        tests_passed = test_results.get("passed", 0)
        tests_failed = test_results.get("failed", 0)
        
        if tests_failed > 0:
            self.current_state = LoopState.DEBUGGING
            self._log_event("TESTS_FAILED", {
                "iteration_id": iteration_id,
                "passed": tests_passed,
                "failed": tests_failed,
                "errors": test_results.get("errors", [])
            })
            return False
        else:
            self.current_state = LoopState.REVIEWING
            self._log_event("TESTS_PASSED", {
                "iteration_id": iteration_id,
                "passed": tests_passed,
                "total": test_results.get("total", tests_passed)
            })
            return True
    
    def record_debug_info(self, iteration_id: str, debug_info: Dict[str, Any]) -> bool:
        """
        Record debugging analysis and proposed fixes.
        
        Args:
            iteration_id: The iteration ID
            debug_info: Debug analysis and fix suggestions
            
        Returns:
            bool: True if recorded successfully
        """
        if self.current_state != LoopState.DEBUGGING:
            self._log_event("STATE_ERROR", {
                "expected": LoopState.DEBUGGING.value,
                "actual": self.current_state.value
            })
            return False
        
        # After debugging, go back to implementing
        self.current_state = LoopState.IMPLEMENTING
        
        self._log_event("DEBUG_RECORDED", {
            "iteration_id": iteration_id,
            "root_causes_identified": len(debug_info.get("root_causes", [])),
            "fixes_proposed": len(debug_info.get("fixes", []))
        })
        
        return True
    
    def record_review(self, iteration_id: str, review_feedback: Dict[str, Any]) -> bool:
        """
        Record code review feedback.
        
        Args:
            iteration_id: The iteration ID
            review_feedback: Review findings and recommendations
            
        Returns:
            bool: True if recorded successfully
        """
        if self.current_state != LoopState.REVIEWING:
            self._log_event("STATE_ERROR", {
                "expected": LoopState.REVIEWING.value,
                "actual": self.current_state.value
            })
            return False
        
        self.current_state = LoopState.COMMITTING
        
        self._log_event("REVIEW_RECORDED", {
            "iteration_id": iteration_id,
            "issues_found": len(review_feedback.get("issues", [])),
            "recommendations": len(review_feedback.get("recommendations", []))
        })
        
        return True
    
    def record_commit(self, iteration_id: str, commit_hash: str) -> bool:
        """
        Record Git commit.
        
        Args:
            iteration_id: The iteration ID
            commit_hash: Git commit hash
            
        Returns:
            bool: True if recorded successfully
        """
        if self.current_state != LoopState.COMMITTING:
            self._log_event("STATE_ERROR", {
                "expected": LoopState.COMMITTING.value,
                "actual": self.current_state.value
            })
            return False
        
        self.current_state = LoopState.REFLECTING
        
        self._log_event("COMMIT_RECORDED", {
            "iteration_id": iteration_id,
            "commit_hash": commit_hash
        })
        
        return True
    
    def record_reflection(self, iteration_id: str, reflection: Dict[str, Any]) -> bool:
        """
        Record reflection analysis from the Reflection Agent.
        
        Args:
            iteration_id: The iteration ID
            reflection: Reflection analysis (learnings, improvements, etc.)
            
        Returns:
            bool: True if recorded successfully; decision on loop continuation
        """
        if self.current_state != LoopState.REFLECTING:
            self._log_event("STATE_ERROR", {
                "expected": LoopState.REFLECTING.value,
                "actual": self.current_state.value
            })
            return False
        
        should_continue = reflection.get("should_continue", False)
        
        if should_continue and self.current_iteration < self.max_iterations:
            self.current_state = LoopState.PLANNING
            self._log_event("REFLECTION_RECORDED_CONTINUE", {
                "iteration_id": iteration_id,
                "learnings": reflection.get("learnings", []),
                "improvements": reflection.get("improvements", [])
            })
            return True
        else:
            self.current_state = LoopState.COMPLETE
            self._log_event("REFLECTION_RECORDED_COMPLETE", {
                "iteration_id": iteration_id,
                "reason": reflection.get("completion_reason", "task_complete")
            })
            return False
    
    def end_iteration(self, iteration_id: str, duration_seconds: float) -> IterationRecord:
        """
        End the current iteration and create a record.
        
        Args:
            iteration_id: The iteration ID
            duration_seconds: How long the iteration took
            
        Returns:
            IterationRecord: The completed iteration record
        """
        record = IterationRecord(
            iteration_id=iteration_id,
            timestamp=datetime.datetime.utcnow().isoformat(),
            state=self.current_state.value,
            duration_seconds=duration_seconds
        )
        
        # Compute anti-simulation proof hash
        record.proof_hash = self._compute_proof_hash(record)
        
        # Log to proof registry
        log_proof("LOOP_ITERATION_COMPLETE", {
            "iteration_id": iteration_id,
            "iteration_number": self.current_iteration,
            "state": self.current_state.value,
            "duration": duration_seconds,
            "proof_hash": record.proof_hash
        })
        
        self.iteration_history.append(record)
        
        self._log_event("ITERATION_END", {
            "iteration_id": iteration_id,
            "duration": duration_seconds,
            "total_iterations": len(self.iteration_history)
        })
        
        return record
    
    def is_complete(self) -> bool:
        """Check if the loop should terminate"""
        return self.current_state == LoopState.COMPLETE or self.current_iteration >= self.max_iterations
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status of the loop"""
        return {
            "task_id": self.task_id,
            "objective": self.task_objective,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "current_state": self.current_state.value,
            "total_iterations_completed": len(self.iteration_history),
            "is_complete": self.is_complete(),
            "task_graph": self.task_graph
        }
    
    def get_history(self) -> List[IterationRecord]:
        """Get all iteration records"""
        return self.iteration_history
    
    def save_checkpoint(self) -> bool:
        """Save current state to disk for persistence"""
        checkpoint = {
            "task_id": self.task_id,
            "task_objective": self.task_objective,
            "current_iteration": self.current_iteration,
            "current_state": self.current_state.value,
            "max_iterations": self.max_iterations,
            "iteration_history": [asdict(r) for r in self.iteration_history],
            "task_graph": self.task_graph,
            "checkpoint_time": datetime.datetime.utcnow().isoformat()
        }
        
        checkpoint_path = BASE / "logs" / f"loop_checkpoint_{self.task_id}.json"
        
        with open(checkpoint_path, "w") as f:
            json.dump(checkpoint, f, indent=2, default=str)
        
        self._log_event("CHECKPOINT_SAVED", {
            "checkpoint_path": str(checkpoint_path),
            "iterations": self.current_iteration
        })
        
        return True
    
    @classmethod
    def load_checkpoint(cls, task_id: str) -> Optional['LoopController']:
        """Load a saved checkpoint"""
        checkpoint_path = BASE / "logs" / f"loop_checkpoint_{task_id}.json"
        
        if not checkpoint_path.exists():
            return None
        
        with open(checkpoint_path, "r") as f:
            checkpoint = json.load(f)
        
        controller = cls(
            task_id=checkpoint["task_id"],
            task_objective=checkpoint["task_objective"],
            max_iterations=checkpoint["max_iterations"]
        )
        
        controller.current_iteration = checkpoint["current_iteration"]
        controller.current_state = LoopState(checkpoint["current_state"])
        controller.task_graph = checkpoint["task_graph"]
        
        # Restore iteration history
        for record_dict in checkpoint["iteration_history"]:
            record = IterationRecord(**record_dict)
            controller.iteration_history.append(record)
        
        return controller


if __name__ == "__main__":
    # Example usage
    controller = LoopController(
        task_id="test_task_001",
        task_objective="Implement and test a new feature",
        max_iterations=5
    )
    
    print("Loop Controller initialized")
    print(f"Status: {json.dumps(controller.get_status(), indent=2)}")
