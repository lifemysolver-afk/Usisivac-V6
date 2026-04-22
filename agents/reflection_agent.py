"""
+----------------------------------------------------------------------+
|  Reflection Agent - Meta-Analysis of Autonomous Coding Loop         |
|  Usisivac V6 | Gemini Orchestrator Extension                        |
|  Anti-Simulation Enforced: All reflections logged and verified      |
+----------------------------------------------------------------------+

The Reflection Agent performs meta-analysis on the coding loop's performance:
  - Failure analysis: Root causes of test failures, debugging logs
  - Iteration summarization: Key learnings and outcomes
  - Improvement proposals: Concrete suggestions for next iteration
"""

import sys, json, time, datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, log_work, log_proof
from core import state_manager as SM


class ReflectionAgent:
    """
    Analyzes the performance of the autonomous coding loop and proposes improvements.
    
    Responsibilities:
    - Analyze test failures and debugging logs
    - Summarize iteration outcomes and learnings
    - Propose improvements to strategy, tools, or code sections
    - Determine whether to continue or terminate the loop
    """
    
    def __init__(self, agent_name: str = "ReflectionAgent"):
        self.agent_name = agent_name
        self.log_path = BASE / "logs" / "reflection_agent.log"
    
    def _log(self, level: str, message: str, data: Optional[Dict] = None):
        """Log reflection agent activity"""
        timestamp = datetime.datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "agent": self.agent_name,
            "message": message,
            "data": data or {}
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def analyze_test_failures(
        self,
        test_results: Dict[str, Any],
        debug_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze test failures to identify root causes.
        
        Args:
            test_results: Test execution results
            debug_info: Optional debug information
            
        Returns:
            Dict with root causes and analysis
        """
        self._log("INFO", "Analyzing test failures", {
            "failed_tests": test_results.get("failed", 0),
            "total_tests": test_results.get("total", 0)
        })
        
        root_causes = []
        
        # Extract error messages
        errors = test_results.get("errors", [])
        for error in errors:
            root_causes.append({
                "error_type": error.get("type", "unknown"),
                "error_message": error.get("message", ""),
                "affected_tests": error.get("affected_tests", []),
                "severity": self._classify_severity(error)
            })
        
        # Analyze patterns
        patterns = self._identify_patterns(root_causes)
        
        analysis = {
            "root_causes": root_causes,
            "patterns": patterns,
            "analysis_time": datetime.datetime.utcnow().isoformat(),
            "debug_info": debug_info or {}
        }
        
        self._log("INFO", "Test failure analysis complete", {
            "root_causes_found": len(root_causes),
            "patterns_identified": len(patterns)
        })
        
        return analysis
    
    def _classify_severity(self, error: Dict[str, Any]) -> str:
        """Classify error severity: critical, high, medium, low"""
        error_type = error.get("type", "").lower()
        
        if "critical" in error_type or "fatal" in error_type:
            return "critical"
        elif "assertion" in error_type or "timeout" in error_type:
            return "high"
        elif "warning" in error_type:
            return "medium"
        else:
            return "low"
    
    def _identify_patterns(self, root_causes: List[Dict]) -> List[Dict]:
        """Identify patterns in root causes"""
        patterns = []
        error_types = {}
        
        for cause in root_causes:
            error_type = cause.get("error_type", "unknown")
            error_types[error_type] = error_types.get(error_type, 0) + 1
        
        # Find recurring error types
        for error_type, count in error_types.items():
            if count > 1:
                patterns.append({
                    "pattern_type": "recurring_error",
                    "error_type": error_type,
                    "occurrences": count,
                    "recommendation": f"This error appears {count} times. Consider systematic fix."
                })
        
        return patterns
    
    def summarize_iteration(
        self,
        iteration_id: str,
        plan: Dict[str, Any],
        implementation: Dict[str, Any],
        test_results: Dict[str, Any],
        review_feedback: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Summarize the outcomes and key learnings from an iteration.
        
        Args:
            iteration_id: The iteration ID
            plan: The plan that was executed
            implementation: The implementation changes
            test_results: The test results
            review_feedback: Optional code review feedback
            
        Returns:
            Dict with iteration summary and learnings
        """
        self._log("INFO", "Summarizing iteration", {"iteration_id": iteration_id})
        
        summary = {
            "iteration_id": iteration_id,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "plan_summary": {
                "strategy": plan.get("strategy", "unknown"),
                "steps_planned": len(plan.get("steps", [])),
                "estimated_complexity": plan.get("complexity", "unknown")
            },
            "implementation_summary": {
                "files_modified": implementation.get("files_modified", 0),
                "lines_added": implementation.get("lines_added", 0),
                "lines_removed": implementation.get("lines_removed", 0),
                "functions_added": implementation.get("functions_added", 0),
                "functions_modified": implementation.get("functions_modified", 0)
            },
            "test_summary": {
                "passed": test_results.get("passed", 0),
                "failed": test_results.get("failed", 0),
                "total": test_results.get("total", 0),
                "success_rate": self._calculate_success_rate(test_results)
            },
            "review_summary": self._summarize_review(review_feedback),
            "learnings": self._extract_learnings(
                plan, implementation, test_results, review_feedback
            )
        }
        
        self._log("INFO", "Iteration summary complete", {
            "iteration_id": iteration_id,
            "success_rate": summary["test_summary"]["success_rate"]
        })
        
        return summary
    
    def _calculate_success_rate(self, test_results: Dict[str, Any]) -> float:
        """Calculate test success rate"""
        total = test_results.get("total", 1)
        passed = test_results.get("passed", 0)
        return (passed / total * 100) if total > 0 else 0.0
    
    def _summarize_review(self, review_feedback: Optional[Dict]) -> Dict:
        """Summarize code review feedback"""
        if not review_feedback:
            return {"issues_found": 0, "recommendations": 0}
        
        return {
            "issues_found": len(review_feedback.get("issues", [])),
            "recommendations": len(review_feedback.get("recommendations", [])),
            "quality_score": review_feedback.get("quality_score", 0)
        }
    
    def _extract_learnings(
        self,
        plan: Dict,
        implementation: Dict,
        test_results: Dict,
        review_feedback: Optional[Dict]
    ) -> List[str]:
        """Extract key learnings from the iteration"""
        learnings = []
        
        # Learning 1: Plan effectiveness
        if test_results.get("passed", 0) == test_results.get("total", 0):
            learnings.append("Plan was effective: all tests passed on first try")
        else:
            learnings.append(f"Plan had gaps: {test_results.get('failed', 0)} tests failed")
        
        # Learning 2: Implementation scope
        lines_added = implementation.get("lines_added", 0)
        if lines_added > 500:
            learnings.append("Implementation was large: consider breaking into smaller steps")
        elif lines_added < 50:
            learnings.append("Implementation was minimal: good incremental approach")
        
        # Learning 3: Code quality
        if review_feedback:
            quality_score = review_feedback.get("quality_score", 0)
            if quality_score > 0.8:
                learnings.append("Code quality was high: good practices followed")
            elif quality_score < 0.5:
                learnings.append("Code quality needs improvement: focus on standards")
        
        # Learning 4: Error patterns
        errors = test_results.get("errors", [])
        if errors:
            error_types = set(e.get("type", "unknown") for e in errors)
            learnings.append(f"Encountered error types: {', '.join(error_types)}")
        
        return learnings
    
    def propose_improvements(
        self,
        iteration_summary: Dict[str, Any],
        iteration_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Propose concrete improvements for the next iteration.
        
        Args:
            iteration_summary: Summary of current iteration
            iteration_history: History of previous iterations
            
        Returns:
            Dict with improvement proposals
        """
        self._log("INFO", "Proposing improvements")
        
        improvements = {
            "strategy_improvements": [],
            "implementation_improvements": [],
            "testing_improvements": [],
            "should_continue": True,
            "estimated_next_success_rate": 0.0,
            "confidence": 0.0
        }
        
        # Strategy improvements
        test_summary = iteration_summary.get("test_summary", {})
        success_rate = test_summary.get("success_rate", 0)
        
        if success_rate < 50:
            improvements["strategy_improvements"].append({
                "suggestion": "Reduce scope: plan is too ambitious",
                "rationale": f"Only {success_rate:.1f}% tests passed",
                "action": "Break into smaller, more focused steps"
            })
        
        # Implementation improvements
        impl_summary = iteration_summary.get("implementation_summary", {})
        if impl_summary.get("lines_added", 0) > 500:
            improvements["implementation_improvements"].append({
                "suggestion": "Reduce implementation size",
                "rationale": f"{impl_summary.get('lines_added')} lines added in one iteration",
                "action": "Implement incrementally with more frequent testing"
            })
        
        # Testing improvements
        if test_summary.get("failed", 0) > 0:
            improvements["testing_improvements"].append({
                "suggestion": "Add more test coverage",
                "rationale": f"{test_summary.get('failed')} tests failed",
                "action": "Write tests before implementation (TDD approach)"
            })
        
        # Determine if should continue
        if success_rate >= 80:
            improvements["should_continue"] = False
            improvements["estimated_next_success_rate"] = 95.0
            improvements["confidence"] = 0.9
        else:
            improvements["should_continue"] = True
            improvements["estimated_next_success_rate"] = min(success_rate + 15, 100)
            improvements["confidence"] = 0.7
        
        self._log("INFO", "Improvements proposed", {
            "strategy_improvements": len(improvements["strategy_improvements"]),
            "implementation_improvements": len(improvements["implementation_improvements"]),
            "testing_improvements": len(improvements["testing_improvements"]),
            "should_continue": improvements["should_continue"]
        })
        
        return improvements
    
    def generate_reflection_report(
        self,
        iteration_id: str,
        iteration_summary: Dict[str, Any],
        improvements: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive reflection report.
        
        Args:
            iteration_id: The iteration ID
            iteration_summary: Summary of the iteration
            improvements: Proposed improvements
            
        Returns:
            Comprehensive reflection report
        """
        report = {
            "iteration_id": iteration_id,
            "report_time": datetime.datetime.utcnow().isoformat(),
            "summary": iteration_summary,
            "improvements": improvements,
            "completion_reason": self._determine_completion_reason(
                iteration_summary, improvements
            ),
            "next_steps": self._determine_next_steps(improvements),
            "confidence_score": improvements.get("confidence", 0.0)
        }
        
        # Log to proof registry
        log_proof("REFLECTION_REPORT_GENERATED", {
            "iteration_id": iteration_id,
            "should_continue": improvements.get("should_continue", False),
            "confidence": improvements.get("confidence", 0.0)
        })
        
        self._log("INFO", "Reflection report generated", {
            "iteration_id": iteration_id,
            "should_continue": improvements.get("should_continue", False)
        })
        
        return report
    
    def _determine_completion_reason(
        self,
        iteration_summary: Dict,
        improvements: Dict
    ) -> str:
        """Determine the reason for loop completion or continuation"""
        success_rate = iteration_summary.get("test_summary", {}).get("success_rate", 0)
        
        if success_rate >= 95:
            return "high_success_rate"
        elif success_rate >= 80:
            return "acceptable_success_rate"
        elif improvements.get("should_continue", False):
            return "improvements_proposed"
        else:
            return "max_iterations_or_user_decision"
    
    def _determine_next_steps(self, improvements: Dict) -> List[str]:
        """Determine recommended next steps"""
        steps = []
        
        if improvements.get("strategy_improvements"):
            steps.append("Revise strategy based on failure analysis")
        
        if improvements.get("implementation_improvements"):
            steps.append("Refactor implementation for better modularity")
        
        if improvements.get("testing_improvements"):
            steps.append("Expand test coverage and add edge cases")
        
        if not steps:
            steps.append("Task complete: no further improvements needed")
        
        return steps


def run(
    iteration_id: str,
    iteration_summary: Dict[str, Any],
    iteration_history: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Run the Reflection Agent.
    
    Args:
        iteration_id: The iteration ID
        iteration_summary: Summary of current iteration
        iteration_history: History of previous iterations
        
    Returns:
        Reflection report
    """
    agent = ReflectionAgent()
    
    # Propose improvements
    improvements = agent.propose_improvements(iteration_summary, iteration_history)
    
    # Generate reflection report
    report = agent.generate_reflection_report(
        iteration_id, iteration_summary, improvements
    )
    
    return report


if __name__ == "__main__":
    # Example usage
    agent = ReflectionAgent()
    
    # Example test results
    test_results = {
        "passed": 8,
        "failed": 2,
        "total": 10,
        "errors": [
            {"type": "AssertionError", "message": "Expected 5, got 4", "affected_tests": ["test_calc"]},
            {"type": "TimeoutError", "message": "Test exceeded timeout", "affected_tests": ["test_slow"]}
        ]
    }
    
    # Analyze failures
    analysis = agent.analyze_test_failures(test_results)
    print("Failure Analysis:")
    print(json.dumps(analysis, indent=2))
