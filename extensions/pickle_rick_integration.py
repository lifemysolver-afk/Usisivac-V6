"""
+----------------------------------------------------------------------+
|  Pickle Rick Extension Integration Module                           |
|  Usisivac V6 | Gemini CLI Extension Integration                     |
|  Anti-Simulation Enforced: All integration logged to proof_registry |
+----------------------------------------------------------------------+

Integrates the Pickle Rick extension (autonomous coding loop) with
Usisivac V6 system for iterative, self-improving code development.

Pickle Rick provides:
  - Autonomous iterative development loops
  - Session state management
  - Completion promise verification
  - Git-based version control integration
"""

import sys, json, time, datetime, subprocess, os
from pathlib import Path
from typing import Dict, List, Optional, Any

BASE = Path(__file__).parent.parent
sys.path.insert(0, str(BASE))

from core.anti_simulation import enforce, log_work, log_proof


class PickleRickIntegration:
    """
    Integration layer between Pickle Rick extension and Usisivac V6.
    
    Responsibilities:
    - Initialize Pickle Rick sessions
    - Monitor session state
    - Manage completion promises
    - Log all actions to proof registry
    """
    
    def __init__(self, extension_root: str = None):
        """
        Initialize Pickle Rick Integration.
        
        Args:
            extension_root: Root directory of pickle-rick-extension
        """
        self.extension_root = extension_root or "/home/ubuntu/pickle-rick-extension"
        self.extension_path = Path(self.extension_root)
        
        if not self.extension_path.exists():
            raise ValueError(f"Pickle Rick extension not found at {self.extension_root}")
        
        self.log_path = BASE / "logs" / "pickle_rick_integration.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        
        self._log("INFO", "Pickle Rick Integration initialized", {
            "extension_root": self.extension_root
        })
    
    def _log(self, level: str, message: str, data: Optional[Dict] = None):
        """Log integration activity"""
        timestamp = datetime.datetime.utcnow().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "level": level,
            "message": message,
            "data": data or {}
        }
        
        with open(self.log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    def verify_installation(self) -> bool:
        """
        Verify that Pickle Rick extension is properly installed.
        
        Returns:
            bool: True if installation is valid
        """
        self._log("INFO", "Verifying Pickle Rick installation")
        
        required_files = [
            "extension/package.json",
            "extension/src/bin/spawn-rick.ts",
            "extension/src/services/session-state.ts",
            "README.md"
        ]
        
        for file_path in required_files:
            full_path = self.extension_path / file_path
            if not full_path.exists():
                self._log("ERROR", f"Missing required file: {file_path}")
                return False
        
        self._log("INFO", "Pickle Rick installation verified")
        log_proof("PICKLE_RICK_VERIFIED", {
            "extension_root": str(self.extension_root),
            "artifact_path": str(self.extension_path / "extension/package.json")
        })
        
        return True
    
    def build_extension(self) -> bool:
        """
        Build the Pickle Rick extension (TypeScript compilation).
        
        Returns:
            bool: True if build succeeded
        """
        self._log("INFO", "Building Pickle Rick extension")
        
        extension_dir = self.extension_path / "extension"
        
        try:
            result = subprocess.run(
                ["npm", "run", "build"],
                cwd=str(extension_dir),
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode != 0:
                self._log("ERROR", "Build failed", {
                    "stderr": result.stderr,
                    "stdout": result.stdout
                })
                return False
            
            self._log("INFO", "Build succeeded")
            log_proof("PICKLE_RICK_BUILD_SUCCESS", {
                "script_run": str(extension_dir / "package.json"),
                "script_output": result.stdout[:500]
            })
            
            return True
        
        except subprocess.TimeoutExpired:
            self._log("ERROR", "Build timed out")
            return False
        except Exception as e:
            self._log("ERROR", f"Build failed: {str(e)}")
            return False
    
    def run_tests(self) -> Dict[str, Any]:
        """
        Run the Pickle Rick extension tests.
        
        Returns:
            Dict with test results
        """
        self._log("INFO", "Running Pickle Rick tests")
        
        extension_dir = self.extension_path / "extension"
        
        try:
            result = subprocess.run(
                ["npm", "test"],
                cwd=str(extension_dir),
                capture_output=True,
                text=True,
                timeout=120
            )
            
            # Parse test output
            test_results = {
                "passed": "passed" in result.stdout,
                "return_code": result.returncode,
                "stdout": result.stdout[:1000],
                "stderr": result.stderr[:1000]
            }
            
            if result.returncode == 0:
                self._log("INFO", "Tests passed")
                log_proof("PICKLE_RICK_TESTS_PASSED", {
                    "script_run": str(extension_dir / "package.json"),
                    "script_output": result.stdout[:500]
                })
            else:
                self._log("ERROR", "Tests failed", test_results)
            
            return test_results
        
        except subprocess.TimeoutExpired:
            self._log("ERROR", "Tests timed out")
            return {"passed": False, "error": "timeout"}
        except Exception as e:
            self._log("ERROR", f"Tests failed: {str(e)}")
            return {"passed": False, "error": str(e)}
    
    def create_session(
        self,
        task_description: str,
        max_iterations: int = 5,
        max_time_minutes: int = 60,
        completion_promise: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Create a new Pickle Rick session.
        
        Args:
            task_description: Description of the task to execute
            max_iterations: Maximum number of iterations
            max_time_minutes: Maximum time in minutes
            completion_promise: Optional completion promise string
            
        Returns:
            Dict with session information
        """
        self._log("INFO", "Creating Pickle Rick session", {
            "task": task_description[:100],
            "max_iterations": max_iterations,
            "max_time_minutes": max_time_minutes
        })
        
        # Create session directory
        sessions_root = self.extension_path / ".gemini" / "extensions" / "pickle-rick" / "sessions"
        sessions_root.mkdir(parents=True, exist_ok=True)
        
        # Generate session ID
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        import random
        session_hash = f"{random.randint(0, 0xFFFFFFFF):08x}"
        session_id = f"{today}-{session_hash}"
        session_dir = sessions_root / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Create state.json
        state = {
            "active": True,
            "working_dir": os.getcwd(),
            "step": "prd",
            "iteration": 1,
            "max_iterations": max_iterations,
            "max_time_minutes": max_time_minutes,
            "start_time_epoch": int(time.time()),
            "original_prompt": task_description,
            "session_dir": str(session_dir),
            "completion_promise": completion_promise,
            "started_at": datetime.datetime.utcnow().isoformat()
        }
        
        state_file = session_dir / "state.json"
        with open(state_file, "w") as f:
            json.dump(state, f, indent=2)
        
        self._log("INFO", "Session created", {
            "session_id": session_id,
            "session_dir": str(session_dir)
        })
        
        log_proof("PICKLE_RICK_SESSION_CREATED", {
            "artifact_path": str(state_file)
        })
        
        return {
            "session_id": session_id,
            "session_dir": str(session_dir),
            "state_file": str(state_file),
            "state": state
        }
    
    def get_session_status(self, session_dir: str) -> Optional[Dict[str, Any]]:
        """
        Get the status of a Pickle Rick session.
        
        Args:
            session_dir: Path to the session directory
            
        Returns:
            Dict with session status or None if not found
        """
        state_file = Path(session_dir) / "state.json"
        
        if not state_file.exists():
            self._log("WARN", f"Session state file not found: {state_file}")
            return None
        
        try:
            with open(state_file, "r") as f:
                state = json.load(f)
            
            return {
                "session_dir": session_dir,
                "active": state.get("active", False),
                "iteration": state.get("iteration", 0),
                "max_iterations": state.get("max_iterations", 0),
                "step": state.get("step", "unknown"),
                "original_prompt": state.get("original_prompt", ""),
                "completion_promise": state.get("completion_promise"),
                "started_at": state.get("started_at")
            }
        
        except Exception as e:
            self._log("ERROR", f"Failed to read session state: {str(e)}")
            return None
    
    def get_session_history(self, session_dir: str) -> List[Dict[str, Any]]:
        """
        Get the history of actions from a Pickle Rick session.
        
        Args:
            session_dir: Path to the session directory
            
        Returns:
            List of history entries
        """
        history_file = Path(session_dir) / "history.jsonl"
        
        if not history_file.exists():
            return []
        
        history = []
        try:
            with open(history_file, "r") as f:
                for line in f:
                    if line.strip():
                        history.append(json.loads(line))
        except Exception as e:
            self._log("ERROR", f"Failed to read session history: {str(e)}")
        
        return history
    
    def generate_report(self, session_dir: str) -> Dict[str, Any]:
        """
        Generate a comprehensive report of a Pickle Rick session.
        
        Args:
            session_dir: Path to the session directory
            
        Returns:
            Dict with session report
        """
        status = self.get_session_status(session_dir)
        history = self.get_session_history(session_dir)
        
        report = {
            "session_dir": session_dir,
            "status": status,
            "history_entries": len(history),
            "report_time": datetime.datetime.utcnow().isoformat(),
            "history": history[:10]  # Last 10 entries
        }
        
        log_proof("PICKLE_RICK_REPORT_GENERATED", {
            "artifact_path": str(Path(session_dir) / "state.json")
        })
        
        return report


def run(
    task_description: str,
    max_iterations: int = 5,
    max_time_minutes: int = 60,
    completion_promise: Optional[str] = None
) -> Dict[str, Any]:
    """
    Run a Pickle Rick session.
    
    Args:
        task_description: Description of the task
        max_iterations: Maximum iterations
        max_time_minutes: Maximum time in minutes
        completion_promise: Optional completion promise
        
    Returns:
        Dict with session information and results
    """
    integration = PickleRickIntegration()
    
    # Verify installation
    if not integration.verify_installation():
        return {"status": "error", "message": "Pickle Rick installation verification failed"}
    
    # Build extension
    if not integration.build_extension():
        return {"status": "error", "message": "Pickle Rick build failed"}
    
    # Run tests
    test_results = integration.run_tests()
    if not test_results.get("passed"):
        return {"status": "error", "message": "Pickle Rick tests failed", "tests": test_results}
    
    # Create session
    session = integration.create_session(
        task_description,
        max_iterations,
        max_time_minutes,
        completion_promise
    )
    
    return {
        "status": "success",
        "session": session,
        "tests": test_results
    }


if __name__ == "__main__":
    # Example usage
    result = run(
        task_description="Test Pickle Rick integration with Usisivac V6",
        max_iterations=3,
        max_time_minutes=30
    )
    print(json.dumps(result, indent=2))
