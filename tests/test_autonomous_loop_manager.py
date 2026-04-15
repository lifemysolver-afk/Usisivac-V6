import inspect

from orchestrator.autonomous_loop import AutonomousLoopManager


class FakeController:
    def __init__(self):
        self.events = []
        self._complete = False
        self.current_iteration = 0
        self._history = []

    def is_complete(self):
        return self._complete

    def start_iteration(self):
        if self.current_iteration >= 1:
            return None
        self.current_iteration += 1
        return f"iter_{self.current_iteration}"

    def end_iteration(self, iteration_id, duration):
        pass

    def save_checkpoint(self):
        pass

    def get_status(self):
        return {"state": "running", "iteration": self.current_iteration}

    def record_plan(self, iteration_id, plan):
        self.events.append(("plan", iteration_id, plan))
        return True

    def record_implementation(self, iteration_id, implementation):
        self.events.append(("implementation", iteration_id, implementation))
        return True

    def record_test_results(self, iteration_id, test_results):
        self.events.append(("test", iteration_id, test_results))
        return test_results.get("failed", 0) == 0

    def record_debug_info(self, iteration_id, debug_info):
        self.events.append(("debug", iteration_id, debug_info))
        return True

    def record_review(self, iteration_id, review_feedback):
        self.events.append(("review", iteration_id, review_feedback))
        return True

    def record_commit(self, iteration_id, commit_hash):
        self.events.append(("commit", iteration_id, commit_hash))
        return True

    def record_reflection(self, iteration_id, reflection):
        self.events.append(("reflection", iteration_id, reflection))
        return True

    def get_history(self):
        return self._history


def _build_manager(max_debug_retries=3):
    manager = AutonomousLoopManager(
        task_id="test_task",
        task_objective="test objective",
        max_iterations=1,
        max_debug_retries=max_debug_retries,
    )
    manager.controller = FakeController()
    return manager


def test_run_iteration_recovers_within_retry_limit():
    manager = _build_manager(max_debug_retries=2)

    test_results_sequence = iter([
        {"passed": 0, "failed": 1},
        {"passed": 1, "failed": 0},
    ])
    counters = {"implement": 0, "test": 0, "debug": 0}

    def call_agent(agent_name, *args, **kwargs):
        if agent_name == "plan_agent":
            return {"steps": ["s1"]}
        if agent_name == "implement_agent":
            counters["implement"] += 1
            return {"files_modified": 1}
        if agent_name == "test_agent":
            counters["test"] += 1
            return next(test_results_sequence)
        if agent_name == "debug_agent":
            counters["debug"] += 1
            return {"fixes": ["f1"]}
        if agent_name == "review_agent":
            return {"issues": []}
        if agent_name == "commit_agent":
            return "abc123"
        if agent_name == "reflection_agent":
            return {"summary": "ok"}
        raise AssertionError(f"Unexpected agent: {agent_name}")

    manager._call_agent = call_agent
    assert manager._run_iteration("iter_1") is True
    assert counters == {"implement": 2, "test": 2, "debug": 1}


def test_run_iteration_stops_after_max_debug_retries():
    manager = _build_manager(max_debug_retries=2)

    counters = {"implement": 0, "test": 0, "debug": 0}
    log_messages = []

    def capture_log(level, message, data=None):
        log_messages.append((level, message, data or {}))

    manager._log = capture_log

    def call_agent(agent_name, *args, **kwargs):
        if agent_name == "plan_agent":
            return {"steps": ["s1"]}
        if agent_name == "implement_agent":
            counters["implement"] += 1
            return {"files_modified": 1}
        if agent_name == "test_agent":
            counters["test"] += 1
            return {"passed": 0, "failed": 1}
        if agent_name == "debug_agent":
            counters["debug"] += 1
            return {"fixes": ["retry"]}
        return None

    manager._call_agent = call_agent
    assert manager._run_iteration("iter_2") is False
    assert counters == {"implement": 3, "test": 3, "debug": 2}
    assert any("Debug retry limit exhausted" in message for _, message, _ in log_messages)


def test_run_iteration_retry_loop_does_not_grow_stack_depth():
    manager = _build_manager(max_debug_retries=3)

    implement_stack_depths = []
    test_results_sequence = iter([
        {"passed": 0, "failed": 1},
        {"passed": 0, "failed": 1},
        {"passed": 1, "failed": 0},
    ])

    def call_agent(agent_name, *args, **kwargs):
        if agent_name == "plan_agent":
            return {"steps": ["s1"]}
        if agent_name == "implement_agent":
            implement_stack_depths.append(len(inspect.stack()))
            return {"files_modified": 1}
        if agent_name == "test_agent":
            return next(test_results_sequence)
        if agent_name == "debug_agent":
            return {"fixes": ["f1"]}
        if agent_name == "review_agent":
            return {"issues": []}
        if agent_name == "commit_agent":
            return "abc123"
        if agent_name == "reflection_agent":
            return {"summary": "ok"}
        raise AssertionError(f"Unexpected agent: {agent_name}")

    manager._call_agent = call_agent
    assert manager._run_iteration("iter_3") is True
    assert len(implement_stack_depths) == 3
    assert max(implement_stack_depths) - min(implement_stack_depths) <= 2


# ── Early-exit when required agents return None ───────────────────────────────

def test_run_iteration_returns_false_when_plan_agent_returns_none():
    """If plan_agent returns None the iteration must short-circuit to False."""
    manager = _build_manager()

    def call_agent(agent_name, *args, **kwargs):
        if agent_name == "plan_agent":
            return None
        raise AssertionError(f"Unexpected agent called: {agent_name}")

    manager._call_agent = call_agent
    assert manager._run_iteration("iter_none_plan") is False


def test_run_iteration_returns_false_when_implement_agent_returns_none():
    """If implement_agent returns None the iteration must short-circuit to False."""
    manager = _build_manager()

    def call_agent(agent_name, *args, **kwargs):
        if agent_name == "plan_agent":
            return {"steps": ["s1"]}
        if agent_name == "implement_agent":
            return None
        raise AssertionError(f"Unexpected agent called: {agent_name}")

    manager._call_agent = call_agent
    assert manager._run_iteration("iter_none_impl") is False


def test_run_iteration_returns_false_when_test_agent_returns_none():
    """If test_agent returns None the iteration must short-circuit to False."""
    manager = _build_manager()

    def call_agent(agent_name, *args, **kwargs):
        if agent_name == "plan_agent":
            return {"steps": ["s1"]}
        if agent_name == "implement_agent":
            return {"files_modified": 1}
        if agent_name == "test_agent":
            return None
        raise AssertionError(f"Unexpected agent called: {agent_name}")

    manager._call_agent = call_agent
    assert manager._run_iteration("iter_none_test") is False


def test_run_iteration_returns_false_when_debug_agent_returns_none():
    """If debug_agent returns None after a test failure, iteration must return False."""
    manager = _build_manager(max_debug_retries=1)

    def call_agent(agent_name, *args, **kwargs):
        if agent_name == "plan_agent":
            return {"steps": ["s1"]}
        if agent_name == "implement_agent":
            return {"files_modified": 1}
        if agent_name == "test_agent":
            return {"passed": 0, "failed": 1}
        if agent_name == "debug_agent":
            return None
        raise AssertionError(f"Unexpected agent called: {agent_name}")

    manager._call_agent = call_agent
    assert manager._run_iteration("iter_none_debug") is False


# ── Debug retry boundary: max_debug_retries=0 ────────────────────────────────

def test_run_iteration_with_zero_max_debug_retries_fails_immediately():
    """With max_debug_retries=0, any test failure immediately terminates the iteration."""
    manager = _build_manager(max_debug_retries=0)
    debug_called = []

    def call_agent(agent_name, *args, **kwargs):
        if agent_name == "plan_agent":
            return {"steps": ["s1"]}
        if agent_name == "implement_agent":
            return {"files_modified": 1}
        if agent_name == "test_agent":
            return {"passed": 0, "failed": 1}
        if agent_name == "debug_agent":
            debug_called.append(1)
            return {"fixes": ["x"]}
        return None

    manager._call_agent = call_agent
    result = manager._run_iteration("iter_zero_retries")
    assert result is False
    assert debug_called == [], "debug_agent must not be called when max_debug_retries=0"


# ── Reflection indicating task completion ─────────────────────────────────────

def test_run_iteration_succeeds_when_reflection_indicates_completion():
    """
    When record_reflection returns False the loop should treat the iteration
    as successful (task is done).
    """
    manager = _build_manager()

    class CompletingController(FakeController):
        def record_reflection(self, iteration_id, reflection):
            super().record_reflection(iteration_id, reflection)
            return False  # signals task complete

    manager.controller = CompletingController()

    def call_agent(agent_name, *args, **kwargs):
        if agent_name == "plan_agent":
            return {"steps": ["s1"]}
        if agent_name == "implement_agent":
            return {"files_modified": 1}
        if agent_name == "test_agent":
            return {"passed": 1, "failed": 0}
        if agent_name in ("review_agent", "commit_agent"):
            return {"issues": []} if agent_name == "review_agent" else "commit-xyz"
        if agent_name == "reflection_agent":
            return {"summary": "done"}
        raise AssertionError(f"Unexpected agent: {agent_name}")

    manager._call_agent = call_agent
    assert manager._run_iteration("iter_complete") is True


# ── get_metrics ───────────────────────────────────────────────────────────────

def test_get_metrics_returns_zero_success_rate_with_no_iterations():
    manager = _build_manager()
    metrics = manager.get_metrics()
    assert metrics["total_iterations"] == 0
    assert metrics["success_rate"] == 0.0
    assert metrics["successful_iterations"] == 0
    assert metrics["failed_iterations"] == 0


def test_get_metrics_computes_success_rate_correctly():
    manager = _build_manager()
    manager.metrics["total_iterations"] = 4
    manager.metrics["successful_iterations"] = 3
    manager.metrics["failed_iterations"] = 1
    metrics = manager.get_metrics()
    assert metrics["success_rate"] == 75.0


def test_get_metrics_all_failed_gives_zero_success_rate():
    manager = _build_manager()
    manager.metrics["total_iterations"] = 2
    manager.metrics["successful_iterations"] = 0
    manager.metrics["failed_iterations"] = 2
    metrics = manager.get_metrics()
    assert metrics["success_rate"] == 0.0


# ── get_status ────────────────────────────────────────────────────────────────

def test_get_status_contains_expected_keys():
    manager = _build_manager()
    status = manager.get_status()
    assert "task_id" in status
    assert "objective" in status
    assert "controller_status" in status
    assert "metrics" in status
    assert "log_path" in status
    assert status["task_id"] == "test_task"
    assert status["objective"] == "test objective"


# ── resume_from_checkpoint ────────────────────────────────────────────────────

def test_resume_from_checkpoint_returns_false_when_no_checkpoint(monkeypatch):
    """If no checkpoint exists, resume_from_checkpoint must return False."""
    from orchestrator import loop_controller as lc

    monkeypatch.setattr(lc.LoopController, "load_checkpoint", staticmethod(lambda task_id: None))
    manager = _build_manager()
    result = manager.resume_from_checkpoint()
    assert result is False


# ── _call_agent with no registered callbacks ──────────────────────────────────

def test_call_agent_returns_none_when_callback_not_registered():
    """_call_agent must return None and log a WARN for unknown agent names."""
    manager = _build_manager()
    log_messages = []

    def capture_log(level, message, data=None):
        log_messages.append((level, message))

    manager._log = capture_log
    result = manager._call_agent("nonexistent_agent", "iter_x")
    assert result is None
    assert any("WARN" == level for level, _ in log_messages)


# ── Regression: review_agent None does not crash iteration ───────────────────

def test_run_iteration_succeeds_when_review_agent_returns_none():
    """review_agent is optional — a None result must not fail the iteration."""
    manager = _build_manager()

    def call_agent(agent_name, *args, **kwargs):
        if agent_name == "plan_agent":
            return {"steps": ["s1"]}
        if agent_name == "implement_agent":
            return {"files_modified": 1}
        if agent_name == "test_agent":
            return {"passed": 1, "failed": 0}
        if agent_name == "review_agent":
            return None  # optional — should be tolerated
        if agent_name == "commit_agent":
            return "commit-abc"
        if agent_name == "reflection_agent":
            return {"summary": "ok"}
        raise AssertionError(f"Unexpected agent: {agent_name}")

    manager._call_agent = call_agent
    assert manager._run_iteration("iter_no_review") is True