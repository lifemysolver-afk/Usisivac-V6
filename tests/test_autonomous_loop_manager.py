import inspect
from orchestrator.autonomous_loop import AutonomousLoopManager

class FakeController:
    def __init__(self):
        self.events = []

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
        return []

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
    # implement: initial + 2 retries = 3
    # test: initial + 2 retries = 3
    # debug: 3 failures, but after 3rd failure it stops, so 3 debug calls? 
    # Let's check logic: 
    # 1st fail -> debug_retries_left: 2 -> debug call 1
    # 2nd fail -> debug_retries_left: 1 -> debug call 2
    # 3rd fail -> debug_retries_left: 0 -> debug call 3
    # 4th fail -> debug_retries_left: -1 -> stop
    # With max_debug_retries=2:
    # 1st fail -> left: 1 -> debug 1
    # 2nd fail -> left: 0 -> debug 2
    # 3rd fail -> left: -1 -> stop
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
    # Stack depth should be constant in a loop
    assert max(implement_stack_depths) == min(implement_stack_depths)
