"""
╔══════════════════════════════════════════════════════════════════════╗
║  Test: VetoBoard — PR Changes                                       ║
║  Covers changes introduced in the parallelized VetoBoard PR:        ║
║    1. Thread error message format changed to "Thread error: {e}"    ║
║    2. LEGAL VETO from LLM now includes "reasonings" key in result   ║
║    3. LEGAL VETO reason uses get('LEGAL') with no default (→ None)  ║
║    4. All personas are submitted to ThreadPoolExecutor               ║
║    5. Quorum result always includes "reasonings" key                 ║
╚══════════════════════════════════════════════════════════════════════╝
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock
from concurrent.futures import Future

sys.path.insert(0, str(Path(__file__).parent.parent))

from loptica.veto_board import VetoBoard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_future_returning(value):
    """Return a completed Future whose result() returns `value`."""
    f = Future()
    f.set_result(value)
    return f


def _make_future_raising(exc):
    """Return a completed Future whose result() raises `exc`."""
    f = Future()
    f.set_exception(exc)
    return f


# ---------------------------------------------------------------------------
# 1. Thread error message format
# ---------------------------------------------------------------------------

def test_thread_error_message_does_not_include_defaulting_pass():
    """Thread exceptions should produce 'Thread error: ...' (no 'defaulting PASS')."""
    board = VetoBoard(use_llm=True)
    err = RuntimeError("network timeout")

    # Patch _get_vote to raise for ALL personas so we can observe message format
    with patch.object(board, "_get_vote", side_effect=err):
        result = board.evaluate_action("Fetch training data from S3")

    # Every persona should have failed with the new message format
    for persona, reasoning in result["reasonings"].items():
        assert reasoning.startswith("Thread error:"), (
            f"Persona {persona} reasoning should start with 'Thread error:' but got: {reasoning!r}"
        )
        assert "defaulting PASS" not in reasoning, (
            f"Old message format found in reasoning for {persona}: {reasoning!r}"
        )


def test_thread_error_vote_defaults_to_pass():
    """A persona whose future raises an exception should still vote PASS."""
    board = VetoBoard(use_llm=True)

    # Only CRITIC raises; others return PASS
    def _get_vote_side_effect(persona, persona_prompt, action, context):
        if persona == "CRITIC":
            raise ValueError("critic exploded")
        return "PASS", f"{persona} is fine"

    with patch.object(board, "_get_vote", side_effect=_get_vote_side_effect):
        result = board.evaluate_action("Run hyperparameter sweep")

    assert result["votes"]["CRITIC"] == "PASS", (
        "Thread exception for a persona should default its vote to PASS"
    )
    assert "Thread error:" in result["reasonings"]["CRITIC"]
    assert "critic exploded" in result["reasonings"]["CRITIC"]


def test_thread_error_message_includes_exception_text():
    """The thread error reasoning must embed the original exception message."""
    board = VetoBoard(use_llm=True)
    unique_msg = "unique-exception-text-xyz"

    def _raise_for_cto(persona, persona_prompt, action, context):
        if persona == "CTO":
            raise IOError(unique_msg)
        return "PASS", "ok"

    with patch.object(board, "_get_vote", side_effect=_raise_for_cto):
        result = board.evaluate_action("Deploy model to production")

    assert unique_msg in result["reasonings"]["CTO"], (
        "Exception message must appear in the thread error reasoning"
    )


# ---------------------------------------------------------------------------
# 2. LEGAL VETO from LLM includes "reasonings" key
# ---------------------------------------------------------------------------

def test_llm_legal_veto_result_contains_reasonings_key():
    """When LLM LEGAL persona votes VETO, result dict must include 'reasonings'."""
    board = VetoBoard(use_llm=True)

    def _get_vote_legal_veto(persona, persona_prompt, action, context):
        if persona == "LEGAL":
            return "VETO", "Detected credential exposure"
        return "PASS", f"{persona} approves"

    with patch.object(board, "_get_vote", side_effect=_get_vote_legal_veto):
        result = board.evaluate_action("Print all environment variables to log")

    assert "reasonings" in result, (
        "LEGAL VETO result must include 'reasonings' key (added in this PR)"
    )


def test_llm_legal_veto_reasonings_contains_all_personas():
    """'reasonings' dict in a LEGAL VETO result should cover all personas that ran."""
    board = VetoBoard(use_llm=True)

    def _get_vote_fn(persona, persona_prompt, action, context):
        if persona == "LEGAL":
            return "VETO", "Sensitive data exposure"
        return "PASS", f"{persona} ok"

    with patch.object(board, "_get_vote", side_effect=_get_vote_fn):
        result = board.evaluate_action("Export user emails to public bucket")

    reasonings = result["reasonings"]
    for persona in VetoBoard.PERSONAS:
        assert persona in reasonings, f"Missing reasoning for persona: {persona}"


def test_llm_legal_veto_verdict_and_structure():
    """LEGAL VETO from LLM must set verdict=VETO and quorum_reached=False."""
    board = VetoBoard(use_llm=True)

    def _get_vote_fn(persona, persona_prompt, action, context):
        return ("VETO", "Legal issue") if persona == "LEGAL" else ("PASS", "ok")

    with patch.object(board, "_get_vote", side_effect=_get_vote_fn):
        result = board.evaluate_action("Share private keys in API response")

    assert result["verdict"] == "VETO"
    assert result["quorum_reached"] is False
    assert "votes" in result
    assert result["votes"]["LEGAL"] == "VETO"


# ---------------------------------------------------------------------------
# 3. LEGAL VETO reason uses get('LEGAL') with no default → may be None
# ---------------------------------------------------------------------------

def test_llm_legal_veto_reason_string_contains_reasoning():
    """Reason string should embed the LEGAL persona's reasoning text."""
    board = VetoBoard(use_llm=True)
    legal_reasoning = "Path traversal vulnerability detected"

    def _get_vote_fn(persona, persona_prompt, action, context):
        return ("VETO", legal_reasoning) if persona == "LEGAL" else ("PASS", "ok")

    with patch.object(board, "_get_vote", side_effect=_get_vote_fn):
        result = board.evaluate_action("Read file at ../../secret.txt")

    assert legal_reasoning in result["reason"], (
        "LEGAL reasoning text should appear in the result 'reason' field"
    )
    assert result["reason"].startswith("LEGAL VETO:")


def test_llm_legal_veto_reason_when_reasoning_is_none():
    """
    With no default in get('LEGAL'), if reasoning is somehow None the reason
    string becomes 'LEGAL VETO: None'. Verify the code doesn't crash.
    """
    board = VetoBoard(use_llm=True)

    # Simulate a future that returns ("VETO", None) for LEGAL
    def _get_vote_fn(persona, persona_prompt, action, context):
        if persona == "LEGAL":
            return "VETO", None
        return "PASS", "ok"

    with patch.object(board, "_get_vote", side_effect=_get_vote_fn):
        result = board.evaluate_action("Access internal network config")

    # Should not raise; reason will contain "None" as string
    assert result["verdict"] == "VETO"
    assert "LEGAL VETO:" in result["reason"]
    # The reason should be "LEGAL VETO: None" (no fallback default string)
    assert result["reason"] == "LEGAL VETO: None"


def test_llm_legal_veto_reason_no_fallback_default_string():
    """
    The PR removed the fallback 'No reason provided' default.
    Verify that when LEGAL reasoning is missing/None, the string is NOT
    'LEGAL VETO: No reason provided'.
    """
    board = VetoBoard(use_llm=True)

    def _get_vote_fn(persona, persona_prompt, action, context):
        if persona == "LEGAL":
            return "VETO", None
        return "PASS", "ok"

    with patch.object(board, "_get_vote", side_effect=_get_vote_fn):
        result = board.evaluate_action("Drop production database")

    assert "No reason provided" not in result["reason"], (
        "Fallback 'No reason provided' string was removed in this PR and should not appear"
    )


# ---------------------------------------------------------------------------
# 4. ThreadPoolExecutor: all PERSONAS are submitted
# ---------------------------------------------------------------------------

def test_all_personas_are_evaluated_in_parallel():
    """evaluate_action must call _get_vote for every persona in PERSONAS."""
    board = VetoBoard(use_llm=True)
    called_personas = []

    def _get_vote_fn(persona, persona_prompt, action, context):
        called_personas.append(persona)
        return "PASS", f"{persona} ok"

    with patch.object(board, "_get_vote", side_effect=_get_vote_fn):
        board.evaluate_action("Train a random forest classifier")

    assert set(called_personas) == set(VetoBoard.PERSONAS.keys()), (
        f"Expected all personas {set(VetoBoard.PERSONAS.keys())} to be called, "
        f"got {set(called_personas)}"
    )


def test_all_personas_called_exactly_once():
    """Each persona should be submitted to the executor exactly once."""
    board = VetoBoard(use_llm=True)
    call_counts: dict = {}

    def _get_vote_fn(persona, persona_prompt, action, context):
        call_counts[persona] = call_counts.get(persona, 0) + 1
        return "PASS", "ok"

    with patch.object(board, "_get_vote", side_effect=_get_vote_fn):
        board.evaluate_action("Evaluate model on validation set")

    for persona in VetoBoard.PERSONAS:
        assert call_counts.get(persona, 0) == 1, (
            f"Persona {persona} should be called exactly once, got {call_counts.get(persona, 0)}"
        )


# ---------------------------------------------------------------------------
# 5. Quorum result always includes "reasonings" key
# ---------------------------------------------------------------------------

def test_quorum_pass_result_includes_reasonings():
    """A quorum PASS result should include the 'reasonings' key."""
    board = VetoBoard(use_llm=True)

    with patch.object(board, "_get_vote", return_value=("PASS", "looks good")):
        result = board.evaluate_action("Load dataset from disk")

    assert "reasonings" in result
    assert result["verdict"] == "PASS"
    assert result["quorum_reached"] is True


def test_quorum_veto_result_includes_reasonings():
    """A quorum VETO result (insufficient PASS votes) should include 'reasonings'."""
    board = VetoBoard(use_llm=True)

    # All personas vote VETO → quorum not reached
    with patch.object(board, "_get_vote", return_value=("VETO", "risky")):
        result = board.evaluate_action("Delete all checkpoints immediately")

    assert "reasonings" in result
    assert result["verdict"] == "VETO"
    assert result["quorum_reached"] is False


# ---------------------------------------------------------------------------
# 6. Quorum boundary cases
# ---------------------------------------------------------------------------

def test_exactly_quorum_of_3_passes():
    """Exactly 3 PASS votes (the QUORUM value) should yield verdict PASS."""
    board = VetoBoard(use_llm=True)
    personas = list(VetoBoard.PERSONAS.keys())  # 5 total

    def _get_vote_fn(persona, persona_prompt, action, context):
        # First 3 personas pass, last 2 veto
        if persona in personas[:3]:
            return "PASS", "ok"
        return "VETO", "no"

    with patch.object(board, "_get_vote", side_effect=_get_vote_fn):
        result = board.evaluate_action("Run cross-validation")

    pass_count = sum(1 for v in result["votes"].values() if v == "PASS")
    assert pass_count == 3
    assert result["verdict"] == "PASS"
    assert result["quorum_reached"] is True


def test_below_quorum_2_passes_yields_veto():
    """2 PASS votes (below quorum of 3) should yield verdict VETO."""
    board = VetoBoard(use_llm=True)
    personas = list(VetoBoard.PERSONAS.keys())

    def _get_vote_fn(persona, persona_prompt, action, context):
        if persona in personas[:2]:
            return "PASS", "ok"
        return "VETO", "no"

    with patch.object(board, "_get_vote", side_effect=_get_vote_fn):
        result = board.evaluate_action("Retrain model with unvalidated data")

    pass_count = sum(1 for v in result["votes"].values() if v == "PASS")
    assert pass_count == 2
    assert result["verdict"] == "VETO"
    assert result["quorum_reached"] is False


# ---------------------------------------------------------------------------
# 7. Regression: deterministic LEGAL veto still bypasses LLM (pre-existing,
#    included as regression guard since the PR touches that code path)
# ---------------------------------------------------------------------------

def test_deterministic_legal_veto_does_not_call_get_vote():
    """Keyword-based instant LEGAL veto must short-circuit before any LLM call."""
    board = VetoBoard(use_llm=True)
    call_log = []

    def _get_vote_fn(persona, persona_prompt, action, context):
        call_log.append(persona)
        return "PASS", "ok"

    with patch.object(board, "_get_vote", side_effect=_get_vote_fn):
        result = board.evaluate_action("Store the password in plaintext")

    assert result["verdict"] == "VETO"
    assert len(call_log) == 0, (
        "_get_vote should never be called when keyword-based LEGAL veto fires"
    )


def test_deterministic_legal_veto_result_structure():
    """Instant LEGAL veto result should have known keys and no 'reasonings'."""
    board = VetoBoard(use_llm=False)
    result = board.evaluate_action("Execute: rm -rf /")

    assert result["verdict"] == "VETO"
    assert "LEGAL VETO" in result["reason"]
    assert result["quorum_reached"] is False
    # Instant veto has no reasonings (added only in LLM path by this PR)
    assert "reasonings" not in result
