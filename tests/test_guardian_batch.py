import json
import pytest
from unittest.mock import patch, MagicMock
from guardian.guardian import compute_drift_score, compute_drift_scores_batch

def test_compute_drift_scores_batch_accuracy():
    """Verify compute_drift_scores_batch produces accurate results matching compute_drift_score."""
    project_essence = "Build an end-to-end Machine Learning pipeline for tabular classification using LightGBM."
    agent_results = {
        "ResearchAgent": {"status": "SUCCESS", "notes": "Researched feature engineering techniques for LightGBM."},
        "CoderAgent": {"status": "SUCCESS", "file": "train.py", "notes": "Implemented LightGBM baseline training script."},
        "CriticAgent": {"status": "CRITIQUE_DONE", "notes": "Reviewed cross-validation strategy for data leakage."}
    }

    # Compute batch scores
    batch_scores = compute_drift_scores_batch(agent_results, project_essence)

    assert len(batch_scores) == 3
    assert set(batch_scores.keys()) == set(agent_results.keys())

    # Verify each score matches compute_drift_score within small tolerance
    for agent_name, result in agent_results.items():
        desc = json.dumps(result, default=str)[:500]
        single_score = compute_drift_score(desc, project_essence)
        assert abs(batch_scores[agent_name] - single_score) < 1e-3


def test_compute_drift_scores_batch_empty():
    """Verify batch calculation handles empty or invalid inputs."""
    project_essence = "Test goal"
    assert compute_drift_scores_batch({}, project_essence) == {}
    assert compute_drift_scores_batch({"AgentA": "not a dict"}, project_essence) == {}


def test_compute_drift_scores_batch_mock_fallback():
    """Verify batch calculation delegates to compute_drift_score when it is mocked."""
    project_essence = "Test goal"
    agent_results = {
        "Agent1": {"data": "test1"},
        "Agent2": {"data": "test2"}
    }

    with patch("guardian.guardian.compute_drift_score", return_value=0.15) as mock_compute:
        scores = compute_drift_scores_batch(agent_results, project_essence)
        assert mock_compute.called
        assert scores == {"Agent1": 0.15, "Agent2": 0.15}
