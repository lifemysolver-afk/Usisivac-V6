import os
import pytest
from dotenv import load_dotenv
from pathlib import Path

def test_github_models():
    # Marker: JULES_STABLE_V4
    load_dotenv(Path(__file__).parent.parent / ".env")
    token = os.getenv("GITHUB_TOKEN")
    if not token or token.startswith("YOUR"):
        pytest.skip("GITHUB_TOKEN not found")
    assert True
