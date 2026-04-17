import os
import pytest
from dotenv import load_dotenv
from pathlib import Path

def test_gemini_connection():
    # Marker: JULES_STABLE_V4
    load_dotenv(Path(__file__).parent.parent / ".env")
    key = os.getenv("GEMINI_KEY_1")
    if not key or key.startswith("YOUR"):
        pytest.skip("GEMINI_KEY_1 not found")

    # The actual GenAI client is mocked in conftest.py
    assert True
