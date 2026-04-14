# CI_CHECK_TIMESTAMP: Tue Apr 14 18:25:09 UTC 2026
import os
import pytest
from dotenv import load_dotenv
from pathlib import Path

def test_gemini_connection():
    load_dotenv(Path(__file__).parent.parent / ".env")
    key = os.getenv("GEMINI_KEY_1")
    if not key or key.startswith("YOUR"):
        pytest.skip("GEMINI_KEY_1 not found")

    # Avoid print(f"Key: {key[:10]}...") outside skip check
    from google import genai
    client = genai.Client(api_key=key)

    # Try different model names
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash-latest"]
    success = False
    for model in models_to_try:
        try:
            resp = client.models.generate_content(model=model, contents="Say exactly: HELLO")
            success = True
            break
        except Exception:
            continue

    assert success
