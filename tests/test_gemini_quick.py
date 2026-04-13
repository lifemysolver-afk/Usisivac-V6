import os
import pytest
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(Path(__file__).parent.parent / ".env")
key = os.getenv("GEMINI_KEY_1")

def test_gemini_connection():
    if not key:
        pytest.skip("GEMINI_KEY_1 not found")

    from google import genai
    client = genai.Client(api_key=key)

    # Try different model names
    models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-2.0-flash-lite"]
    success = False
    for model in models_to_try:
        try:
            resp = client.models.generate_content(model=model, contents="Say exactly: HELLO_TEST_OK")
            print(f"  {model}: {resp.text[:80]}")
            success = True
            break
        except Exception as e:
            print(f"  {model}: FAIL - {e}")

    assert success, "Failed to connect to any Gemini model"

if __name__ == "__main__":
    if key:
        print(f"Key: {key[:10]}...")
        test_gemini_connection()
    else:
        print("GEMINI_KEY_1 not found in environment.")
