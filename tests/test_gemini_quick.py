import os
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent.parent / ".env")

key = os.getenv("GEMINI_KEY_1")
print(f"Key: {key[:10]}...")

from google import genai
client = genai.Client(api_key=key)

# Try different model names
models_to_try = ["gemini-2.0-flash", "gemini-1.5-flash-latest", "gemini-2.0-flash-lite"]
for model in models_to_try:
    try:
        resp = client.models.generate_content(model=model, contents="Say exactly: HELLO_TEST_OK")
        print(f"  {model}: {resp.text[:80]}")
        break
    except Exception as e:
        print(f"  {model}: FAIL - {e}")
