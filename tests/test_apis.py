"""Test GitHub Models API"""
import sys, os
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

PROMPT = "Say exactly: HELLO_GITHUB_MODELS_OK"

def test_github_models():
    print("=== GITHUB MODELS ===")
    token = os.getenv("GITHUB_TOKEN", "")
    if not token or token.startswith("YOUR"):
        pytest.skip("GITHUB_TOKEN not found or not configured")
    
    import requests
    
    url = "https://models.inference.ai.azure.com/chat/completions"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": 50,
    }

    response = requests.post(url, headers=headers, json=data, timeout=30)

    if response.status_code == 200:
        resp_json = response.json()
        text = resp_json["choices"][0]["message"]["content"]
        print(f"  Response: {text[:100]}")
        print(f"  OK ✓")
        assert True
    else:
        print(f"  ERROR: {response.status_code}")
        print(f"  {response.text[:200]}")
        pytest.fail(f"API returned status code {response.status_code}")

if __name__ == "__main__":
    try:
        test_github_models()
    except pytest.skip.Exception:
        print("Skipped: GITHUB_TOKEN not found")
    except Exception as e:
        print(f"Error: {e}")
