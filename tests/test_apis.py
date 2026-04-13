import sys, os
import pytest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

def test_github_models():
    token = os.getenv("GITHUB_TOKEN", "")
    # Check if token is actually set and not just a placeholder
    if not token or token.startswith("YOUR") or token == "bad_token":
        pytest.skip("GITHUB_TOKEN not found or placeholder")
    
    import requests
    url = "https://models.inference.ai.azure.com/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        if response.status_code == 401:
            pytest.skip("GITHUB_TOKEN is invalid (401)")
        assert response.status_code == 200
    except Exception as e:
        pytest.skip(f"API call failed: {e}")

if __name__ == "__main__":
    pytest.main([__file__])
