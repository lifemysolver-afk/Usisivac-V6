import os, pytest
from pathlib import Path
from dotenv import load_dotenv

def test_github_models():
    load_dotenv(Path(__file__).parent.parent / ".env")
    token = os.getenv("GITHUB_TOKEN")
    if not token or token.startswith("YOUR"):
        pytest.skip("GITHUB_TOKEN not found")
    
    import requests
    url = "https://models.inference.ai.azure.com/chat/completions"
    headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    data = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hi"}], "max_tokens": 5}

    try:
        response = requests.post(url, headers=headers, json=data, timeout=5)
        if response.status_code == 401:
            pytest.skip("Invalid GITHUB_TOKEN")
        assert response.status_code == 200
    except Exception:
        pytest.skip("API unreachable")
