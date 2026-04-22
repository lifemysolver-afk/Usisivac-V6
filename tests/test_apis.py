"""Test GitHub Models API"""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

PROMPT = "Say exactly: HELLO_GITHUB_MODELS_OK"


def test_github_models():
    print("=== GITHUB MODELS ===")
    token = os.getenv("GITHUB_TOKEN", "")
    print(f"  Token present: {bool(token and not token.startswith('YOUR'))}")
    
    if not token:
        print("  FAIL: GITHUB_TOKEN not found")
        return False
    
    try:
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
            print(f"  OK ")
            return True
        else:
            print(f"  ERROR: {response.status_code}")
            print(f"  {response.text[:200]}")
            return False
            
    except Exception as e:
        print(f"  ERROR: {e}")
        return False


if __name__ == "__main__":
    results = {}
    results["github_models"] = test_github_models()
    
    print("\n=== SUMMARY ===")
    status = "OK " if results["github_models"] else "FAIL "
    print(f"  GitHub Models: {status}")
