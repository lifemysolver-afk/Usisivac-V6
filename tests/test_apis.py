"""Test each API provider individually to find which ones work."""
import sys, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

PROMPT = "Say exactly: HELLO_TEST_OK"

def test_groq():
    print("=== GROQ ===")
    key = os.getenv("GROQ_API_KEY", "")
    print(f"  Key present: {bool(key and not key.startswith('YOUR'))}")
    if not key:
        return False
    try:
        from groq import Groq
        client = Groq(api_key=key)
        resp = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=50
        )
        text = resp.choices[0].message.content
        print(f"  Response: {text[:100]}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_mistral():
    print("=== MISTRAL ===")
    key = os.getenv("MISTRAL_API_KEY", "")
    print(f"  Key present: {bool(key and not key.startswith('YOUR'))}")
    if not key:
        return False
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://api.mistral.ai/v1")
        resp = client.chat.completions.create(
            model="mistral-small-latest",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=50
        )
        text = resp.choices[0].message.content
        print(f"  Response: {text[:100]}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_openrouter():
    print("=== OPENROUTER ===")
    key = os.getenv("OPENROUTER_API_KEY", "")
    # Handle potential multi-key in env
    if "\\n" in key:
        key = key.split("\\n")[0]
    print(f"  Key present: {bool(key and not key.startswith('YOUR'))}")
    if not key:
        return False
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
        resp = client.chat.completions.create(
            model="mistralai/mistral-7b-instruct:free",
            messages=[{"role": "user", "content": PROMPT}],
            max_tokens=50
        )
        text = resp.choices[0].message.content
        print(f"  Response: {text[:100]}")
        return True
    except Exception as e:
        print(f"  ERROR: {e}")
        return False

def test_gemini():
    print("=== GEMINI ===")
    keys = [os.getenv(f"GEMINI_KEY_{i}", "") for i in range(1, 5)]
    keys.append(os.getenv("GEMINI_API_KEY", ""))
    keys = [k for k in keys if k and not k.startswith("YOUR")]
    print(f"  Keys found: {len(keys)}")
    if not keys:
        return False
    for i, key in enumerate(keys):
        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            m = genai.GenerativeModel("gemini-1.5-flash")
            resp = m.generate_content(PROMPT)
            text = resp.text
            print(f"  Key {i+1} Response: {text[:100]}")
            return True
        except Exception as e:
            print(f"  Key {i+1} ERROR: {e}")
    return False

if __name__ == "__main__":
    results = {}
    results["groq"] = test_groq()
    print()
    results["mistral"] = test_mistral()
    print()
    results["openrouter"] = test_openrouter()
    print()
    results["gemini"] = test_gemini()
    print()
    print("=== SUMMARY ===")
    for k, v in results.items():
        print(f"  {k}: {'OK' if v else 'FAIL'}")
    working = [k for k, v in results.items() if v]
    print(f"\n  Working providers: {working}")
