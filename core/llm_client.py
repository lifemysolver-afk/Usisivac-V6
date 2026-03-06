"""
╔══════════════════════════════════════════════════════════════════════╗
║  LLM Client — Multi-Provider (Groq, Mistral, Gemini, OpenRouter)    ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Prioritet: Groq (najbrži) → Mistral → Gemini → OpenRouter → fallback
Sve su FREE tier opcije.
"""

import os, json, time
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass


def _call_groq(prompt: str, model: str = "llama-3.3-70b-versatile", system: str = "") -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=4096)
    return resp.choices[0].message.content


def _call_mistral(prompt: str, model: str = "mistral-small-latest", system: str = "") -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("MISTRAL_API_KEY"),
        base_url="https://api.mistral.ai/v1"
    )
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=4096)
    return resp.choices[0].message.content


def _call_gemini(prompt: str, model: str = "gemini-2.0-flash", system: str = "") -> str:
    """Uses new google-genai SDK with key rotation."""
    from google import genai
    
    # Try all available Gemini keys
    keys = []
    for i in range(1, 5):
        k = os.getenv(f"GEMINI_KEY_{i}", "")
        if k:
            keys.append(k)
    main_key = os.getenv("GEMINI_API_KEY", "")
    if main_key and main_key not in keys:
        keys.append(main_key)
    
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    
    for key in keys:
        try:
            client = genai.Client(api_key=key)
            resp = client.models.generate_content(model=model, contents=full_prompt)
            return resp.text
        except Exception:
            time.sleep(2)
            continue
    raise RuntimeError("All Gemini keys exhausted or rate-limited")


def _call_openrouter(prompt: str, model: str = "mistralai/mistral-7b-instruct:free", system: str = "") -> str:
    from openai import OpenAI
    key = os.getenv("OPENROUTER_API_KEY", "")
    if "\\n" in key:
        key = key.split("\\n")[0]
    client = OpenAI(api_key=key, base_url="https://openrouter.ai/api/v1")
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=4096)
    return resp.choices[0].message.content


# Provider registry with priority order
PROVIDERS = {
    "groq": {"call": _call_groq, "env_key": "GROQ_API_KEY"},
    "mistral": {"call": _call_mistral, "env_key": "MISTRAL_API_KEY"},
    "gemini": {"call": _call_gemini, "env_key": "GEMINI_KEY_1"},
    "openrouter": {"call": _call_openrouter, "env_key": "OPENROUTER_API_KEY"},
}


def call(prompt: str,
         system: str = "",
         provider: str = None,
         model: str = None,
         retries: int = 2) -> str:
    """
    Poziva LLM sa automatskim fallback-om između providera.
    Redosled: groq → mistral → gemini → openrouter
    """
    # Build provider order
    if provider:
        order = [provider]
        # Add fallbacks
        all_p = ["groq", "mistral", "gemini", "openrouter"]
        order += [p for p in all_p if p != provider]
    else:
        pref = os.getenv("PRIMARY_LLM", "groq")
        all_p = ["groq", "mistral", "gemini", "openrouter"]
        order = [pref] + [p for p in all_p if p != pref]

    last_err = None
    for prov in order:
        cfg = PROVIDERS.get(prov)
        if not cfg:
            continue
        
        api_key = os.getenv(cfg["env_key"], "")
        if not api_key or api_key.startswith("YOUR_"):
            continue

        for attempt in range(retries):
            try:
                kwargs = {"prompt": prompt, "system": system}
                if model:
                    kwargs["model"] = model
                return cfg["call"](**kwargs)
            except Exception as e:
                last_err = str(e)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)
                continue

    # Fallback mock
    return json.dumps({
        "status": "MOCK_RESPONSE",
        "note": "All providers failed. Check API keys.",
        "error": last_err,
        "prompt_preview": prompt[:100],
    }, indent=2)
