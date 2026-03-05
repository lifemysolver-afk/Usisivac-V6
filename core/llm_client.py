"""
╔══════════════════════════════════════════════════════════════════════╗
║  LLM Client — Multi-Provider (Groq, Mistral, Gemini, OpenRouter)    ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Prioritet: Groq (najbrži) → Mistral → OpenRouter → Gemini → fallback
Sve su FREE tier opcije.
"""

import os, json, time
from pathlib import Path
from typing import Optional

# Učitaj .env
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass

# ─── Provider configs ─────────────────────────────────────────────────────────
PROVIDERS = {
    "groq": {
        "models": [
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        "env_key": "GROQ_API_KEY",
    },
    "mistral": {
        "models": [
            "mistral-small-latest",
            "open-mistral-7b",
            "open-mixtral-8x7b",
        ],
        "env_key": "MISTRAL_API_KEY",
    },
    "openrouter": {
        "models": [
            "mistralai/mistral-7b-instruct:free",
            "google/gemma-2-9b-it:free",
            "meta-llama/llama-3.2-3b-instruct:free",
        ],
        "env_key": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
    },
    "gemini": {
        "models": [
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
        ],
        "env_key": "GEMINI_API_KEY",
    },
}


def _call_groq(prompt: str, model: str, system: str = "") -> str:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=4096)
    return resp.choices[0].message.content


def _call_mistral(prompt: str, model: str, system: str = "") -> str:
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


def _call_openrouter(prompt: str, model: str, system: str = "") -> str:
    from openai import OpenAI
    client = OpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1"
    )
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=4096)
    return resp.choices[0].message.content


def _call_gemini(prompt: str, model: str, system: str = "") -> str:
    import google.generativeai as genai
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    m = genai.GenerativeModel(model)
    full = f"{system}\n\n{prompt}" if system else prompt
    resp = m.generate_content(full)
    return resp.text


def call(prompt: str,
         system: str = "",
         provider: str = None,
         model: str = None,
         retries: int = 3) -> str:
    """
    Poziva LLM sa automatskim fallback-om između providera.
    Redosled: groq → mistral → openrouter → gemini
    """
    # Odredi provider redosled
    order = []
    if provider:
        order = [provider]
    else:
        pref = os.getenv("PRIMARY_LLM", "groq")
        all_p = list(PROVIDERS.keys())
        order = [pref] + [p for p in all_p if p != pref]

    last_err = None
    for prov in order:
        cfg = PROVIDERS.get(prov, {})
        api_key = os.getenv(cfg.get("env_key", ""), "")
        if not api_key or api_key.startswith("YOUR_"):
            continue  # Preskoci ako nema ključa

        use_model = model or cfg["models"][0]
        for attempt in range(retries):
            try:
                if prov == "groq":
                    return _call_groq(prompt, use_model, system)
                elif prov == "mistral":
                    return _call_mistral(prompt, use_model, system)
                elif prov == "openrouter":
                    return _call_openrouter(prompt, use_model, system)
                elif prov == "gemini":
                    return _call_gemini(prompt, use_model, system)
            except Exception as e:
                last_err = str(e)
                if attempt < retries - 1:
                    time.sleep(2 ** attempt)  # exponential backoff
                continue

    # Fallback: lokalni mock za testiranje bez API ključa
    return _mock_response(prompt, last_err)


def _mock_response(prompt: str, error: str = None) -> str:
    """
    Mock odgovor kada nema API ključa.
    Koristi se samo za lokalno testiranje.
    """
    return json.dumps({
        "status": "MOCK_RESPONSE",
        "note": "No API key configured. Add keys to .env file.",
        "error": error,
        "prompt_preview": prompt[:100],
        "suggestion": "Register free at: https://console.groq.com/"
    }, indent=2)
