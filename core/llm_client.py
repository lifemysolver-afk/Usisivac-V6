"""
╔══════════════════════════════════════════════════════════════════════╗
║  LLM Client — Multi-Provider (Groq, Mistral, Gemini, OpenRouter)    ║
║  Usisivac V6 | Trinity Protocol                                     ║
╚══════════════════════════════════════════════════════════════════════╝

Prioritet: configurable preko PRIMARY_LLM.
Ako je ONLY_PRIMARY_LLM=true, koristi se isključivo primarni provider.
"""

import os, json, time, functools
from pathlib import Path
from typing import Optional
import requests

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except Exception:
    pass


@functools.lru_cache(maxsize=10)
def _get_groq_client(api_key: str):
    from groq import Groq
    return Groq(api_key=api_key)


@functools.lru_cache(maxsize=10)
def _get_openai_client(api_key: str, base_url: str = None):
    from openai import OpenAI
    return OpenAI(api_key=api_key, base_url=base_url)


@functools.lru_cache(maxsize=10)
def _get_gemini_client(api_key: str):
    from google import genai
    return genai.Client(api_key=api_key)


def _call_groq(prompt: str, model: str = "llama-3.3-70b-versatile", system: str = "") -> str:
    client = _get_groq_client(os.getenv("GROQ_API_KEY"))
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=4096)
    return resp.choices[0].message.content


def _call_mistral(prompt: str, model: str = "mistral-small-latest", system: str = "") -> str:
    client = _get_openai_client(
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
            client = _get_gemini_client(key)
            resp = client.models.generate_content(model=model, contents=full_prompt)
            return resp.text
        except Exception:
            time.sleep(2)
            continue
    raise RuntimeError("All Gemini keys exhausted or rate-limited")


def _call_openrouter(prompt: str, model: str = "mistralai/mistral-7b-instruct:free", system: str = "") -> str:
    key = os.getenv("OPENROUTER_API_KEY", "")
    if "\\n" in key:
        key = key.split("\\n")[0]
    client = _get_openai_client(api_key=key, base_url="https://openrouter.ai/api/v1")
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    resp = client.chat.completions.create(model=model, messages=msgs, max_tokens=4096)
    return resp.choices[0].message.content


def _call_huggingface(prompt: str, model: str = "HuggingFaceH4/zephyr-7b-beta", system: str = "") -> str:
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise ValueError("HF_API_KEY not set for Hugging Face provider")

    headers = {"Authorization": f"Bearer {api_key}"}
    API_URL = f"https://api-inference.huggingface.co/models/{model}"

    payload = {
        "inputs": f"{system}\n\n{prompt}",
        "parameters": {"max_new_tokens": 4096}
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()[0]["generated_text"]


# Provider registry with priority order
PROVIDERS = {
    "groq": {"call": _call_groq, "env_key": "GROQ_API_KEY"},
    "mistral": {"call": _call_mistral, "env_key": "MISTRAL_API_KEY"},
    "gemini": {"call": _call_gemini, "env_key": "GEMINI_KEY_1"},
    "openrouter": {"call": _call_openrouter, "env_key": "OPENROUTER_API_KEY"},
    "huggingface": {"call": _call_huggingface, "env_key": "HF_API_KEY"},
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
    all_providers = ["groq", "mistral", "gemini", "openrouter", "huggingface"]
    only_primary_llm = os.getenv("ONLY_PRIMARY_LLM", "false").lower() in {"1", "true", "yes", "on"}
    primary_llm = provider or os.getenv("PRIMARY_LLM", "gemini")

    # Build provider order based on PRIMARY_LLM and ONLY_PRIMARY_LLM
    if only_primary_llm:
        order = [primary_llm]
    else:
        order = [primary_llm] + [p for p in all_providers if p != primary_llm]

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
                chosen_model = model or os.getenv("PRIMARY_MODEL")
                if not chosen_model and prov == "gemini":
                    chosen_model = "gemini-2.5-flash" # Default for Gemini if not specified
                if chosen_model:
                    kwargs["model"] = chosen_model
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
