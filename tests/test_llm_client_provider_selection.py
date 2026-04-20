from core import llm_client


def test_call_uses_only_primary_provider_when_enabled(monkeypatch):
    calls = []

    monkeypatch.setenv("ONLY_PRIMARY_LLM", "true")
    monkeypatch.setenv("PRIMARY_LLM", "gemini")
    monkeypatch.setenv("GEMINI_KEY_1", "dummy-key")
    monkeypatch.setenv("PRIMARY_MODEL", "gemini-2.5-flash")

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: calls.append(kwargs) or "ok-gemini"
        },
        "groq": {
            "env_key": "GROQ_API_KEY",
            "call": lambda **kwargs: "ok-groq"
        }
    }
    try:
        result = llm_client.call("hello")
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-gemini"
    assert len(calls) == 1
    assert calls[0]["model"] == "gemini-2.5-flash"


def test_call_sets_default_gemini_model_when_missing(monkeypatch):
    calls = []

    monkeypatch.setenv("ONLY_PRIMARY_LLM", "true")
    monkeypatch.setenv("PRIMARY_LLM", "gemini")
    monkeypatch.setenv("GEMINI_KEY_1", "dummy-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: calls.append(kwargs) or "ok-gemini"
        }
    }
    try:
        result = llm_client.call("hello")
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-gemini"
    assert calls[0]["model"] == "gemini-2.5-flash"


def test_call_uses_huggingface_provider(monkeypatch):
    calls = []

    monkeypatch.setenv("ONLY_PRIMARY_LLM", "true")
    monkeypatch.setenv("PRIMARY_LLM", "huggingface")
    monkeypatch.setenv("HF_API_KEY", "dummy-hf-key")
    monkeypatch.setenv("PRIMARY_MODEL", "some-hf-model")

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "huggingface": {
            "env_key": "HF_API_KEY",
            "call": lambda **kwargs: calls.append(kwargs) or "ok-hf"
        }
    }
    try:
        result = llm_client.call("hello")
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-hf"
    assert len(calls) == 1
    assert calls[0]["model"] == "some-hf-model"
