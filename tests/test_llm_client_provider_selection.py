import pytest
from core import llm_client


@pytest.fixture(autouse=True)
def clear_llm_client_caches():
    """Clear LRU caches before and after each test to ensure test isolation."""
    llm_client._get_groq_client.cache_clear()
    llm_client._get_openai_client.cache_clear()
    llm_client._get_gemini_client.cache_clear()
    llm_client._get_hf_session.cache_clear()
    yield
    llm_client._get_groq_client.cache_clear()
    llm_client._get_openai_client.cache_clear()
    llm_client._get_gemini_client.cache_clear()
    llm_client._get_hf_session.cache_clear()


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


def test_client_factories_memoization():
    """Verify that memoized helpers return the exact same instance on repeated calls."""
    c1 = llm_client._get_groq_client("test-groq-key")
    c2 = llm_client._get_groq_client("test-groq-key")
    assert c1 is c2

    o1 = llm_client._get_openai_client("test-key", "https://api.mistral.ai/v1")
    o2 = llm_client._get_openai_client("test-key", "https://api.mistral.ai/v1")
    assert o1 is o2

    s1 = llm_client._get_hf_session()
    s2 = llm_client._get_hf_session()
    assert s1 is s2
