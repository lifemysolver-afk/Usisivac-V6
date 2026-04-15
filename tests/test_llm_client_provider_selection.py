import json

from core import llm_client


def _patch_providers(monkeypatch, providers_dict):
    """Context-manager-style helper that restores PROVIDERS after the test."""
    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = providers_dict
    return original


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


# ── ONLY_PRIMARY_LLM truthy-value variants ───────────────────────────────────

def test_only_primary_llm_enabled_by_value_1(monkeypatch):
    """ONLY_PRIMARY_LLM='1' should also suppress fallback."""
    groq_called = []

    monkeypatch.setenv("ONLY_PRIMARY_LLM", "1")
    monkeypatch.setenv("PRIMARY_LLM", "gemini")
    monkeypatch.setenv("GEMINI_KEY_1", "dummy-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: "ok-gemini"
        },
        "groq": {
            "env_key": "GROQ_API_KEY",
            "call": lambda **kwargs: groq_called.append(1) or "ok-groq"
        },
    }
    try:
        result = llm_client.call("test")
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-gemini"
    assert groq_called == [], "groq should not be called when ONLY_PRIMARY_LLM=1"


def test_only_primary_llm_enabled_by_value_yes(monkeypatch):
    """ONLY_PRIMARY_LLM='yes' should suppress fallback."""
    groq_called = []

    monkeypatch.setenv("ONLY_PRIMARY_LLM", "yes")
    monkeypatch.setenv("PRIMARY_LLM", "gemini")
    monkeypatch.setenv("GEMINI_KEY_1", "dummy-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: "ok-gemini"
        },
        "groq": {
            "env_key": "GROQ_API_KEY",
            "call": lambda **kwargs: groq_called.append(1) or "ok-groq"
        },
    }
    try:
        result = llm_client.call("test")
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-gemini"
    assert groq_called == []


def test_only_primary_llm_enabled_by_value_on(monkeypatch):
    """ONLY_PRIMARY_LLM='on' should suppress fallback."""
    groq_called = []

    monkeypatch.setenv("ONLY_PRIMARY_LLM", "on")
    monkeypatch.setenv("PRIMARY_LLM", "gemini")
    monkeypatch.setenv("GEMINI_KEY_1", "dummy-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: "ok-gemini"
        },
        "groq": {
            "env_key": "GROQ_API_KEY",
            "call": lambda **kwargs: groq_called.append(1) or "ok-groq"
        },
    }
    try:
        result = llm_client.call("test")
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-gemini"
    assert groq_called == []


# ── Fallback behaviour when ONLY_PRIMARY_LLM is disabled ─────────────────────

def test_call_falls_back_to_next_provider_when_primary_fails(monkeypatch):
    """When ONLY_PRIMARY_LLM=false and primary fails, fallback providers are tried."""
    monkeypatch.setenv("ONLY_PRIMARY_LLM", "false")
    monkeypatch.setenv("PRIMARY_LLM", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.setenv("GEMINI_KEY_1", "gemini-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "groq": {
            "env_key": "GROQ_API_KEY",
            "call": lambda **kwargs: (_ for _ in ()).throw(RuntimeError("groq down"))
        },
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: "ok-gemini-fallback"
        },
        "mistral": {
            "env_key": "MISTRAL_API_KEY",
            "call": lambda **kwargs: "ok-mistral"
        },
        "openrouter": {
            "env_key": "OPENROUTER_API_KEY",
            "call": lambda **kwargs: "ok-openrouter"
        },
    }
    try:
        result = llm_client.call("test", retries=1)
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-gemini-fallback"


def test_call_explicit_provider_arg_overrides_fallback(monkeypatch):
    """Passing `provider` explicitly forces only that provider, even when ONLY_PRIMARY_LLM=false."""
    groq_called = []

    monkeypatch.setenv("ONLY_PRIMARY_LLM", "false")
    monkeypatch.setenv("PRIMARY_LLM", "gemini")
    monkeypatch.setenv("GEMINI_KEY_1", "gemini-key")
    monkeypatch.setenv("GROQ_API_KEY", "groq-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "groq": {
            "env_key": "GROQ_API_KEY",
            "call": lambda **kwargs: groq_called.append(1) or "ok-groq"
        },
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: "ok-gemini"
        },
    }
    try:
        result = llm_client.call("test", provider="groq", retries=1)
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-groq"
    assert len(groq_called) == 1


# ── API key filtering ─────────────────────────────────────────────────────────

def test_call_skips_provider_with_placeholder_key(monkeypatch):
    """Providers with 'YOUR_' placeholder keys must be skipped entirely."""
    monkeypatch.setenv("ONLY_PRIMARY_LLM", "false")
    monkeypatch.setenv("PRIMARY_LLM", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "YOUR_GROQ_KEY_HERE")
    monkeypatch.setenv("GEMINI_KEY_1", "real-gemini-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "groq": {
            "env_key": "GROQ_API_KEY",
            "call": lambda **kwargs: (_ for _ in ()).throw(AssertionError("groq must not be called"))
        },
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: "ok-gemini"
        },
        "mistral": {"env_key": "MISTRAL_API_KEY", "call": lambda **kwargs: "ok-mistral"},
        "openrouter": {"env_key": "OPENROUTER_API_KEY", "call": lambda **kwargs: "ok-or"},
    }
    try:
        result = llm_client.call("test", retries=1)
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-gemini"


def test_call_skips_provider_with_empty_key(monkeypatch):
    """Providers with empty API keys must be skipped."""
    monkeypatch.setenv("ONLY_PRIMARY_LLM", "false")
    monkeypatch.setenv("PRIMARY_LLM", "mistral")
    monkeypatch.setenv("MISTRAL_API_KEY", "")
    monkeypatch.setenv("GEMINI_KEY_1", "real-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "mistral": {
            "env_key": "MISTRAL_API_KEY",
            "call": lambda **kwargs: (_ for _ in ()).throw(AssertionError("mistral must not be called"))
        },
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: "ok-gemini"
        },
        "groq": {"env_key": "GROQ_API_KEY", "call": lambda **kwargs: "ok-groq"},
        "openrouter": {"env_key": "OPENROUTER_API_KEY", "call": lambda **kwargs: "ok-or"},
    }
    try:
        result = llm_client.call("test", retries=1)
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-gemini"


# ── Mock response on total failure ───────────────────────────────────────────

def test_call_returns_mock_response_when_all_providers_fail(monkeypatch):
    """When all providers fail, the function returns a JSON mock response."""
    monkeypatch.setenv("ONLY_PRIMARY_LLM", "true")
    monkeypatch.setenv("PRIMARY_LLM", "gemini")
    monkeypatch.setenv("GEMINI_KEY_1", "some-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: (_ for _ in ()).throw(RuntimeError("network error"))
        },
    }
    try:
        result = llm_client.call("hello", retries=1)
    finally:
        llm_client.PROVIDERS = original

    parsed = json.loads(result)
    assert parsed["status"] == "MOCK_RESPONSE"
    assert "error" in parsed
    assert "prompt_preview" in parsed


def test_mock_response_contains_prompt_preview(monkeypatch):
    """Mock response includes first 100 chars of the prompt."""
    monkeypatch.setenv("ONLY_PRIMARY_LLM", "true")
    monkeypatch.setenv("PRIMARY_LLM", "gemini")
    monkeypatch.setenv("GEMINI_KEY_1", "some-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    long_prompt = "A" * 200

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: (_ for _ in ()).throw(RuntimeError("fail"))
        },
    }
    try:
        result = llm_client.call(long_prompt, retries=1)
    finally:
        llm_client.PROVIDERS = original

    parsed = json.loads(result)
    assert parsed["prompt_preview"] == "A" * 100


# ── Default PRIMARY_LLM is gemini ─────────────────────────────────────────────

def test_default_primary_llm_is_gemini_when_env_not_set(monkeypatch):
    """When PRIMARY_LLM env var is absent, the default provider used is 'gemini'."""
    calls = []

    monkeypatch.delenv("PRIMARY_LLM", raising=False)
    monkeypatch.setenv("ONLY_PRIMARY_LLM", "true")
    monkeypatch.setenv("GEMINI_KEY_1", "gemini-key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: calls.append("gemini") or "ok-gemini"
        },
        "groq": {
            "env_key": "GROQ_API_KEY",
            "call": lambda **kwargs: calls.append("groq") or "ok-groq"
        },
    }
    try:
        result = llm_client.call("test")
    finally:
        llm_client.PROVIDERS = original

    assert result == "ok-gemini"
    assert calls == ["gemini"]


# ── Model selection logic ─────────────────────────────────────────────────────

def test_call_passes_explicit_model_arg_to_provider(monkeypatch):
    """An explicit `model` argument takes priority over PRIMARY_MODEL."""
    calls = []

    monkeypatch.setenv("ONLY_PRIMARY_LLM", "true")
    monkeypatch.setenv("PRIMARY_LLM", "gemini")
    monkeypatch.setenv("GEMINI_KEY_1", "key")
    monkeypatch.setenv("PRIMARY_MODEL", "gemini-2.5-flash")

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: calls.append(kwargs) or "ok"
        }
    }
    try:
        llm_client.call("hi", model="custom-model-v1")
    finally:
        llm_client.PROVIDERS = original

    assert calls[0]["model"] == "custom-model-v1"


def test_call_non_gemini_provider_does_not_get_gemini_default_model(monkeypatch):
    """The gemini-2.5-flash default model must NOT be injected for non-gemini providers."""
    calls = []

    monkeypatch.setenv("ONLY_PRIMARY_LLM", "true")
    monkeypatch.setenv("PRIMARY_LLM", "groq")
    monkeypatch.setenv("GROQ_API_KEY", "key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "groq": {
            "env_key": "GROQ_API_KEY",
            "call": lambda **kwargs: calls.append(kwargs) or "ok-groq"
        }
    }
    try:
        llm_client.call("test")
    finally:
        llm_client.PROVIDERS = original

    assert "model" not in calls[0], "No model should be injected for non-gemini providers without PRIMARY_MODEL"


# ── System prompt forwarding ──────────────────────────────────────────────────

def test_call_forwards_system_prompt_to_provider(monkeypatch):
    """The `system` argument must be passed through to the provider call."""
    calls = []

    monkeypatch.setenv("ONLY_PRIMARY_LLM", "true")
    monkeypatch.setenv("PRIMARY_LLM", "gemini")
    monkeypatch.setenv("GEMINI_KEY_1", "key")
    monkeypatch.delenv("PRIMARY_MODEL", raising=False)

    original = llm_client.PROVIDERS
    llm_client.PROVIDERS = {
        "gemini": {
            "env_key": "GEMINI_KEY_1",
            "call": lambda **kwargs: calls.append(kwargs) or "ok"
        }
    }
    try:
        llm_client.call("user-msg", system="be concise")
    finally:
        llm_client.PROVIDERS = original

    assert calls[0]["system"] == "be concise"
    assert calls[0]["prompt"] == "user-msg"