import httpx
import pytest

from app.ai.providers import (
    GPT6_LUNA_HIGH,
    NarratorRegistry,
    NarratorUnavailable,
    build_narrator_registry,
    select_provider,
)
from app.core.settings import Settings


def test_registry_defaults_to_local_qwen_without_bridge_secret():
    settings = Settings(_env_file=None)
    assert settings.luna_bridge_base_url == "http://host.docker.internal:18765"
    assert settings.luna_bridge_token == ""
    assert settings.luna_model_id == "gpt-6-luna"


def test_registry_never_places_bridge_secret_in_model_listing():
    settings = Settings(_env_file=None, luna_bridge_token="x" * 32)
    registry = build_narrator_registry(settings, object())

    assert set(registry.providers) == {"local_qwen", "gpt6_luna_high"}
    assert registry.public_configuration() == [
        {"id": "local_qwen", "name": "Local Qwen", "model": settings.llm_model_alias, "default": True},
        {"id": "gpt6_luna_high", "name": "GPT-6 Luna High", "model": "gpt-6-luna", "default": False},
    ]
    assert "x" * 32 not in repr(registry.public_configuration())


@pytest.mark.asyncio
async def test_luna_selection_fails_closed_when_bridge_is_unavailable():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    registry = build_narrator_registry(
        Settings(_env_file=None, luna_bridge_token="x" * 32), client
    )

    assert await registry.is_ready("gpt6_luna_high") is False
    # Readiness is asynchronous; provider resolution never silently routes to Qwen.
    assert registry.require_configured("gpt6_luna_high").model_alias == "gpt-6-luna"
    unconfigured = build_narrator_registry(Settings(_env_file=None, luna_bridge_token=""), client)
    with pytest.raises(ValueError, match="unavailable"):
        unconfigured.require_configured("gpt6_luna_high")
    assert unconfigured.require_configured("local_qwen").model_alias == "ddu-narrator"
    await client.aclose()


def test_provider_routing_never_falls_back_for_a_luna_session():
    qwen = object()
    luna = object()
    registry = NarratorRegistry(
        {"local_qwen": qwen, GPT6_LUNA_HIGH: luna}, luna_configured=True
    )
    assert select_provider(registry, GPT6_LUNA_HIGH) is luna
    assert select_provider(qwen, "local_qwen") is qwen
    with pytest.raises(NarratorUnavailable):
        select_provider(qwen, GPT6_LUNA_HIGH)
