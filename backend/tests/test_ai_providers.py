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


def test_registry_defaults_to_local_qwen_without_gateway_token():
    settings = Settings(_env_file=None)
    assert settings.inference_gateway_base_url == "http://ddu-gateway:8080"
    assert settings.inference_gateway_token == ""
    assert settings.luna_model_id == "gpt-6-luna"


def test_registry_routes_both_models_to_same_gateway_with_same_consumer_token():
    settings = Settings(_env_file=None, inference_gateway_token="x" * 32)
    registry = build_narrator_registry(settings, object())

    assert set(registry.providers) == {"local_qwen", "gpt6_luna_high"}
    assert registry.public_configuration() == [
        {"id": "local_qwen", "name": "Local Qwen", "model": settings.llm_model_alias, "default": True},
        {"id": "gpt6_luna_high", "name": "GPT-6 Luna High", "model": "gpt-6-luna", "default": False},
    ]
    assert "x" * 32 not in repr(registry.public_configuration())
    local = registry.providers["local_qwen"]
    luna = registry.providers[GPT6_LUNA_HIGH]
    assert local.base_url == luna.base_url == "http://ddu-gateway:8080"
    assert local.headers == luna.headers == {"Authorization": "Bearer " + "x" * 32}
    assert local.model_alias == "ddu-narrator"
    assert luna.model_alias == "gpt-6-luna"
    assert local.enable_thinking is None
    assert local.require_provider_header and luna.require_provider_header


@pytest.mark.asyncio
async def test_narrator_selection_fails_closed_without_gateway_token():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(503)))
    registry = build_narrator_registry(
        Settings(_env_file=None, inference_gateway_token="x" * 32), client
    )

    assert await registry.is_ready("gpt6_luna_high") is False
    # Readiness is asynchronous; provider resolution never silently routes to Qwen.
    assert registry.require_configured("gpt6_luna_high").model_alias == "gpt-6-luna"
    unconfigured = build_narrator_registry(Settings(_env_file=None, inference_gateway_token=""), client)
    with pytest.raises(ValueError, match="unavailable"):
        unconfigured.require_configured("gpt6_luna_high")
    with pytest.raises(ValueError, match="unavailable"):
        unconfigured.require_configured("local_qwen")
    await client.aclose()


def test_provider_routing_never_falls_back_for_a_luna_session():
    qwen = object()
    luna = object()
    registry = NarratorRegistry(
        {"local_qwen": qwen, GPT6_LUNA_HIGH: luna}, gateway_configured=True
    )
    assert select_provider(registry, GPT6_LUNA_HIGH) is luna
    assert select_provider(qwen, "local_qwen") is qwen
    with pytest.raises(NarratorUnavailable):
        select_provider(qwen, GPT6_LUNA_HIGH)
