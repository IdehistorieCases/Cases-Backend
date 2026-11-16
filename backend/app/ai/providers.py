"""Provider construction, readiness, and safe public narrator capabilities."""

from app.ai.llm import InferenceClient
from app.core.settings import Settings

LOCAL_QWEN = "local_qwen"
GPT6_LUNA_HIGH = "gpt6_luna_high"


class NarratorUnavailable(RuntimeError):
    """A requested provider cannot service this session; callers must not fallback."""


def select_provider(generator, narrator: str):
    """Resolve a pinned provider; legacy single-client callers remain Qwen-only."""
    if isinstance(generator, NarratorRegistry):
        try:
            return generator.require_configured(narrator)
        except ValueError as exc:
            raise NarratorUnavailable("selected narrator is unavailable") from exc
    if narrator != LOCAL_QWEN:
        raise NarratorUnavailable("selected narrator is unavailable")
    return generator


class NarratorRegistry:
    def __init__(self, providers: dict[str, InferenceClient], *, luna_configured: bool):
        self.providers = providers
        self.luna_configured = luna_configured

    def require_configured(self, narrator: str) -> InferenceClient:
        if narrator not in self.providers:
            raise ValueError("unknown narrator selection")
        if narrator == GPT6_LUNA_HIGH and not self.luna_configured:
            raise ValueError("selected narrator is unavailable")
        return self.providers[narrator]

    async def is_ready(self, narrator: str) -> bool:
        if narrator not in self.providers:
            return False
        if narrator == GPT6_LUNA_HIGH and not self.luna_configured:
            return False
        provider = self.providers[narrator]
        return await provider.is_ready() and await provider.serves_expected_model()

    async def public_status(self) -> list[dict]:
        local = self.providers[LOCAL_QWEN]
        luna = self.providers[GPT6_LUNA_HIGH]
        return [
            {
                "id": LOCAL_QWEN,
                "name": "Local Qwen",
                "model": local.model_alias,
                "default": True,
                "configured": True,
                "ready": await self.is_ready(LOCAL_QWEN),
            },
            {
                "id": GPT6_LUNA_HIGH,
                "name": "GPT-6 Luna High",
                "model": luna.model_alias,
                "reasoning_effort": "high",
                "default": False,
                "configured": self.luna_configured,
                "ready": await self.is_ready(GPT6_LUNA_HIGH),
            },
        ]

    def public_configuration(self) -> list[dict]:
        return [
            {
                "id": LOCAL_QWEN,
                "name": "Local Qwen",
                "model": self.providers[LOCAL_QWEN].model_alias,
                "default": True,
            },
            {
                "id": GPT6_LUNA_HIGH,
                "name": "GPT-6 Luna High",
                "model": self.providers[GPT6_LUNA_HIGH].model_alias,
                "default": False,
            },
        ]


def build_narrator_registry(settings: Settings, client) -> NarratorRegistry:
    return NarratorRegistry(
        {
            LOCAL_QWEN: InferenceClient(
                settings.llm_base_url,
                settings.llm_model_alias,
                client,
                enable_thinking=settings.llm_enable_thinking,
                provider_label=LOCAL_QWEN,
            ),
            GPT6_LUNA_HIGH: InferenceClient(
                settings.luna_bridge_base_url,
                settings.luna_model_id,
                client,
                headers=(
                    {"Authorization": f"Bearer {settings.luna_bridge_token}"}
                    if settings.luna_bridge_token
                    else {}
                ),
                provider_label="codex_cli",
            ),
        },
        luna_configured=bool(settings.luna_bridge_token),
    )
