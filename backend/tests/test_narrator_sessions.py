import os
import secrets
from uuid import UUID, uuid4

import pytest

pytestmark = pytest.mark.skipif(
    os.getenv("DDU_TEST_DB") != "1", reason="isolated Postgres not configured"
)


@pytest.mark.asyncio
async def test_narrator_is_pinned_and_idempotency_rejects_a_changed_choice(
    turn_harness, monkeypatch
):
    from app.game import routes

    class ReadyRegistry:
        luna_configured = True

        async def is_ready(self, _narrator):
            return True

    monkeypatch.setattr(routes, "build_narrator_registry", lambda *_: ReadyRegistry())
    key = uuid4()
    capability = secrets.token_urlsafe(32)
    response = await turn_harness.client.post(
        "/api/v1/sessions",
        json={"scenario": turn_harness.slug, "narrator": "gpt6_luna_high"},
        headers={
            "Idempotency-Key": str(key),
            "X-Session-Capability": capability,
        },
    )

    assert response.status_code == 201, response.text
    assert response.json()["narrator"] == "gpt6_luna_high"
    turn_harness.created_sessions.add(UUID(response.json()["id"]))
    changed = await turn_harness.client.post(
        "/api/v1/sessions",
        json={"scenario": turn_harness.slug, "narrator": "local_qwen"},
        headers={
            "Idempotency-Key": str(key),
            "X-Session-Capability": capability,
        },
    )
    assert changed.status_code == 409
    resumed = await turn_harness.client.get(
        f"/api/v1/sessions/{response.json()['id']}",
        headers={"Authorization": f"Bearer {capability}"},
    )
    assert resumed.json()["narrator"] == "gpt6_luna_high"

    class UnavailableRegistry:
        luna_configured = True

        async def is_ready(self, _narrator):
            return False

    monkeypatch.setattr(
        routes, "build_narrator_registry", lambda *_: UnavailableRegistry()
    )
    unavailable = await turn_harness.client.post(
        "/api/v1/sessions",
        json={"scenario": turn_harness.slug, "narrator": "gpt6_luna_high"},
        headers={
            "Idempotency-Key": str(uuid4()),
            "X-Session-Capability": secrets.token_urlsafe(32),
        },
    )
    assert unavailable.status_code == 503
