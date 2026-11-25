import os

import httpx
import pytest

from app.db.connection import engine
from app.main import app


@pytest.mark.asyncio
async def test_liveness_when_db_is_offline(monkeypatch):
    async def down():
        return False

    monkeypatch.setattr("app.main.database_ready", down)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        assert (await c.get("/healthz")).json() == {"status": "alive"}
        response = await c.get("/readyz")
        assert response.status_code == 503
        assert response.json()["database"] is False


@pytest.mark.asyncio
async def test_ready_with_healthy_database(monkeypatch):
    async def up():
        return True

    monkeypatch.setattr("app.main.database_ready", up)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        response = await c.get("/readyz")
        assert response.status_code == 200
        assert response.json()["status"] == "ready"


@pytest.mark.asyncio
async def test_create_session_rejects_an_unknown_narrator_before_database_access():
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/api/v1/sessions", json={"scenario": "example", "narrator": "unknown"}
        )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_narrator_listing_reports_models_without_secrets(monkeypatch):
    class Registry:
        async def public_status(self):
            return [
                {"id": "local_qwen", "name": "Local Qwen", "model": "ddu-narrator", "default": False,
                 "configured": True, "ready": False},
                {"id": "gpt6_luna_high", "name": "GPT-6 Luna High", "model": "gpt-6-luna", "default": True,
                 "reasoning_effort": "high", "configured": False, "ready": False},
            ]

    monkeypatch.setattr("app.game.routes.build_narrator_registry", lambda *_: Registry())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/api/v1/narrators")

    assert response.status_code == 200
    assert response.json()[1]["reasoning_effort"] == "high"
    assert "Authorization" not in response.text


@pytest.mark.skipif(os.getenv("DDU_TEST_DB") != "1", reason="Postgres not configured")
@pytest.mark.asyncio
async def test_private_status_reports_the_database_used_by_the_api():
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as c:
            response = await c.get("/api/v1/status")
    finally:
        await engine.dispose()
    assert response.status_code == 200
    assert response.json()["database_name"] == os.environ["DB_NAME"]
