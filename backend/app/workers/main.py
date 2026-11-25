"""Inference monitor and durable structured turn and initialization workers."""
import asyncio
import logging

import httpx
from sqlalchemy import func
from sqlalchemy.dialects.postgresql import insert

from app.ai.providers import build_narrator_registry
from app.core.settings import WORKER_HEALTH_TIMEOUT_SECONDS, get_settings
from app.db.connection import engine
from app.db.models import ServiceHeartbeat
from app.game.turn_worker import run_turn_worker
from app.runtime.orchestrator import TurnOrchestrator
from app.workers.initialization import build_retrieval_service, run_initialization_worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def record_heartbeat(inference_ready: bool, message: str) -> None:
    table = ServiceHeartbeat.__table__
    statement = insert(table).values(
        service_name="turn-worker", inference_ready=inference_ready, message=message
    )
    statement = statement.on_conflict_do_update(
        index_elements=[table.c.service_name],
        set_={
            "inference_ready": statement.excluded.inference_ready,
            "message": statement.excluded.message,
            "last_seen": func.now(),
        },
    )
    async with engine.begin() as connection:
        await connection.execute(statement)


async def monitor_inference() -> None:
    settings = get_settings()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(WORKER_HEALTH_TIMEOUT_SECONDS), trust_env=False
    ) as client:
        registry = build_narrator_registry(settings, client)
        while True:
            ready = await registry.is_ready("local_qwen")
            luna_ready = await registry.is_ready("gpt6_luna_high")
            message = f"qwen={'ready' if ready else 'down'}; luna={'ready' if luna_ready else 'down'}"
            inference_ready = luna_ready
            try:
                await record_heartbeat(inference_ready, message)
            except Exception:
                logger.exception("Could not persist worker heartbeat; retrying")
            else:
                logger.info("worker heartbeat: inference_ready=%s", inference_ready)
            await asyncio.sleep(settings.worker_poll_seconds)


async def run_real_turn_worker() -> None:
    settings = get_settings()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(settings.turn_inference_timeout_seconds), trust_env=False
    ) as http:
        generator = build_narrator_registry(settings, http)
        retrieval_service = build_retrieval_service(settings, http)
        orchestrator = TurnOrchestrator(
            generator=generator,
            retrieval_service=retrieval_service,
            settings=settings,
        )
        await run_turn_worker(orchestrator=orchestrator, settings=settings)


async def run() -> None:
    await asyncio.gather(
        monitor_inference(), run_real_turn_worker(), run_initialization_worker()
    )


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except KeyboardInterrupt:
        logger.info("Worker stopped")
