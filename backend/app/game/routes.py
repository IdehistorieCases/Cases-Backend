"""Session-scoped M2 API; generation is handled by the separate worker."""
import hashlib
import hmac
import logging
import re
import secrets
from datetime import date, datetime, timedelta, timezone
from typing import Annotated, Literal
from uuid import UUID, uuid4

import httpx
from fastapi import APIRouter, Depends, Header, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, ConfigDict, Field, field_validator
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.providers import (
    GPT6_LUNA_HIGH,
    NarratorUnavailable,
    build_narrator_registry,
    select_provider,
)
from app.core.settings import INFERENCE_READINESS_TIMEOUT_SECONDS, get_settings
from app.db.connection import session_factory
from app.db.models import (
    CommittedTurn,
    GameEvent,
    GameSession,
    ScenarioVersion,
    SessionInitializationJob,
    TurnRequest,
)
from app.game.initialization import (
    InitializationNotRetryable,
    enqueue_initialization,
    get_session_job,
    request_retry,
)
from app.runtime.context import (
    ContextBudget,
    ContextBuildFailure,
    SessionPromptContext,
    build_session_context,
)
from app.runtime.proposals import SceneCompletedEvent
from app.runtime.reflection import (
    ReflectionBuildError,
    ReflectionGenerationError,
    build_reflection_input,
    build_reflection_search_query,
    generate_reflection,
    get_reflection_generator,
    get_reflection_retrieval_service,
    public_reflection_result,
    revalidate_reflection_sources,
    validate_reflection_events,
)
from app.runtime.state import project_state
from app.scenarios.cli import scenario_readiness_report
from app.scenarios.repository import get_active_scenario
from app.scenarios.runtime_compatibility import (
    ScenarioRuntimeIncompatible,
    require_runtime_compatible,
)
from app.scenarios.schema import ScenarioDefinition
from app.scenarios.schema_v2 import ScenarioDefinitionV2
from app.scenarios.validation import validate_definition, validate_v2_definition

router = APIRouter(prefix="/api/v1")
logger = logging.getLogger(__name__)
bearer = HTTPBearer(auto_error=False)
ACTIVE_STATUSES = ("queued", "running", "retryable_failure")
CAPABILITY_LIFETIME = timedelta(hours=24)
SESSION_CAPABILITY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{43}$")


class CreateSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    scenario: str = Field(min_length=1, max_length=80)
    role_id: str | None = Field(default=None, min_length=2, max_length=64)
    narrator: Literal["local_qwen", "gpt6_luna_high"] = GPT6_LUNA_HIGH


class SubmitTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_turn_id: UUID
    expected_turn: int = Field(ge=0)
    text: str = Field(min_length=1, max_length=2000)
    input_language: Literal["da"]

    @field_validator("text")
    @classmethod
    def nonblank_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value


class CompleteSession(BaseModel):
    model_config = ConfigDict(extra="forbid")
    client_turn_id: UUID
    expected_turn: int = Field(ge=0)


def require_token(
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
) -> str:
    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(status_code=401, detail="Bearer token required", headers={
            "WWW-Authenticate": "Bearer"
        })
    return credentials.credentials


async def owned_session(db: AsyncSession, session_id: UUID, token: str, lock: bool = False):
    statement = select(GameSession).where(GameSession.id == session_id)
    if lock:
        statement = statement.with_for_update()
    session = (await db.execute(statement)).scalar_one_or_none()
    digest = hashlib.sha256(token.encode("utf-8")).hexdigest()
    if (
        session is None
        or not hmac.compare_digest(session.capability_hash, digest)
        or session.capability_expires_at <= datetime.now(timezone.utc)
    ):
        raise HTTPException(status_code=404, detail="Session not found")
    return session


def public_request(turn: TurnRequest) -> dict:
    return {
        "request_id": str(turn.id),
        "status": turn.status,
        "poll_after_ms": 1000,
    }


def public_turn_failure(turn: TurnRequest) -> dict | None:
    if turn.failure_code is None:
        return None
    return {
        "code": turn.failure_code,
        "stage": turn.failure_stage or "worker",
        "retryable": turn.status == "retryable_failure",
        "message": turn.failure_message or "The turn could not be processed.",
    }


def public_initialization(
    job: SessionInitializationJob | None, runtime_version: str
) -> dict:
    if runtime_version == "v1_legacy":
        return {
            "status": "not_applicable",
            "lifecycle_status": "ready",
            "runtime_version": runtime_version,
            "job_id": None,
            "attempts": 0,
            "failure": None,
            "opening": None,
            "provenance": None,
        }
    if job is None:
        return {
            "status": "uninitialized",
            "lifecycle_status": "uninitialized",
            "runtime_version": runtime_version,
            "job_id": None,
            "attempts": 0,
            "failure": None,
            "opening": None,
            "provenance": None,
        }
    failure = None
    if job.failure_code is not None:
        retryable = job.status == "retryable_failure"
        failure = {
            "code": job.failure_code,
            "stage": job.failure_stage,
            "retryable": retryable,
            "message": (
                "Initialiseringen kan forsøges igen."
                if retryable
                else "Initialiseringen mislykkedes."
            ),
        }
    return {
        "status": job.status,
        "lifecycle_status": "ready" if job.status == "ready" else "uninitialized",
        "runtime_version": runtime_version,
        "job_id": str(job.id),
        "attempts": job.attempts,
        "failure": failure,
        "opening": job.result.get("opening") if job.status == "ready" else None,
        "provenance": job.result.get("provenance") if job.status == "ready" else None,
    }


def public_created_session(
    session: GameSession,
    scenario: ScenarioVersion,
    job: SessionInitializationJob | None,
    *,
    token: str | None,
    replay: bool,
) -> dict:
    initialization = public_initialization(job, session.runtime_version)
    return {
        "id": str(session.id),
        "token": token,
        "turn_number": session.turn_number,
        "scenario": scenario.slug,
        "scenario_version": scenario.version,
        "role_id": session.player_role_id,
        "narrator": session.narrator,
        "runtime_version": session.runtime_version,
        "lifecycle_status": initialization["lifecycle_status"],
        "token_expires_at": session.capability_expires_at.isoformat(),
        "initialization_status": initialization["status"],
        "initialization_job_id": initialization["job_id"],
        "idempotent_replay": replay,
    }


@router.get("/scenarios")
async def list_scenarios():
    async with session_factory() as db:
        rows = (await db.execute(select(ScenarioVersion).order_by(
            ScenarioVersion.slug,
            ScenarioVersion.is_active.desc(),
            ScenarioVersion.created_at,
            ScenarioVersion.version,
        ))).scalars().all()
        result = []
        for row in rows:
            definition = row.definition
            metadata = definition.get("metadata", {}) if definition else {}
            result.append({
                "slug": row.slug,
                "version": row.version,
                "schema_version": row.schema_version,
                "runtime_version": row.runtime_version,
                "title": metadata.get("title"),
                "summary": metadata.get("summary"),
                "default_language": metadata.get("default_language") or metadata.get("language"),
                "supported_roles": [
                    role["id"] for role in definition.get("roles", []) if role.get("playable")
                ] if definition else [],
                "is_active": row.is_active,
                "is_authored": definition is not None,
            })
        return result


@router.get("/narrators")
async def list_narrators():
    settings = get_settings()
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(INFERENCE_READINESS_TIMEOUT_SECONDS), trust_env=False
    ) as client:
        return await build_narrator_registry(settings, client).public_status()


@router.get("/scenarios/{slug}")
async def get_scenario(slug: str):
    async with session_factory() as db:
        scenario = await get_active_scenario(db, slug)
        if scenario is None:
            raise HTTPException(status_code=404, detail="Scenario not found")
        if scenario.definition is None:
            raise HTTPException(
                status_code=409,
                detail="Active scenario version is an unauthored legacy stub and is not playable",
            )
        definition = scenario.definition
        common = {
            "slug": scenario.slug,
            "version": scenario.version,
            "schema_version": scenario.schema_version,
            "runtime_version": scenario.runtime_version,
            "metadata": definition["metadata"],
            "historical_context": definition["historical_context"],
            "actors": definition["actors"],
            "roles": [role for role in definition["roles"] if role["playable"]],
            "default_role_id": definition["default_role_id"],
            "learning_objectives": definition["learning_objectives"],
        }
        if scenario.schema_version == 1:
            try:
                parsed = ScenarioDefinition.model_validate(definition)
                validate_definition(parsed)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail="Active scenario definition is invalid") from exc
            return {**common, "phases": definition["phases"], "dilemmas": definition["dilemmas"]}
        if scenario.schema_version == 2:
            try:
                parsed = ScenarioDefinitionV2.model_validate(definition)
                validate_v2_definition(parsed)
            except ValueError as exc:
                raise HTTPException(status_code=409, detail="Active scenario definition is invalid") from exc
            return {
                **common,
                "chronology": definition["chronology"],
                "interaction_guidance": definition["interaction_guidance"],
                "completion_policy": definition["completion_policy"],
                "optional_beats": definition["optional_beats"],
            }
        raise HTTPException(status_code=409, detail="Active scenario schema version is unsupported")


@router.post("/sessions", status_code=201)
async def create_session(
    payload: CreateSession,
    idempotency_key: Annotated[UUID | None, Header(alias="Idempotency-Key")] = None,
    session_capability: Annotated[
        str | None,
        Header(
            alias="X-Session-Capability",
            description=(
                "Client-generated 256-bit URL-safe bearer used with Idempotency-Key "
                "for safe create retries. The API stores only its hash."
            ),
        ),
    ] = None,
):
    requested_creation_key_hash = (
        hashlib.sha256(str(idempotency_key).encode("ascii")).hexdigest()
        if idempotency_key is not None
        else None
    )
    valid_client_capability = (
        session_capability is not None
        and SESSION_CAPABILITY_PATTERN.fullmatch(session_capability) is not None
    )
    async with session_factory() as db:
        async with db.begin():
            if requested_creation_key_hash is not None:
                # Serialize same-key creates, including the response-lost retry path.
                await db.execute(
                    select(
                        func.pg_advisory_xact_lock(
                            func.hashtextextended(requested_creation_key_hash, 0)
                        )
                    )
                )
                existing = (
                    await db.execute(
                        select(GameSession)
                        .where(GameSession.creation_key_hash == requested_creation_key_hash)
                        .with_for_update()
                    )
                ).scalar_one_or_none()
                if existing is not None:
                    scenario = await db.get(ScenarioVersion, existing.scenario_version_id)
                    if not valid_client_capability:
                        raise HTTPException(
                            status_code=422,
                            detail=(
                                "Schema-v2 session replay requires Idempotency-Key and "
                                "X-Session-Capability"
                            ),
                        )
                    requested_capability_hash = hashlib.sha256(
                        session_capability.encode("utf-8")
                    ).hexdigest()
                    expected_role = None
                    if scenario is not None and scenario.definition is not None:
                        expected_role = (
                            payload.role_id or scenario.definition.get("default_role_id")
                        )
                    if (
                        scenario is None
                        or scenario.runtime_version != "v2_ai"
                        or scenario.slug != payload.scenario
                        or existing.narrator != payload.narrator
                        or expected_role != existing.player_role_id
                        or not hmac.compare_digest(
                            existing.capability_hash, requested_capability_hash
                        )
                    ):
                        raise HTTPException(
                            status_code=409,
                            detail="Idempotency-Key was already used for a different session",
                        )
                    if existing.capability_expires_at <= datetime.now(timezone.utc):
                        raise HTTPException(status_code=404, detail="Session not found")
                    job = await get_session_job(db, existing.id)
                    if scenario.schema_version == 2 and job is None:
                        job = await enqueue_initialization(db, existing.id)
                    return public_created_session(
                        existing, scenario, job, token=None, replay=True
                    )

            scenario = await get_active_scenario(db, payload.scenario, lock=True)
            if scenario is None:
                raise HTTPException(status_code=404, detail="Scenario not found")
            if payload.narrator == GPT6_LUNA_HIGH and scenario.schema_version != 2:
                raise HTTPException(
                    status_code=422,
                    detail="GPT-6 Luna High is only available for schema-v2 scenarios",
                )
            player_role_id = None
            if scenario.definition is not None:
                if scenario.schema_version == 1:
                    try:
                        require_runtime_compatible(scenario)
                    except ScenarioRuntimeIncompatible as exc:
                        raise HTTPException(status_code=409, detail=str(exc)) from exc
                elif scenario.schema_version == 2:
                    try:
                        definition = ScenarioDefinitionV2.model_validate(scenario.definition)
                        validate_v2_definition(definition)
                    except ValueError as exc:
                        raise HTTPException(
                            status_code=409, detail="Active scenario definition is invalid"
                        ) from exc
                    readiness = await scenario_readiness_report(db, scenario)
                    if not readiness["playable"]:
                        raise HTTPException(
                            status_code=409,
                            detail={
                                "code": "SCENARIO_NOT_PLAYABLE",
                                "message": "Active schema-v2 scenario is not ready for gameplay",
                                "checks": readiness["checks"],
                            },
                        )
                else:
                    raise HTTPException(
                        status_code=409, detail="Active scenario schema version is unsupported"
                    )
                playable_roles = {
                    role["id"] for role in scenario.definition["roles"] if role["playable"]
                }
                player_role_id = payload.role_id or scenario.definition["default_role_id"]
                if player_role_id not in playable_roles:
                    raise HTTPException(status_code=422, detail="role_id is not a playable role")
            elif payload.role_id is not None:
                raise HTTPException(status_code=422, detail="Legacy scenario does not support roles")
            if scenario.schema_version == 2 and (
                requested_creation_key_hash is None or not valid_client_capability
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "Schema-v2 session creation requires Idempotency-Key and "
                        "X-Session-Capability"
                    ),
                )
            if scenario.runtime_version == "v2_ai":
                token = session_capability
                creation_key_hash = requested_creation_key_hash
            else:
                # Preserve v1's server-issued capability and ignore the v2-only headers.
                token = secrets.token_urlsafe(32)
                creation_key_hash = None
            capability_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            if scenario.runtime_version == "v2_ai":
                # Serialize reuse across different idempotency keys before the unique check.
                await db.execute(
                    select(
                        func.pg_advisory_xact_lock(func.hashtextextended(capability_hash, 1))
                    )
                )
                capability_owner = (
                    await db.execute(
                        select(GameSession.id).where(
                            GameSession.capability_hash == capability_hash
                        )
                    )
                ).scalar_one_or_none()
                if capability_owner is not None:
                    raise HTTPException(
                        status_code=409,
                        detail="X-Session-Capability is already assigned to another session",
                    )
            if payload.narrator == GPT6_LUNA_HIGH:
                settings = get_settings()
                async with httpx.AsyncClient(
                    timeout=httpx.Timeout(INFERENCE_READINESS_TIMEOUT_SECONDS),
                    trust_env=False,
                ) as client:
                    registry = build_narrator_registry(settings, client)
                    if not registry.luna_configured or not await registry.is_ready(GPT6_LUNA_HIGH):
                        raise HTTPException(
                            status_code=503,
                            detail={"code": "NARRATOR_UNAVAILABLE", "narrator": GPT6_LUNA_HIGH},
                        )
            session = GameSession(
                id=uuid4(), scenario_version_id=scenario.id,
                player_role_id=player_role_id,
                capability_hash=capability_hash,
                creation_key_hash=creation_key_hash,
                runtime_version=scenario.runtime_version,
                narrator=payload.narrator,
                capability_expires_at=datetime.now(timezone.utc) + CAPABILITY_LIFETIME,
                turn_number=0,
                state={} if scenario.schema_version == 2 else dict(scenario.initial_state),
            )
            db.add(session)
            await db.flush()
            job = (
                await enqueue_initialization(db, session.id)
                if scenario.schema_version == 2
                else None
            )
            return public_created_session(session, scenario, job, token=token, replay=False)


@router.get("/sessions/{session_id}")
async def resume_session(session_id: UUID, token: Annotated[str, Depends(require_token)]):
    async with session_factory() as db:
        # Keep completion from changing the state before the pending request is read.
        session = await owned_session(db, session_id, token, lock=True)
        pending = (await db.execute(select(TurnRequest).where(
            TurnRequest.session_id == session.id, TurnRequest.status.in_(ACTIVE_STATUSES)
        ))).scalar_one_or_none()
        scenario = await db.get(ScenarioVersion, session.scenario_version_id)
        initialization = public_initialization(
            await get_session_job(db, session.id), session.runtime_version
        )
        return {
            "id": str(session.id), "turn_number": session.turn_number, "state": session.state,
            "pending_request": public_request(pending) if pending else None,
            "scenario": scenario.slug,
            "scenario_version": scenario.version,
            "runtime_version": session.runtime_version,
            "narrator": session.narrator,
            "lifecycle_status": initialization["lifecycle_status"],
            "player_role_id": session.player_role_id,
            "initialization_status": initialization["status"],
            "initialization": initialization,
        }


@router.get("/sessions/{session_id}/initialization")
async def get_initialization_status(
    session_id: UUID,
    token: Annotated[str, Depends(require_token)],
):
    async with session_factory() as db:
        session = await owned_session(db, session_id, token)
        return public_initialization(
            await get_session_job(db, session.id), session.runtime_version
        )


@router.post("/sessions/{session_id}/initialization/retry", status_code=202)
async def retry_session_initialization(
    session_id: UUID,
    token: Annotated[str, Depends(require_token)],
):
    async with session_factory() as db:
        async with db.begin():
            session = await owned_session(db, session_id, token, lock=True)
            scenario = await db.get(ScenarioVersion, session.scenario_version_id)
            if scenario is None or scenario.schema_version != 2:
                raise HTTPException(
                    status_code=409,
                    detail="Only schema-v2 sessions have an initialization job",
                )
            try:
                job = await request_retry(db, session.id)
            except InitializationNotRetryable as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            return public_initialization(job, session.runtime_version)

@router.post("/sessions/{session_id}/turns", status_code=202)
async def submit_turn(
    session_id: UUID,
    payload: SubmitTurn,
    response: Response,
    token: Annotated[str, Depends(require_token)],
    idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
):
    if idempotency_key != payload.client_turn_id:
        raise HTTPException(status_code=400, detail="Idempotency key must match client_turn_id")
    async with session_factory() as db:
        async with db.begin():
            session = await owned_session(db, session_id, token, lock=True)
            scenario = await db.get(ScenarioVersion, session.scenario_version_id)
            if scenario is None:
                raise HTTPException(status_code=404, detail="Session scenario not found")
            if session.runtime_version == "v2_ai":
                initialization = await get_session_job(db, session.id)
                if initialization is None or initialization.status != "ready":
                    raise HTTPException(
                        status_code=409,
                        detail="Session initialization is not ready",
                    )
            existing = (await db.execute(select(TurnRequest).where(
                TurnRequest.session_id == session.id,
                TurnRequest.client_turn_id == payload.client_turn_id,
            ))).scalar_one_or_none()
            if existing:
                if (existing.expected_turn != payload.expected_turn or existing.text != payload.text
                        or existing.input_language != payload.input_language):
                    raise HTTPException(status_code=409, detail="Idempotency key already used")
                response.headers["Location"] = f"/api/v1/turn-requests/{existing.id}"
                return public_request(existing)
            if session.runtime_version != "v2_ai":
                raise HTTPException(
                    status_code=409,
                    detail="Legacy sessions are read-only and do not accept new turns",
                )
            if session.state.get("completion_status") == "complete":
                raise HTTPException(status_code=409, detail="Session is already complete")
            try:
                require_runtime_compatible(scenario)
            except ScenarioRuntimeIncompatible as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            pending = (await db.execute(select(TurnRequest).where(
                TurnRequest.session_id == session.id, TurnRequest.status.in_(ACTIVE_STATUSES)
            ))).scalar_one_or_none()
            if pending:
                return JSONResponse(status_code=409, content={
                    "detail": "A turn is already in progress", "request_id": str(pending.id)
                })
            if payload.expected_turn != session.turn_number:
                raise HTTPException(status_code=409, detail="Expected turn does not match session")
            turn = TurnRequest(
                id=uuid4(), session_id=session.id, client_turn_id=payload.client_turn_id,
                expected_turn=payload.expected_turn, text=payload.text,
                input_language=payload.input_language, status="queued", attempts=0,
            )
            db.add(turn)
            response.headers["Location"] = f"/api/v1/turn-requests/{turn.id}"
            return public_request(turn)


@router.post("/sessions/{session_id}/completion", status_code=201)
async def complete_session(
    session_id: UUID,
    payload: CompleteSession,
    response: Response,
    token: Annotated[str, Depends(require_token)],
    idempotency_key: Annotated[UUID, Header(alias="Idempotency-Key")],
):
    if idempotency_key != payload.client_turn_id:
        raise HTTPException(status_code=400, detail="Idempotency key must match client_turn_id")
    player_action = "Afslut sagen."
    async with session_factory() as db, db.begin():
        session = await owned_session(db, session_id, token, lock=True)
        existing = (
            await db.execute(
                select(TurnRequest).where(
                    TurnRequest.session_id == session.id,
                    TurnRequest.client_turn_id == payload.client_turn_id,
                )
            )
        ).scalar_one_or_none()
        if existing is not None:
            if (
                existing.expected_turn != payload.expected_turn
                or existing.text != player_action
                or existing.input_language != "da"
            ):
                raise HTTPException(status_code=409, detail="Idempotency key already used")
            committed = (
                await db.execute(
                    select(CommittedTurn).where(CommittedTurn.request_id == existing.id)
                )
            ).scalar_one_or_none()
            if (
                committed is None
                or not isinstance(committed.result, dict)
                or committed.result.get("completion_reason") != "player_end"
            ):
                raise HTTPException(status_code=409, detail="Idempotency key already used")
            response.headers["Location"] = f"/api/v1/turn-requests/{existing.id}"
            return {**public_request(existing), "result": committed.result}

        if session.runtime_version != "v2_ai":
            raise HTTPException(status_code=409, detail="Only schema-v2 sessions can be completed")
        if session.state.get("completion_status") == "complete":
            raise HTTPException(status_code=409, detail="Session is already complete")
        initialization = await get_session_job(db, session.id)
        if initialization is None or initialization.status != "ready":
            raise HTTPException(status_code=409, detail="Session initialization is not ready")
        scenario = await db.get(ScenarioVersion, session.scenario_version_id)
        if scenario is None or scenario.schema_version != 2 or scenario.definition is None:
            raise HTTPException(status_code=409, detail="Pinned schema-v2 scenario is unavailable")
        try:
            definition = ScenarioDefinitionV2.model_validate(scenario.definition)
            validate_v2_definition(definition)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Pinned scenario definition is invalid") from exc
        if definition.completion_policy.mode not in {"goal_or_player_end", "player_end"}:
            raise HTTPException(
                status_code=409,
                detail="The pinned completion policy requires its objectives to be completed",
            )
        pending = (
            await db.execute(
                select(TurnRequest).where(
                    TurnRequest.session_id == session.id,
                    TurnRequest.status.in_(ACTIVE_STATUSES),
                )
            )
        ).scalar_one_or_none()
        if pending is not None:
            raise HTTPException(
                status_code=409,
                detail={"message": "A turn is already in progress", "request_id": str(pending.id)},
            )
        if payload.expected_turn != session.turn_number:
            raise HTTPException(status_code=409, detail="Expected turn does not match session")

        turn_number = session.turn_number + 1
        event = SceneCompletedEvent(event_type="scene_completed", reason="player_end")
        state = project_state(session.state, [event])
        state["turn_number"] = turn_number
        state["last_turn_number"] = turn_number
        event_result = event.model_dump(mode="json")
        result = {
            "turn_number": turn_number,
            "player_text": player_action,
            "reply": "Forløbet afsluttes efter dit ønske.",
            "dialogue": [],
            "outcome_summary": "Spilleren afsluttede forløbet.",
            "next_player_prompt": None,
            "scene_status": "complete",
            "completion_reason": "player_end",
            "provider": "player-requested-completion",
            "scenario_version_id": str(session.scenario_version_id),
            "scenario_version": scenario.version,
            "events": [event_result],
            "state": state,
        }
        turn = TurnRequest(
            id=uuid4(),
            session_id=session.id,
            client_turn_id=payload.client_turn_id,
            expected_turn=session.turn_number,
            text=player_action,
            input_language="da",
            status="completed",
        )
        db.add(turn)
        await db.flush()
        committed = CommittedTurn(
            id=uuid4(),
            session_id=session.id,
            request_id=turn.id,
            turn_number=turn_number,
            result=result,
        )
        db.add(committed)
        await db.flush()
        db.add_all(
            [
                GameEvent(
                    id=uuid4(),
                    committed_turn_id=committed.id,
                    event_index=0,
                    event_type="player_argument",
                    payload={"text": player_action},
                ),
                GameEvent(
                    id=uuid4(),
                    committed_turn_id=committed.id,
                    event_index=1,
                    event_type="scene_completed",
                    payload={"reason": "player_end"},
                ),
            ]
        )
        session.turn_number = turn_number
        session.state = state
        response.headers["Location"] = f"/api/v1/turn-requests/{turn.id}"
        return {**public_request(turn), "result": result}


def _reflection_turn_manifest_query(session_id: UUID):
    """Fetch only per-turn integrity fields, never the cumulative state snapshot."""
    return (
        select(
            CommittedTurn.turn_number,
            CommittedTurn.result["events"].label("event_manifest"),
            CommittedTurn.result["turn_number"].as_integer().label("result_turn_number"),
            CommittedTurn.result["scenario_version_id"]
            .as_string()
            .label("result_scenario_version_id"),
            CommittedTurn.result["player_text"].as_string().label("result_player_text"),
            TurnRequest.session_id,
            TurnRequest.expected_turn,
            TurnRequest.text.label("request_text"),
            TurnRequest.status.label("request_status"),
        )
        .select_from(CommittedTurn)
        .join(TurnRequest, TurnRequest.id == CommittedTurn.request_id)
        .where(CommittedTurn.session_id == session_id)
        .order_by(CommittedTurn.turn_number)
    )


@router.post("/sessions/{session_id}/reflection")
async def reflect_session(
    session_id: UUID,
    token: Annotated[str, Depends(require_token)],
    generator=Depends(get_reflection_generator),
    retrieval_service=Depends(get_reflection_retrieval_service),
):
    async with session_factory() as db:
        session = await owned_session(db, session_id, token)
        if session.runtime_version != "v2_ai":
            raise HTTPException(status_code=409, detail="Only schema-v2 sessions support reflection")
        if session.state.get("completion_status") != "complete":
            raise HTTPException(status_code=409, detail="Session must be complete before reflection")

        scenario = await db.get(ScenarioVersion, session.scenario_version_id)
        if scenario is None or scenario.schema_version != 2 or not isinstance(scenario.definition, dict):
            raise HTTPException(status_code=409, detail="Pinned schema-v2 scenario is unavailable")
        try:
            definition = ScenarioDefinitionV2.model_validate(scenario.definition)
            validate_v2_definition(definition)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail="Pinned scenario definition is invalid") from exc
        if definition.slug != scenario.slug or definition.version != scenario.version:
            raise HTTPException(status_code=409, detail="Pinned scenario identity is invalid")

        turn_rows = (
            await db.execute(_reflection_turn_manifest_query(session.id))
        ).all()
        committed_turns: dict[int, dict] = {}
        for row in turn_rows:
            turn_number = row.turn_number
            if (
                row.session_id != session.id
                or row.expected_turn != turn_number - 1
                or row.request_status != "completed"
                or row.result_turn_number != turn_number
                or row.result_scenario_version_id != str(session.scenario_version_id)
                or row.result_player_text != row.request_text
                or not isinstance(row.request_text, str)
            ):
                raise HTTPException(status_code=409, detail="Committed reflection turn identity is invalid")
            committed_turns[turn_number] = {
                "manifest": row.event_manifest,
                "player_action": row.request_text,
                "events": [],
            }
        if sorted(committed_turns) != list(range(1, session.turn_number + 1)):
            raise HTTPException(status_code=409, detail="Committed reflection history is incomplete")

        event_rows = (
            await db.execute(
                select(
                    CommittedTurn.turn_number,
                    GameEvent.event_index,
                    GameEvent.event_type,
                    GameEvent.payload,
                )
                .select_from(CommittedTurn)
                .join(GameEvent, GameEvent.committed_turn_id == CommittedTurn.id)
                .where(CommittedTurn.session_id == session.id)
                .order_by(CommittedTurn.turn_number, GameEvent.event_index)
            )
        ).all()
        for turn_number, event_index, event_type, payload in event_rows:
            committed_turns[turn_number]["events"].append(
                {
                    "turn_number": turn_number,
                    "event_index": event_index,
                    "event_type": event_type,
                    "payload": payload,
                }
            )

        events = []
        player_actions_by_turn = {}
        for turn_number, entry in committed_turns.items():
            turn_events = entry["events"]
            manifest = entry["manifest"]
            if (
                not isinstance(manifest, list)
                or any(
                    not isinstance(item, dict) or not isinstance(item.get("event_type"), str)
                    for item in manifest
                )
                or len(turn_events) != len(manifest) + 1
                or [item["event_index"] for item in turn_events]
                != list(range(len(turn_events)))
            ):
                raise HTTPException(status_code=409, detail="Committed reflection history is incomplete")
            events.extend(turn_events)
            player_actions_by_turn[turn_number] = entry["player_action"]
        role = next(
            (item for item in definition.roles if item.id == session.player_role_id), None
        )
        if role is None:
            raise HTTPException(status_code=409, detail="Pinned player role is unavailable")
        try:
            validate_reflection_events(
                events,
                player_actions_by_turn=player_actions_by_turn,
                expected_actor_id=role.actor_id,
            )
        except ReflectionBuildError as exc:
            raise HTTPException(status_code=409, detail="Committed reflection events are invalid") from exc
        completion_events = [
            event
            for event in events
            if event["event_type"] == "scene_completed"
        ]
        completion_reason = session.state.get("completion_reason")
        if (
            not completion_events
            or len(completion_events) != 1
            or completion_events[0]["payload"].get("reason") != completion_reason
            or completion_events[0] != events[-1]
        ):
            raise HTTPException(status_code=409, detail="Committed completion history is unavailable")

        query = build_reflection_search_query(definition, events)
        try:
            postgame_context = await build_session_context(
                db,
                session,
                stage="reflection",
                mode="postgame",
                query=query,
                retrieval_service=retrieval_service,
                budget=ContextBudget(max_chars=12_000, top_k=5),
                release_transaction_before_retrieval=True,
            )
        except ContextBuildFailure as exc:
            if exc.status != "retrieval_failure":
                status_code = 503 if exc.status == "database_failure" else 409
                raise HTTPException(
                    status_code=status_code,
                    detail="Pinned postgame context is unavailable",
                ) from exc
            try:
                world_date = date.fromisoformat(session.state["world_date"])
            except (KeyError, TypeError, ValueError) as date_exc:
                raise HTTPException(
                    status_code=409, detail="Pinned postgame context is invalid"
                ) from date_exc
            postgame_context = SessionPromptContext(
                payload={
                    "pinned_scenario": {
                        "slug": scenario.slug,
                        "version": scenario.version,
                        "schema_version": 2,
                        "release_id": str(scenario.id),
                    },
                    "generation_mode": "postgame",
                    "evidence": {"status": "retrieval_failure", "passages": []},
                },
                world_date=world_date,
                evidence_status="retrieval_failure",
                serialized_chars=0,
                stage="reflection",
                mode="postgame",
                metadata={"evidence": {"status": "retrieval_failure"}},
            )

        try:
            reflection_input = build_reflection_input(
                definition=definition,
                events=events,
                postgame_context=postgame_context,
                completion_state=session.state,
            )
            try:
                selected_generator = select_provider(generator, session.narrator)
            except NarratorUnavailable as exc:
                raise HTTPException(
                    status_code=503,
                    detail={"code": "NARRATOR_UNAVAILABLE", "narrator": session.narrator},
                ) from exc
            generation = await generate_reflection(selected_generator, reflection_input)
            await revalidate_reflection_sources(
                db=db,
                session=session,
                definition=definition,
                events=events,
                retrieval_service=retrieval_service,
                reflection_input=reflection_input,
                generation=generation,
            )
            return public_reflection_result(generation, reflection_input)
        except ReflectionBuildError as exc:
            raise HTTPException(status_code=409, detail="Committed reflection history is invalid") from exc
        except ReflectionGenerationError as exc:
            logger.warning(
                "reflection_generation_unavailable reason=%s",
                exc.diagnostic_code or "reflection_rejected",
            )
            raise HTTPException(
                status_code=503,
                detail={"code": "REFLECTION_UNAVAILABLE", "message": "Refleksionen er midlertidigt utilgængelig."},
            ) from exc
        except ContextBuildFailure as exc:
            raise HTTPException(
                status_code=503,
                detail={"code": "REFLECTION_UNAVAILABLE", "message": "Refleksionen er midlertidigt utilgængelig."},
            ) from exc


@router.get("/turn-requests/{request_id}")
async def get_turn_request(request_id: UUID, token: Annotated[str, Depends(require_token)]):
    async with session_factory() as db:
        turn = await db.get(TurnRequest, request_id)
        if turn is None:
            raise HTTPException(status_code=404, detail="Turn request not found")
        await owned_session(db, turn.session_id, token, lock=True)
        # A completion may have committed while the session lock was acquired.
        await db.refresh(turn)
        result = (await db.execute(
            select(CommittedTurn).where(CommittedTurn.request_id == turn.id)
        )).scalar_one_or_none()
        return {
            **public_request(turn),
            "failure_code": turn.failure_code,
            "failure": public_turn_failure(turn),
            "attempt_metadata": turn.attempt_metadata or [],
            "result": result.result if result else None,
        }


@router.post("/turn-requests/{request_id}/retry", status_code=202)
async def retry_turn_request(
    request_id: UUID,
    token: Annotated[str, Depends(require_token)],
):
    async with session_factory() as db, db.begin():
        initial = (
            await db.execute(select(TurnRequest).where(TurnRequest.id == request_id))
        ).scalar_one_or_none()
        if initial is None:
            raise HTTPException(status_code=404, detail="Turn request not found")
        await owned_session(db, initial.session_id, token, lock=True)
        turn = (
            await db.execute(
                select(TurnRequest)
                .where(TurnRequest.id == request_id)
                .with_for_update()
            )
        ).scalar_one_or_none()
        if turn is None:
            raise HTTPException(status_code=404, detail="Turn request not found")
        if turn.status in {"queued", "running", "completed"}:
            return public_request(turn)
        if turn.status != "retryable_failure" or turn.client_retry_used:
            raise HTTPException(status_code=409, detail="Turn request has no retry remaining")
        now = (await db.execute(select(func.clock_timestamp()))).scalar_one()
        turn.status = "queued"
        turn.attempts = 0
        turn.client_retry_used = True
        turn.failure_code = None
        turn.failure_stage = None
        turn.failure_message = None
        turn.lease_token = None
        turn.leased_until = None
        turn.next_attempt_at = now
        turn.updated_at = now
        return public_request(turn)


@router.get("/sessions/{session_id}/turns")
async def get_turns(session_id: UUID, token: Annotated[str, Depends(require_token)]):
    async with session_factory() as db:
        await owned_session(db, session_id, token)
        rows = (await db.execute(select(CommittedTurn).where(
            CommittedTurn.session_id == session_id
        ).order_by(CommittedTurn.turn_number))).scalars().all()
        return {"turns": [row.result for row in rows]}
