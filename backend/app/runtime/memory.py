"""Rebuild a v2 session projection from its durable committed event stream."""
from __future__ import annotations

from copy import deepcopy
from uuid import UUID

from sqlalchemy import select

from app.db.connection import session_factory
from app.db.models import (
    CommittedTurn,
    GameEvent,
    GameSession,
    SessionInitializationJob,
    TurnRequest,
)
from app.runtime.state import StateProjectionError, project_state


class SessionMemoryRebuildError(ValueError):
    """A session's durable initialization or committed event history is incomplete."""


async def rebuild_session_memory(session_id: UUID, *, persist: bool = False) -> dict:
    """Reconstruct a v2 session's state projection, optionally repairing its JSONB cache.

    The committed initialization result supplies the turn-zero seed. Every later state
    change is folded from the session's ordered ``GameEvent`` rows. ``CommittedTurn``
    results and the current ``GameSession.state`` are treated as caches, not history.
    """
    if not isinstance(session_id, UUID):
        raise SessionMemoryRebuildError("session id is invalid")

    async with session_factory() as db, db.begin():
        game = (
            await db.execute(
                select(GameSession).where(GameSession.id == session_id).with_for_update()
            )
        ).scalar_one_or_none()
        if game is None:
            raise SessionMemoryRebuildError("session does not exist")
        if game.runtime_version != "v2_ai":
            raise SessionMemoryRebuildError("event memory rebuild requires a v2 session")

        initialization = (
            await db.execute(
                select(SessionInitializationJob).where(
                    SessionInitializationJob.session_id == session_id
                )
            )
        ).scalar_one_or_none()
        if (
            initialization is None
            or initialization.status != "ready"
            or not isinstance(initialization.result, dict)
            or not isinstance(initialization.result.get("initial_state"), dict)
        ):
            raise SessionMemoryRebuildError("ready initialization state is unavailable")

        try:
            state = project_state(initialization.result["initial_state"], [])
        except StateProjectionError as exc:
            raise SessionMemoryRebuildError("ready initialization state is invalid") from exc

        if persist:
            active_request_id = await db.scalar(
                select(TurnRequest.id)
                .where(
                    TurnRequest.session_id == session_id,
                    TurnRequest.status.in_(("queued", "running", "retryable_failure")),
                )
                .limit(1)
            )
            if active_request_id is not None:
                raise SessionMemoryRebuildError(
                    "cannot persist rebuilt memory while a turn request is active"
                )

        rows = (
            await db.execute(
                select(
                    CommittedTurn.turn_number,
                    CommittedTurn.result,
                    GameEvent.event_index,
                    GameEvent.event_type,
                    GameEvent.payload,
                )
                .select_from(CommittedTurn)
                .outerjoin(GameEvent, GameEvent.committed_turn_id == CommittedTurn.id)
                .where(CommittedTurn.session_id == session_id)
                .order_by(CommittedTurn.turn_number, GameEvent.event_index)
            )
        ).all()

        events_by_turn: dict[int, list[tuple[int, str, dict]]] = {}
        result_by_turn: dict[int, dict] = {}
        for turn_number, result, event_index, event_type, payload in rows:
            turn_events = events_by_turn.setdefault(turn_number, [])
            result_by_turn[turn_number] = result
            if event_index is None:
                continue
            if not isinstance(event_type, str) or not isinstance(payload, dict):
                raise SessionMemoryRebuildError("committed event envelope is invalid")
            turn_events.append((event_index, event_type, payload))

        turn_numbers = sorted(events_by_turn)
        expected_turn_numbers = list(range(1, game.turn_number + 1))
        if turn_numbers != expected_turn_numbers:
            raise SessionMemoryRebuildError("committed turn history is incomplete")

        for turn_number in turn_numbers:
            turn_events = events_by_turn[turn_number]
            result = result_by_turn[turn_number]
            manifest_events = result.get("events") if isinstance(result, dict) else None
            if (
                not isinstance(manifest_events, list)
                or any(
                    not isinstance(event, dict)
                    or not isinstance(event.get("event_type"), str)
                    for event in manifest_events
                )
            ):
                raise SessionMemoryRebuildError(
                    "committed turn event count manifest is invalid"
                )
            if len(turn_events) != len(manifest_events) + 1:
                raise SessionMemoryRebuildError("committed event history is incomplete")
            indices = [event_index for event_index, _, _ in turn_events]
            if indices != list(range(len(turn_events))):
                raise SessionMemoryRebuildError("committed event indexes are incomplete")
            if not turn_events or turn_events[0][1] != "player_argument":
                raise SessionMemoryRebuildError("committed player action is missing")
            if sum(event_type == "player_argument" for _, event_type, _ in turn_events) != 1:
                raise SessionMemoryRebuildError("committed turn has invalid player actions")
            player_payload = turn_events[0][2]
            if not isinstance(player_payload.get("text"), str):
                raise SessionMemoryRebuildError("committed player action is invalid")

            state_events = [
                {"event_type": event_type, "payload": payload}
                for _, event_type, payload in turn_events[1:]
            ]
            try:
                state = project_state(state, state_events)
            except StateProjectionError as exc:
                raise SessionMemoryRebuildError(
                    f"committed state events are invalid at turn {turn_number}"
                ) from exc
            state["turn_number"] = turn_number
            state["last_turn_number"] = turn_number

        rebuilt = state
        if persist:
            game.state = rebuilt
        return deepcopy(rebuilt)
