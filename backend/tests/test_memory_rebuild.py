from __future__ import annotations

import copy
import importlib
from uuid import UUID, uuid4

import pytest
from sqlalchemy import func, select

from app.db.connection import session_factory
from app.db.models import CommittedTurn, GameEvent, GameSession, TurnRequest


def _memory_service():
    try:
        return importlib.import_module("app.runtime.memory")
    except ModuleNotFoundError as exc:
        if exc.name == "app.runtime.memory":
            pytest.fail("event-derived session memory rebuild service is not implemented")
        raise


async def _insert_committed_turn(
    db,
    session_id: UUID,
    turn_number: int,
    player_text: str,
    events: list[tuple[str, dict]],
    *,
    attempts: int = 1,
    include_player_argument: bool = True,
) -> None:
    request = TurnRequest(
        id=uuid4(),
        session_id=session_id,
        client_turn_id=uuid4(),
        expected_turn=turn_number - 1,
        text=player_text,
        input_language="da",
        status="completed",
        attempts=attempts,
    )
    db.add(request)
    await db.flush()

    committed = CommittedTurn(
        id=uuid4(),
        session_id=session_id,
        request_id=request.id,
        turn_number=turn_number,
        # Replay must use the ordered event stream, never this cached result.
        result={"state": {"stale_result_snapshot": turn_number}},
    )
    db.add(committed)
    await db.flush()
    if include_player_argument:
        db.add(
            GameEvent(
                id=uuid4(),
                committed_turn_id=committed.id,
                event_index=0,
                event_type="player_argument",
                payload={"text": player_text},
            )
        )
    for index, (event_type, payload) in enumerate(events, start=1):
        db.add(
            GameEvent(
                id=uuid4(),
                committed_turn_id=committed.id,
                event_index=index,
                event_type=event_type,
                payload=payload,
            )
        )


@pytest.mark.asyncio
async def test_rebuild_replays_ten_turns_after_retries_and_restart(turn_harness):
    memory = _memory_service()
    session_body = await turn_harness.create_ready_session()
    session_id = UUID(session_body["id"])
    seed = copy.deepcopy(session_body["state"])
    date_at_start = seed["world_date"]

    turn_events = [
        [("decision_recorded", {"summary": "Keep the exit open", "player_action": "Keep the exit open."})],
        [("thread_opened", {"thread_id": "safety_inspection", "summary": "Inspect the south exit."})],
        [("thread_opened", {"thread_id": "worker_interviews", "summary": "Interview workers."})],
        [(
            "relation_changed",
            {
                "actor_a_id": "factory_owner",
                "actor_b_id": "worker_delegate",
                "relation": "strained",
                "reason": "The inspection plan is disputed.",
            },
        )],
        [(
            "simulation_event_recorded",
            {"summary": "Production pauses during inspection.", "actor_ids": ["worker_delegate"]},
        )],
        [(
            "historical_claim_recorded",
            {
                "truth_type": "historical_evidence",
                "claim": "The reviewed passage describes crowded loft buildings.",
                "evidence_ids": ["passage-reviewed-1"],
                "world_date": date_at_start,
                "actor_id": "commission",
            },
        )],
        [(
            "player_claim_recorded",
            {
                "truth_type": "player_claim",
                "text": "The south exit was always clear.",
                "world_date": date_at_start,
                "actor_id": "worker_delegate",
            },
        )],
        [(
            "learning_observation_recorded",
            {
                "objective_id": "assess_workplace_evidence",
                "observation": "Compared a report passage with a player claim.",
                "world_date": date_at_start,
                "actor_id": "commission",
            },
        )],
        [
            (
                "thread_resolved",
                {"thread_id": "worker_interviews", "resolution": "The interview list is complete."},
            ),
            (
                "relation_changed",
                {
                    "actor_a_id": "factory_owner",
                    "actor_b_id": "worker_delegate",
                    "relation": "contested",
                    "reason": "The parties disagree about the recorded conditions.",
                },
            ),
        ],
        [
            (
                "world_time_advanced",
                {"from_date": date_at_start, "to_date": "1888-05-08"},
            )
        ],
    ]

    async with session_factory() as db, db.begin():
        for turn_number, events in enumerate(turn_events, start=1):
            player_text = "Keep the exit open." if turn_number == 1 else f"Player action {turn_number}"
            await _insert_committed_turn(
                db,
                session_id,
                turn_number,
                player_text,
                events,
                attempts=3 if turn_number == 5 else 1,
            )

        session = await db.get(GameSession, session_id)
        assert session is not None
        expected = copy.deepcopy(seed)
        expected.update(
            {
                "turn_number": 10,
                "last_turn_number": 10,
                "world_date": "1888-05-08",
                "decisions": [
                    {
                        "summary": "Keep the exit open",
                        "player_action": "Keep the exit open.",
                        "truth_type": "player_decision",
                    }
                ],
                "open_threads": [
                    {"id": "safety_inspection", "text": "Inspect the south exit.", "status": "open"},
                    {
                        "id": "worker_interviews",
                        "text": "Interview workers.",
                        "status": "resolved",
                        "resolution": "The interview list is complete.",
                    },
                ],
                "relations": [
                    {
                        "actor_a_id": "factory_owner",
                        "actor_b_id": "worker_delegate",
                        "relation": "contested",
                        "reason": "The parties disagree about the recorded conditions.",
                    }
                ],
                "simulation_events": [
                    {
                        "summary": "Production pauses during inspection.",
                        "actor_ids": ["worker_delegate"],
                        "truth_type": "simulation",
                    }
                ],
                "historical_claims": [
                    {
                        "claim": "The reviewed passage describes crowded loft buildings.",
                        "evidence_ids": ["passage-reviewed-1"],
                        "truth_type": "historical_evidence",
                        "world_date": date_at_start,
                        "actor_id": "commission",
                    }
                ],
                "player_claims": [
                    {
                        "text": "The south exit was always clear.",
                        "truth_type": "player_claim",
                        "world_date": date_at_start,
                        "actor_id": "worker_delegate",
                    }
                ],
                "learning_observations": [
                    {
                        "objective_id": "assess_workplace_evidence",
                        "observation": "Compared a report passage with a player claim.",
                        "truth_type": "learning_observation",
                        "world_date": date_at_start,
                        "actor_id": "commission",
                    }
                ],
                "author_context_references": [],
            }
        )
        session.turn_number = 10
        session.state = expected

    rebuilt = await memory.rebuild_session_memory(session_id)
    assert rebuilt == expected
    assert rebuilt["historical_claims"] == expected["historical_claims"]
    assert rebuilt["player_claims"][0]["truth_type"] == "player_claim"
    assert rebuilt["simulation_events"][0]["truth_type"] == "simulation"
    assert rebuilt["open_threads"][0]["status"] == "open"

    async with session_factory() as db, db.begin():
        session = await db.get(GameSession, session_id)
        assert session is not None
        session.state = {"stale_projection": True}
    assert await memory.rebuild_session_memory(session_id, persist=True) == expected
    async with session_factory() as db:
        session = await db.get(GameSession, session_id)
        assert session is not None
        assert session.state == expected

    # A fresh service call models replay after worker restart; the retry attempt
    # count remains one committed choice, not a duplicated event.
    assert await memory.rebuild_session_memory(session_id) == expected
    async with session_factory() as db:
        turn_count = await db.scalar(
            select(func.count()).select_from(CommittedTurn).where(
                CommittedTurn.session_id == session_id
            )
        )
        player_event_count = await db.scalar(
            select(func.count())
            .select_from(GameEvent)
            .join(CommittedTurn, CommittedTurn.id == GameEvent.committed_turn_id)
            .where(
                CommittedTurn.session_id == session_id,
                GameEvent.event_type == "player_argument",
            )
        )
    assert turn_count == 10
    assert player_event_count == 10


@pytest.mark.asyncio
async def test_rebuild_session_memory_isolates_session_histories(turn_harness):
    memory = _memory_service()
    first = await turn_harness.create_ready_session()
    second = await turn_harness.create_ready_session()
    first_id, second_id = UUID(first["id"]), UUID(second["id"])

    async with session_factory() as db, db.begin():
        for session_id, text in ((first_id, "First session choice"), (second_id, "Second session choice")):
            await _insert_committed_turn(
                db,
                session_id,
                1,
                text,
                [("decision_recorded", {"summary": text, "player_action": text})],
            )
            session = await db.get(GameSession, session_id)
            assert session is not None
            session.turn_number = 1
            session.state = {**session.state, "turn_number": 1, "last_turn_number": 1}

    first_projection = await memory.rebuild_session_memory(first_id)
    second_projection = await memory.rebuild_session_memory(second_id)
    assert [item["summary"] for item in first_projection["decisions"]] == [
        "First session choice"
    ]
    assert [item["summary"] for item in second_projection["decisions"]] == [
        "Second session choice"
    ]


@pytest.mark.asyncio
async def test_incomplete_event_history_cannot_overwrite_saved_projection(turn_harness):
    memory = _memory_service()
    session_body = await turn_harness.create_ready_session()
    session_id = UUID(session_body["id"])
    original_state = {**session_body["state"], "local_marker": "must survive failure"}

    async with session_factory() as db, db.begin():
        await _insert_committed_turn(
            db,
            session_id,
            1,
            "The committed player action is missing",
            [],
            include_player_argument=False,
        )
        session = await db.get(GameSession, session_id)
        assert session is not None
        session.turn_number = 1
        session.state = original_state

    with pytest.raises(memory.SessionMemoryRebuildError):
        await memory.rebuild_session_memory(session_id, persist=True)

    async with session_factory() as db:
        session = await db.get(GameSession, session_id)
        assert session is not None
        assert session.state == original_state
