from __future__ import annotations

import importlib
from types import SimpleNamespace
from uuid import UUID, uuid4

import pytest
from sqlalchemy import select
from sqlalchemy.dialects import postgresql

from app.db.connection import session_factory
from app.db.models import (
    CommittedTurn,
    GameEvent,
    HistoricalPassage,
    HistoricalPassageReview,
)
from app.game import routes
from app.main import app
from app.rag.policy import EvidencePassage, PassageApproval
from app.rag.retrieval import RAG_NO_EVIDENCE, RagSearchResult, SearchCandidate
from app.scenarios.loader import load_scenario

FIXTURE = "tests/fixtures/scenarios/industrial-inquiry/1.0.yaml"
PASSAGE_ID = "a6a00000-0000-4000-8000-000000000001"
OTHER_PASSAGE_ID = "a6a00000-0000-4000-8000-000000000002"


def _reflection_service():
    try:
        return importlib.import_module("app.runtime.reflection")
    except ModuleNotFoundError as exc:
        if exc.name == "app.runtime.reflection":
            pytest.fail("generic session reflection service is not implemented")
        raise


def _definition():
    return load_scenario(FIXTURE).definition


def _event_metadata(definition):
    role = next(role for role in definition.roles if role.id == definition.default_role_id)
    return {
        "actor_id": role.actor_id,
        "world_date": definition.chronology.opening_date.isoformat(),
    }


def _postgame_context(passages):
    definition = _definition()
    return SimpleNamespace(
        payload={
            "pinned_scenario": {"slug": definition.slug, "version": definition.version},
            "evidence": {"status": "evidence_included", "passages": passages},
        },
        evidence_status="evidence_included",
        stage="reflection",
        mode="postgame",
    )


def _event(turn, index, event_type, payload):
    return {
        "turn_number": turn,
        "event_index": index,
        "event_type": event_type,
        "payload": payload,
    }


def test_reflection_input_keeps_historical_evidence_separate_from_playthrough_claims():
    reflection = _reflection_service()
    definition = _definition()
    eligible_passage = {
        "passage_id": PASSAGE_ID,
        "scenario_slug": definition.slug,
        "scenario_version": definition.version,
        "passage_text": "Et godkendt uddrag beskriver arbejdsdagen.",
        "source_identifier": "R01",
        "source_title": "Arbejdsrapport",
        "source_url": "https://example.test/report",
        "source_anchor": "side 2",
        "document_year": "1888",
        "citation_ready": True,
        "trust_category": "historical_evidence",
    }
    wrong_release_passage = {
        **eligible_passage,
        "passage_id": OTHER_PASSAGE_ID,
        "scenario_slug": "another-case",
    }
    events = [
        _event(1, 0, "player_argument", {"text": "Sammenlign løn og arbejdstid."}),
        _event(
            1,
            1,
            "decision_recorded",
            {
                "summary": "Undersøg lønnen før konklusionen.",
                "player_action": "Sammenlign løn og arbejdstid.",
            },
        ),
        _event(
            1,
            2,
            "simulation_event_recorded",
            {"summary": "En simuleret samtale afdækker uenighed.", "actor_ids": ["factory_owner"]},
        ),
        _event(
            1,
            3,
            "player_claim_recorded",
            {
                "text": "Arbejdstiden blev halveret.",
                "truth_type": "player_claim",
                **_event_metadata(definition),
            },
        ),
        _event(
            1,
            4,
            "historical_claim_recorded",
            {
                "claim": "Kilden omtaler arbejdsdagen.",
                "evidence_ids": [PASSAGE_ID],
                "truth_type": "historical_evidence",
                **_event_metadata(definition),
            },
        ),
        _event(
            1,
            5,
            "historical_claim_recorded",
            {
                "claim": "En anden sag dokumenterer påstanden.",
                "evidence_ids": [OTHER_PASSAGE_ID],
                "truth_type": "historical_evidence",
                **_event_metadata(definition),
            },
        ),
        _event(
            1,
            6,
            "learning_observation_recorded",
            {
                "objective_id": "weigh_evidence",
                "observation": "Sammenholdt to forklaringer.",
                "truth_type": "learning_observation",
                **_event_metadata(definition),
            },
        ),
    ]

    result = reflection.build_reflection_input(
        definition=definition,
        events=events,
        postgame_context=_postgame_context([eligible_passage, wrong_release_passage]),
        completion_state={
            "completion_status": "complete",
            "completion_reason": "player_end",
            "turn_number": 1,
        },
    )

    assert result["player_choices"] == [
        {
            "turn_number": 1,
            "event_index": 1,
            "action": "Sammenlign løn og arbejdstid.",
            "summary": "Undersøg lønnen før konklusionen.",
        }
    ]
    assert result["simulated_consequences"][0]["truth_type"] == "simulation"
    assert result["player_claims"][0]["truth_type"] == "player_claim"
    assert result["documented_history"] == [
        {
            "turn_number": 1,
            "claim": "Kilden omtaler arbejdsdagen.",
            "evidence_ids": [PASSAGE_ID],
            "truth_type": "historical_evidence",
        }
    ]
    assert result["objectives"][0]["id"] == "weigh_evidence"
    assert result["objectives"][0]["observation_status"] == "observed"
    assert result["citation_ready_evidence_ids"] == [PASSAGE_ID]
    assert result["unsupported_historical_claim_count"] == 1
    assert result["completion"]["reason"] == "player_end"


def test_reflection_reports_evidence_unavailable_instead_of_reusing_old_citations():
    reflection = _reflection_service()
    definition = _definition()
    result = reflection.build_reflection_input(
        definition=definition,
        events=[
            _event(
                1,
                2,
                "historical_claim_recorded",
                {
                    "claim": "Påstanden blev tidligere kildebelagt.",
                    "evidence_ids": [PASSAGE_ID],
                    "truth_type": "historical_evidence",
                    **_event_metadata(definition),
                },
            )
        ],
        postgame_context=_postgame_context([]),
        completion_state={
            "completion_status": "complete",
            "completion_reason": "turn_limit",
            "turn_number": 24,
        },
    )

    assert result["evidence_status"] == "evidence_unavailable"
    assert result["documented_history"] == []
    assert result["citation_ready_evidence_ids"] == []
    assert result["unsupported_historical_claim_count"] == 1


def test_reflection_preserves_retrieval_failure_as_uncertainty():
    reflection = _reflection_service()
    definition = _definition()
    context = _postgame_context([])
    context.evidence_status = "retrieval_failure"
    result = reflection.build_reflection_input(
        definition=definition,
        events=[],
        postgame_context=context,
        completion_state={
            "completion_status": "complete",
            "completion_reason": "player_end",
            "turn_number": 1,
        },
    )

    assert result["evidence_status"] == "retrieval_failure"


def test_reflection_rejects_unknown_and_malformed_committed_event_types():
    reflection = _reflection_service()
    definition = _definition()
    completion_state = {
        "completion_status": "complete",
        "completion_reason": "player_end",
        "turn_number": 1,
    }

    for event in (
        _event(1, 0, "unrecognized_event", {"summary": "Skjult hændelse."}),
        _event(1, 0, "decision_recorded", {"summary": "Beslutning uden handling."}),
    ):
        with pytest.raises(reflection.ReflectionBuildError):
            reflection.build_reflection_input(
                definition=definition,
                events=[event],
                postgame_context=_postgame_context([]),
                completion_state=completion_state,
            )


def test_reflection_history_query_selects_event_manifest_not_cumulative_state():
    statement = routes._reflection_turn_manifest_query(UUID(int=1))
    selected = [getattr(item, "element", item) for item in statement.selected_columns]

    assert not any(
        getattr(item, "key", None) == "result"
        and getattr(item, "table", None) is CommittedTurn.__table__
        for item in selected
    )
    manifest = next(
        item
        for item in selected
        if getattr(item, "left", None) is not None
        and getattr(item.left, "key", None) == "result"
    )
    assert manifest.right.value == "events"
    assert "result[" in str(statement.compile(dialect=postgresql.dialect()))


class FakeReflectionGenerator:
    def __init__(self, output):
        self.output = output
        self.calls = []

    async def generate_structured(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output=self.output,
            requested_model="test-reflector",
            served_model="test-reflector",
            model_revision="test-revision",
            provider_request_id="reflection-test-request",
            prompt_tokens=100,
            completion_tokens=80,
            duration_ms=12,
            raw_output_sha256="b" * 64,
        )


class NoEvidenceRetriever:
    async def search(self, query, context, *, top_k):
        return RagSearchResult(status=RAG_NO_EVIDENCE, passages=[])


class FailedRetriever:
    async def search(self, query, context, *, top_k):
        raise RuntimeError("retrieval service unavailable")


class ApprovedPassageRetriever:
    def __init__(self, candidate):
        self.candidate = candidate
        self.calls = 0

    async def search(self, query, context, *, top_k):
        self.calls += 1
        async with session_factory() as db:
            approval = (
                await db.execute(
                    select(HistoricalPassageReview).where(
                        HistoricalPassageReview.passage_id == self.candidate.passage.id
                    )
                    .order_by(HistoricalPassageReview.revision.desc())
                    .limit(1)
                )
            ).scalar_one()
        candidate = SearchCandidate(
            passage=self.candidate.passage,
            approval=PassageApproval(
                review_status=approval.review_status,
                rights_status=approval.rights_status,
                revision=approval.revision,
            ),
        )
        return RagSearchResult(status="ok", passages=[candidate])


@pytest.mark.asyncio
async def test_reflection_endpoint_requires_completion_and_reports_missing_evidence(turn_harness):
    reflection = _reflection_service()
    session = await turn_harness.create_ready_session()
    headers = {"Authorization": f"Bearer {session['token']}"}
    endpoint = f"/api/v1/sessions/{session['id']}/reflection"

    incomplete = await turn_harness.client.post(endpoint, headers=headers)
    assert incomplete.status_code == 409

    completion_id = UUID(int=1234567)
    completion = await turn_harness.client.post(
        f"/api/v1/sessions/{session['id']}/completion",
        json={"client_turn_id": str(completion_id), "expected_turn": 0},
        headers={**headers, "Idempotency-Key": str(completion_id)},
    )
    assert completion.status_code == 201
    generator = FakeReflectionGenerator(
        {
            "language": "da",
            "summary": "Du afsluttede forløbet før du undersøgte sagen nærmere.",
            "objective_feedback": [
                {
                    "objective_id": "weigh_evidence",
                    "reflection": "Der blev ikke registreret en observation for målet.",
                }
            ],
            "historical_comparison": None,
            "historical_evidence_ids": [],
            "uncertainty_notes": ["Forløbet indeholder endnu ingen undersøgte valg."],
        }
    )
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[reflection.get_reflection_generator] = lambda: generator
    app.dependency_overrides[reflection.get_reflection_retrieval_service] = (
        lambda: NoEvidenceRetriever()
    )
    try:
        response = await turn_harness.client.post(endpoint, headers=headers)
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "ready"
    assert body["evidence_status"] == "evidence_unavailable"
    assert body["historical_comparison"] == (
        "Der var ikke godkendt, citation-egnet historisk belæg tilgængeligt til denne refleksion."
    )
    assert body["sources"] == []
    assert body["objectives"][0]["observation_status"] == "not_observed"
    assert "grade" not in body

    app.dependency_overrides[reflection.get_reflection_generator] = lambda: generator
    app.dependency_overrides[reflection.get_reflection_retrieval_service] = (
        lambda: FailedRetriever()
    )
    try:
        failed_search = await turn_harness.client.post(endpoint, headers=headers)
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)
    assert failed_search.status_code == 200, failed_search.text
    assert failed_search.json()["evidence_status"] == "retrieval_failure"
    assert failed_search.json()["historical_comparison"] == (
        "Den historiske kildesøgning kunne ikke gennemføres, så sammenligningen er uafklaret."
    )


@pytest.mark.asyncio
async def test_reflection_discards_citations_revoked_during_generation(turn_harness):
    reflection = _reflection_service()
    session = await turn_harness.create_ready_session()
    headers = {"Authorization": f"Bearer {session['token']}"}
    completion_id = UUID(int=1234568)
    completion = await turn_harness.client.post(
        f"/api/v1/sessions/{session['id']}/completion",
        json={"client_turn_id": str(completion_id), "expected_turn": 0},
        headers={**headers, "Idempotency-Key": str(completion_id)},
    )
    assert completion.status_code == 201, completion.text

    async with session_factory() as db:
        passage = (
            await db.execute(
                select(HistoricalPassage).where(
                    HistoricalPassage.scenario_slug == turn_harness.slug
                )
            )
        ).scalar_one()
    candidate = SearchCandidate(
        passage=EvidencePassage(
            id=passage.id,
            scenario_version_id=passage.scenario_version_id,
            scenario_slug=passage.scenario_slug,
            scenario_version=passage.scenario_version,
            passage_key=passage.passage_key,
            source_identifier=passage.source_identifier,
            source_title=passage.source_title,
            source_url=passage.source_url,
            source_anchor=passage.source_anchor,
            text=passage.passage_text,
            passage_sha256=passage.passage_sha256,
            document_date=passage.document_date,
            in_world_available_from=passage.in_world_available_from,
            postgame_only=passage.postgame_only,
            actor_ids=tuple(passage.actor_ids),
            visibility_scope=passage.visibility_scope,
        ),
        approval=PassageApproval("approved", "approved", 1),
    )
    retriever = ApprovedPassageRetriever(candidate)

    class RevokingGenerator(FakeReflectionGenerator):
        async def generate_structured(self, **kwargs):
            response = await super().generate_structured(**kwargs)
            async with session_factory() as db, db.begin():
                current_review = (
                    await db.execute(
                        select(HistoricalPassageReview).where(
                            HistoricalPassageReview.passage_id == passage.id
                        )
                        .order_by(HistoricalPassageReview.revision.desc())
                        .limit(1)
                    )
                ).scalar_one()
                db.add(
                    HistoricalPassageReview(
                        id=uuid4(),
                        passage_id=passage.id,
                        revision=current_review.revision + 1,
                        review_status="approved",
                        rights_status="pending",
                        reviewer="reflection-revocation-test",
                    )
                )
            return response

    generator = RevokingGenerator(
        {
            "language": "da",
            "summary": "Du sammenholdt et vidneudsagn med journalen.",
            "objective_feedback": [
                {
                    "objective_id": "weigh_evidence",
                    "reflection": "Der blev ikke registreret en observation for målet.",
                }
            ],
            "historical_comparison": "Journalen beskriver arbejdsdagen.",
            "historical_evidence_ids": [str(passage.id)],
            "uncertainty_notes": ["Journaluddraget dækker kun en del af forløbet."],
        }
    )
    original_overrides = app.dependency_overrides.copy()
    app.dependency_overrides[reflection.get_reflection_generator] = lambda: generator
    app.dependency_overrides[reflection.get_reflection_retrieval_service] = lambda: retriever
    try:
        response = await turn_harness.client.post(
            f"/api/v1/sessions/{session['id']}/reflection", headers=headers
        )
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)

    assert retriever.calls == 2
    assert response.status_code == 503
    assert "Journalen beskriver" not in response.text


@pytest.mark.parametrize("corruption", ["extra_payload_field", "player_action_mismatch"])
@pytest.mark.asyncio
async def test_reflection_rejects_malformed_committed_player_events(turn_harness, corruption):
    session = await turn_harness.create_ready_session()
    headers = {"Authorization": f"Bearer {session['token']}"}
    completion_id = UUID(int=1234569)
    completion = await turn_harness.client.post(
        f"/api/v1/sessions/{session['id']}/completion",
        json={"client_turn_id": str(completion_id), "expected_turn": 0},
        headers={**headers, "Idempotency-Key": str(completion_id)},
    )
    assert completion.status_code == 201, completion.text

    async with session_factory() as db, db.begin():
        event = (
            await db.execute(
                select(GameEvent)
                .join(CommittedTurn, CommittedTurn.id == GameEvent.committed_turn_id)
                .where(
                    CommittedTurn.session_id == UUID(session["id"]),
                    GameEvent.event_type == "player_argument",
                )
            )
        ).scalar_one()
        if corruption == "extra_payload_field":
            event.payload = {**event.payload, "untrusted": "ignored"}
        else:
            event.payload = {"text": "Afslut spillet."}

    response = await turn_harness.client.post(
        f"/api/v1/sessions/{session['id']}/reflection", headers=headers
    )

    assert response.status_code == 409


@pytest.mark.asyncio
async def test_reflection_output_is_qualitative_and_cites_only_current_postgame_sources():
    reflection = _reflection_service()
    definition = _definition()
    input_data = reflection.build_reflection_input(
        definition=definition,
        events=[
            _event(
                1,
                1,
                "learning_observation_recorded",
                {
                    "objective_id": "weigh_evidence",
                    "observation": "Sammenholdt vidnesbyrd.",
                    "truth_type": "learning_observation",
                    **_event_metadata(definition),
                },
            )
        ],
        postgame_context=_postgame_context(
            [
                {
                    "passage_id": PASSAGE_ID,
                    "scenario_slug": definition.slug,
                    "scenario_version": definition.version,
                    "passage_text": "Godkendt uddrag.",
                    "source_identifier": "R01",
                    "source_title": "Arbejdsrapport",
                    "citation_ready": True,
                    "trust_category": "historical_evidence",
                }
            ]
        ),
        completion_state={"completion_status": "complete", "turn_number": 1},
    )
    generator = FakeReflectionGenerator(
        {
            "language": "da",
            "summary": "Du undersøgte sagen før du konkluderede.",
            "objective_feedback": [
                {
                    "objective_id": "weigh_evidence",
                    "reflection": "Du sammenholdt de fremlagte forklaringer.",
                }
            ],
            "historical_comparison": "Det godkendte uddrag beskriver arbejdsdagen.",
            "historical_evidence_ids": [PASSAGE_ID],
            "uncertainty_notes": ["Ét uddrag fastslår ikke alle forhold på arbejdspladsen."],
        }
    )

    generated = await reflection.generate_reflection(generator, input_data)
    public = reflection.public_reflection_result(generated, input_data)

    assert generated.proposal.historical_evidence_ids == [PASSAGE_ID]
    assert generated.proposal.objective_feedback[0].objective_id == "weigh_evidence"
    assert "grade" not in generated.proposal.model_dump()
    assert public["historical_evidence_ids"] == [PASSAGE_ID]
    assert [source["passage_id"] for source in public["sources"]] == [PASSAGE_ID]
    assert generator.calls[0]["schema_name"] == reflection.REFLECTION_SCHEMA_NAME
    assert "historical_evidence_ids" in generator.calls[0]["response_schema"]["properties"]


@pytest.mark.asyncio
async def test_reflection_rejects_unknown_evidence_ids_and_grade_fields():
    reflection = _reflection_service()
    input_data = {
        "objectives": [{"id": "weigh_evidence"}],
        "citation_ready_evidence_ids": [PASSAGE_ID],
        "evidence_status": "evidence_included",
    }
    base = {
        "language": "da",
        "summary": "En kort refleksion.",
        "objective_feedback": [
            {"objective_id": "weigh_evidence", "reflection": "Du begrundede dit valg."}
        ],
        "historical_comparison": "Kilden giver et afgrænset sammenligningspunkt.",
        "historical_evidence_ids": [OTHER_PASSAGE_ID],
        "uncertainty_notes": ["Kilden dækker ikke alle forhold."],
    }

    with pytest.raises(reflection.ReflectionGenerationError):
        await reflection.generate_reflection(FakeReflectionGenerator(base), input_data)

    base["historical_evidence_ids"] = [PASSAGE_ID]
    base["grade"] = "A"
    with pytest.raises(reflection.ReflectionGenerationError):
        await reflection.generate_reflection(FakeReflectionGenerator(base), input_data)


@pytest.mark.asyncio
async def test_reflection_requires_comparison_or_abstention_when_evidence_is_available():
    reflection = _reflection_service()
    input_data = {
        "objectives": [{"id": "weigh_evidence", "observation_status": "observed"}],
        "citation_ready_evidence_ids": [PASSAGE_ID],
        "evidence_status": "evidence_included",
    }
    proposal = {
        "language": "da",
        "summary": "Du sammenholdt to beskrivelser.",
        "objective_feedback": [
            {"objective_id": "weigh_evidence", "reflection": "Du sammenholdt beskrivelserne."}
        ],
        "historical_comparison": None,
        "historical_evidence_ids": [],
        "uncertainty_notes": ["Kildematerialet er afgrænset."],
    }

    with pytest.raises(reflection.ReflectionGenerationError):
        await reflection.generate_reflection(FakeReflectionGenerator(proposal), input_data)

    proposal["historical_abstention"] = (
        "Det tilgængelige uddrag belyser ikke den konkrete beslutning."
    )
    generated = await reflection.generate_reflection(FakeReflectionGenerator(proposal), input_data)
    public = reflection.public_reflection_result(generated, input_data)
    assert public["historical_comparison"] is None
    assert public["historical_abstention"] == proposal["historical_abstention"]


@pytest.mark.asyncio
async def test_reflection_rejects_grade_and_mastery_claims_in_text_fields():
    reflection = _reflection_service()
    input_data = {
        "objectives": [
            {"id": "weigh_evidence", "observation_status": "not_observed", "observations": []}
        ],
        "citation_ready_evidence_ids": [],
        "evidence_status": "evidence_unavailable",
    }
    proposal = {
        "language": "da",
        "summary": "Din karakter er 12.",
        "objective_feedback": [
            {"objective_id": "weigh_evidence", "reflection": "Du har mestret målet."}
        ],
        "historical_comparison": None,
        "historical_evidence_ids": [],
        "uncertainty_notes": ["Der blev ikke registreret en observation for målet."],
    }

    with pytest.raises(reflection.ReflectionGenerationError):
        await reflection.generate_reflection(FakeReflectionGenerator(proposal), input_data)


@pytest.mark.asyncio
async def test_unobserved_objective_uses_grounded_public_feedback():
    reflection = _reflection_service()
    input_data = {
        "objectives": [
            {"id": "weigh_evidence", "observation_status": "not_observed", "observations": []}
        ],
        "citation_ready_evidence_ids": [],
        "evidence_status": "evidence_unavailable",
    }
    proposal = {
        "language": "da",
        "summary": "Du undersøgte sagen.",
        "objective_feedback": [
            {"objective_id": "weigh_evidence", "reflection": "Du begrundede dit valg."}
        ],
        "historical_comparison": None,
        "historical_evidence_ids": [],
        "uncertainty_notes": ["Der blev ikke registreret en observation for målet."],
    }
    generated = await reflection.generate_reflection(FakeReflectionGenerator(proposal), input_data)

    public = reflection.public_reflection_result(generated, input_data)

    assert public["objectives"][0]["reflection"] == (
        "Der blev ikke registreret en observation for dette mål."
    )
