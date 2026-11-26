"""Strict, bounded contracts for untrusted schema-v2 turn proposals and events."""
from datetime import date
from typing import Annotated, Literal, Union

from pydantic import Field, StrictInt, StringConstraints, field_validator, model_validator

from app.scenarios.schema import Id, ShortText, StrictModel, Text

MAX_STATE_OPERATIONS = 20
MAX_PROPOSAL_JSON_CHARS = 24_000
MAX_EVIDENCE_SEARCH_TOOL_CALLS = 3

CanonicalDate = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
    ),
]
ClaimText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
DialogueText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=600)]
OutcomeSummary = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=500)]
PlayerActionText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2000),
]
ActionExcerpt = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=2, max_length=240),
]


def _canonical_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("date must be a valid ISO calendar date") from exc
    if parsed.isoformat() != value:
        raise ValueError("date must be a canonical ISO calendar date")
    return value


class DialogueLine(StrictModel):
    actor_id: Id
    text: DialogueText


class DelegationIssueResponse(StrictModel):
    actor_id: Id
    stance: Literal["accept", "counter", "reject", "defer"]
    position: ShortText
    counteroffer: Text | None = None

    @model_validator(mode="after")
    def counter_requires_a_term(self):
        if self.stance == "counter" and not self.counteroffer:
            raise ValueError("a counter response requires a counteroffer")
        return self


class NegotiationIssueUpdate(StrictModel):
    issue_id: Id
    action_type: Literal["proposal", "discussion"]
    player_action_excerpt: ActionExcerpt
    recommendation: Text
    responses: list[DelegationIssueResponse] = Field(max_length=16)
    consequence: ShortText
    remaining_disagreement: ShortText


class NegotiationAssessment(StrictModel):
    consequence: ShortText
    remaining_disagreement: ShortText


class RecordDecisionOperation(StrictModel):
    type: Literal["record_decision"]
    summary: ShortText


class OpenThreadOperation(StrictModel):
    type: Literal["open_thread"]
    thread_id: Id
    summary: ShortText


class ResolveThreadOperation(StrictModel):
    type: Literal["resolve_thread"]
    thread_id: Id
    resolution: ShortText


class RecordSimulationEventOperation(StrictModel):
    type: Literal["record_simulation_event"]
    summary: ShortText
    actor_ids: list[Id] = Field(default_factory=list, max_length=8)

    @field_validator("actor_ids")
    @classmethod
    def unique_actor_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("actor_ids must be unique")
        return values


class SetRelationOperation(StrictModel):
    type: Literal["set_relation"]
    actor_a_id: Id
    actor_b_id: Id
    relation: Literal["cooperative", "strained", "contested", "unknown"]
    reason: ShortText

    @model_validator(mode="after")
    def actors_must_differ(self):
        if self.actor_a_id == self.actor_b_id:
            raise ValueError("a relation requires two different actors")
        return self


class AdvanceWorldTimeOperation(StrictModel):
    type: Literal["advance_world_time"]
    target_date: CanonicalDate

    @field_validator("target_date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        return _canonical_date(value)


StateOperation = Annotated[
    Union[
        RecordDecisionOperation,
        OpenThreadOperation,
        ResolveThreadOperation,
        RecordSimulationEventOperation,
        SetRelationOperation,
        AdvanceWorldTimeOperation,
    ],
    Field(discriminator="type"),
]


class HistoricalEvidenceClaim(StrictModel):
    truth_type: Literal["historical_evidence"]
    claim: ClaimText
    evidence_ids: list[Annotated[str, StringConstraints(min_length=1, max_length=80)]] = Field(
        min_length=1, max_length=4
    )

    @field_validator("evidence_ids")
    @classmethod
    def unique_evidence_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("evidence_ids must be unique")
        return values


class AuthorContextReference(StrictModel):
    truth_type: Literal["author_context"]
    item_key: Id


class PlayerClaim(StrictModel):
    truth_type: Literal["player_claim"]
    text: ClaimText


class LearningObservation(StrictModel):
    objective_id: Id
    observation: ShortText


class TurnProposal(StrictModel):
    """Narrative and references remain separate from accepted state operations."""

    schema_version: Literal[1]
    language: Literal["da"]
    narration: Text
    dialogue: list[DialogueLine] = Field(max_length=16)
    outcome_summary: OutcomeSummary
    negotiation_assessment: NegotiationAssessment
    issue_updates: list[NegotiationIssueUpdate] = Field(max_length=12)
    state_operations: list[StateOperation] = Field(max_length=MAX_STATE_OPERATIONS)
    historical_claims: list[HistoricalEvidenceClaim] = Field(max_length=10)
    author_context_references: list[AuthorContextReference] = Field(max_length=10)
    player_claims: list[PlayerClaim] = Field(max_length=3)
    learning_observations: list[LearningObservation] = Field(max_length=20)
    scene_status: Literal["continue", "complete"]
    next_player_prompt: Text | None

    @model_validator(mode="after")
    def prompt_matches_status(self):
        if self.scene_status == "continue" and self.next_player_prompt is None:
            raise ValueError("a continuing scene requires a next-player prompt")
        if self.scene_status == "complete" and self.next_player_prompt is not None:
            raise ValueError("a completed scene cannot prompt for another action")
        return self


class EvidenceSearchToolCall(StrictModel):
    """A model request for the one bounded historical-search tool."""

    name: Literal["search_approved_evidence"]
    query: Annotated[
        str,
        StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
    ]


class TurnAgentResponse(StrictModel):
    """Request bounded evidence searches or return one final validated proposal."""

    schema_version: Literal[1]
    tool_calls: list[EvidenceSearchToolCall] = Field(max_length=MAX_EVIDENCE_SEARCH_TOOL_CALLS)
    proposal: TurnProposal | None

    @model_validator(mode="after")
    def response_is_tools_or_proposal(self):
        normalized_queries = [" ".join(call.query.casefold().split()) for call in self.tool_calls]
        if len(normalized_queries) != len(set(normalized_queries)):
            raise ValueError("evidence search tool calls must have unique queries")
        if bool(self.tool_calls) == (self.proposal is not None):
            raise ValueError("response must contain tool calls or a final proposal, but not both")
        return self


class DecisionRecordedEvent(StrictModel):
    event_type: Literal["decision_recorded"]
    summary: ShortText
    player_action: PlayerActionText


class ThreadOpenedEvent(StrictModel):
    event_type: Literal["thread_opened"]
    thread_id: Id
    summary: ShortText


class ThreadResolvedEvent(StrictModel):
    event_type: Literal["thread_resolved"]
    thread_id: Id
    resolution: ShortText


class SimulationEventRecordedEvent(StrictModel):
    event_type: Literal["simulation_event_recorded"]
    summary: ShortText
    actor_ids: list[Id] = Field(max_length=8)

    @field_validator("actor_ids")
    @classmethod
    def unique_actor_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("actor_ids must be unique")
        return values


class RelationChangedEvent(StrictModel):
    event_type: Literal["relation_changed"]
    actor_a_id: Id
    actor_b_id: Id
    relation: Literal["cooperative", "strained", "contested", "unknown"]
    reason: ShortText

    @model_validator(mode="after")
    def actors_must_differ(self):
        if self.actor_a_id == self.actor_b_id:
            raise ValueError("a relation requires two different actors")
        return self


class WorldTimeAdvancedEvent(StrictModel):
    event_type: Literal["world_time_advanced"]
    from_date: CanonicalDate
    to_date: CanonicalDate

    @field_validator("from_date", "to_date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        return _canonical_date(value)


class HistoricalClaimRecordedEvent(HistoricalEvidenceClaim):
    event_type: Literal["historical_claim_recorded"]
    world_date: CanonicalDate
    actor_id: Id

    @field_validator("world_date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        return _canonical_date(value)


class AuthorContextReferencedEvent(AuthorContextReference):
    event_type: Literal["author_context_referenced"]
    world_date: CanonicalDate
    actor_id: Id

    @field_validator("world_date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        return _canonical_date(value)


class PlayerClaimRecordedEvent(PlayerClaim):
    event_type: Literal["player_claim_recorded"]
    world_date: CanonicalDate
    actor_id: Id

    @field_validator("world_date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        return _canonical_date(value)


class LearningObservationRecordedEvent(LearningObservation):
    event_type: Literal["learning_observation_recorded"]
    truth_type: Literal["learning_observation"] = "learning_observation"
    world_date: CanonicalDate
    actor_id: Id

    @field_validator("world_date")
    @classmethod
    def validate_date(cls, value: str) -> str:
        return _canonical_date(value)


class NegotiationIssueSeed(StrictModel):
    id: Id
    title: ShortText
    description: Text
    baseline_disagreement: Text
    required_responder_ids: list[Id] = Field(min_length=1, max_length=16)

    @field_validator("required_responder_ids")
    @classmethod
    def unique_responder_ids(cls, values: list[str]) -> list[str]:
        if len(values) != len(set(values)):
            raise ValueError("required_responder_ids must be unique")
        return values


class NegotiationBoardInitializedEvent(StrictModel):
    event_type: Literal["negotiation_board_initialized"]
    issues: list[NegotiationIssueSeed] = Field(min_length=1, max_length=12)


class NegotiationTurnAssessedEvent(StrictModel):
    event_type: Literal["negotiation_turn_assessed"]
    turn_number: StrictInt = Field(ge=1)
    consequence: ShortText
    remaining_disagreement: ShortText


class NegotiationIssueUpdatedEvent(StrictModel):
    event_type: Literal["negotiation_issue_updated"]
    issue_id: Id
    action_type: Literal["proposal", "discussion"]
    turn_number: StrictInt = Field(ge=1)
    player_action_excerpt: ActionExcerpt
    proposer_role_id: Id
    authority: Literal["advisory"]
    binding_status: Literal["non_binding"]
    status: Literal["open", "proposed", "contested", "agreed"]
    recommendation: Text
    responses: list[DelegationIssueResponse] = Field(min_length=1, max_length=16)
    consequence: ShortText
    remaining_disagreement: ShortText

    @field_validator("responses")
    @classmethod
    def unique_response_actors(
        cls, values: list[DelegationIssueResponse]
    ) -> list[DelegationIssueResponse]:
        actor_ids = [item.actor_id for item in values]
        if len(actor_ids) != len(set(actor_ids)):
            raise ValueError("responses must contain one entry per actor")
        return values


class SceneCompletedEvent(StrictModel):
    event_type: Literal["scene_completed"]
    reason: Literal["player_end", "objectives_complete", "turn_limit", "validated_terminal"] = (
        "validated_terminal"
    )


StateEvent = Annotated[
    Union[
        DecisionRecordedEvent,
        ThreadOpenedEvent,
        ThreadResolvedEvent,
        SimulationEventRecordedEvent,
        RelationChangedEvent,
        WorldTimeAdvancedEvent,
        HistoricalClaimRecordedEvent,
        AuthorContextReferencedEvent,
        PlayerClaimRecordedEvent,
        LearningObservationRecordedEvent,
        NegotiationBoardInitializedEvent,
        NegotiationTurnAssessedEvent,
        NegotiationIssueUpdatedEvent,
        SceneCompletedEvent,
    ],
    Field(discriminator="event_type"),
]
