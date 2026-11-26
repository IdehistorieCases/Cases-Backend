from __future__ import annotations

from app.scenarios.schema_v2 import ScenarioDefinitionV2


def initial_negotiation_issue_state(definition: ScenarioDefinitionV2) -> list[dict]:
    """Create the server-owned, unresolved issue board for a pinned scenario."""
    return [
        {
            "id": issue.id,
            "title": issue.title,
            "description": issue.description,
            "baseline_disagreement": issue.baseline_disagreement,
            "required_responder_ids": list(issue.required_responder_ids),
            "status": "open",
            "current_recommendation": "",
            "proposer_role_id": "",
            "authority": "advisory",
            "responses": [],
            "last_consequence": "",
            "remaining_disagreement": issue.baseline_disagreement,
            "last_updated_turn": 0,
            "history": [],
        }
        for issue in definition.negotiation_issues
    ]


def negotiation_issue_seeds(definition: ScenarioDefinitionV2) -> list[dict]:
    """Return immutable issue definitions for the lazy board-initialization event."""
    return [
        {
            "id": issue.id,
            "title": issue.title,
            "description": issue.description,
            "baseline_disagreement": issue.baseline_disagreement,
            "required_responder_ids": list(issue.required_responder_ids),
        }
        for issue in definition.negotiation_issues
    ]
