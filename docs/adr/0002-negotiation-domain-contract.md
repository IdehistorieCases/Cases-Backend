# ADR 0002: Negotiation domain contract for CASES v2

- Status: Accepted contract for the first negotiation release
- Date: 2026-11-23
- Scope: domain semantics and authored test fixtures; no SQL or runtime changes in this stage
- Depends on: ADR 0001, CASES v2 runtime contract

## Context

The v2 runtime already commits validated dialogue, state operations, claims, and learning observations to an ordered event stream. It does not yet define a durable negotiation model for offers, authority, signatures, or agreements. A convincing dialogue is not itself a negotiation outcome: a sentence such as “we can accept” must not silently transfer rights, bind another delegation, or count as a signed agreement.

The Versailles 3.2-ai dossier describes a fictional British adviser at the 22 April 1919 Shantung meeting. Britain, France, and the United States were represented; the Chinese plenipotentiaries were present; Japan was absent. The authoring fixture explicitly permits a meaningful process result and an unresolved end. Those facts constrain the first domain model, while all generated offers, reactions, signatures, and alternative outcomes remain simulated.

## Decision

Add an explicit, versioned negotiation layer to the v2 domain. It is additive to the existing event-authoritative runtime. This ADR freezes meaning and validation rules; a later implementation stage will define migrations and API payloads without changing v1 interpretation or rewriting existing v2 events.

### 1. Keep communication separate from agreement state

The domain distinguishes these acts:

| Act | Meaning | Agreement-state effect |
|---|---|---|
| Speech | A participant's utterance, question, argument, or explanation | None |
| Signal | A typed indication such as interest, reservation, or willingness to explore | None |
| Advice or draft | A recommendation or unissued package prepared by an adviser | None |
| Commitment | A clearly attributed promise with a scope and any conditions | Records the speaker's simulated commitment; does not bind another party by itself |
| Offer | A versioned package formally issued by an authorized principal to named recipients | Opens that exact package revision for response |
| Response | An explicit accept, reject, or counteroffer against one exact revision | Records only that authorized actor's response |
| Signature | An authorized principal's explicit execution of an agreed package | Records that principal's signature; does not imply missing signatures |
| Ratification | A separate, explicit approval by an actor whose release-scoped mandate allows ratification | Makes the simulated agreement ratified only after all required approvals |

A narrative line, positive signal, silence, continued participation, deadline expiry, or model-generated summary can never stand in for an offer, acceptance, signature, or ratification. A response must name the package ID and current revision. Acceptance is not ratification.

A commitment must name its speaker, addressee or beneficiary, scope, conditions, and expiry where applicable. A statement with ambiguous intent stays speech. A promise by one actor cannot bind another actor or an absent delegation.

### 2. Bound package terms and conditions

The first implementation supports three term families:

1. **Disposition and sovereignty:** proposed holder or treatment of a right, territory, title, or claim.
2. **Administration and access:** proposed control or access for railways, ports, policing, military presence, or local administration.
3. **Process and undertaking:** proposed review, consultation, timetable, invitation, information exchange, or other nonbinding next step.

Each term is typed, identifies the affected parties and scope, and says whether it changes a right, creates an undertaking, or only proposes a process. A package may combine terms, but it is atomic for acceptance: partial acceptance creates a new package revision or a separately identified package. The runtime must not silently accept a subset.

A conditional term carries machine-checkable named conditions and an explicit result for each condition. Conditions are conjunctive in the first implementation. They must state which actor supplies evidence of satisfaction and who is allowed to confirm it. Free-text narration may explain a condition but cannot satisfy it. No condition can manufacture consent, alter an actor's authority, or waive a required signature.

An offer names its issuer, recipients, affected parties, required signatories, visibility, expiry or deadline, package ID, revision, and terms. The immutable revision history is retained. A revised offer creates the next revision and marks the previous revision superseded in the same commit. A superseded offer cannot be accepted. A counteroffer is a new revision only when the authorized issuer adopts it; an adviser's draft remains a draft.

### 3. Make authority explicit and release-scoped

Authority is data in the immutable scenario release, not an inference from an actor name, historical office, dialogue, or the model. For each actor and role, the release must declare allowed actions, subjects, scope, required co-signers, and any conditions or limits. Runtime validators enforce those declarations before recording an offer, commitment, response, signature, ratification, or state effect.

For Versailles 3.2-ai:

- The player is a fictional British adviser. The player may ask, explain, advise, and prepare drafts. The player cannot issue a binding British offer, accept on behalf of Britain, speak for China or Japan, sign an agreement, or ratify a treaty.
- A named principal may speak and make only the commitments explicitly permitted by the scenario's authority declaration. A historical office or a plausible reaction is not itself proof of authority.
- The Chinese delegation must express its own consent. Koo's documented rejection of the two alternatives is specific to that meeting; the runtime must not turn it into a permanent veto over every materially different proposal.
- Japan is absent. No Japanese reply, acceptance, signature, or consent may be generated in this scene. Any substantive term changing Japanese rights requires an authorized Japanese representative and explicit Japanese consent before it can be agreed. The scene cannot ratify the Treaty of Versailles.
- A process-only agreement among present, authorized principals may proceed without a Chinese signature only when its terms impose no duty on China, change no Chinese or Japanese right, and preserve China's choice to participate or decline. It cannot be described as China's agreement.

The current 3.2-ai manifest is a draft candidate and does not yet encode this authority table as a runtime-enforced policy. Therefore, these rules do not make that release playable; they are a gate for its later implementation and review. A test fixture may declare explicit counterfactual authority for a simulated process memorandum, but that fixture is not evidence that the historical participants actually signed one.

NPCs may pursue their own release-defined objectives, ask questions, refuse, make authorized offers, and counteroffer without waiting for a player command. Their private goals or dossiers must not be exposed to the player or other NPCs unless the release explicitly grants that visibility. Retrieval and prompt assembly must filter private content before it is supplied to a model; a visibility label in a manifest is not sufficient by itself.

### 4. Preserve audience and privacy boundaries

Every negotiation record has an audience and visibility policy: public to the session, restricted to named delegations or roles, or private to one actor. A private message is never evidence of another party's consent. An agreement that needs multiple parties must use explicit, separately attributed responses and signatures from each required principal.

The event stream retains the information needed for authorized replay and audit. Public transcript rendering reveals only public records; private records are returned only to an authorized audience. Rebuilding a projection must not widen access. Model prompts receive only the actor's permitted dossier and negotiation records visible to that actor.

### 5. Use a versioned lifecycle and reject stale responses

A negotiation package follows this lifecycle:

- draft
- offered
- accepted by the required parties for the current revision
- signed by every required signatory
- ratified by every required ratifier, if the instrument requires ratification
- concluded, rejected, withdrawn, superseded, or expired

The lifecycle records both the status and the actors who caused each transition. Each transition is an explicit validated event. An agreement is not concluded merely because all participants spoke positively, and an agreed package is not ratified merely because it was signed.

When a response references an old or superseded revision, the API rejects it as stale and leaves the package, event-derived state, and turn number unchanged. The client must fetch the current revision and ask the participant to respond again. Retrying an identical request is idempotent. A revised offer does not erase the earlier record.

An instrument that requires signatures cannot reach signed or ratified status until every required, authorized signatory has acted on the same current revision. The scenario may require additional ratifiers. No defaults add a party, remove a required party, or substitute an absent actor.

### 6. Make closure independent of agreement

A negotiation may close as concluded, unresolved, or abandoned. Closure reasons include an agreement concluded, a declared deadline, deadlock, participant withdrawal, or player exit. The close event records the reason and any current open package IDs.

A deadline or player exit is not acceptance. A no-agreement result is valid and may still demonstrate sound reasoning, accurate historical distinctions, or respect for representation. Closing without agreement does not create a hidden loss, false concession, or ratification. Reopening a session replays the same committed close state.

### 7. Keep learning objectives and negotiation objectives separate

Negotiation objectives are authored, case-specific measures of process and outcomes, such as preserving a party's decision rights, testing a tradeoff, or making useful process progress. They may have qualitative success, partial, failure, and veto conditions. A veto condition overrides positive observations where it applies.

Learning objectives describe what the learner should be able to explain or distinguish. They are evaluated separately from the negotiation objective rubric. A player can learn from a failed or unresolved negotiation, and a process agreement is not automatically a historically correct settlement. Rubric feedback must cite the player's concrete action, the affected party, the source anchor or uncertainty, and what remains unresolved. No hidden scalar score is the authoritative grade.

### 8. Preserve event, replay, and compatibility invariants

The accepted, ordered event stream remains the source of truth. The scenario and application validators—not the model—authorize domain transitions. Each accepted negotiation event, committed result, and rebuildable state projection is written atomically. Failed validation, failed inference, stale responses, raw drafts, and uncommitted proposals do not mutate the canonical negotiation state.

The future event vocabulary is additive and versioned. It may include speech/signal records, offer draft and issue records, revision/supersession, explicit responses, commitments, signatures, ratification, and closure. Each event records its schema version, actor, target package/revision where relevant, timestamp, audience, and validated authority basis. Replay is deterministic and does not rerun inference. Existing v1 event meanings and committed v2 events remain unchanged.

This stage makes no SQL migration, API behavior change, scenario import, or activation. The later implementation must use an isolated database, include an additive migration and rollback/recovery plan, and preserve the existing v2 session history.

## Adversarial contract fixtures

The linked fixture file, docs/negotiation/stage2-adversarial-transcripts.yaml, is part of this ADR. It contains two constructed, explicitly counterfactual transcripts and named invariants:

- A process-only memorandum is ratified in the simulation by three principals with fixture-declared authority. It preserves Chinese choice and changes no Chinese or Japanese rights. The test asserts that the Chinese delegation did not consent and that the Treaty of Versailles was not ratified.
- A substantive package is revised conditionally, rejected by China, receives an attempted stale response to the superseded revision, and reaches its deadline without an agreement. The test asserts no implicit Japanese consent and no false ratification.

These transcripts are protocol tests, not historical dialogue, not claims that these events occurred, and not runtime implementation.

## Acceptance gate for Stage 2

Stage 2 is complete when:

1. The authority, term, condition, response, signature, ratification, privacy, closure, and evaluation rules above are accepted.
2. The fixture pair collectively covers speech and signals, an authorized offer, a conditional revision, explicit refusal, authorized acceptance, stale-response rejection, and ratified and unratified endings.
3. Every listed invariant has a concrete expected result in the fixture file.
4. No SQL change, migration, import, or release activation is made as part of this stage.
5. A later implementation can add these semantics without changing v1 contracts or reinterpreting existing v2 events.

## Consequences

The simulator can distinguish convincing dialogue from a valid commitment and a valid agreement. Advisers can participate meaningfully without being granted authority they do not have. Absent or affected parties cannot be simulated into consent. The current CASES UI and API do not yet implement these domain objects; a later stage must add the durable types, server-side checks, and UI affordances before this candidate release is playable.