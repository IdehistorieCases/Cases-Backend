# Selectable GPT-6 Luna High Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax.

**Goal:** Let CASES players choose and complete an authored scenario with Qwen or GPT-6 Luna High.

**Architecture:** Add a Windows-host loopback Codex CLI bridge and backend provider adapter, then persist a safe provider choice per session and route all model-backed stages through it. Keep Qwen as the existing default and keep retrieval/validation in CASES.

**Tech Stack:** Python 3.12, FastAPI, SQLAlchemy/Alembic, httpx, standard-library Windows HTTP/process bridge, pytest, Ruff.

**Spec:** `docs/superpowers/specs/2026-11-16-selectable-luna-narrator-design.md`

## Global Constraints

- Only this feature checkout is writable; never access live checkout or database.
- Do not read or move ChatGPT authentication files; no undocumented auth endpoint or generic API-key assumption.
- Default Qwen behavior and schema-v1 behavior remain unchanged.
- Luna unavailable means fail closed, never silently switch provider.
- Readiness/provenance must not expose secrets or falsely identify the serving model.

## Review Focus

- Invalid/unavailable provider and idempotency mismatch fail without session mutation.
- Old rows migrate to `local_qwen` and schema-v1 stays legacy.
- Initialization, turns, and reflection all use the persisted choice.
- Bridge rejects unauthorized, oversized, malformed, timed-out, and non-JSON requests safely.
- Host-authentication/filesystem isolation limitations remain clear in runbook.

---

### Task 1: Isolated host bridge

**Files:** Create `bridge/codex_bridge.py`, `bridge/README.md`, and bridge pytest coverage.

- [ ] Test bearer auth, payload limits, schema validation, timeout, malformed output, and truthful model response with a fake Codex executable.
- [ ] Run tests to verify failure.
- [ ] Implement fixed-argv Codex CLI service bound to `127.0.0.1:18765`, read-only/temp workspace, bearer auth, limits, `/health`, `/v1/models`, `/v1/chat/completions`.
- [ ] Run bridge tests and Ruff.

### Task 2: Backend client and configuration

**Files:** Modify `backend/app/core/settings.py`; create provider registry/adapter in `backend/app/ai/providers.py`; add client/config tests.

- [ ] Test optional bridge config, no secret exposure, health/model readiness, structured JSON parsing, and fail-closed behavior.
- [ ] Observe red, implement bridge adapter while retaining `InferenceClient` for Qwen.
- [ ] Run targeted tests and Ruff.

### Task 3: Session persistence and API selection

**Files:** Modify `backend/app/db/models.py`, `backend/app/game/routes.py`; create migration `0015_session_narrator.py`; update API/migration tests.

- [ ] Test migration default, narrator listing, create/resume pin, bad selection/unavailability, and idempotent narrator mismatch.
- [ ] Observe red, add validated constrained enum and additive migration; only permit Luna on schema-v2.
- [ ] Run targeted tests and migration preservation suite.

### Task 4: Runtime routing and provenance

**Files:** Modify worker initialization, turn orchestration, reflection and capability/status reporting; add routing tests.

- [ ] Test all three stages route by stored provider and preserve requested/served provider provenance; test failure does not fall back.
- [ ] Observe red, route through persisted registry selection without changing retrieval/validation.
- [ ] Run targeted runtime tests and full backend suite.

### Task 5: Runbook and final verification

**Files:** Add Windows bridge setup, environment, network, isolated smoke test, and API examples to `bridge/README.md` or `docs`.

- [ ] Run Ruff and full backend plus bridge suites from their correct working directories.
- [ ] Inspect full diff and report coverage/limitations; do not deploy or merge.
