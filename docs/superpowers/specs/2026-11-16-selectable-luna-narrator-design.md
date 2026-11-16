# Selectable GPT-6 Luna High narrator

## Intent and constraints

Allow a player to choose `local_qwen` (the unchanged default) or `gpt6_luna_high` when creating an authored schema-v2 CASES session. Pin the choice for the session so initialization, gameplay turns, repairs, and postgame reflection all use the same provider. Invalid/unavailable selections fail closed. Keep RAG retrieval and evidence/proposal validation in CASES. No live checkout/database access, auth-token file handling, or deployment.

## Approaches

- Embed Codex through its supported Python SDK: clean agent boundary, but this repository is Python in Docker while the authorized ChatGPT sign-in and Codex installation are on Windows.
- Run the supported Codex CLI from a narrow Windows loopback bridge: reuses the host sign-in, keeps credentials off Docker, and speaks the existing structured client protocol. This is the chosen approach; subprocess/file isolation still cannot make a general agent equivalent to a dedicated model API.
- Call undocumented auth endpoints or pass host credentials into Docker: rejected because it violates the stated credential boundary.

## Design

Persist a constrained narrator enum on each session through additive migration `0015`, defaulting existing rows to `local_qwen`. Expose readiness and configuration for the two known providers, validate selection during session creation, and include the pin in create/resume responses and idempotency checks. Schema-v1 remains on its legacy route. Inject a provider registry into background runtime operations while leaving Qwen defaults intact. The Windows bridge binds only to loopback, uses a strong bearer secret supplied out of band, caps requests/work/output, validates structured JSON, invokes `codex exec` with fixed argv in an empty isolated directory and read-only sandbox, and returns truthful model/provenance metadata. The backend receives only the bridge secret and URL. Document setup and isolated smoke procedure.

## Goals, non-goals, and open implementation choices

Goal: complete selected-provider session flow and covered failure handling. Non-goals: deployment/merge, changing history evidence rules, modifying Qwen behavior, changing live systems, or promising a true API-model endpoint. The builder chooses standard-library HTTP/CLI implementation and tests bridge routing with a fake executable; no external dependency is needed.
