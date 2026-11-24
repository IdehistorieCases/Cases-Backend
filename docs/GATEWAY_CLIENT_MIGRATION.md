# CASES gateway client migration

CASES routes both pinned narrators through the canonical Cases-AI service at
`http://ddu-gateway:8080` on `ddu_inference_bus`. Set `INFERENCE_GATEWAY_TOKEN`
to a consumer bearer token issued for CASES. It is separate from gateway admin
credentials and any provider credentials. Never put provider credentials in the
CASES environment.

`LLM_MODEL_ALIAS` selects the local narrator ID (default `ddu-narrator`), while
`LUNA_MODEL_ID` selects Luna (default `gpt-6-luna`). `LLM_ENABLE_THINKING`
continues to control the optional Qwen template setting. Without a gateway
token, neither real narrator is ready or selectable. If Luna is disabled, the
gateway omits it from `/v1/models` and returns `503` for requests; a session
pinned to Luna fails without switching models. CASES continues to own retrieval,
structured output validation, and gameplay safety checks.

## Local rollout

1. Configure the gateway and create the consumer token in the Cases-AI
   deployment. Use Cases-Startup for the complete bridge, gateway, backend, and GUI sequence. Attach its gateway service to `ddu_inference_bus`.
2. Copy `deployment/backend/.env.example` to `.env`, set the database password
   and `INFERENCE_GATEWAY_TOKEN`, then verify the gateway `/health` and the
   authenticated `/v1/models` response contain the expected enabled model IDs.
3. Start the CASES stack using the normal local compose procedure. Verify its
   narrator readiness endpoint and a local narrator turn, then a Luna-pinned
   turn if Luna is enabled. No schema migration is needed; database migration
   `0015` remains current.

## Rollback

For a local rollback, restore the previous CASES client revision and its
matching configuration, then restart only the local CASES stack under the
operator's normal procedure. The previous client used direct provider endpoints
and a separate bridge credential; those settings are intentionally absent from
the migrated CASES code and must not be copied into this revision. No database
rollback is required because this change does not alter schema or stored session
narrator pins.
