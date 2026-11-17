# Legacy CASES-host bridge (transition only)

The bridge source remains in this repository temporarily as a rollback artifact while the unified gateway migration is staged. It is not used by the migrated CASES client. Do not start this copy alongside the Local-AI-API-LITE-owned bridge, and do not add its bearer token or loopback URL to the migrated CASES environment.

The active ownership and startup instructions are in [Local-AI-API-LITE's Codex bridge runbook](https://github.com/MarcusFunt/Local-AI-API-LITE/tree/main/integrations/codex-bridge). CASES uses only `INFERENCE_GATEWAY_BASE_URL` and its separately scoped `INFERENCE_GATEWAY_TOKEN`.

If staged gateway acceptance fails, roll back the CASES client revision and configuration as described in [the gateway migration runbook](../docs/GATEWAY_CLIENT_MIGRATION.md). Retire this legacy bridge copy after the new bridge has passed live probes and the rollback window has closed.
