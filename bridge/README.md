# CASES GPT-6 Luna High bridge (Windows host)

This service adapts the authenticated Codex CLI to CASES' existing structured-completion interface. The CLI and ChatGPT sign-in stay on Windows; Docker receives only the bridge URL and a separately generated bearer token. The bridge binds exclusively to `127.0.0.1:18765` and supports only `gpt-6-luna` with high reasoning.

## Host setup

1. In the same Windows account that will run the service, confirm the supported CLI sign-in with `npx --yes @openai/codex login status`. The bridge can use a direct `codex` executable, or on Windows it invokes `node` plus npm's `npx-cli.js` with an argv array (never a shell command). On the first npx run, npm may download the CLI package.
2. Generate a separate random bridge secret. In PowerShell:

   ```powershell
   $bridgeToken = [Convert]::ToHexString([Security.Cryptography.RandomNumberGenerator]::GetBytes(32))
   $env:CASES_CODEX_BRIDGE_TOKEN = $bridgeToken
   ```

   Copy that value to `LUNA_BRIDGE_TOKEN` in the feature deployment's untracked `deployment/backend/.env`. Do not use, copy, or expose Codex authentication material. The bearer token is not a substitute for the ChatGPT sign-in.
3. From this checkout, run `python bridge/codex_bridge.py`. The process binds to loopback only. The default concurrency is two Codex jobs; request/schema/output limits and a 210-second process deadline are enforced. The bridge runs Codex with an empty temporary working directory, `read-only`, `--ignore-user-config` (which keeps authentication under the same `CODEX_HOME`), ephemeral sessions, hooks/apps/MCP/plugins/web search/shell/multi-agent disabled, and no user rules. To select an explicit executable argv, set `CASES_CODEX_COMMAND_JSON` to a JSON array, for example `["C:\\Program Files\\nodejs\\node.exe","C:\\Program Files\\nodejs\\node_modules\\npm\\bin\\npx-cli.js","--yes","@openai/codex"]`.
4. In a second PowerShell window, use the same untracked feature `.env` and an isolated Compose project/database name when starting CASES. For example, set `COMPOSE_PROJECT_NAME=cases-luna-smoke`, `POSTGRES_DB=cases_luna_smoke`, and `LUNA_BRIDGE_TOKEN` to the same value, then start only this checkout's backend Compose file. This creates a separate named Compose volume. Never point it at `ddu_game` and never reset or remove volumes.

The backend reaches the Windows host at `http://host.docker.internal:18765`. `GET /api/v1/narrators` reports each provider's model, configuration, and readiness without returning either secret. Luna is not selectable until the bridge reports the expected model and the CLI reports signed-in status. New sessions may still choose the existing `local_qwen` default.

## API example

```json
{
  "scenario": "versailles-1919",
  "role_id": "british_adviser",
  "narrator": "gpt6_luna_high"
}
```

Send this to `POST /api/v1/sessions` with the existing schema-v2 `Idempotency-Key` and `X-Session-Capability` headers. The response includes `narrator`; the same value appears when the session is resumed. A retry that changes narrator returns `409`. Omit `narrator` or set it to `local_qwen` to retain current behavior. Luna is available only for schema-v2 scenarios. No provider fallback occurs after creation.

## Isolated real smoke test

1. Ensure no `DB_NAME=ddu_game` environment override is active. Use the unique Compose project and database names above so Docker creates a separate database volume for this feature checkout.
2. Start the host bridge, then start the isolated database and CASES API/worker using `deployment/backend/compose.yaml`; do not use `down -v`, `volume rm`, or any database reset command.
3. Check `GET http://127.0.0.1:8000/api/v1/narrators`. Confirm `gpt6_luna_high` has `configured: true` and `ready: true`.
4. Create a new schema-v2 session with the sample payload and fresh idempotency/capability values. Poll the returned initialization job until ready, submit at least two turns, complete the scenario, and request reflection. Confirm each output's provenance identifies `OpenAI Codex CLI`, `gpt-6-luna`, and `high`; confirm historical evidence references remain CASES-produced and validated.
5. Stop the bridge between turns in a disposable session. The selected Luna session should fail/retry according to its existing bounded worker policy; it must not generate with Qwen. Restore the bridge and retry only through the existing session retry endpoints.
6. Stop only the uniquely named smoke-test Compose project when finished. Keep the resulting volume unless its contents have been reviewed and a separate cleanup is authorized.

## Isolation limits

The bridge rejects arbitrary model IDs, non-schema completions, oversized requests, and concurrent excess work. It passes player/system text only through stdin, starts the CLI in a fresh empty temporary directory, uses the CLI's read-only sandbox, suppresses CLI stdout/stderr, removes temporary schema/output files, and never logs prompts or bearer values. It does not install or configure MCP plugins. The CLI still runs as the Windows user and can read whatever the CLI itself can read outside that temporary working directory; `read-only` prevents writes but is not a dedicated security boundary against host reads. Use a dedicated Windows account/profile with minimal readable data if that residual access is unacceptable. The bridge is a private adapter around Codex CLI, not a generic OpenAI API endpoint or a guarantee that the CLI's model service stays available.
