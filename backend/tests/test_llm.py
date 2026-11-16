import json

import httpx
import pytest
from pydantic import ValidationError

from app.ai.llm import InferenceClient, StructuredOutputError


def completion(content, finish_reason="stop"):
    return {
        "choices": [
            {"finish_reason": finish_reason, "message": {"content": content}}
        ]
    }


@pytest.mark.asyncio
async def test_health_accepts_empty_non_json_http_200_body():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, content=b""))
    ) as http:
        assert await InferenceClient("http://llm:8080", "ddu-narrator", http).is_ready()


@pytest.mark.asyncio
async def test_health_503_returns_false():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503, text="loading"))
    ) as http:
        assert not await InferenceClient("http://llm:8080", "ddu-narrator", http).is_ready()


@pytest.mark.parametrize(
    "failure", [httpx.ReadTimeout("timed out"), httpx.ConnectError("offline")]
)
@pytest.mark.asyncio
async def test_health_network_failures_return_false(failure):
    def handler(request):
        raise failure

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        assert not await InferenceClient("http://llm:8080", "ddu-narrator", http).is_ready()


@pytest.mark.asyncio
async def test_explicit_model_readiness_checks_the_expected_alias():
    def handler(request):
        assert request.url.path == "/v1/models"
        return httpx.Response(200, json={"data": [{"id": "other-model"}, {"id": "ddu-narrator"}]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = InferenceClient("http://llm:8080", "ddu-narrator", http)
        assert await client.serves_expected_model()


@pytest.mark.asyncio
async def test_explicit_model_readiness_rejects_alias_mismatch():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json={"data": [{"id": "other-model"}]})
        )
    ) as http:
        client = InferenceClient("http://llm:8080", "ddu-narrator", http)
        assert not await client.serves_expected_model()


@pytest.mark.parametrize("enable_thinking", [None, False, True])
@pytest.mark.asyncio
async def test_structured_probe_applies_the_configured_thinking_mode(enable_thinking):
    def handler(request):
        body = json.loads(request.content)
        assert body["response_format"]["type"] == "json_schema"
        schema = body["response_format"]["json_schema"]["schema"]
        assert schema["properties"]["reply"]["pattern"] == r"\S"
        if enable_thinking is None:
            assert "chat_template_kwargs" not in body
        else:
            assert body["chat_template_kwargs"] == {"enable_thinking": enable_thinking}
        assert body["model"] == "ddu-narrator"
        return httpx.Response(
            200,
            json=completion('{"language":"da","reply":"Fred gennem garantier."}'),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = InferenceClient(
            "http://llm:8080", "ddu-narrator", http, enable_thinking=enable_thinking
        )
        result = await client.generate_danish_probe()
        assert result.language == "da"
        assert "garantier" in result.reply


@pytest.mark.asyncio
async def test_generic_structured_output_uses_call_schema_and_records_provider_metadata():
    schema = {
        "type": "object",
        "properties": {"status": {"type": "string", "enum": ["ok"]}},
        "required": ["status"],
        "additionalProperties": False,
    }

    def handler(request):
        body = json.loads(request.content)
        assert body["messages"] == [
            {"role": "system", "content": "Return status JSON."},
            {"role": "user", "content": "Say ok."},
        ]
        assert body["response_format"]["json_schema"] == {
            "name": "test_status",
            "strict": True,
            "schema": schema,
        }
        return httpx.Response(
            200,
            json={
                **completion('{"status":"ok"}'),
                "id": "req-structured-42",
                "model": "ddu-narrator",
                "model_revision": "revision-42",
                "usage": {"prompt_tokens": 18, "completion_tokens": 4},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await InferenceClient("http://llm:8080", "ddu-narrator", http).generate_structured(
            system_prompt="Return status JSON.",
            user_prompt="Say ok.",
            schema_name="test_status",
            response_schema=schema,
            max_tokens=32,
        )

    assert result.output == {"status": "ok"}
    assert result.model_revision == "revision-42"
    assert result.provider_request_id == "req-structured-42"
    assert result.prompt_tokens == 18
    assert result.completion_tokens == 4
    assert result.provider == "local_qwen"
    assert result.reasoning_effort is None
    assert result.duration_ms >= 0
    assert len(result.raw_output_sha256) == 64


@pytest.mark.asyncio
async def test_direct_client_keeps_explicit_provider_metadata_compatibility():
    def handler(request):
        assert request.headers["Authorization"] == "Bearer " + "b" * 32
        return httpx.Response(
            200,
            json={
                **completion('{"status":"ok"}'),
                "model": "gpt-6-luna",
                "provider": "spoofed-value",
                "reasoning_effort": "low",
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await InferenceClient(
            "http://inference:8080",
            "gpt-6-luna",
            http,
            headers={"Authorization": "Bearer " + "b" * 32},
            provider_label="codex_cli",
        ).generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema_name="test_status",
            response_schema={"type": "object"},
            max_tokens=16,
        )

    assert result.requested_model == "gpt-6-luna"
    assert result.served_model == "gpt-6-luna"
    assert result.provider == "codex_cli"
    assert result.reasoning_effort == "high"


@pytest.mark.asyncio
async def test_gateway_routed_models_verify_auth_model_and_provider_headers():
    for model, provider, reasoning in (
        ("ddu-narrator", "local_qwen", None),
        ("gpt-6-luna", "codex_cli", "high"),
    ):
        def handler(request):
            assert request.url.host == "ddu-gateway"
            assert request.headers["Authorization"] == "Bearer " + "g" * 32
            headers = {"X-Provider-ID": provider}
            if reasoning:
                headers["X-Reasoning-Effort"] = reasoning
            return httpx.Response(
                200,
                headers=headers,
                json={**completion('{"status":"ok"}'), "model": model},
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            result = await InferenceClient(
                "http://ddu-gateway:8080",
                model,
                http,
                headers={"Authorization": "Bearer " + "g" * 32},
                provider_label="codex_cli" if provider == "codex_cli" else "local_qwen",
                expected_provider=provider,
                require_provider_header=True,
            ).generate_structured(
                system_prompt="system",
                user_prompt="user",
                schema_name="test_status",
                response_schema={"type": "object"},
                max_tokens=16,
            )
        assert result.requested_model == result.served_model == model
        assert result.provider == provider
        assert result.reasoning_effort == ("high" if provider == "codex_cli" else None)


@pytest.mark.parametrize(
    ("served_model", "provider_header", "reasoning_header"),
    [("other-model", "local_qwen", None), ("ddu-narrator", "codex_cli", None)],
)
@pytest.mark.asyncio
async def test_gateway_rejects_model_or_provider_provenance_mismatch(
    served_model, provider_header, reasoning_header
):
    def handler(_request):
        headers = {"X-Provider-ID": provider_header}
        if reasoning_header:
            headers["X-Reasoning-Effort"] = reasoning_header
        return httpx.Response(
            200,
            headers=headers,
            json={**completion('{"status":"ok"}'), "model": served_model},
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        with pytest.raises(ValueError, match="mismatched"):
            await InferenceClient(
                "http://ddu-gateway:8080",
                "ddu-narrator",
                http,
                expected_provider="local_qwen",
                require_provider_header=True,
            ).generate_structured(
                system_prompt="system",
                user_prompt="user",
                schema_name="test_status",
                response_schema={"type": "object"},
                max_tokens=16,
            )


@pytest.mark.asyncio
async def test_gateway_routed_client_rejects_missing_provider_or_model_metadata():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=completion('{"status":"ok"}'))
        )
    ) as http:
        with pytest.raises(ValueError, match="provider metadata"):
            await InferenceClient(
                "http://ddu-gateway:8080",
                "gpt-6-luna",
                http,
                expected_provider="codex_cli",
                provider_label="codex_cli",
                require_provider_header=True,
            ).generate_structured(
                system_prompt="system",
                user_prompt="user",
                schema_name="test_status",
                response_schema={"type": "object"},
                max_tokens=16,
            )


@pytest.mark.asyncio
async def test_generic_structured_output_reports_truncation_with_a_safe_hash():
    raw = '{"status":"cu'
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=completion(raw, "length"))
        )
    ) as http:
        client = InferenceClient("http://llm:8080", "ddu-narrator", http)
        with pytest.raises(StructuredOutputError) as rejected:
            await client.generate_structured(
                system_prompt="system",
                user_prompt="user",
                schema_name="test_status",
                response_schema={"type": "object"},
                max_tokens=16,
            )

    assert rejected.value.code == "truncated_output"
    assert rejected.value.raw_output == raw


@pytest.mark.asyncio
async def test_structured_output_can_disable_model_thinking_for_json_reliability():
    def handler(request):
        body = json.loads(request.content)
        assert body["chat_template_kwargs"] == {"enable_thinking": False}
        return httpx.Response(
            200,
            json=completion('{"status":"ok"}'),
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        result = await InferenceClient(
            "http://llm:8080", "ddu-narrator", http, enable_thinking=True
        ).generate_structured(
            system_prompt="system",
            user_prompt="user",
            schema_name="test_status",
            response_schema={"type": "object"},
            max_tokens=16,
            enable_thinking=False,
        )

    assert result.output == {"status": "ok"}


@pytest.mark.asyncio
async def test_empty_length_limited_completion_is_classified_as_truncated():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(
            lambda _: httpx.Response(200, json=completion(None, "length"))
        )
    ) as http:
        client = InferenceClient("http://llm:8080", "ddu-narrator", http)
        with pytest.raises(StructuredOutputError) as rejected:
            await client.generate_structured(
                system_prompt="system",
                user_prompt="user",
                schema_name="test_status",
                response_schema={"type": "object"},
                max_tokens=16,
            )

    assert rejected.value.code == "truncated_output"
    assert rejected.value.raw_output is None


@pytest.mark.parametrize(
    ("response", "error", "message"),
    [
        (completion('{"language":"da","reply":"Hej"}', "length"), ValueError, "truncat"),
        (completion('{"language":"da","reply":"Hej"}', "content_filter"), ValueError, "finish successfully"),
        (completion(""), ValueError, "non-empty"),
        (completion(None), ValueError, "non-empty"),
        (completion("not JSON"), ValueError, "JSON"),
        (completion('{"language":"da","reply":"Hej","extra":1}'), ValidationError, None),
        (completion('{"language":"en","reply":"Hello"}'), ValidationError, None),
        (completion('{"language":"da","reply":"   "}'), ValidationError, "blank"),
    ],
)
@pytest.mark.asyncio
async def test_structured_probe_rejects_invalid_or_truncated_output(response, error, message):
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=response))
    ) as http:
        client = InferenceClient("http://llm:8080", "ddu-narrator", http)
        with pytest.raises(error, match=message):
            await client.generate_danish_probe()


@pytest.mark.parametrize("failure", [httpx.ReadTimeout("timed out"), httpx.ConnectError("offline")])
@pytest.mark.asyncio
async def test_structured_probe_propagates_network_failures(failure):
    async def handler(request):
        raise failure

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        client = InferenceClient("http://llm:8080", "ddu-narrator", http)
        with pytest.raises(httpx.HTTPError):
            await client.generate_danish_probe()


@pytest.mark.asyncio
async def test_structured_probe_propagates_http_errors():
    async with httpx.AsyncClient(
        transport=httpx.MockTransport(lambda _: httpx.Response(503))
    ) as http:
        client = InferenceClient("http://llm:8080", "ddu-narrator", http)
        with pytest.raises(httpx.HTTPStatusError):
            await client.generate_danish_probe()
