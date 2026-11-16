"""Loopback-only, bearer-protected Codex CLI bridge for structured CASES output."""

from __future__ import annotations

import hmac
import json
import os
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

MODEL_ID = "gpt-6-luna"
MAX_PROMPT_CHARS = 48_000
SCHEMA_NAME = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


@dataclass(frozen=True)
class BridgeConfig:
    bearer_token: str
    command: tuple[str, ...] = ("codex",)
    model_id: str = MODEL_ID
    timeout_seconds: float = 210.0
    max_request_bytes: int = 96_000
    max_schema_bytes: int = 24_000
    max_output_bytes: int = 32_000
    max_concurrency: int = 2

    def __post_init__(self) -> None:
        if len(self.bearer_token) < 24:
            raise ValueError("bridge bearer token must contain at least 24 characters")
        if not self.command or not self.model_id or self.timeout_seconds <= 0:
            raise ValueError("bridge command, model, and positive timeout are required")
        if min(self.max_request_bytes, self.max_schema_bytes, self.max_output_bytes) < 1:
            raise ValueError("bridge size limits must be positive")
        if self.max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")


def _json_schema(value: Any, depth: int = 0) -> bool:
    """Reject malformed or pathological schemas before passing them to Codex."""
    if depth > 16:
        return False
    if isinstance(value, (str, int, float, bool)) or value is None:
        return True
    if isinstance(value, list):
        return len(value) <= 256 and all(_json_schema(item, depth + 1) for item in value)
    if isinstance(value, dict):
        return len(value) <= 256 and all(
            isinstance(key, str) and _json_schema(item, depth + 1)
            for key, item in value.items()
        )
    return False


def _matches_schema(
    value: Any, schema: dict, *, root: dict | None = None, depth: int = 0
) -> bool:
    if depth > 32:
        return False
    root = schema if root is None else root
    reference = schema.get("$ref")
    if isinstance(reference, str):
        if not reference.startswith("#/"):
            return False
        target: Any = root
        for part in reference[2:].split("/"):
            part = part.replace("~1", "/").replace("~0", "~")
            if not isinstance(target, dict) or part not in target:
                return False
            target = target[part]
        return isinstance(target, dict) and _matches_schema(
            value, target, root=root, depth=depth + 1
        )
    has_combinator = False
    for keyword in ("allOf", "anyOf", "oneOf"):
        choices = schema.get(keyword)
        if choices is None:
            continue
        has_combinator = True
        if not isinstance(choices, list):
            return False
        matches = [
            _matches_schema(value, item, root=root, depth=depth + 1)
            for item in choices
            if isinstance(item, dict)
        ]
        if len(matches) != len(choices):
            return False
        if keyword == "allOf" and not all(matches):
            return False
        if keyword == "anyOf" and not any(matches):
            return False
        if keyword == "oneOf" and sum(matches) != 1:
            return False
    if "const" in schema and value != schema["const"]:
        return False
    if "enum" in schema and value not in schema["enum"]:
        return False
    kind = schema.get("type")
    if isinstance(kind, list):
        return any(
            _matches_schema(value, {**schema, "type": option}, root=root, depth=depth + 1)
            for option in kind
        )
    if kind is None:
        return has_combinator
    if kind == "object":
        if not isinstance(value, dict):
            return False
        properties = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(properties, dict) or not isinstance(required, list):
            return False
        if any(key not in value for key in required):
            return False
        if schema.get("additionalProperties") is False and set(value) - set(properties):
            return False
        return all(
            key not in properties
            or _matches_schema(item, properties[key], root=root, depth=depth + 1)
            for key, item in value.items()
        )
    if kind == "array":
        return isinstance(value, list) and all(
            _matches_schema(
                item, schema.get("items", {}), root=root, depth=depth + 1
            )
            for item in value
        )
    if kind == "string":
        return (
            isinstance(value, str)
            and len(value) >= schema.get("minLength", 0)
            and len(value) <= schema.get("maxLength", float("inf"))
            and ("pattern" not in schema or re.search(schema["pattern"], value) is not None)
        )
    if kind == "integer":
        return (
            isinstance(value, int)
            and not isinstance(value, bool)
            and value >= schema.get("minimum", float("-inf"))
            and value <= schema.get("maximum", float("inf"))
        )
    if kind == "number":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and value >= schema.get("minimum", float("-inf"))
            and value <= schema.get("maximum", float("inf"))
        )
    if kind == "boolean":
        return isinstance(value, bool)
    if kind == "null":
        return value is None
    return False


def _codex_authenticated(config: BridgeConfig) -> bool:
    try:
        with tempfile.TemporaryDirectory(prefix="cases-codex-health-") as directory:
            result = subprocess.run(
                [*config.command, "login", "status"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                cwd=directory,
                timeout=min(8.0, config.timeout_seconds),
                check=False,
                shell=False,
            )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _request_fields(document: dict, config: BridgeConfig) -> tuple[str, str, str, dict, int]:
    if document.get("model") != config.model_id:
        raise ValueError("unsupported model")
    if document.get("stream", False) is not False:
        raise ValueError("streaming is unsupported")
    messages = document.get("messages")
    if not isinstance(messages, list) or not messages or len(messages) > 16:
        raise ValueError("messages are invalid")
    content: dict[str, str] = {}
    for message in messages:
        if (
            not isinstance(message, dict)
            or not isinstance(message.get("role"), str)
            or message.get("role") not in {"system", "user"}
            or not isinstance(message.get("content"), str)
            or not message["content"].strip()
        ):
            raise ValueError("messages are invalid")
        role = message["role"]
        content[role] = content.get(role, "") + "\n\n" + message["content"]
    system = content.get("system", "Follow the supplied JSON schema exactly.")
    user = content.get("user", "")
    if not user or len(system) + len(user) > MAX_PROMPT_CHARS:
        raise ValueError("prompt is empty or too large")
    response_format = document.get("response_format")
    if not isinstance(response_format, dict):
        raise TypeError("strict object JSON schema is required")
    json_schema = response_format.get("json_schema") if isinstance(response_format, dict) else None
    schema = json_schema.get("schema") if isinstance(json_schema, dict) else None
    name = json_schema.get("name") if isinstance(json_schema, dict) else None
    if (
        not isinstance(json_schema, dict)
        or response_format.get("type") != "json_schema"
        or json_schema.get("strict") is not True
        or not isinstance(name, str)
        or not SCHEMA_NAME.fullmatch(name)
        or not isinstance(schema, dict)
        or schema.get("type") != "object"
        or not _json_schema(schema)
        or len(json.dumps(schema, separators=(",", ":")).encode("utf-8"))
        > config.max_schema_bytes
    ):
        raise ValueError("strict object JSON schema is required")
    max_tokens = document.get("max_tokens", 2048)
    if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or not 1 <= max_tokens <= 8192:
        raise ValueError("max_tokens must be between 1 and 8192")
    return system, user, name, schema, max_tokens


def _strict_codex_schema(value: Any) -> Any:
    """Convert optional Pydantic object fields to nullable required fields for Codex.

    OpenAI strict JSON schemas require every object property to be listed as
    required. CASES still validates returned data against the original schema.
    """
    if isinstance(value, list):
        return [_strict_codex_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {
        key: _strict_codex_schema(item)
        for key, item in value.items()
        if key != "default"
    }
    properties = result.get("properties")
    if result.get("type") == "object" and isinstance(properties, dict):
        required = set(result.get("required", []))
        for key, field in list(properties.items()):
            if key not in required and isinstance(field, dict):
                branches = field.get("anyOf")
                nullable = (
                    isinstance(branches, list)
                    and any(
                        isinstance(branch, dict) and branch.get("type") == "null"
                        for branch in branches
                    )
                )
                types = field.get("type")
                nullable = nullable or (
                    isinstance(types, list) and "null" in types
                ) or types == "null"
                if not nullable:
                    properties[key] = {
                        "anyOf": [field, {"type": "null"}]
                    }
        result["required"] = list(properties)
        result["additionalProperties"] = False
    return result


def _run_codex(
    config: BridgeConfig,
    *,
    system: str,
    user: str,
    schema: dict,
    max_tokens: int,
) -> dict:
    with tempfile.TemporaryDirectory(prefix="cases-codex-") as directory:
        workspace = Path(directory)
        schema_path = workspace / "schema.json"
        output_path = workspace / "result.json"
        schema_path.write_text(json.dumps(_strict_codex_schema(schema), ensure_ascii=False), encoding="utf-8")
        prompt = (
            "Return a single JSON object matching the required schema. Do not use tools, "
            "shell commands, files, browsers, or connectors. "
            f"Keep the result within approximately {max_tokens} output tokens.\n\n"
            "SYSTEM INSTRUCTIONS:\n" + system + "\n\nUSER REQUEST:\n" + user
        )
        # Prompt goes only on stdin. CLI is pinned to a fresh directory and read-only mode.
        argv = [
            *config.command,
            "exec",
            "-m",
            config.model_id,
            "-c",
            "model_reasoning_effort=high",
            "-c",
            "features.hooks=false",
            "-c",
            "features.shell_tool=false",
            "-c",
            "features.multi_agent=false",
            "-c",
            "features.apps=false",
            "-c",
            "features.remote_plugin=false",
            "-c",
            "features.web_search=false",
            "-c",
            "features.view_image=false",
            "-c",
            "features.skill_mcp_dependency_install=false",
            "--ignore-user-config",
            "--ignore-rules",
            "--ephemeral",
            "--skip-git-repo-check",
            "-s",
            "read-only",
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
            "-",
        ]
        with open(os.devnull, "wb") as null_output:
            process = subprocess.Popen(
                argv,
                stdin=subprocess.PIPE,
                stdout=null_output,
                stderr=null_output,
                text=True,
                encoding="utf-8",
                cwd=workspace,
                shell=False,
            )
            try:
                process.communicate(input=prompt, timeout=config.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                if os.name == "nt":
                    taskkill = shutil.which("taskkill.exe")
                    if taskkill:
                        try:
                            subprocess.run(
                                [taskkill, "/PID", str(process.pid), "/T", "/F"],
                                stdin=subprocess.DEVNULL,
                                stdout=null_output,
                                stderr=null_output,
                                timeout=5,
                                check=False,
                                shell=False,
                            )
                        except (OSError, subprocess.TimeoutExpired):
                            pass
                if process.poll() is None:
                    process.kill()
                process.communicate()
                raise TimeoutError("Codex completion timed out") from exc
        if process.returncode != 0 or not output_path.is_file():
            raise RuntimeError("Codex completion failed")
        if not output_path.is_file() or not 0 < output_path.stat().st_size <= config.max_output_bytes:
            raise ValueError("Codex output was empty or exceeded the output limit")
        raw = output_path.read_bytes()
        try:
            output = json.loads(raw)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Codex output was not valid JSON") from exc
        if not isinstance(output, dict) or not _matches_schema(output, schema):
            raise ValueError("Codex output did not match the required schema")
        return output


def create_server(config: BridgeConfig, *, host: str = "127.0.0.1", port: int = 18765):
    if host != "127.0.0.1":
        raise ValueError("Codex bridge must bind to 127.0.0.1")
    slots = threading.BoundedSemaphore(config.max_concurrency)
    request_slots = threading.BoundedSemaphore(config.max_concurrency * 2)

    class Handler(BaseHTTPRequestHandler):
        server_version = "CASES-Codex-Bridge"
        sys_version = ""

        def log_message(self, _format: str, *_args: Any) -> None:
            # Default access logs contain paths and can leak request metadata.
            return

        def send_json(self, status: int, document: dict) -> None:
            data = json.dumps(document, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self) -> bool:
            supplied = self.headers.get("Authorization", "")
            expected = f"Bearer {config.bearer_token}"
            return hmac.compare_digest(supplied.encode(), expected.encode())

        def do_GET(self) -> None:
            if self.path == "/health":
                self.send_json(200, {"status": "ok"})
            elif self.path == "/v1/models" and self.authorized():
                if _codex_authenticated(config):
                    self.send_json(200, {"object": "list", "data": [{"id": config.model_id, "object": "model"}]})
                else:
                    self.send_json(503, {"error": {"message": "Codex CLI is not authenticated", "type": "service_unavailable"}})
            elif self.path == "/v1/models":
                self.send_json(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
            else:
                self.send_json(404, {"error": {"message": "Not found", "type": "not_found"}})

        def do_POST(self) -> None:
            if not self.authorized():
                self.send_json(401, {"error": {"message": "Unauthorized", "type": "authentication_error"}})
                return
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": {"message": "Not found", "type": "not_found"}})
                return
            length = self.headers.get("Content-Length", "")
            if not length.isdigit() or int(length) > config.max_request_bytes:
                self.send_json(413, {"error": {"message": "Request too large", "type": "invalid_request_error"}})
                return
            try:
                document = json.loads(self.rfile.read(int(length)))
                if not isinstance(document, dict):
                    raise TypeError("request must be an object")
                system, user, _name, schema, max_tokens = _request_fields(document, config)
            except (ValueError, TypeError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                self.send_json(400, {"error": {"message": str(exc), "type": "invalid_request_error"}})
                return
            if not slots.acquire(blocking=False):
                self.send_json(503, {"error": {"message": "Bridge is busy", "type": "server_error"}})
                return
            try:
                output = _run_codex(
                    config, system=system, user=user, schema=schema, max_tokens=max_tokens
                )
                content = json.dumps(output, ensure_ascii=False, separators=(",", ":"))
                if len(content.encode("utf-8")) > config.max_output_bytes:
                    raise ValueError("Codex output exceeded the output limit")
                self.send_json(
                    200,
                    {
                        "object": "chat.completion",
                        "model": config.model_id,
                        "provider": "codex_cli",
                        "reasoning_effort": "high",
                        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                    },
                )
            except TimeoutError:
                self.send_json(504, {"error": {"message": "Codex completion timed out", "type": "timeout_error"}})
            except (RuntimeError, TypeError, ValueError):
                self.send_json(502, {"error": {"message": "Codex completion failed validation", "type": "upstream_error"}})
            finally:
                slots.release()

    class BoundedHTTPServer(ThreadingHTTPServer):
        daemon_threads = True

        def process_request(self, request, client_address):
            if not request_slots.acquire(blocking=False):
                self.shutdown_request(request)
                return
            try:
                super().process_request(request, client_address)
            except Exception:
                request_slots.release()
                raise

        def process_request_thread(self, request, client_address):
            try:
                super().process_request_thread(request, client_address)
            finally:
                request_slots.release()

    return BoundedHTTPServer((host, port), Handler)


def main() -> None:
    if os.name != "nt":
        raise SystemExit("The CASES Codex bridge must run on the Windows host")
    token = os.environ.get("CASES_CODEX_BRIDGE_TOKEN", "")
    if len(token) < 24:
        raise SystemExit("Set CASES_CODEX_BRIDGE_TOKEN to a random value of at least 24 characters")
    configured_command = os.environ.get("CASES_CODEX_COMMAND_JSON")
    if configured_command:
        try:
            parsed_command = json.loads(configured_command)
        except json.JSONDecodeError as exc:
            raise SystemExit("CASES_CODEX_COMMAND_JSON must be a JSON string array") from exc
        if (
            not isinstance(parsed_command, list)
            or not parsed_command
            or any(not isinstance(item, str) or not item for item in parsed_command)
        ):
            raise SystemExit("CASES_CODEX_COMMAND_JSON must be a non-empty JSON string array")
        command = tuple(parsed_command)
    else:
        codex_executable = shutil.which("codex")
        if codex_executable and Path(codex_executable).suffix.lower() not in {".cmd", ".bat"}:
            command = (codex_executable,)
        elif os.name == "nt":
            node = shutil.which("node")
            npm_cli = Path(node).parent / "node_modules" / "npm" / "bin" / "npx-cli.js" if node else None
            if node and npm_cli and npm_cli.is_file():
                command = (node, str(npm_cli), "--yes", "@openai/codex")
            else:
                raise SystemExit("Install Codex CLI or provide CASES_CODEX_COMMAND_JSON")
        else:
            command = (codex_executable or "codex",)
    server = create_server(
        BridgeConfig(
            bearer_token=token,
            command=command,
            timeout_seconds=float(os.environ.get("CASES_CODEX_TIMEOUT_SECONDS", "210")),
        )
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
