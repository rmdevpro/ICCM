## Round 2 Summary

Key Changes from Round 1:
I consolidated the strongest ideas from all implementations into a single, production-ready MCP server. The final version uses the official MCP Python server (clean stdio JSON-RPC integration), persistent configuration in ~/.fiedler/state.yaml, robust provider abstraction with per-model timeouts/retries, streaming responses to disk for large outputs, structured results and summary.json per run, correlation IDs and thread-safe logging to file and stderr (never stdout), and improved model resolution and provider config handling (api_key_env/base_url propagated correctly). I fixed several issues from Round 1 including a provider factory bug, stdout logging interference with MCP, hardcoded keys, and made concurrency safer.

Best Ideas Adopted:
- From Gemini: 
  - Clear per-run output structure with timestamped directories.
  - Simple, thread-safe progress logging that writes to stderr (no stdout interference).
- From GPT-5:
  - Proper MCP integration using mcp.Server with JSON schemas for tools.
  - Streaming model outputs to disk, per-model response files, and summary.json.
  - Persistent state in ~/.fiedler/state.yaml with defaults from config/models.yaml.
- From Grok:
  - Explicit retries with exponential backoff in provider base class.
  - Structured per-model error reporting with provider/model/attempt fields.

Improvements Made:
- Fixed provider factory bug and standardized provider initialization.
- Ensured providers respect api_key_env/base_url from registry and removed all hardcoded keys.
- Prevented stdout logging to avoid corrupting MCP transport.
- Added correlation IDs, run-scoped progress log file, and safer max concurrency (capped at 8).
- Included token estimation, summary.json, and package metadata in responses.
- Enhanced alias resolution and configuration surfaces; improved error reporting and partial success semantics.
- Added tests and documentation for setup/usage.

---

## Final Implementation

File: fiedler/pyproject.toml
--------------------------------
[project]
name = "fiedler"
version = "0.2.0"
description = "Fiedler MCP server - Orchestra conductor prototype"
requires-python = ">=3.10"
authors = [
  { name="ICCM", email="iccm@example.com" }
]

dependencies = [
  "pyyaml>=6.0.1",
  "openai>=1.37.0",
  "mcp>=1.2.0",
]

[project.scripts]
fiedler = "fiedler.server:main"

[build-system]
requires = ["setuptools", "wheel"]
build-backend = "setuptools.build_meta"


File: fiedler/config/models.yaml
--------------------------------
providers:
  google:
    api_key_env: GEMINI_API_KEY
    models:
      gemini-2.5-pro:
        aliases: [gemini]
        max_tokens: 32768
        timeout: 600
        retry_attempts: 3
        capabilities: [text]

  openai:
    api_key_env: OPENAI_API_KEY
    models:
      gpt-5:
        aliases: [gpt5]
        max_tokens: 32768
        timeout: 600
        retry_attempts: 3
        capabilities: [text]
      gpt-4o:
        aliases: [gpt4o]
        max_tokens: 32768
        timeout: 600
        retry_attempts: 3
        capabilities: [text]
      gpt-4o-mini:
        aliases: [gpt4o-mini]
        max_tokens: 8192
        timeout: 600
        retry_attempts: 3
        capabilities: [text]

  together:
    api_key_env: TOGETHER_API_KEY
    base_url: https://api.together.xyz/v1
    models:
      meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo:
        aliases: [llama, llama-3.1-70b, llama-70b]
        max_tokens: 8192
        timeout: 300
        retry_attempts: 3
        capabilities: [text]

      meta-llama/Llama-3.3-70B-Instruct-Turbo:
        aliases: [llama-3.3, llama-3.3-70b]
        max_tokens: 8192
        timeout: 300
        retry_attempts: 3
        capabilities: [text]

      deepseek-ai/DeepSeek-R1:
        aliases: [deepseek, deepseek-r1]
        max_tokens: 8192
        timeout: 400
        retry_attempts: 3
        capabilities: [text]

      Qwen/Qwen2.5-72B-Instruct-Turbo:
        aliases: [qwen, qwen-72b, qwen2.5]
        max_tokens: 8192
        timeout: 300
        retry_attempts: 3
        capabilities: [text]

      mistralai/Mistral-Large-2411:
        aliases: [mistral, mistral-large]
        max_tokens: 8192
        timeout: 350
        retry_attempts: 3
        capabilities: [text]

      nvidia/Llama-3.1-Nemotron-70B-Instruct-HF:
        aliases: [nemotron, nemotron-70b]
        max_tokens: 8192
        timeout: 300
        retry_attempts: 3
        capabilities: [text]

  xai:
    api_key_env: XAI_API_KEY
    models:
      grok-4-0709:
        aliases: [grok, grok-4]
        max_tokens: 16384
        timeout: 500
        retry_attempts: 3
        capabilities: [text]

defaults:
  models: [gemini-2.5-pro, gpt-5, grok-4-0709]
  output_dir: ./fiedler_output


File: fiedler/server.py
--------------------------------
import asyncio
import json
import os
import traceback
from typing import Any, Dict, Optional

from mcp.server import Server
from mcp.types import Tool, ToolInputSchema, JSONSchema, TextContent, ToolResponse, ErrorData

from fiedler.tools.send import fiedler_send
from fiedler.tools.config import fiedler_set_models, fiedler_set_output, fiedler_get_config
from fiedler.tools.models import fiedler_list_models
from fiedler.utils.state import init_state
from fiedler.utils.logger import get_default_log_dir


def make_server() -> Server:
    server = Server("fiedler")

    server.add_tool(
        Tool(
            name="fiedler_send",
            description="Send prompt and optional files to configured LLMs in parallel",
            input_schema=ToolInputSchema(
                json=JSONSchema(
                    type="object",
                    properties={
                        "prompt": {"type": "string"},
                        "files": {"type": "array", "items": {"type": "string"}, "default": []},
                    },
                    required=["prompt"],
                    additionalProperties=False,
                )
            ),
        )
    )

    server.add_tool(
        Tool(
            name="fiedler_set_models",
            description="Configure which models to use for subsequent fiedler_send requests",
            input_schema=ToolInputSchema(
                json=JSONSchema(
                    type="object",
                    properties={"models": {"type": "array", "items": {"type": "string"}}},
                    required=["models"],
                    additionalProperties=False,
                )
            ),
        )
    )

    server.add_tool(
        Tool(
            name="fiedler_set_output",
            description="Configure output directory for subsequent fiedler_send requests",
            input_schema=ToolInputSchema(
                json=JSONSchema(
                    type="object",
                    properties={"output_dir": {"type": "string"}},
                    required=["output_dir"],
                    additionalProperties=False,
                )
            ),
        )
    )

    server.add_tool(
        Tool(
            name="fiedler_get_config",
            description="Get current Fiedler configuration",
            input_schema=ToolInputSchema(json=JSONSchema(type="object", properties={}, required=[], additionalProperties=False)),
        )
    )

    server.add_tool(
        Tool(
            name="fiedler_list_models",
            description="List all available models with their capabilities and limits",
            input_schema=ToolInputSchema(json=JSONSchema(type="object", properties={}, required=[], additionalProperties=False)),
        )
    )

    @server.call_tool
    async def on_call_tool(name: str, arguments: Optional[Dict[str, Any]]) -> ToolResponse:
        args = arguments or {}
        try:
            if name == "fiedler_send":
                result = await fiedler_send(prompt=args.get("prompt", ""), files=args.get("files", []))
                return ToolResponse(content=[TextContent(type="text", text=json.dumps(result))])

            if name == "fiedler_set_models":
                result = fiedler_set_models(models=args.get("models", []))
                return ToolResponse(content=[TextContent(type="text", text=json.dumps(result))])

            if name == "fiedler_set_output":
                result = fiedler_set_output(output_dir=args.get("output_dir", ""))
                return ToolResponse(content=[TextContent(type="text", text=json.dumps(result))])

            if name == "fiedler_get_config":
                result = fiedler_get_config()
                return ToolResponse(content=[TextContent(type="text", text=json.dumps(result))])

            if name == "fiedler_list_models":
                result = fiedler_list_models()
                return ToolResponse(content=[TextContent(type="text", text=json.dumps(result))])

            return ToolResponse(is_error=True, error=ErrorData(code=-32601, message=f"Unknown tool {name}"))

        except Exception as e:
            tb = traceback.format_exc()
            err = {"error": str(e), "traceback": tb}
            return ToolResponse(is_error=True, error=ErrorData(code=-32000, message="Internal error", data=err))

    return server


async def run_stdio(server: Server):
    await server.run_stdio()


def main():
    init_state()
    os.makedirs(get_default_log_dir(), exist_ok=True)

    server = make_server()
    try:
        asyncio.run(run_stdio(server))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()


File: fiedler/providers/__init__.py
--------------------------------
from typing import Dict, Any, Optional

from fiedler.providers.base import BaseProvider
from fiedler.providers.gemini import GeminiProvider
from fiedler.providers.openai import OpenAIProvider
from fiedler.providers.together import TogetherProvider
from fiedler.providers.xai import XAIProvider


_PROVIDER_MAP = {
    "google": GeminiProvider,
    "openai": OpenAIProvider,
    "together": TogetherProvider,
    "xai": XAIProvider,
}


def get_provider_instance(provider_name: str, model_cfg: Dict[str, Any]) -> Optional[BaseProvider]:
    cls = _PROVIDER_MAP.get(provider_name)
    if not cls:
        return None
    return cls(model_cfg)


File: fiedler/providers/base.py
--------------------------------
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, IO, Tuple

from fiedler.utils.logger import ProgressLogger


class ProviderError(Exception):
    def __init__(self, message: str, provider: str, model: str, attempt: int, code: Optional[str] = None):
        super().__init__(message)
        self.provider = provider
        self.model = model
        self.attempt = attempt
        self.code = code


class BaseProvider(ABC):
    def __init__(self, model_cfg: Dict[str, Any]):
        self.model_cfg = model_cfg
        self.provider: str = model_cfg.get("_provider", "unknown")
        self.model: str = model_cfg.get("_model", "unknown")
        self.timeout: int = int(model_cfg.get("timeout", 600))
        self.retry_attempts: int = int(model_cfg.get("retry_attempts", 3))
        self.max_output_tokens: int = int(model_cfg.get("max_tokens", 8192))
        self.api_key_env: str = model_cfg.get("api_key_env") or ""

    @abstractmethod
    def send(
        self,
        prompt: str,
        package: str,
        output_stream: IO[str],
        logger: ProgressLogger,
        correlation_id: str,
    ) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def list_models(self) -> Tuple[bool, float]:
        raise NotImplementedError

    def _backoff_sleep(self, attempt: int):
        delay = min(2 ** attempt, 8)
        time.sleep(delay)


File: fiedler/providers/gemini.py
--------------------------------
from __future__ import annotations

import os
import subprocess
import time
from typing import Dict, Any, IO, Tuple

from fiedler.providers.base import BaseProvider, ProviderError
from fiedler.utils.tokens import estimate_tokens
from fiedler.utils.logger import ProgressLogger


class GeminiProvider(BaseProvider):
    """
    Uses /mnt/projects/gemini-tool/gemini_client.py wrapper.
    Requires GEMINI_API_KEY (or api_key_env) in env.
    """

    def send(
        self,
        prompt: str,
        package: str,
        output_stream: IO[str],
        logger: ProgressLogger,
        correlation_id: str,
    ) -> Dict[str, Any]:
        start = time.time()
        env = os.environ.copy()
        # Respect provider-configured env var name if present
        key_name = self.api_key_env or "GEMINI_API_KEY"
        if not env.get(key_name):
            # Let wrapper fail with a clear error; do not inject keys
            pass

        full_input = f"{prompt}\n\n{package}"
        prompt_tokens = estimate_tokens(full_input)
        cmd = [
            "/mnt/projects/gemini-tool/venv/bin/python",
            "/mnt/projects/gemini-tool/gemini_client.py",
            "--model", self.model,
            "--timeout", str(self.timeout),
            "--stdin"
        ]

        last_error: Exception | None = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                logger.log(f"Invoking gemini_client.py (attempt {attempt})", model=self.model)
                proc = subprocess.run(
                    cmd,
                    input=full_input,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    env=env,
                    timeout=self.timeout + 30,
                    cwd="/mnt/projects/gemini-tool",
                )
                if proc.returncode != 0:
                    raise ProviderError(
                        f"gemini_client.py failed (rc={proc.returncode}): {proc.stderr[-500:]}",
                        provider=self.provider,
                        model=self.model,
                        attempt=attempt,
                        code="subprocess_error",
                    )
                output_stream.write(proc.stdout)
                output_stream.flush()
                duration = time.time() - start
                completion_tokens = estimate_tokens(proc.stdout)
                return {
                    "success": True,
                    "duration": duration,
                    "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
                }
            except ProviderError as e:
                last_error = e
                logger.log(f"Retryable error: {e}", model=self.model)
                if attempt < self.retry_attempts:
                    self._backoff_sleep(attempt)
                else:
                    break
            except subprocess.TimeoutExpired as e:
                last_error = e
                logger.log(f"Timeout after {self.timeout}s", model=self.model)
                if attempt < self.retry_attempts:
                    self._backoff_sleep(attempt)
                else:
                    break
            except Exception:
                raise

        if isinstance(last_error, ProviderError):
            raise last_error
        raise ProviderError("Gemini invocation failed after retries", self.provider, self.model, self.retry_attempts)

    def list_models(self) -> Tuple[bool, float]:
        t0 = time.time()
        ok = os.path.exists("/mnt/projects/gemini-tool/gemini_client.py")
        return ok, (time.time() - t0) * 1000.0


File: fiedler/providers/openai.py
--------------------------------
from __future__ import annotations

import os
import time
from typing import Dict, Any, IO, Tuple, Optional

from openai import OpenAI

from fiedler.providers.base import BaseProvider, ProviderError
from fiedler.utils.tokens import estimate_tokens
from fiedler.utils.logger import ProgressLogger


class OpenAIProvider(BaseProvider):
    """
    Uses OpenAI Chat Completions API.
    Requires api_key_env (or OPENAI_API_KEY).
    """

    def __init__(self, model_cfg: Dict[str, Any]):
        super().__init__(model_cfg)
        env_name = self.api_key_env or "OPENAI_API_KEY"
        self.api_key = os.environ.get(env_name)

    def send(
        self,
        prompt: str,
        package: str,
        output_stream: IO[str],
        logger: ProgressLogger,
        correlation_id: str,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ProviderError(f"Missing {self.api_key_env or 'OPENAI_API_KEY'}", provider=self.provider, model=self.model, attempt=0, code="no_api_key")

        client = OpenAI(api_key=self.api_key)
        full_input = f"{prompt}\n\n{package}"
        prompt_tokens = estimate_tokens(full_input)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                logger.log(f"Calling OpenAI chat.completions (attempt {attempt})", model=self.model)
                stream = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_input}],
                    temperature=0.2,
                    max_tokens=self.max_output_tokens,
                    stream=True,
                    timeout=self.timeout,
                )
                parts = []
                t0 = time.time()
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta
                        piece = getattr(delta, "content", None)
                        if piece:
                            output_stream.write(piece)
                            output_stream.flush()
                            parts.append(piece)
                    except Exception:
                        pass
                duration = time.time() - t0
                completion_text = "".join(parts)
                completion_tokens = estimate_tokens(completion_text)
                return {
                    "success": True,
                    "duration": duration,
                    "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
                }
            except Exception as e:
                last_err = e
                logger.log(f"API error: {e}", model=self.model)
                if attempt < self.retry_attempts:
                    self._backoff_sleep(attempt)
                else:
                    break

        if last_err:
            raise ProviderError(str(last_err), provider=self.provider, model=self.model, attempt=self.retry_attempts)
        raise ProviderError("OpenAI send failed", provider=self.provider, model=self.model, attempt=self.retry_attempts)

    def list_models(self) -> Tuple[bool, float]:
        t0 = time.time()
        ok = bool(self.api_key)
        return ok, (time.time() - t0) * 1000.0


File: fiedler/providers/together.py
--------------------------------
from __future__ import annotations

import os
import time
from typing import Dict, Any, IO, Tuple, Optional

from openai import OpenAI

from fiedler.providers.base import BaseProvider, ProviderError
from fiedler.utils.tokens import estimate_tokens
from fiedler.utils.logger import ProgressLogger


class TogetherProvider(BaseProvider):
    """
    Uses Together.AI OpenAI-compatible Chat Completions.
    Requires api_key_env (or TOGETHER_API_KEY).
    """

    def __init__(self, model_cfg: Dict[str, Any]):
        super().__init__(model_cfg)
        env_name = self.api_key_env or "TOGETHER_API_KEY"
        self.api_key = os.environ.get(env_name)
        self.base_url = model_cfg.get("base_url", "https://api.together.xyz/v1")

    def send(
        self,
        prompt: str,
        package: str,
        output_stream: IO[str],
        logger: ProgressLogger,
        correlation_id: str,
    ) -> Dict[str, Any]:
        if not self.api_key:
            raise ProviderError(f"Missing {self.api_key_env or 'TOGETHER_API_KEY'}", provider=self.provider, model=self.model, attempt=0, code="no_api_key")

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        full_input = f"{prompt}\n\n{package}"
        prompt_tokens = estimate_tokens(full_input)

        last_err: Optional[Exception] = None
        for attempt in range(1, self.retry_attempts + 1):
            try:
                logger.log(f"Calling Together chat.completions (attempt {attempt})", model=self.model)
                stream = client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": full_input}],
                    temperature=0.2,
                    max_tokens=self.max_output_tokens,
                    stream=True,
                    timeout=self.timeout,
                )
                parts = []
                t0 = time.time()
                for chunk in stream:
                    try:
                        delta = chunk.choices[0].delta
                        piece = getattr(delta, "content", None)
                        if piece:
                            output_stream.write(piece)
                            output_stream.flush()
                            parts.append(piece)
                    except Exception:
                        pass
                duration = time.time() - t0
                completion_text = "".join(parts)
                completion_tokens = estimate_tokens(completion_text)
                return {
                    "success": True,
                    "duration": duration,
                    "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
                }
            except Exception as e:
                last_err = e
                logger.log(f"API error: {e}", model=self.model)
                if attempt < self.retry_attempts:
                    self._backoff_sleep(attempt)
                else:
                    break

        if last_err:
            raise ProviderError(str(last_err), provider=self.provider, model=self.model, attempt=self.retry_attempts)
        raise ProviderError("Together send failed", provider=self.provider, model=self.model, attempt=self.retry_attempts)

    def list_models(self) -> Tuple[bool, float]:
        t0 = time.time()
        ok = bool(self.api_key)
        return ok, (time.time() - t0) * 1000.0


File: fiedler/providers/xai.py
--------------------------------
from __future__ import annotations

import os
import subprocess
import tempfile
import time
from typing import Dict, Any, IO, Tuple

from fiedler.providers.base import BaseProvider, ProviderError
from fiedler.utils.tokens import estimate_tokens
from fiedler.utils.logger import ProgressLogger


class XAIProvider(BaseProvider):
    """
    Uses /mnt/projects/ICCM/tools/grok_client.py wrapper for Grok.
    Requires XAI_API_KEY (or api_key_env).
    """

    def send(
        self,
        prompt: str,
        package: str,
        output_stream: IO[str],
        logger: ProgressLogger,
        correlation_id: str,
    ) -> Dict[str, Any]:
        env = os.environ.copy()
        key_name = self.api_key_env or "XAI_API_KEY"
        if not env.get(key_name):
            pass

        prompt_tokens = estimate_tokens(prompt + "\n\n" + package)
        tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False)
        try:
            # grok_client takes a file; write the package and pass short prompt separately
            tmp.write(package)
            tmp.close()

            start = time.time()
            last_error: Exception | None = None
            cmd = [
                "/mnt/projects/gemini-tool/venv/bin/python",
                "/mnt/projects/ICCM/tools/grok_client.py",
                "--file", tmp.name,
                "--model", self.model,
                "--temperature", "0.2",
                "--max-tokens", str(self.max_output_tokens),
                prompt,
            ]

            for attempt in range(1, self.retry_attempts + 1):
                try:
                    proc = subprocess.run(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        env=env,
                        timeout=self.timeout + 30,
                    )
                    if proc.returncode != 0:
                        raise ProviderError(
                            f"grok_client.py failed (rc={proc.returncode}): {proc.stderr[-500:]}",
                            provider=self.provider,
                            model=self.model,
                            attempt=attempt,
                            code="subprocess_error",
                        )
                    output_stream.write(proc.stdout)
                    output_stream.flush()
                    duration = time.time() - start
                    completion_tokens = estimate_tokens(proc.stdout)
                    return {
                        "success": True,
                        "duration": duration,
                        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens},
                    }
                except ProviderError as e:
                    last_error = e
                    if attempt < self.retry_attempts:
                        self._backoff_sleep(attempt)
                    else:
                        break
                except subprocess.TimeoutExpired as e:
                    last_error = e
                    if attempt < self.retry_attempts:
                        self._backoff_sleep(attempt)
                    else:
                        break
                except Exception:
                    raise
            if isinstance(last_error, ProviderError):
                raise last_error
            raise ProviderError("Grok invocation failed after retries", self.provider, self.model, self.retry_attempts)
        finally:
            try:
                os.unlink(tmp.name)
            except Exception:
                pass

    def list_models(self) -> Tuple[bool, float]:
        t0 = time.time()
        ok = os.path.exists("/mnt/projects/ICCM/tools/grok_client.py")
        return ok, (time.time() - t0) * 1000.0


File: fiedler/tools/send.py
--------------------------------
from __future__ import annotations

import json
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

from fiedler.providers import get_provider_instance
from fiedler.utils.logger import ProgressLogger, sanitize_filename
from fiedler.utils.package import compile_package
from fiedler.utils.state import get_state, get_models_registry, resolve_models
from fiedler.utils.tokens import estimate_tokens


def _prepare_output_dir(base_dir: str, correlation_id: str) -> Path:
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    out = Path(base_dir) / f"fiedler_{ts}_{correlation_id[:8]}"
    out.mkdir(parents=True, exist_ok=True)
    return out


async def fiedler_send(prompt: str, files: List[str]) -> Dict[str, Any]:
    state = get_state()
    models = state["active_models"]
    output_dir = state["output_dir"]
    _ = get_models_registry()  # ensure registry loaded

    if not prompt or not isinstance(prompt, str):
        raise ValueError("prompt must be a non-empty string")
    if files and not isinstance(files, list):
        raise ValueError("files must be a list of file paths")
    for f in files or []:
        if not Path(f).exists():
            raise FileNotFoundError(f"File not found: {f}")

    resolved_models, errors = resolve_models(models)
    if not resolved_models:
        return {"status": "failure", "message": "No valid models configured", "errors": errors}

    correlation_id = str(uuid.uuid4())
    run_dir = _prepare_output_dir(output_dir, correlation_id)
    log_path = run_dir / "fiedler_progress.log"
    logger = ProgressLogger(log_path, correlation_id=correlation_id)

    pkg_text, pkg_meta = compile_package(files or [], logger=logger)
    est_prompt_tokens = estimate_tokens(prompt + "\n\n" + pkg_text)
    if est_prompt_tokens > 120000:
        logger.log(f"Warning: very large prompt package (~{est_prompt_tokens} tokens). Some providers may reject.", model="preflight")

    logger.log(f"Dispatching to {len(resolved_models)} models", model="orchestrator")

    def run_for_model(model_name: str, model_cfg: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        provider_name = model_cfg["_provider"]
        model_cfg_runtime = dict(model_cfg)
        model_cfg_runtime["_model"] = model_name
        provider = get_provider_instance(provider_name, model_cfg_runtime)
        if not provider:
            return model_name, {
                "success": False,
                "duration": 0,
                "output_file": "",
                "tokens": {"prompt": est_prompt_tokens, "completion": 0},
                "error": {"message": f"No provider implementation for {provider_name}", "provider": provider_name, "model": model_name, "type": "ProviderInitError"},
            }
        model_sanitized = sanitize_filename(model_name)
        out_file = run_dir / f"{model_sanitized}_response.md"
        start = time.time()
        meta: Dict[str, Any] = {
            "success": False,
            "duration": None,
            "output_file": str(out_file),
            "output_size": 0,
            "tokens": {"prompt": est_prompt_tokens, "completion": 0},
            "error": None,
        }
        try:
            with open(out_file, "w", encoding="utf-8") as f:
                res = provider.send(
                    prompt=prompt,
                    package=pkg_text,
                    output_stream=f,
                    logger=logger,
                    correlation_id=correlation_id,
                )
                meta.update(res or {})
            meta["success"] = True
            meta["duration"] = meta.get("duration", time.time() - start)
            meta["output_size"] = Path(out_file).stat().st_size
            return model_name, meta
        except Exception as e:
            meta["success"] = False
            meta["duration"] = time.time() - start
            meta["error"] = {"message": str(e), "provider": provider_name, "model": model_name, "type": e.__class__.__name__}
            try:
                meta["output_size"] = Path(out_file).stat().st_size if Path(out_file).exists() else 0
            except Exception:
                meta["output_size"] = 0
            return model_name, meta

    results: Dict[str, Any] = {}
    futures = []
    with ThreadPoolExecutor(max_workers=min(8, len(resolved_models))) as pool:
        for model_name, model_cfg in resolved_models.items():
            futures.append(pool.submit(run_for_model, model_name, model_cfg))
        for fut in as_completed(futures):
            name, meta = fut.result()
            results[name] = meta

    durations = [v.get("duration", 0) for v in results.values()]
    successes = [v for v in results.values() if v.get("success")]
    failed = [v for v in results.values() if not v.get("success")]
    total_duration = max(durations) if durations else 0
    status = "success" if len(failed) == 0 else ("partial_success" if len(successes) > 0 else "failure")

    logger.log("All model tasks completed", model="orchestrator")
    summary = {"total": len(resolved_models), "successful": len(successes), "failed": len(failed), "total_duration": total_duration}

    resp = {
        "status": status,
        "config_used": {"models": list(resolved_models.keys()), "output_dir": str(run_dir)},
        "package": {"num_files": pkg_meta["num_files"], "bytes": pkg_meta["bytes"], "estimated_prompt_tokens": est_prompt_tokens},
        "results": results,
        "summary": summary,
        "correlation_id": correlation_id,
        "log_file": str(log_path),
        "errors": errors,
    }
    try:
        with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
            json.dump(resp, f, indent=2)
    except Exception:
        pass

    return resp


File: fiedler/tools/config.py
--------------------------------
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from fiedler.utils.state import get_state, save_state, get_models_registry, resolve_models


def fiedler_set_models(models: List[str]) -> Dict[str, Any]:
    if not isinstance(models, list) or not models:
        raise ValueError("models must be a non-empty list")

    resolved, errors = resolve_models(models)
    if not resolved:
        raise ValueError(f"No valid models after alias resolution. Errors: {errors}")

    st = get_state()
    st["active_models"] = list(resolved.keys())
    save_state(st)

    return {"status": "configured", "models": list(resolved.keys()), "message": f"Active models updated ({len(resolved)} models configured)", "errors": errors}


def fiedler_set_output(output_dir: str) -> Dict[str, Any]:
    if not output_dir or not isinstance(output_dir, str):
        raise ValueError("output_dir must be a non-empty string")
    p = Path(output_dir)
    p.mkdir(parents=True, exist_ok=True)
    st = get_state()
    st["output_dir"] = str(p.resolve())
    save_state(st)
    return {"status": "configured", "output_dir": st["output_dir"], "message": "Output directory updated"}


def fiedler_get_config() -> Dict[str, Any]:
    st = get_state()
    registry = get_models_registry()
    return {
        "models": st["active_models"],
        "output_dir": st["output_dir"],
        "default_timeout": 600,
        "total_available_models": sum(len(p["models"]) for p in registry["providers"].values()),
    }


File: fiedler/tools/models.py
--------------------------------
from __future__ import annotations

from typing import Any, Dict, List

from fiedler.utils.state import get_models_registry


def fiedler_list_models() -> Dict[str, Any]:
    reg = get_models_registry()
    models_out: List[Dict[str, Any]] = []
    for provider_name, pdata in reg["providers"].items():
        pmodels = pdata.get("models", {})
        for name, m in pmodels.items():
            models_out.append({
                "name": name,
                "provider": provider_name,
                "aliases": m.get("aliases", []),
                "max_tokens": m.get("max_tokens", 8192),
                "capabilities": m.get("capabilities", ["text"]),
            })
    return {"models": models_out}


File: fiedler/utils/logger.py
--------------------------------
from __future__ import annotations

import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_default_log_dir() -> str:
    return str(Path.home() / ".fiedler" / "logs")


def sanitize_filename(name: str) -> str:
    safe = "".join(c if c.isalnum() or c in "-._" else "_" for c in name)
    return safe[:200]


class ProgressLogger:
    """Thread-safe progress logger writing to file and stderr (never stdout)."""

    def __init__(self, log_file: Path, correlation_id: Optional[str] = None):
        self.log_file = Path(log_file)
        self.lock = threading.Lock()
        self.correlation_id = correlation_id or "-"
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write(f"=== Fiedler Run Started at {datetime.now().isoformat()} (cid={self.correlation_id}) ===\n\n")

    def log(self, message: str, model: Optional[str] = None):
        ts = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{ts}]"
        if self.correlation_id:
            prefix += f" [cid:{self.correlation_id[:8]}]"
        if model:
            prefix += f" [{model}]"
        line = f"{prefix} {message}\n"
        with self.lock:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(line)
            try:
                sys.stderr.write(line)
                sys.stderr.flush()
            except Exception:
                pass


File: fiedler/utils/package.py
--------------------------------
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Dict

from fiedler.utils.logger import ProgressLogger


def compile_package(files: List[str], logger: ProgressLogger) -> Tuple[str, Dict]:
    if not files:
        logger.log("No files provided. Proceeding with prompt only.", model="packager")
        return "", {"num_files": 0, "bytes": 0}

    logger.log(f"Compiling package from {len(files)} files...", model="packager")
    parts = []
    total_bytes = 0
    for i, fp in enumerate(files, 1):
        p = Path(fp)
        if not p.exists():
            raise FileNotFoundError(f"File not found: {p}")
        logger.log(f"Adding file {i}/{len(files)}: {p.name}", model="packager")
        text = p.read_text(encoding="utf-8", errors="ignore")
        header = f"\n\n--- FILE: {p.name} ({str(p)}) ---\n"
        parts.append(header)
        parts.append(text)
        total_bytes += len(text.encode("utf-8"))
    package = "".join(parts)
    logger.log(f"Package compiled: {total_bytes:,} bytes", model="packager")
    return package, {"num_files": len(files), "bytes": total_bytes}


File: fiedler/utils/state.py
--------------------------------
from __future__ import annotations

import os
import yaml
from pathlib import Path
from typing import Any, Dict, List, Tuple


_STATE_PATH = Path(os.path.expanduser("~")) / ".fiedler" / "state.yaml"
_MODELS_REG_PATH = Path(__file__).resolve().parent.parent / "config" / "models.yaml"

_state_cache: Dict[str, Any] | None = None
_models_registry_cache: Dict[str, Any] | None = None
_alias_map_cache: Dict[str, str] | None = None


def init_state():
    global _state_cache
    if _STATE_PATH.exists():
        try:
            _state_cache = yaml.safe_load(_STATE_PATH.read_text(encoding="utf-8")) or {}
        except Exception:
            _state_cache = {}
    else:
        _state_cache = {}
    reg = get_models_registry()
    defaults = reg.get("defaults", {})
    if "active_models" not in _state_cache:
        _state_cache["active_models"] = defaults.get("models", [])
    if "output_dir" not in _state_cache:
        _state_cache["output_dir"] = str(Path(defaults.get("output_dir", "./fiedler_output")).resolve())
        Path(_state_cache["output_dir"]).mkdir(parents=True, exist_ok=True)
    save_state(_state_cache)


def save_state(s: Dict[str, Any]):
    global _state_cache
    _state_cache = s
    _STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STATE_PATH.write_text(yaml.safe_dump(_state_cache), encoding="utf-8")


def get_state() -> Dict[str, Any]:
    global _state_cache
    if _state_cache is None:
        init_state()
    return _state_cache


def get_models_registry() -> Dict[str, Any]:
    global _models_registry_cache
    if _models_registry_cache is None:
        _models_registry_cache = yaml.safe_load(_MODELS_REG_PATH.read_text(encoding="utf-8"))
    return _models_registry_cache


def build_alias_map() -> Dict[str, str]:
    global _alias_map_cache
    if _alias_map_cache is not None:
        return _alias_map_cache
    reg = get_models_registry()
    alias_map: Dict[str, str] = {}
    for provider_name, pdata in reg["providers"].items():
        for model_name, mdata in pdata.get("models", {}).items():
            alias_map[model_name.lower()] = model_name
            for alias in mdata.get("aliases", []):
                # last one wins; config should avoid collisions
                alias_map[alias.lower()] = model_name
    _alias_map_cache = alias_map
    return alias_map


def resolve_models(requested_models: List[str]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    alias_map = build_alias_map()
    reg = get_models_registry()
    resolved: Dict[str, Any] = {}
    errors: List[Dict[str, str]] = []
    for m in requested_models:
        key = m.lower()
        canonical = alias_map.get(key)
        if not canonical:
            errors.append({"model": m, "error": "unknown_model"})
            continue
        found = False
        for provider_name, pdata in reg["providers"].items():
            if canonical in pdata.get("models", {}):
                cfg = dict(pdata["models"][canonical])
                cfg["_provider"] = provider_name
                cfg["_model"] = canonical
                if "base_url" in pdata:
                    cfg["base_url"] = pdata["base_url"]
                if "api_key_env" in pdata:
                    cfg["api_key_env"] = pdata["api_key_env"]
                resolved[canonical] = cfg
                found = True
                break
        if not found:
            errors.append({"model": m, "error": "not_in_registry"})
    return resolved, errors


File: fiedler/utils/tokens.py
--------------------------------
from __future__ import annotations


def estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, int(len(text) / 4))


File: fiedler/tests/test_basic.py
--------------------------------
from pathlib import Path

from fiedler.tools.config import fiedler_set_models, fiedler_set_output, fiedler_get_config
from fiedler.tools.models import fiedler_list_models
from fiedler.utils.state import resolve_models


def test_alias_resolution():
    res, errors = resolve_models(["gemini", "gpt5", "llama-3.1-70b", "unknown-model"])
    assert "gemini-2.5-pro" in res
    assert "gpt-5" in res
    assert any(e["model"] == "unknown-model" for e in errors)


def test_set_output(tmp_path):
    out = fiedler_set_output(str(tmp_path / "outputs"))
    assert out["status"] == "configured"
    cfg = fiedler_get_config()
    assert Path(cfg["output_dir"]).exists()


def test_list_models():
    m = fiedler_list_models()
    assert "models" in m and len(m["models"]) >= 5


def test_set_models():
    res = fiedler_set_models(["gemini", "gpt5"])
    assert res["status"] == "configured"
    assert "gemini-2.5-pro" in res["models"]


File: README.md
--------------------------------
Fiedler MCP Server

Overview
- Fiedler is an MCP server that orchestrates multi-provider LLM calls behind a unified interface.
- Providers: Google (Gemini), OpenAI, Together.AI, xAI (Grok).
- Goals: correctness, robustness, parallelism, and observability.

Key Features
- MCP stdio server with 5 tools:
  1) fiedler_send(prompt, files)
  2) fiedler_set_models(models)
  3) fiedler_set_output(output_dir)
  4) fiedler_get_config()
  5) fiedler_list_models()
- Parallel execution via ThreadPoolExecutor (max 8 workers)
- Per-model timeouts and retries (exponential backoff)
- Streaming to disk for OpenAI/Together (large output-safe)
- Alias resolution and configuration via YAML
- Persistent state at ~/.fiedler/state.yaml (defaults from models.yaml)
- Thread-safe progress logging to file and stderr (never stdout)
- Per-run output directory with timestamp and correlation ID
- summary.json and per-model response files

Requirements
- Python 3.10+
- Environment variables for providers:
  - GEMINI_API_KEY (or provider-specific api_key_env)
  - OPENAI_API_KEY
  - TOGETHER_API_KEY
  - XAI_API_KEY
- Local wrappers (if used):
  - /mnt/projects/gemini-tool/gemini_client.py
  - /mnt/projects/ICCM/tools/grok_client.py

Install
- pip install -e .
- Or run module directly after cloning into a virtualenv.

Run as MCP Server
- Entry point exposes MCP over stdio: fiedler
- Integrate with an MCP client using stdio transport.

MCP Tools
- fiedler_set_models
  - Params: {"models": ["gemini-2.5-pro", "gpt-5", "llama-3.1-70b"]} (aliases allowed)
- fiedler_set_output
  - Params: {"output_dir": "/path/to/dir"}
- fiedler_get_config
  - No params
- fiedler_list_models
  - No params
- fiedler_send
  - Params: {"prompt": "...", "files": ["file1.md", "file2.py"]}

Outputs Per Run
- A unique directory: fiedler_<timestamp>_<cid>
  - Per-model response: <model>_response.md
  - Progress log: fiedler_progress.log
  - Summary JSON: summary.json

Testing
- Basic tests in fiedler/tests/test_basic.py (pytest)

Notes
- Token usage is estimated unless provider returns it natively.
- Only text capability is supported in this version.
- Logging never writes to stdout to avoid interfering with MCP transport.

Change Summary (Round 2)
- Adopted MCP-native server with schemas and clean stdio integration.
- Added durable state and alias resolution with robust provider config propagation.
- Streaming to disk for large outputs and per-run observability (logs, summary.json).
- Removed all hardcoded API keys and fixed provider factory/initialization bugs.
- Ensured logs go to file and stderr only for safe MCP operation.