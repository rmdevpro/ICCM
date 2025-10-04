# Fiedler Implementation Review Request

**Date:** 2025-10-02
**Correlation ID:** FIEDLER-IMPL-R1
**Reviewers:** Fiedler's default models 

## Review Scope

This is the synthesized implementation of the Fiedler MCP Server based on your Round 2 submissions. Please review the complete implementation for:

1. **Correctness**: Does it correctly implement the requirements?
2. **Completeness**: Are all features from requirements present?
3. **Code Quality**: Clean, maintainable, well-structured code?
4. **Error Handling**: Robust error handling and retries?
5. **Documentation**: Clear README and inline documentation?
6. **Integration**: Will it work correctly as an MCP server?

## Original Requirements Reference

The requirements you implemented are in `/mnt/projects/ICCM/architecture/fiedler_requirements.md` (5 MCP tools, multi-provider support, parallel execution, persistent state).

## Implementation Structure

```
fiedler/
├── config/
│   └── models.yaml          # Provider/model registry
├── fiedler/
│   ├── __init__.py
│   ├── server.py            # MCP server entry point
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py        # Thread-safe logging
│   │   ├── state.py         # Persistent state management
│   │   ├── tokens.py        # Token estimation
│   │   └── package.py       # File compilation
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py          # Abstract provider with retry
│   │   ├── gemini.py        # Gemini subprocess wrapper
│   │   ├── openai.py        # OpenAI SDK provider
│   │   ├── together.py      # Together.AI provider
│   │   └── xai.py           # xAI subprocess wrapper
│   └── tools/
│       ├── __init__.py
│       ├── models.py        # fiedler_list_models
│       ├── config.py        # Configuration tools
│       └── send.py          # fiedler_send orchestrator
├── tests/
│   └── test_basic.py        # Basic tests
├── pyproject.toml
└── README.md
```

## Review Questions

1. **Requirements Compliance**: Does implementation match all requirements?
2. **Best Practices**: Are there better approaches for any components?
3. **Security**: Any security concerns with API key handling or file operations?
4. **Performance**: Will parallel execution work efficiently?
5. **Error Cases**: Are all error scenarios handled properly?
6. **Testing**: Are tests adequate for validation?
7. **Deployment**: Will installation/setup work smoothly?
8. **Bugs**: Any obvious bugs or issues?

## Specific Areas for Review

### Provider Abstraction
- Is BaseProvider design sound?
- Are retry mechanisms correctly implemented?
- Do subprocess wrappers handle errors properly?

### MCP Server
- Is MCP SDK usage correct?
- Are tool schemas properly defined?
- Will async/await work correctly with sync tool implementations?

### State Management
- Is YAML state management robust?
- Thread-safe operations?
- Proper fallback to defaults?

### Parallel Execution
- ThreadPoolExecutor usage correct?
- Thread-safe logging working?
- Proper future handling?

## Expected Review Format

Please provide:

1. **Overall Assessment** (1-2 paragraphs)
2. **Strengths** (bullet points)
3. **Issues Found** (by severity: Critical, Major, Minor)
4. **Specific Code Corrections** (with file:line references)
5. **Recommended Changes** (prioritized list)
6. **Testing Recommendations**

## Files to Review

The complete implementation follows below. All files are included in full.

---

# Complete Implementation

## pyproject.toml

```toml
[project]
name = "fiedler"
version = "1.0.0"
description = "Fiedler MCP Server - Orchestra Conductor Prototype"
requires-python = ">=3.10"
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
```

## config/models.yaml

```yaml
providers:
  google:
    api_key_env: GEMINI_API_KEY
    models:
      gemini-2.5-pro:
        aliases: [gemini]
        max_tokens: 32768
        timeout: 600
        retry_attempts: 3

  openai:
    api_key_env: OPENAI_API_KEY
    models:
      gpt-5:
        aliases: [gpt5]
        max_tokens: 32768
        timeout: 600
        retry_attempts: 3

  together:
    api_key_env: TOGETHER_API_KEY
    base_url: https://api.together.xyz/v1
    models:
      meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo:
        aliases: [llama, llama-3.1-70b, llama-70b]
        max_tokens: 8192
        timeout: 300
        retry_attempts: 3
      meta-llama/Llama-3.3-70B-Instruct-Turbo:
        aliases: [llama-3.3, llama-3.3-70b]
        max_tokens: 8192
        timeout: 300
        retry_attempts: 3
      deepseek-ai/DeepSeek-R1:
        aliases: [deepseek, deepseek-r1]
        max_tokens: 8192
        timeout: 400
        retry_attempts: 3
      Qwen/Qwen2.5-72B-Instruct-Turbo:
        aliases: [qwen, qwen-72b, qwen2.5]
        max_tokens: 8192
        timeout: 300
        retry_attempts: 3
      mistralai/Mistral-Large-2411:
        aliases: [mistral, mistral-large]
        max_tokens: 8192
        timeout: 350
        retry_attempts: 3
      nvidia/Llama-3.1-Nemotron-70B-Instruct-HF:
        aliases: [nemotron, nemotron-70b]
        max_tokens: 8192
        timeout: 300
        retry_attempts: 3

  xai:
    api_key_env: XAI_API_KEY
    models:
      grok-4-0709:
        aliases: [grok, grok-4]
        max_tokens: 16384
        timeout: 500
        retry_attempts: 3

defaults:
  models: [gemini-2.5-pro, gpt-5, grok-4-0709]
  output_dir: ./fiedler_output
```

## fiedler/__init__.py

```python
"""Fiedler MCP Server - Orchestra Conductor Prototype."""

__version__ = "1.0.0"
```

## fiedler/server.py

```python
"""Fiedler MCP Server - Orchestra Conductor Prototype."""
import asyncio
import sys
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from .tools import (
    fiedler_list_models,
    fiedler_set_models,
    fiedler_set_output,
    fiedler_get_config,
    fiedler_send,
)


# Create server instance
app = Server("fiedler")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available Fiedler tools."""
    return [
        Tool(
            name="fiedler_list_models",
            description="List all available LLM models with their properties (name, provider, aliases, max_tokens, capabilities).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="fiedler_set_models",
            description="Configure default models for fiedler_send. Accepts list of model IDs or aliases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model IDs or aliases to use as defaults",
                    },
                },
                "required": ["models"],
            },
        ),
        Tool(
            name="fiedler_set_output",
            description="Configure output directory for fiedler_send results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_dir": {
                        "type": "string",
                        "description": "Path to output directory",
                    },
                },
                "required": ["output_dir"],
            },
        ),
        Tool(
            name="fiedler_get_config",
            description="Get current Fiedler configuration (models, output_dir, total_available_models).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="fiedler_send",
            description="Send prompt and optional package files to configured LLMs. Returns results from all models in parallel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "User prompt or question to send to models",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of file paths to compile into package",
                    },
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional override of default models (use model IDs or aliases)",
                    },
                },
                "required": ["prompt"],
            },
        ),
    ]


@app.call_tool()
async def call_tool(name: str, arguments: Any) -> list[TextContent]:
    """Handle tool calls."""
    try:
        if name == "fiedler_list_models":
            result = fiedler_list_models()
        elif name == "fiedler_set_models":
            result = fiedler_set_models(arguments["models"])
        elif name == "fiedler_set_output":
            result = fiedler_set_output(arguments["output_dir"])
        elif name == "fiedler_get_config":
            result = fiedler_get_config()
        elif name == "fiedler_send":
            result = fiedler_send(
                prompt=arguments["prompt"],
                files=arguments.get("files"),
                models=arguments.get("models"),
            )
        else:
            raise ValueError(f"Unknown tool: {name}")

        # Format result as text
        import json
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except Exception as e:
        return [TextContent(type="text", text=f"Error: {str(e)}")]


async def main():
    """Run the MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


if __name__ == "__main__":
    asyncio.run(main())
```

## fiedler/utils/__init__.py

```python
"""Utility modules for Fiedler."""
from .logger import ProgressLogger
from .package import compile_package
from .state import load_state, save_state, get_models, get_output_dir, set_models, set_output_dir
from .tokens import estimate_tokens, check_token_budget

__all__ = [
    "ProgressLogger",
    "compile_package",
    "load_state",
    "save_state",
    "get_models",
    "get_output_dir",
    "set_models",
    "set_output_dir",
    "estimate_tokens",
    "check_token_budget",
]
```

## fiedler/utils/logger.py

```python
"""Thread-safe progress logger for Fiedler."""
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional


class ProgressLogger:
    """Thread-safe logger writing to stderr and optional file."""

    def __init__(self, correlation_id: Optional[str] = None, log_file: Optional[Path] = None):
        self.lock = threading.Lock()
        self.correlation_id = correlation_id or "-"
        self.log_file = log_file

        if self.log_file:
            self.log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.log_file, "w", encoding="utf-8") as f:
                f.write(f"=== Fiedler Run {datetime.now().isoformat()} (cid={self.correlation_id}) ===\n\n")

    def log(self, message: str, model: Optional[str] = None) -> None:
        """Log a message to stderr and file."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}] [cid:{self.correlation_id[:8]}]"
        if model:
            prefix += f" [{model}]"

        full_message = f"{prefix} {message}\n"

        with self.lock:
            # Always write to stderr for real-time progress
            sys.stderr.write(full_message)
            sys.stderr.flush()

            # Optionally write to file
            if self.log_file:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(full_message)
```

## fiedler/utils/state.py

```python
"""State management for Fiedler configuration."""
import os
from pathlib import Path
from typing import Dict, List, Optional
import yaml


STATE_FILE = Path.home() / ".fiedler" / "state.yaml"


def load_state(config_path: Path) -> Dict:
    """
    Load state from ~/.fiedler/state.yaml or initialize from defaults.

    Args:
        config_path: Path to models.yaml config file

    Returns:
        Dict with 'models' and 'output_dir' keys
    """
    # Try to load existing state
    if STATE_FILE.exists():
        try:
            with open(STATE_FILE, "r", encoding="utf-8") as f:
                state = yaml.safe_load(f) or {}
                if "models" in state and "output_dir" in state:
                    return state
        except Exception:
            pass  # Fall through to defaults

    # Load defaults from config
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    defaults = config.get("defaults", {})
    return {
        "models": defaults.get("models", ["gemini-2.5-pro", "gpt-5", "grok-4-0709"]),
        "output_dir": defaults.get("output_dir", "./fiedler_output")
    }


def save_state(models: List[str], output_dir: str) -> None:
    """
    Save state to ~/.fiedler/state.yaml.

    Args:
        models: List of model IDs
        output_dir: Output directory path
    """
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)

    with open(STATE_FILE, "w", encoding="utf-8") as f:
        yaml.dump({
            "models": models,
            "output_dir": output_dir
        }, f)


def get_models() -> List[str]:
    """Get current configured models."""
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    state = load_state(config_path)
    return state["models"]


def get_output_dir() -> str:
    """Get current configured output directory."""
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    state = load_state(config_path)
    return state["output_dir"]


def set_models(models: List[str]) -> None:
    """Set configured models."""
    output_dir = get_output_dir()
    save_state(models, output_dir)


def set_output_dir(output_dir: str) -> None:
    """Set configured output directory."""
    models = get_models()
    save_state(models, output_dir)
```

## fiedler/utils/tokens.py

```python
"""Token estimation utilities."""


def estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Uses rough approximation of ~4 characters per token.

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    return len(text) // 4


def check_token_budget(text: str, max_tokens: int, model_name: str) -> tuple[bool, int, str]:
    """
    Check if text fits within model's token budget.

    Args:
        text: Input text
        max_tokens: Model's maximum input tokens
        model_name: Model name for warning message

    Returns:
        Tuple of (is_ok, estimated_tokens, warning_message)
    """
    estimated = estimate_tokens(text)
    threshold = int(max_tokens * 0.8)  # Warn at 80%

    if estimated > max_tokens:
        return (False, estimated, f"{model_name}: Input ({estimated} tokens) exceeds max ({max_tokens})")
    elif estimated > threshold:
        return (True, estimated, f"{model_name}: Input ({estimated} tokens) near limit ({max_tokens})")
    else:
        return (True, estimated, "")
```

## fiedler/utils/package.py

```python
"""Document package compilation utilities."""
from pathlib import Path
from typing import List, Dict, Tuple

from .logger import ProgressLogger


def compile_package(files: List[str], logger: ProgressLogger) -> Tuple[str, Dict[str, int]]:
    """
    Compile list of files into a single package string.

    Args:
        files: List of file paths
        logger: Progress logger

    Returns:
        Tuple of (package_string, metadata_dict)

    Raises:
        FileNotFoundError: If any file doesn't exist
    """
    if not files:
        logger.log("No files provided")
        return "", {"num_files": 0, "bytes": 0}

    logger.log(f"Compiling package from {len(files)} files...")

    contents = []
    total_bytes = 0

    for i, file_path_str in enumerate(files):
        file_path = Path(file_path_str).resolve()

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise ValueError(f"Not a file: {file_path}")

        logger.log(f"Adding file {i+1}/{len(files)}: {file_path.name}")

        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            contents.append(f"--- {file_path.name} ---\n{content}")
            total_bytes += len(content.encode("utf-8"))

    package = "\n\n".join(contents)
    logger.log(f"✅ Package compiled: {total_bytes:,} bytes")

    return package, {"num_files": len(files), "bytes": total_bytes}
```

## fiedler/providers/__init__.py

```python
"""Provider implementations for different LLM APIs."""
from .base import BaseProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .together import TogetherProvider
from .xai import XAIProvider

__all__ = [
    "BaseProvider",
    "GeminiProvider",
    "OpenAIProvider",
    "TogetherProvider",
    "XAIProvider",
]
```

## fiedler/providers/base.py

```python
"""Base provider abstraction with retry logic."""
import time
from abc import ABC, abstractmethod
from typing import Dict, Any
from pathlib import Path


class BaseProvider(ABC):
    """Abstract base class for LLM providers."""

    def __init__(self, model_id: str, config: Dict[str, Any]):
        """
        Initialize provider.

        Args:
            model_id: Canonical model ID
            config: Model configuration dict (timeout, retry_attempts, max_tokens, etc.)
        """
        self.model_id = model_id
        self.timeout = config.get("timeout", 600)
        self.retry_attempts = config.get("retry_attempts", 3)
        self.max_tokens = config.get("max_tokens", 8192)

    def send(self, package: str, prompt: str, output_file: Path, logger) -> Dict[str, Any]:
        """
        Send request to model with retry logic.

        Args:
            package: Compiled document package
            prompt: User prompt
            output_file: Path to write response
            logger: ProgressLogger instance

        Returns:
            Dict with success, duration, output_file, output_size, tokens, error (if failed)
        """
        start_time = time.time()

        for attempt in range(self.retry_attempts):
            try:
                logger.log(f"Attempt {attempt + 1}/{self.retry_attempts}", self.model_id)
                result = self._send_impl(package, prompt, output_file, logger)
                duration = time.time() - start_time
                return {
                    "success": True,
                    "duration": round(duration, 2),
                    "output_file": str(output_file),
                    "output_size": output_file.stat().st_size if output_file.exists() else 0,
                    "tokens": result.get("tokens", {"prompt": 0, "completion": 0}),
                }
            except Exception as e:
                error_msg = str(e)
                logger.log(f"❌ Error: {error_msg}", self.model_id)

                if attempt < self.retry_attempts - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    logger.log(f"Retrying in {wait_time}s...", self.model_id)
                    time.sleep(wait_time)
                else:
                    # Final attempt failed
                    duration = time.time() - start_time
                    return {
                        "success": False,
                        "duration": round(duration, 2),
                        "output_file": None,
                        "output_size": 0,
                        "tokens": {"prompt": 0, "completion": 0},
                        "error": error_msg,
                    }

    @abstractmethod
    def _send_impl(self, package: str, prompt: str, output_file: Path, logger) -> Dict[str, Any]:
        """
        Concrete implementation of send logic.

        Must write response to output_file and return dict with optional 'tokens' key.

        Args:
            package: Compiled document package
            prompt: User prompt
            output_file: Path to write response
            logger: ProgressLogger instance

        Returns:
            Dict with optional 'tokens' key: {"tokens": {"prompt": int, "completion": int}}

        Raises:
            Exception: On any error (will trigger retry)
        """
        pass
```

## fiedler/providers/gemini.py

```python
"""Gemini provider implementation via subprocess."""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any

from .base import BaseProvider
from ..utils.tokens import estimate_tokens


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini models via gemini_client.py wrapper."""

    def __init__(self, model_id: str, config: Dict[str, Any], api_key_env: str):
        super().__init__(model_id, config)
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing environment variable: {api_key_env}")

    def _send_impl(self, package: str, prompt: str, output_file: Path, logger) -> Dict[str, Any]:
        # Combine prompt and package
        full_input = f"{prompt}\n\n{package}" if package else prompt

        # Call gemini_client.py via subprocess
        cmd = [
            "/mnt/projects/gemini-tool/venv/bin/python",
            "/mnt/projects/gemini-tool/gemini_client.py",
            "--model", self.model_id,
            "--timeout", str(self.timeout),
            "--stdin"
        ]

        env = os.environ.copy()
        env["GEMINI_API_KEY"] = self.api_key

        result = subprocess.run(
            cmd,
            input=full_input,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd="/mnt/projects/gemini-tool",
            timeout=self.timeout + 50  # Give it a bit more time
        )

        if result.returncode != 0:
            raise RuntimeError(f"gemini_client.py failed: {result.stderr}")

        # Write output
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result.stdout)

        # Estimate tokens (Gemini wrapper doesn't return usage)
        tokens = {
            "prompt": estimate_tokens(full_input),
            "completion": estimate_tokens(result.stdout),
        }

        return {"tokens": tokens}
```

## fiedler/providers/openai.py

```python
"""OpenAI provider implementation."""
import os
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models (GPT-4o, GPT-5, etc.)."""

    def __init__(self, model_id: str, config: Dict[str, Any], api_key_env: str):
        super().__init__(model_id, config)
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing environment variable: {api_key_env}")
        self.client = OpenAI(api_key=api_key)

    def _send_impl(self, package: str, prompt: str, output_file: Path, logger) -> Dict[str, Any]:
        # Combine prompt and package
        full_input = f"{prompt}\n\n{package}" if package else prompt

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": full_input}],
            max_completion_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        # Extract content
        content = response.choices[0].message.content

        # Write to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        # Extract token usage
        tokens = {
            "prompt": response.usage.prompt_tokens if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
        }

        return {"tokens": tokens}
```

## fiedler/providers/together.py

```python
"""Together.AI provider implementation."""
import os
from pathlib import Path
from typing import Dict, Any

from openai import OpenAI

from .base import BaseProvider


class TogetherProvider(BaseProvider):
    """Provider for Together.AI models (Llama, DeepSeek, Qwen, etc.)."""

    def __init__(self, model_id: str, config: Dict[str, Any], api_key_env: str, base_url: str):
        super().__init__(model_id, config)
        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(f"Missing environment variable: {api_key_env}")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _send_impl(self, package: str, prompt: str, output_file: Path, logger) -> Dict[str, Any]:
        # Combine prompt and package
        full_input = f"{prompt}\n\n{package}" if package else prompt

        # Call Together.AI API (OpenAI-compatible)
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[{"role": "user", "content": full_input}],
            max_tokens=self.max_tokens,
            timeout=self.timeout,
        )

        # Extract content
        content = response.choices[0].message.content

        # Write to file
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(content)

        # Extract token usage (Together.AI provides this)
        tokens = {
            "prompt": response.usage.prompt_tokens if response.usage else 0,
            "completion": response.usage.completion_tokens if response.usage else 0,
        }

        return {"tokens": tokens}
```

## fiedler/providers/xai.py

```python
"""xAI (Grok) provider implementation via subprocess."""
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Any

from .base import BaseProvider
from ..utils.tokens import estimate_tokens


class XAIProvider(BaseProvider):
    """Provider for xAI Grok models via grok_client.py wrapper."""

    def __init__(self, model_id: str, config: Dict[str, Any], api_key_env: str):
        super().__init__(model_id, config)
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"Missing environment variable: {api_key_env}")

    def _send_impl(self, package: str, prompt: str, output_file: Path, logger) -> Dict[str, Any]:
        # Combine prompt and package
        full_input = f"{prompt}\n\n{package}" if package else prompt

        # Write to temp file (grok_client.py uses --file)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as temp_file:
            temp_file.write(full_input)
            temp_path = temp_file.name

        try:
            # Call grok_client.py via subprocess
            cmd = [
                "/mnt/projects/gemini-tool/venv/bin/python",
                "/mnt/projects/ICCM/tools/grok_client.py",
                "--file", temp_path,
                "--model", self.model_id,
                "--max-tokens", str(self.max_tokens),
                prompt  # Prompt as positional argument
            ]

            env = os.environ.copy()
            env["XAI_API_KEY"] = self.api_key

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=self.timeout + 50
            )

            if result.returncode != 0:
                raise RuntimeError(f"grok_client.py failed: {result.stderr}")

            # Write output
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(result.stdout)

            # Estimate tokens (Grok wrapper doesn't return usage)
            tokens = {
                "prompt": estimate_tokens(full_input),
                "completion": estimate_tokens(result.stdout),
            }

            return {"tokens": tokens}

        finally:
            # Cleanup temp file
            if Path(temp_path).exists():
                Path(temp_path).unlink()
```

## fiedler/tools/__init__.py

```python
"""Fiedler MCP tools."""
from .models import fiedler_list_models
from .config import fiedler_set_models, fiedler_set_output, fiedler_get_config
from .send import fiedler_send

__all__ = [
    "fiedler_list_models",
    "fiedler_set_models",
    "fiedler_set_output",
    "fiedler_get_config",
    "fiedler_send",
]
```

## fiedler/tools/models.py

```python
"""fiedler_list_models tool implementation."""
import yaml
from pathlib import Path
from typing import Dict, List, Any


def build_alias_map(config: Dict) -> Dict[str, str]:
    """Build map of alias -> canonical model ID."""
    alias_map = {}
    for provider_name, provider_config in config["providers"].items():
        for model_id, model_config in provider_config["models"].items():
            # Model ID maps to itself
            alias_map[model_id] = model_id
            # Each alias maps to model ID
            for alias in model_config.get("aliases", []):
                alias_map[alias] = model_id
    return alias_map


def fiedler_list_models() -> Dict[str, Any]:
    """
    List all available models with their properties.

    Returns:
        Dict with 'models' key containing list of model info dicts
    """
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    models = []
    for provider_name, provider_config in config["providers"].items():
        for model_id, model_config in provider_config["models"].items():
            models.append({
                "name": model_id,
                "provider": provider_name,
                "aliases": model_config.get("aliases", []),
                "max_tokens": model_config.get("max_tokens", 8192),
                "capabilities": model_config.get("capabilities", ["text"]),
            })

    return {"models": models}
```

## fiedler/tools/config.py

```python
"""Configuration management tools."""
import yaml
from pathlib import Path
from typing import Dict, Any, List

from ..utils import get_models, get_output_dir, set_models as save_models, set_output_dir as save_output_dir
from .models import build_alias_map


def fiedler_set_models(models: List[str]) -> Dict[str, Any]:
    """
    Configure default models for fiedler_send.

    Args:
        models: List of model IDs or aliases

    Returns:
        Dict with status, resolved models, message
    """
    # Load config for alias resolution
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    alias_map = build_alias_map(config)

    # Resolve aliases to canonical model IDs
    resolved = []
    for model in models:
        if model in alias_map:
            resolved.append(alias_map[model])
        else:
            raise ValueError(f"Unknown model or alias: {model}")

    # Save to state
    save_models(resolved)

    return {
        "status": "configured",
        "models": resolved,
        "message": f"Default models updated ({len(resolved)} models configured)"
    }


def fiedler_set_output(output_dir: str) -> Dict[str, Any]:
    """
    Configure output directory for fiedler_send.

    Args:
        output_dir: Path to output directory

    Returns:
        Dict with status, output_dir, message
    """
    save_output_dir(output_dir)

    return {
        "status": "configured",
        "output_dir": output_dir,
        "message": "Output directory updated"
    }


def fiedler_get_config() -> Dict[str, Any]:
    """
    Get current Fiedler configuration.

    Returns:
        Dict with models, output_dir, total_available_models
    """
    # Get current state
    current_models = get_models()
    current_output_dir = get_output_dir()

    # Count available models
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    total_models = sum(
        len(provider_config["models"])
        for provider_config in config["providers"].values()
    )

    return {
        "models": current_models,
        "output_dir": current_output_dir,
        "total_available_models": total_models
    }
```

## fiedler/tools/send.py

```python
"""fiedler_send tool implementation."""
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import json
import threading

from ..utils import get_models, get_output_dir
from ..utils.logger import ProgressLogger
from ..utils.package import compile_package
from ..utils.tokens import check_token_budget
from ..providers import GeminiProvider, OpenAIProvider, TogetherProvider, XAIProvider


def create_provider(model_id: str, config: Dict[str, Any]):
    """Factory function to create appropriate provider."""
    # Find model in config
    for provider_name, provider_config in config["providers"].items():
        if model_id in provider_config["models"]:
            model_config = provider_config["models"][model_id]
            api_key_env = provider_config["api_key_env"]

            if provider_name == "google":
                return GeminiProvider(model_id, model_config, api_key_env)
            elif provider_name == "openai":
                return OpenAIProvider(model_id, model_config, api_key_env)
            elif provider_name == "together":
                base_url = provider_config.get("base_url", "https://api.together.xyz/v1")
                return TogetherProvider(model_id, model_config, api_key_env, base_url)
            elif provider_name == "xai":
                return XAIProvider(model_id, model_config, api_key_env)

    raise ValueError(f"Unknown model: {model_id}")


def send_to_model(
    model_id: str,
    package: str,
    prompt: str,
    output_dir: Path,
    correlation_id: str,
    config: Dict[str, Any],
    logger: ProgressLogger,
    lock: threading.Lock
) -> Dict[str, Any]:
    """Send to single model (runs in thread)."""
    try:
        # Create provider
        provider = create_provider(model_id, config)

        # Check token budget
        full_input = f"{prompt}\n\n{package}" if package else prompt
        within_budget, estimated, warning = check_token_budget(
            full_input, provider.max_tokens, model_id
        )
        if warning:
            logger.log(warning, model_id)

        # Create output file
        output_file = output_dir / f"{model_id.replace('/', '_')}.md"

        # Send
        logger.log(f"Sending to {model_id}...", model_id)
        result = provider.send(package, prompt, output_file, logger)

        if result["success"]:
            logger.log(f"✓ {model_id} completed in {result['duration']:.1f}s", model_id)
            return {
                "model": model_id,
                "status": "success",
                "output_file": str(output_file),
                "duration": result["duration"],
                "tokens": result.get("tokens", {}),
            }
        else:
            logger.log(f"✗ {model_id} failed: {result.get('error', 'unknown error')}", model_id)
            return {
                "model": model_id,
                "status": "failed",
                "error": result.get("error", "unknown error"),
            }

    except Exception as e:
        logger.log(f"✗ {model_id} exception: {str(e)}", model_id)
        return {
            "model": model_id,
            "status": "failed",
            "error": str(e),
        }


def fiedler_send(
    prompt: str,
    files: Optional[List[str]] = None,
    models: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Send prompt and optional package to configured models.

    Args:
        prompt: User prompt/question
        files: Optional list of file paths to compile into package
        models: Optional override of default models

    Returns:
        Dict with status, correlation_id, output_dir, results per model
    """
    # Load config
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get models (use override or defaults)
    if models is None:
        models = get_models()
        if not models:
            # Fall back to config defaults
            models = config.get("defaults", {}).get("models", ["gemini-2.5-pro"])

    # Get output directory
    output_base = get_output_dir()

    # Create correlation ID and timestamped directory
    correlation_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / f"{timestamp}_{correlation_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    log_file = output_dir / "fiedler.log"
    logger = ProgressLogger(correlation_id, log_file)

    logger.log(f"Starting Fiedler run (correlation_id: {correlation_id})")
    logger.log(f"Models: {', '.join(models)}")
    logger.log(f"Output: {output_dir}")

    # Compile package if files provided
    package = ""
    package_metadata = {}
    if files:
        logger.log(f"Compiling package from {len(files)} file(s)")
        package, package_metadata = compile_package(files, logger)
        logger.log(f"Package compiled: {package_metadata['total_size']} bytes, {package_metadata['total_lines']} lines")

    # Send to models in parallel
    lock = threading.Lock()
    results = []

    logger.log(f"Sending to {len(models)} model(s) in parallel")

    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = {
            executor.submit(
                send_to_model,
                model_id,
                package,
                prompt,
                output_dir,
                correlation_id,
                config,
                logger,
                lock
            ): model_id
            for model_id in models
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

    # Create summary
    summary = {
        "correlation_id": correlation_id,
        "timestamp": timestamp,
        "prompt": prompt,
        "files": files or [],
        "package_metadata": package_metadata,
        "models": models,
        "results": results,
    }

    summary_file = output_dir / "summary.json"
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Determine overall status
    success_count = sum(1 for r in results if r["status"] == "success")
    if success_count == len(models):
        status = "success"
    elif success_count > 0:
        status = "partial_success"
    else:
        status = "failure"

    logger.log(f"Run complete: {success_count}/{len(models)} models succeeded")

    return {
        "status": status,
        "correlation_id": correlation_id,
        "output_dir": str(output_dir),
        "summary_file": str(summary_file),
        "results": results,
        "message": f"{success_count}/{len(models)} models succeeded"
    }
```

## tests/test_basic.py

```python
"""Basic tests for Fiedler MCP Server."""
import pytest
from pathlib import Path
import yaml

from fiedler.tools.models import fiedler_list_models, build_alias_map
from fiedler.tools.config import fiedler_set_models, fiedler_set_output, fiedler_get_config
from fiedler.utils.tokens import estimate_tokens, check_token_budget


def test_list_models():
    """Test that fiedler_list_models returns expected structure."""
    result = fiedler_list_models()

    assert "models" in result
    assert isinstance(result["models"], list)
    assert len(result["models"]) > 0

    # Check first model has required fields
    model = result["models"][0]
    assert "name" in model
    assert "provider" in model
    assert "aliases" in model
    assert "max_tokens" in model
    assert "capabilities" in model


def test_build_alias_map():
    """Test alias resolution."""
    config = {
        "providers": {
            "google": {
                "models": {
                    "gemini-2.5-pro": {
                        "aliases": ["gemini", "gemini-pro"]
                    }
                }
            }
        }
    }

    alias_map = build_alias_map(config)

    assert alias_map["gemini-2.5-pro"] == "gemini-2.5-pro"
    assert alias_map["gemini"] == "gemini-2.5-pro"
    assert alias_map["gemini-pro"] == "gemini-2.5-pro"


def test_set_models():
    """Test fiedler_set_models with valid aliases."""
    result = fiedler_set_models(["gemini", "gpt-5"])

    assert result["status"] == "configured"
    assert "gemini-2.5-pro" in result["models"]
    assert "gpt-5" in result["models"]
    assert "models configured" in result["message"]


def test_set_models_invalid():
    """Test fiedler_set_models with invalid model."""
    with pytest.raises(ValueError, match="Unknown model or alias"):
        fiedler_set_models(["invalid-model"])


def test_set_output():
    """Test fiedler_set_output."""
    test_dir = "/tmp/fiedler_test_output"
    result = fiedler_set_output(test_dir)

    assert result["status"] == "configured"
    assert result["output_dir"] == test_dir


def test_get_config():
    """Test fiedler_get_config returns all fields."""
    result = fiedler_get_config()

    assert "models" in result
    assert "output_dir" in result
    assert "total_available_models" in result
    assert isinstance(result["models"], list)
    assert isinstance(result["total_available_models"], int)


def test_estimate_tokens():
    """Test token estimation."""
    text = "hello world" * 100  # 1100 chars
    tokens = estimate_tokens(text)

    # ~4 chars/token = ~275 tokens
    assert 200 < tokens < 350


def test_check_token_budget():
    """Test token budget checking."""
    # Within budget
    within, estimated, warning = check_token_budget("hello" * 10, 1000, "test-model")
    assert within is True
    assert warning == ""

    # Warning threshold (>80%)
    within, estimated, warning = check_token_budget("x" * 3500, 1000, "test-model")
    assert within is True
    assert "WARNING" in warning

    # Over budget
    within, estimated, warning = check_token_budget("x" * 5000, 1000, "test-model")
    assert within is False
    assert "exceeds" in warning


def test_models_yaml_structure():
    """Test that config/models.yaml has correct structure."""
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    assert "providers" in config
    assert "defaults" in config

    # Check defaults
    assert "models" in config["defaults"]
    assert "output_dir" in config["defaults"]

    # Check each provider
    for provider_name, provider_config in config["providers"].items():
        assert "api_key_env" in provider_config
        assert "models" in provider_config

        # Check each model
        for model_id, model_config in provider_config["models"].items():
            assert "aliases" in model_config
            assert "max_tokens" in model_config
            assert "capabilities" in model_config


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

## README.md

```markdown
# Fiedler MCP Server

**Orchestra Conductor Prototype** - Unified MCP interface for multiple LLM providers.

## Overview

Fiedler is an MCP (Model Context Protocol) server that provides a unified interface for sending prompts to multiple LLM providers simultaneously. It abstracts away provider-specific APIs and presents a consistent set of tools for orchestrating LLM calls.

**Key Features:**
- **Multi-Provider Support**: Google Gemini, OpenAI, Together.AI, xAI (Grok)
- **Parallel Execution**: Send to multiple models simultaneously
- **Persistent Configuration**: Set default models and output directory once
- **Automatic Retries**: Exponential backoff for transient failures
- **Token Management**: Budget checking and warnings
- **Correlation IDs**: Traceability across runs
- **Package Compilation**: Combine multiple files into single context

## Installation

### Prerequisites

- Python 3.10+
- API keys for providers you want to use:
  - `GEMINI_API_KEY` - Google Gemini
  - `OPENAI_API_KEY` - OpenAI
  - `TOGETHER_API_KEY` - Together.AI
  - `XAI_API_KEY` - xAI (Grok)

### Setup

1. **Install dependencies:**
   ```bash
   cd /mnt/projects/ICCM/fiedler
   pip install -e .
   ```

2. **Set environment variables:**
   ```bash
   export GEMINI_API_KEY="your_key_here"
   export OPENAI_API_KEY="your_key_here"
   export TOGETHER_API_KEY="your_key_here"
   export XAI_API_KEY="your_key_here"
   ```

3. **Configure MCP client** (e.g., Claude Desktop):

   Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:
   ```json
   {
     "mcpServers": {
       "fiedler": {
         "command": "python",
         "args": ["-m", "fiedler.server"],
         "cwd": "/mnt/projects/ICCM/fiedler",
         "env": {
           "GEMINI_API_KEY": "your_key_here",
           "OPENAI_API_KEY": "your_key_here",
           "TOGETHER_API_KEY": "your_key_here",
           "XAI_API_KEY": "your_key_here"
         }
       }
     }
   }
   ```

4. **Initialize state** (optional):

   State file created automatically at `~/.fiedler/state.yaml` on first use with config defaults.

## Available Models

| Provider | Model ID | Aliases | Max Tokens |
|----------|----------|---------|------------|
| Google | gemini-2.5-pro | gemini, gemini-pro | 1,000,000 |
| OpenAI | gpt-5 | openai, gpt5 | 128,000 |
| OpenAI | gpt-4o | gpt4 | 128,000 |
| Together.AI | meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo | llama-3.1-70b | 131,072 |
| Together.AI | meta-llama/Llama-3.3-70B-Instruct-Turbo | llama-3.3-70b | 131,072 |
| Together.AI | deepseek-ai/DeepSeek-R1 | deepseek-r1 | 65,536 |
| Together.AI | Qwen/Qwen2.5-72B-Instruct-Turbo | qwen-2.5-72b | 32,768 |
| Together.AI | mistralai/Mistral-Large-2411 | mistral-large | 131,072 |
| Together.AI | nvidia/Llama-3.1-Nemotron-70B-Instruct-HF | nemotron-70b | 131,072 |
| xAI | grok-4-0709 | grok, grok-4 | 131,072 |

## MCP Tools

### 1. fiedler_list_models

List all available models with their properties.

**Input:** None

**Output:**
```json
{
  "models": [
    {
      "name": "gemini-2.5-pro",
      "provider": "google",
      "aliases": ["gemini", "gemini-pro"],
      "max_tokens": 1000000,
      "capabilities": ["text", "vision"]
    },
    ...
  ]
}
```

### 2. fiedler_set_models

Configure default models for `fiedler_send`.

**Input:**
```json
{
  "models": ["gemini", "gpt-5", "grok"]
}
```

**Output:**
```json
{
  "status": "configured",
  "models": ["gemini-2.5-pro", "gpt-5", "grok-4-0709"],
  "message": "Default models updated (3 models configured)"
}
```

### 3. fiedler_set_output

Configure output directory for results.

**Input:**
```json
{
  "output_dir": "/mnt/projects/ICCM/results"
}
```

**Output:**
```json
{
  "status": "configured",
  "output_dir": "/mnt/projects/ICCM/results",
  "message": "Output directory updated"
}
```

### 4. fiedler_get_config

Get current configuration.

**Input:** None

**Output:**
```json
{
  "models": ["gemini-2.5-pro", "gpt-5", "grok-4-0709"],
  "output_dir": "/mnt/projects/ICCM/results",
  "total_available_models": 10
}
```

### 5. fiedler_send

Send prompt to models (uses defaults or override).

**Input:**
```json
{
  "prompt": "Explain quantum computing in simple terms",
  "files": ["/path/to/context1.md", "/path/to/context2.md"],
  "models": ["gemini", "gpt-5"]
}
```

**Parameters:**
- `prompt` (required): User prompt/question
- `files` (optional): List of file paths to compile into package
- `models` (optional): Override default models for this call

**Output:**
```json
{
  "status": "success",
  "correlation_id": "a1b2c3d4",
  "output_dir": "/mnt/projects/ICCM/results/20250102_143052_a1b2c3d4",
  "summary_file": "/mnt/projects/ICCM/results/20250102_143052_a1b2c3d4/summary.json",
  "results": [
    {
      "model": "gemini-2.5-pro",
      "status": "success",
      "output_file": "/mnt/projects/ICCM/results/20250102_143052_a1b2c3d4/gemini-2.5-pro.md",
      "duration": 12.3,
      "tokens": {"prompt": 1500, "completion": 800}
    },
    ...
  ],
  "message": "2/2 models succeeded"
}
```

**Status Values:**
- `success`: All models succeeded
- `partial_success`: Some models succeeded
- `failure`: All models failed

## Usage Examples

### Configure Once

```python
# Set default models
fiedler_set_models(models=["gemini", "gpt-5", "grok"])

# Set output directory
fiedler_set_output(output_dir="/mnt/projects/ICCM/results")
```

### Send to Default Models

```python
result = fiedler_send(
    prompt="Review this architecture for consistency"
)
```

### Send with Package

```python
result = fiedler_send(
    prompt="Synthesize requirements from these papers",
    files=[
        "/mnt/projects/ICCM/docs/papers/01_Primary_Paper.md",
        "/mnt/projects/ICCM/docs/papers/02_Training.md"
    ]
)
```

### Override Models for Single Call

```python
result = fiedler_send(
    prompt="Quick test",
    models=["gemini"]  # Override defaults
)
```

## Output Structure

Each `fiedler_send` creates a timestamped directory:

```
/output_dir/
  20250102_143052_a1b2c3d4/
    summary.json           # Run metadata and results
    fiedler.log            # Progress log
    gemini-2.5-pro.md      # Response from Gemini
    gpt-5.md               # Response from GPT-5
    grok-4-0709.md         # Response from Grok
```

## Error Handling

- **Automatic Retries**: 3 attempts with exponential backoff (1s, 2s, 4s)
- **Partial Success**: Returns results from successful models even if some fail
- **Token Warnings**: Warns at 80% of max_tokens
- **Detailed Logging**: All progress logged to `fiedler.log`

## State Management

Configuration persisted at `~/.fiedler/state.yaml`:

```yaml
models:
  - gemini-2.5-pro
  - gpt-5
  - grok-4-0709
output_dir: /mnt/projects/ICCM/results
```

## Architecture

```
fiedler/
├── config/
│   └── models.yaml          # Provider/model registry
├── fiedler/
│   ├── utils/
│   │   ├── logger.py        # Thread-safe logging
│   │   ├── state.py         # Persistent state
│   │   ├── tokens.py        # Token estimation
│   │   └── package.py       # File compilation
│   ├── providers/
│   │   ├── base.py          # Abstract provider
│   │   ├── gemini.py        # Gemini subprocess wrapper
│   │   ├── openai.py        # OpenAI SDK provider
│   │   ├── together.py      # Together.AI provider
│   │   └── xai.py           # xAI subprocess wrapper
│   ├── tools/
│   │   ├── models.py        # fiedler_list_models
│   │   ├── config.py        # Configuration tools
│   │   └── send.py          # fiedler_send orchestrator
│   └── server.py            # MCP server entry point
└── pyproject.toml
```

## License

Part of the ICCM (Intelligent Context and Conversation Management) system.

## Version

1.0.0 - Initial release
```

---

# End of Implementation

Please provide your detailed review following the format specified above.
