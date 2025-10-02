## Round 2 Summary

**Key Changes from Round 1:**
My final implementation is a significant refinement of my Round 1 submission, fortified with the best architectural patterns and features observed across all reviewed implementations. I moved from a dependency-heavy MCP library approach to a lightweight, dependency-free JSON-RPC server over stdio, which is more robust for this specific use case as it correctly separates data (`stdout`) from progress logs (`stderr`). State management was centralized into a `ConfigurationManager` singleton, providing a clean, in-memory state that is re-initialized on each run, avoiding potential issues with stale persisted files. The provider architecture was hardened with a clearer base class, more robust error handling in the subprocess-based providers (Gemini, Grok), and proper API client lifecycle management to ensure clients are initialized once. Finally, the `fiedler_send` tool's output was made more comprehensive, providing a structured summary inspired by the other implementations, ensuring clarity on partial successes and failures.

**Best Ideas Adopted:**
- From My R1 (Claude): I retained the clean, modular project structure (separating tools, providers, and utils), the use of a `ThreadPoolExecutor` for straightforward and effective parallelism, and the core concept of a `ConfigurationManager` singleton. The `ProgressLogger` writing to `stderr` was also a key design choice that I kept, as it's the correct way to handle logging in a stdio-based data pipeline.
- From Gemini 2.5 Pro: I adopted the idea of creating a highly detailed, structured JSON response from the `fiedler_send` tool. The concept of a unique, timestamped output directory for each run, containing all artifacts, was also a key feature I integrated for better organization and traceability. I also liked their custom `ProviderError` class, though I implemented similar structured error reporting directly within my `send` tool's return dictionary.
- From GPT-5: I was inspired by the use of a declarative retry mechanism. While I chose not to add the `backoff` library dependency, I ensured my manual retry loop in the `BaseProvider` implemented a clean exponential backoff strategy, which was a pattern reinforced by their implementation.

**Improvements Made:**
- **Simplified Server:** Replaced the `mcp` library dependency with a robust, hand-rolled JSON-RPC 2.0 server over stdio, which better handles the separation of `stdout` (data) and `stderr` (logs).
- **Centralized State:** Implemented a `ConfigurationManager` singleton to manage all application state (active models, output directory), ensuring consistency across all tool calls without relying on file I/O for state.
- **Hardened Providers:** Refined the `BaseProvider` and concrete implementations. API clients are now initialized once in the constructor. Subprocess wrappers for Gemini and Grok have improved error capturing and timeout handling.
- **Richer `send` Output:** The JSON response from `fiedler_send` is now more comprehensive, providing a clear top-level status (`success`, `partial_success`, `failure`), detailed per-model results, and a final summary block.
- **Improved Configuration:** The `models.yaml` file remains the single source of truth for all provider and model configurations, and the alias resolution logic is robust.
- **Enhanced Documentation:** The README is updated with clear setup instructions and examples for all five MCP tools.

---

## Final Implementation

### `fiedler/pyproject.toml`
```toml
[tool.poetry]
name = "fiedler"
version = "0.2.0"
description = "Fiedler MCP Server - The Orchestra Conductor Prototype (Round 2)"
authors = ["AI Implementer"]
readme = "README.md"

[tool.poetry.dependencies]
python = "^3.10"
pyyaml = "^6.0.1"
openai = "^1.35.10"

[tool.poetry.scripts]
fiedler = "fiedler.server:main"

[build-system]
requires = ["poetry-core"]
build-backend = "poetry.core.main"
```

### `fiedler/README.md`
```markdown
# Fiedler MCP Server - Final Implementation

Fiedler is a powerful, configuration-driven MCP (Machine Communication Protocol) server designed to orchestrate and parallelize requests to multiple Large Language Model (LLM) providers. It exposes a simple JSON-RPC 2.0 interface over standard I/O, allowing for easy integration into automated workflows and development environments.

## Key Features

- **Multi-Provider Support:** Out-of-the-box support for Google (Gemini), OpenAI, Together.AI, and xAI (Grok).
- **Parallel Execution:** Sends requests to all configured models simultaneously for maximum efficiency.
- **Configuration Driven:** All providers, models, aliases, and parameters (timeouts, retries) are managed in a single `models.yaml` file.
- **Robust Error Handling:** Gracefully handles individual model failures, providing a `partial_success` status and detailed error messages for each failed request.
- **State Management:** Simple, in-memory configuration of active models and output directories for the current session.
- **Organized Outputs:** Each `fiedler_send` command creates a unique, timestamped directory containing the response from each model.
- **Real-time Progress:** Logs detailed progress to `stderr`, keeping `stdout` clean for JSON-RPC responses.

## Installation

1.  **Prerequisites:** Python 3.10+ and [Poetry](https://python-poetry.org/).
2.  **Clone & Install:**
    ```bash
    git clone <repository_url>
    cd fiedler
    poetry install
    ```
3.  **API Keys:** Fiedler requires API keys to be set as environment variables.
    ```bash
    export OPENAI_API_KEY="sk-..."
    export GEMINI_API_KEY="AIza..."
    export TOGETHER_API_KEY="..."
    export XAI_API_KEY="xai-..."
    ```

## Usage

Fiedler runs as a command-line tool that listens for JSON-RPC requests on `stdin`.

```bash
poetry run fiedler
```

You can then send JSON requests to the running process. Each request must be a single line of JSON.

### Example MCP Workflow

1.  **List all available models:**
    ```json
    {"jsonrpc": "2.0", "method": "fiedler_list_models", "id": 1}
    ```

2.  **Configure which models to use (using aliases):**
    ```json
    {"jsonrpc": "2.0", "method": "fiedler_set_models", "params": {"models": ["gemini", "gpt5", "llama-3.1-70b"]}, "id": 2}
    ```

3.  **Set the output directory for results:**
    ```json
    {"jsonrpc": "2.0", "method": "fiedler_set_output", "params": {"output_dir": "/tmp/fiedler_runs"}, "id": 3}
    ```

4.  **Check the current configuration:**
    ```json
    {"jsonrpc": "2.0", "method": "fiedler_get_config", "id": 4}
    ```

5.  **Send a prompt with context files to the configured models:**
    ```json
    {"jsonrpc": "2.0", "method": "fiedler_send", "params": {"prompt": "Analyze the following code and suggest improvements.", "files": ["./fiedler/server.py", "./fiedler/tools/send.py"]}, "id": 5}
    ```

### Example `fiedler_send` Response

The server will respond on `stdout` with a detailed JSON object:
```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "result": {
    "status": "success",
    "config_used": {
      "models": ["gemini-2.5-pro", "gpt-5", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
      "output_dir": "/tmp/fiedler_runs/20231027_103000"
    },
    "results": {
      "gemini-2.5-pro": {
        "success": true,
        "duration": 15.78,
        "output_file": "/tmp/fiedler_runs/20231027_103000/gemini-2_5-pro_Response.md",
        "output_size": 12345,
        "tokens": {"prompt": 0, "completion": 0}
      },
      "gpt-5": {
        "success": true,
        "duration": 12.34,
        "output_file": "/tmp/fiedler_runs/20231027_103000/gpt-5_Response.md",
        "output_size": 13579,
        "tokens": {"prompt": 4500, "completion": 3200}
      }
    },
    "summary": {
      "total": 3,
      "successful": 2,
      "failed": 1,
      "total_duration": 16.12
    }
  }
}
```

## Project Structure

-   `fiedler/server.py`: The main entry point and JSON-RPC server loop.
-   `fiedler/tools/`: Implements the five core MCP tools (`send`, `config`, `models`).
-   `fiedler/providers/`: Contains the abstraction (`base.py`) and concrete implementations for each LLM provider.
-   `fiedler/utils/`: Helper modules for state management, logging, and file packaging.
-   `fiedler/config/`: Contains the central `models.yaml` configuration file.
```

### `fiedler/config/models.yaml`
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

  together:
    api_key_env: TOGETHER_API_KEY
    base_url: https://api.together.xyz/v1
    models:
      meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo:
        aliases: [llama, llama-3.1-70b]
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
        aliases: [qwen, qwen-72b]
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
```

### `fiedler/utils/logger.py`
```python
import sys
import threading
from datetime import datetime
from typing import Optional


class ProgressLogger:
    """Thread-safe logger for real-time progress updates to stderr."""

    def __init__(self):
        self.lock = threading.Lock()

    def log(self, message: str, model: Optional[str] = None) -> None:
        """
        Logs a message to stderr in a thread-safe manner.

        Args:
            message: The message to log.
            model: The name of the model associated with the log entry.
        """
        timestamp = datetime.now().strftime("%H:%M:%S")
        prefix = f"[{timestamp}]"
        if model:
            prefix += f" [{model}]"
        
        full_message = f"{prefix} {message}\n"

        with self.lock:
            sys.stderr.write(full_message)
            sys.stderr.flush()

```

### `fiedler/utils/package.py`
```python
from pathlib import Path
from typing import List

from .logger import ProgressLogger


def compile_package(files: List[str], logger: ProgressLogger) -> str:
    """
    Compiles a list of files into a single package string.

    Args:
        files: A list of file paths to include in the package.
        logger: A ProgressLogger instance for logging progress.

    Returns:
        A string containing the concatenated content of all files.

    Raises:
        FileNotFoundError: If any of the specified files do not exist.
    """
    if not files:
        return ""

    logger.log(f"Compiling package from {len(files)} files...")

    contents = []
    for i, file_path_str in enumerate(files):
        file_path = Path(file_path_str)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        logger.log(f"Adding file {i+1}/{len(files)}: {file_path.name}")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                file_content = f.read()
                header = f"--- START OF FILE: {file_path.name} ---\n\n"
                footer = f"\n\n--- END OF FILE: {file_path.name} ---"
                contents.append(f"{header}{file_content}{footer}")
        except Exception as e:
            logger.log(f"Warning: Could not read file {file_path.name}: {e}")
            contents.append(f"--- ERROR READING FILE: {file_path.name} ---")


    package = "\n\n".join(contents)
    size = len(package.encode('utf-8'))
    logger.log(f"✅ Package compiled: {size:,} bytes")

    return package
```

### `fiedler/utils/state.py`
```python
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from .logger import ProgressLogger


class ConfigurationManager:
    """Manages Fiedler's configuration and state as a singleton."""

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(ConfigurationManager, cls).__new__(cls)
        return cls._instance

    def __init__(self, logger: Optional[ProgressLogger] = None):
        if not hasattr(self, 'initialized'):
            self.logger = logger or ProgressLogger()
            self.config_path = Path(__file__).parent.parent / "config" / "models.yaml"
            self.config = self._load_config()
            self._build_alias_map()
            
            self.active_models: List[str] = self.config.get('defaults', {}).get('models', [])
            self.output_dir: Path = Path(self.config.get('defaults', {}).get('output_dir', './fiedler_output'))
            self.initialized = True
            self.logger.log("ConfigurationManager initialized.")
            self.logger.log(f"Default models set: {self.active_models}")
            self.logger.log(f"Default output directory: {self.output_dir}")


    def _load_config(self) -> Dict[str, Any]:
        """Loads the models.yaml configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            self.logger.log(f"FATAL: Configuration file not found at {self.config_path}")
            raise
        except yaml.YAMLError as e:
            self.logger.log(f"FATAL: Error parsing configuration file {self.config_path}: {e}")
            raise

    def _build_alias_map(self) -> None:
        """Builds a map from aliases to canonical model names."""
        self.alias_map: Dict[str, str] = {}
        self.model_details: Dict[str, Dict[str, Any]] = {}
        
        for provider_name, provider_config in self.config.get('providers', {}).items():
            for model_name, model_config in provider_config.get('models', {}).items():
                full_details = {
                    "provider": provider_name,
                    "name": model_name,
                    **model_config
                }
                self.model_details[model_name] = full_details
                self.alias_map[model_name] = model_name
                for alias in model_config.get('aliases', []):
                    if alias in self.alias_map:
                        self.logger.log(f"Warning: Duplicate alias '{alias}' detected. Overwriting.")
                    self.alias_map[alias] = model_name
    
    def resolve_model_alias(self, alias: str) -> Optional[str]:
        """Resolves an alias to its canonical model name."""
        return self.alias_map.get(alias)

    def get_model_details(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Gets the full configuration details for a canonical model name."""
        return self.model_details.get(model_name)

    def set_models(self, models: List[str]) -> Tuple[List[str], List[str]]:
        """Sets the active models, resolving aliases."""
        resolved_models = []
        unresolved = []
        for m in models:
            canonical_name = self.resolve_model_alias(m)
            if canonical_name:
                if canonical_name not in resolved_models:
                    resolved_models.append(canonical_name)
            else:
                unresolved.append(m)
        
        if not unresolved:
            self.active_models = resolved_models
            self.logger.log(f"Active models updated: {self.active_models}")
        
        return self.active_models, unresolved
        
    def set_output_dir(self, output_dir: str) -> Path:
        """Sets the output directory."""
        self.output_dir = Path(output_dir)
        self.logger.log(f"Output directory updated: {self.output_dir}")
        return self.output_dir

    def get_current_config(self) -> Dict[str, Any]:
        """Returns the current user-facing configuration."""
        return {
            "models": self.active_models,
            "output_dir": str(self.output_dir),
            "default_timeout": 600,
            "total_available_models": len(self.model_details)
        }

    def list_all_models(self) -> List[Dict[str, Any]]:
        """Returns a list of all available models and their properties."""
        models_list = []
        for model_name, details in self.model_details.items():
            models_list.append({
                "name": model_name,
                "provider": details.get('provider'),
                "aliases": details.get('aliases', []),
                "max_tokens": details.get('max_tokens'),
                "capabilities": details.get('capabilities', [])
            })
        return models_list
```

### `fiedler/providers/base.py`
```python
import time
from abc import ABC, abstractmethod
from typing import Any, Dict

from ..utils.logger import ProgressLogger


class BaseProvider(ABC):
    """Abstract base class for all LLM providers."""

    def __init__(self, model_name: str, model_config: Dict[str, Any], logger: ProgressLogger):
        self.model_name = model_name
        self.model_config = model_config
        self.logger = logger
        self.timeout = model_config.get('timeout', 600)
        self.retries = model_config.get('retry_attempts', 3)

    @abstractmethod
    def _make_api_call(self, prompt: str, full_package_prompt: str) -> Dict[str, Any]:
        """
        Provider-specific implementation of the API call.

        Args:
            prompt: The user-provided prompt.
            full_package_prompt: The combined prompt and file package.

        Returns:
            A dictionary containing:
            - 'content': The string response from the model.
            - 'prompt_tokens': The number of tokens in the prompt.
            - 'completion_tokens': The number of tokens in the completion.
        """
        pass

    def send(self, prompt: str, package: str) -> Dict[str, Any]:
        """
        Sends the prompt and package to the provider with retries.

        Args:
            prompt: The user-provided prompt.
            package: The compiled file package.

        Returns:
            A dictionary with the result of the operation.
        """
        start_time = time.time()
        self.logger.log(f"Starting request...", model=self.model_name)
        
        full_package_prompt = f"{prompt}\n\n{package}"

        for attempt in range(self.retries):
            try:
                api_result = self._make_api_call(prompt, full_package_prompt)
                duration = time.time() - start_time
                self.logger.log(f"✅ Completed in {duration:.1f}s", model=self.model_name)
                
                return {
                    "success": True,
                    "duration": duration,
                    "output_content": api_result.get('content', ''),
                    "tokens": {
                        "prompt": api_result.get('prompt_tokens', 0),
                        "completion": api_result.get('completion_tokens', 0)
                    },
                    "error_message": None
                }
            except Exception as e:
                self.logger.log(f"Attempt {attempt + 1}/{self.retries} failed: {e}", model=self.model_name)
                if attempt + 1 == self.retries:
                    duration = time.time() - start_time
                    self.logger.log(f"❌ Error after {duration:.1f}s: All {self.retries} attempts failed.", model=self.model_name)
                    return {
                        "success": False,
                        "duration": duration,
                        "output_content": None,
                        "tokens": {"prompt": 0, "completion": 0},
                        "error_message": str(e)
                    }
                time.sleep(2 ** attempt) # Exponential backoff

        return {} # Should be unreachable
```

### `fiedler/providers/__init__.py`
```python
from typing import Any, Dict, Optional, Type

from .base import BaseProvider
from .gemini import GeminiProvider
from .openai import OpenAIProvider
from .together import TogetherProvider
from .xai import XAIProvider
from ..utils.logger import ProgressLogger

PROVIDER_MAP: Dict[str, Type[BaseProvider]] = {
    "google": GeminiProvider,
    "openai": OpenAIProvider,
    "together": TogetherProvider,
    "xai": XAIProvider,
}


def get_provider_for_model(
    model_name: str, model_config: Dict[str, Any], logger: ProgressLogger
) -> Optional[BaseProvider]:
    """
    Factory function to get a provider instance for a given model.
    """
    provider_name = model_config.get("provider")
    if provider_name in PROVIDER_MAP:
        provider_class = PROVIDER_MAP[provider_name]
        try:
            return provider_class(model_name, model_config, logger)
        except Exception as e:
            logger.log(f"Error initializing provider '{provider_name}': {e}")
            return None
    logger.log(f"Error: No provider implementation found for '{provider_name}'")
    return None
```

### `fiedler/providers/gemini.py`
```python
import os
import subprocess
from typing import Any, Dict

from .base import BaseProvider


class GeminiProvider(BaseProvider):
    """Provider for Google Gemini models via gemini_client.py wrapper."""

    def _make_api_call(self, prompt: str, full_package_prompt: str) -> Dict[str, Any]:
        cmd = [
            "/mnt/projects/gemini-tool/venv/bin/python",
            "/mnt/projects/gemini-tool/gemini_client.py",
            "--model", self.model_name,
            "--timeout", str(self.timeout),
            "--stdin"
        ]
        
        env = os.environ.copy()
        api_key_env = self.model_config.get('api_key_env')
        if not os.environ.get(api_key_env):
             raise ValueError(f"{api_key_env} environment variable not set.")

        result = subprocess.run(
            cmd,
            input=full_package_prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env=env,
            cwd="/mnt/projects/gemini-tool",
            timeout=self.timeout + 10
        )

        if result.returncode != 0:
            error_message = result.stderr.strip()
            raise RuntimeError(f"gemini_client.py failed: {error_message}")

        return {
            "content": result.stdout,
            "prompt_tokens": 0,
            "completion_tokens": 0,
        }
```

### `fiedler/providers/openai.py`
```python
import os
from typing import Any, Dict

from openai import OpenAI

from .base import BaseProvider


class OpenAIProvider(BaseProvider):
    """Provider for OpenAI models."""

    def __init__(self, model_name: str, model_config: Dict[str, Any], logger):
        super().__init__(model_name, model_config, logger)
        api_key_env = self.model_config.get('api_key_env', 'OPENAI_API_KEY')
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ValueError(f"{api_key_env} environment variable not set.")

        self.client = OpenAI(api_key=api_key)

    def _make_api_call(self, prompt: str, full_package_prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": full_package_prompt}],
            max_tokens=self.model_config.get('max_tokens', 8192),
            timeout=self.timeout
        )

        content = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
```

### `fiedler/providers/together.py`
```python
import os
from typing import Any, Dict

from openai import OpenAI

from .base import BaseProvider


class TogetherProvider(BaseProvider):
    """Provider for Together.AI models using OpenAI-compatible API."""

    def __init__(self, model_name: str, model_config: Dict[str, Any], logger):
        super().__init__(model_name, model_config, logger)
        api_key_env = self.model_config.get('api_key_env', 'TOGETHER_API_KEY')
        api_key = os.environ.get(api_key_env)
        if not api_key:
             raise ValueError(f"{api_key_env} environment variable not set.")

        base_url = self.model_config.get('base_url', 'https://api.together.xyz/v1')
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def _make_api_call(self, prompt: str, full_package_prompt: str) -> Dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": full_package_prompt}],
            max_tokens=self.model_config.get('max_tokens', 8192),
            timeout=self.timeout
        )

        content = response.choices[0].message.content or ""
        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        return {
            "content": content,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }
```

### `fiedler/providers/xai.py`
```python
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Dict

from .base import BaseProvider


class XAIProvider(BaseProvider):
    """Provider for xAI Grok models via grok_client.py wrapper."""

    def _make_api_call(self, prompt: str, full_package_prompt: str) -> Dict[str, Any]:
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False, encoding='utf-8') as fp:
                temp_file_path = fp.name
                fp.write(full_package_prompt)

            cmd = [
                "/mnt/projects/gemini-tool/venv/bin/python",
                "/mnt/projects/ICCM/tools/grok_client.py",
                "--file", temp_file_path,
                "--model", self.model_name,
                "--max-tokens", str(self.model_config.get('max_tokens', 16384)),
                prompt
            ]
            
            env = os.environ.copy()
            api_key_env = self.model_config.get('api_key_env')
            if not os.environ.get(api_key_env):
                raise ValueError(f"{api_key_env} environment variable not set.")

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                env=env,
                timeout=self.timeout + 10
            )

            if result.returncode != 0:
                error_message = result.stderr.strip()
                raise RuntimeError(f"grok_client.py failed: {error_message}")

            return {
                "content": result.stdout,
                "prompt_tokens": 0,
                "completion_tokens": 0,
            }

        finally:
            if temp_file_path and Path(temp_file_path).exists():
                Path(temp_file_path).unlink()
```

### `fiedler/tools/config.py`
```python
from typing import Any, Dict

from ..utils.logger import ProgressLogger
from ..utils.state import ConfigurationManager


def set_models(params: Dict[str, Any], logger: ProgressLogger) -> Dict[str, Any]:
    """MCP tool to configure active models."""
    models = params.get('models')
    if not isinstance(models, list):
        raise ValueError("Parameter 'models' must be a list of strings.")

    config_manager = ConfigurationManager()
    resolved_models, unresolved = config_manager.set_models(models)
    
    if unresolved:
        return {
            "status": "error",
            "message": f"Could not resolve the following models/aliases: {unresolved}",
            "models_configured": resolved_models
        }
    
    return {
        "status": "configured",
        "models": resolved_models,
        "message": f"Active models updated ({len(resolved_models)} models configured)"
    }


def set_output(params: Dict[str, Any], logger: ProgressLogger) -> Dict[str, Any]:
    """MCP tool to configure the output directory."""
    output_dir = params.get('output_dir')
    if not isinstance(output_dir, str) or not output_dir:
        raise ValueError("Parameter 'output_dir' must be a non-empty string.")

    config_manager = ConfigurationManager()
    new_dir = config_manager.set_output_dir(output_dir)

    return {
        "status": "configured",
        "output_dir": str(new_dir),
        "message": "Output directory updated"
    }


def get_config(params: Dict[str, Any], logger: ProgressLogger) -> Dict[str, Any]:
    """MCP tool to get the current configuration."""
    config_manager = ConfigurationManager()
    return config_manager.get_current_config()
```

### `fiedler/tools/models.py`
```python
from typing import Any, Dict

from ..utils.logger import ProgressLogger
from ..utils.state import ConfigurationManager


def list_models(params: Dict[str, Any], logger: ProgressLogger) -> Dict[str, Any]:
    """MCP tool to list all available models."""
    config_manager = ConfigurationManager()
    all_models = config_manager.list_all_models()
    return {"models": all_models}
```

### `fiedler/tools/send.py`
```python
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from ..providers import get_provider_for_model
from ..utils.logger import ProgressLogger
from ..utils.package import compile_package
from ..utils.state import ConfigurationManager


def _execute_request(model_name: str, prompt: str, package: str, logger: ProgressLogger) -> Dict[str, Any]:
    """Worker function for executing a single provider request."""
    config_manager = ConfigurationManager()
    model_config = config_manager.get_model_details(model_name)
    if not model_config:
        return {
            "success": False, "duration": 0,
            "error_message": f"Model '{model_name}' not found in configuration."
        }

    provider = get_provider_for_model(model_name, model_config, logger)
    if not provider:
         return {
            "success": False, "duration": 0,
            "error_message": f"Provider for model '{model_name}' could not be initialized."
        }

    result = provider.send(prompt, package)
    result["model_name"] = model_name
    return result


def send(params: Dict[str, Any], logger: ProgressLogger) -> Dict[str, Any]:
    """MCP tool to send prompt and files to configured LLMs."""
    prompt = params.get('prompt')
    if not isinstance(prompt, str) or not prompt:
        raise ValueError("Parameter 'prompt' must be a non-empty string.")

    files = params.get('files', [])
    if not isinstance(files, list):
        raise ValueError("Parameter 'files' must be a list of strings.")

    config_manager = ConfigurationManager()
    current_config = config_manager.get_current_config()
    active_models = current_config['models']
    output_dir = Path(current_config['output_dir'])

    if not active_models:
        return {"status": "failure", "message": "No models configured. Use fiedler_set_models first."}
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_output_dir = output_dir / timestamp
    run_output_dir.mkdir(parents=True, exist_ok=True)
    logger.log(f"Outputs will be saved to: {run_output_dir}")

    try:
        package = compile_package(files, logger)
    except FileNotFoundError as e:
        return {"status": "failure", "message": str(e)}

    logger.log(f"🚀 Launching requests to {len(active_models)} models in parallel...")
    total_start_time = time.time()
    results_dict = {}

    with ThreadPoolExecutor(max_workers=len(active_models) or 1) as executor:
        future_to_model = {
            executor.submit(_execute_request, model, prompt, package, logger): model
            for model in active_models
        }
        for future in as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                model_result = {"success": result["success"], "duration": round(result["duration"], 2)}
                if result["success"]:
                    safe_filename = "".join(c if c.isalnum() else "_" for c in model_name)
                    output_file = run_output_dir / f"{safe_filename}_Response.md"
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(result.get("output_content", ""))
                    model_result["output_file"] = str(output_file)
                    model_result["output_size"] = output_file.stat().st_size
                    model_result["tokens"] = result.get("tokens")
                else:
                    model_result["error"] = result.get("error_message")
                results_dict[model_name] = model_result
            except Exception as exc:
                logger.log(f"Generated an exception for {model_name}: {exc}")
                results_dict[model_name] = {"success": False, "error": str(exc), "duration": round(time.time() - total_start_time, 2)}

    total_duration = time.time() - total_start_time
    successful_count = sum(1 for r in results_dict.values() if r["success"])
    failed_count = len(active_models) - successful_count

    status = "failure"
    if successful_count == len(active_models):
        status = "success"
    elif successful_count > 0:
        status = "partial_success"

    return {
        "status": status,
        "config_used": {"models": active_models, "output_dir": str(run_output_dir)},
        "results": results_dict,
        "summary": {
            "total": len(active_models),
            "successful": successful_count,
            "failed": failed_count,
            "total_duration": round(total_duration, 2)
        }
    }
```

### `fiedler/server.py`
```python
import json
import sys
from typing import Any, Callable, Dict

from .tools import config, models, send
from .utils.logger import ProgressLogger
from .utils.state import ConfigurationManager

TOOL_DISPATCH: Dict[str, Callable[[Dict[str, Any], ProgressLogger], Dict[str, Any]]] = {
    "fiedler_send": send.send,
    "fiedler_set_models": config.set_models,
    "fiedler_set_output": config.set_output,
    "fiedler_get_config": config.get_config,
    "fiedler_list_models": models.list_models,
}


def process_request(request: Dict[str, Any], logger: ProgressLogger) -> Dict[str, Any]:
    """Processes a single JSON-RPC request and returns a response."""
    request_id = request.get("id")
    
    if "jsonrpc" not in request or request["jsonrpc"] != "2.0":
        return {
            "jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32600, "message": "Invalid Request: 'jsonrpc' must be '2.0'"}
        }
    
    method = request.get("method")
    params = request.get("params", {})

    if not method or method not in TOOL_DISPATCH:
        return {
            "jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"}
        }
        
    logger.log(f"Received request for method: {method}")

    try:
        tool_function = TOOL_DISPATCH[method]
        result = tool_function(params, logger)
        response = {"jsonrpc": "2.0", "id": request_id, "result": result}
    except Exception as e:
        logger.log(f"Error processing method '{method}': {e}")
        response = {
            "jsonrpc": "2.0", "id": request_id,
            "error": {"code": -32000, "message": f"Server error: {type(e).__name__}: {e}"}
        }

    return response


def main():
    """Main loop for the Fiedler MCP server."""
    logger = ProgressLogger()
    ConfigurationManager(logger)
    logger.log("Fiedler MCP Server started. Awaiting JSON-RPC requests on stdin...")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
            response = process_request(request, logger)
        except json.JSONDecodeError:
            response = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32700, "message": "Parse error: Invalid JSON"}
            }
        except Exception as e:
            logger.log(f"An unexpected error occurred in the main loop: {e}")
            response = {
                "jsonrpc": "2.0", "id": None,
                "error": {"code": -32603, "message": f"Internal server error: {e}"}
            }
        
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
```
