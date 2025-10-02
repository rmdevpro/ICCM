Overall Assessment

The Fiedler MCP Server shows a solid architectural foundation: a clean separation between the MCP entrypoint, tool layer, providers, and utility modules. The concurrency model (threaded provider fan-out with a thread-safe logger and retry logic in a base provider) is straightforward and easy to reason about. Security hardening around secrets is appreciably thorough (OS-native keyring with secure-backend enforcement, explicit fallback policy, and non-leaky logging). The tool suite is cohesive and covers a practical end-to-end workflow.

However, a few production-grade issues remain. The most serious blockers are: resolving the configuration file path using a relative ascent that will break in installed environments; hard-coded, environment-specific subprocess client paths for Gemini and xAI; and passing the prompt as a command-line argument to the Grok client (risking data exposure). There are also some correctness/compatibility gaps (OpenAI/Together request timeout usage), packaging concerns for shipping config files, and operational polish (capping parallelism, output path normalization, and better binary file handling). Addressing the items below would elevate the package to production readiness.

Issues Found

Critical

1) Config path resolution breaks after installation
- File: fiedler/utils/paths.py (get_config_path)
- Problem: Uses Path(__file__).resolve().parents[2] to reach project root. This works in a source tree but resolves to site-packages when installed, breaking config lookup.
- Impact: Server will fail to start or resolve models in most installed environments.

2) Hard-coded subprocess paths for providers
- Files:
  - fiedler/providers/gemini.py (_send_impl)
  - fiedler/providers/xai.py (_send_impl)
- Problem: Absolute paths to external clients and a specific venv (/mnt/projects/...) make the code non-portable and fragile outside a bespoke environment.
- Impact: Immediate failure on any machine that doesn’t mirror that path. Not production-ready.

3) Prompt leakage via process arguments (xAI/Grok)
- File: fiedler/providers/xai.py (_send_impl)
- Problem: Passing prompt as a positional argument to grok_client.py makes it visible via process listings to other users on the host.
- Impact: Sensitive data exposure risk. Must be treated as a security bug.

4) Config/models.yaml packaging and discovery
- Files: pyproject.toml, fiedler/utils/paths.py, config/models.yaml
- Problem: models.yaml is outside the package and not configured as package data; it won’t be available post-install. Coupled with the current path resolution this guarantees runtime failure outside the repo.
- Impact: Breaks production installs. Needs packaging and importlib.resources-based lookup.

Major

1) OpenAI/Together timeout parameter usage may be incompatible
- Files:
  - fiedler/providers/openai.py (_send_impl)
  - fiedler/providers/together.py (_send_impl)
- Problem: Passing timeout=self.timeout into chat.completions.create may not be supported across OpenAI client versions. The canonical approach is to set the client-level timeout or pass request_timeout.
- Impact: Runtime TypeError in some environments.

2) Unlimited parallelism by number of models
- File: fiedler/tools/send.py (ThreadPoolExecutor(max_workers=len(models)))
- Problem: No cap on max_workers; scaling model list can exhaust system resources or hit provider rate limits.
- Impact: Performance instability and API throttling. Add an upper bound.

3) Output directory not normalized
- Files:
  - fiedler/tools/config.py (fiedler_set_output)
  - fiedler/utils/state.py (save_state / get_output_dir)
- Problem: output_dir is stored as-is without expanduser/absolute normalization; relative paths depend on cwd.
- Impact: Surprising file placements and fragility in daemonized/packaged runs.

4) Binary/encoding robustness in package compilation
- File: fiedler/utils/package.py (compile_package)
- Problem: Always reads files as UTF-8; will error or mis-handle binary files or unknown encodings.
- Impact: Failures on arbitrary inputs. Should detect/handle binary or fallback with errors='replace'.

5) Incomplete error structuring for MCP tools
- File: fiedler/server.py (call_tool)
- Problem: Errors are returned as TextContent "Error: ..." strings rather than structured tool errors. Harder for clients to distinguish failures programmatically.
- Impact: Poor client ergonomics and observability in MCP ecosystems.

6) Token budgeting semantics
- Files:
  - fiedler/utils/tokens.py
  - fiedler/providers/* (use of max_tokens)
- Problem: Model max_tokens are treated as total context window in budget checks, but also passed as completion max_tokens to APIs (OpenAI/Together). This can easily overshoot or mis-budget completion lengths.
- Impact: Truncation and API errors. Needs separation of input budget and completion budget.

7) Alias collision risk
- File: fiedler/tools/models.py (build_alias_map)
- Problem: Last-write wins for duplicate aliases across providers, silently shadowing models.
- Impact: Hard-to-debug routing errors. Should detect collisions and error out.

Minor

1) Type hints mismatch
- File: fiedler/utils/secrets.py (get_backend_info)
- Problem: Annotates return type as dict[str, str], but returns booleans under secure/available.
- Impact: Minor typing inconsistency.

2) Unused imports
- Files:
  - fiedler/providers/openai.py (os)
  - fiedler/providers/together.py (os)
  - fiedler/providers/gemini.py (tempfile)
  - fiedler/tools/send.py (threading)
- Impact: Cosmetic.

3) Duplicate models in set_models
- File: fiedler/tools/config.py (fiedler_set_models)
- Problem: No deduplication; same canonical model can be set multiple times.
- Impact: Redundant work and clutter.

4) delete_api_key error semantics
- File: fiedler/utils/secrets.py (delete_api_key)
- Problem: Exceptions during delete map to False and tool reports “not_found”. Caller loses signal that there was an error.
- Impact: Diagnostics confusion. Consider surfacing error state.

5) Logging and privacy hygiene
- Files:
  - fiedler/tools/send.py (summary.json, logger)
- Problem: summary.json stores full prompt; for sensitive use cases this may be undesirable.
- Impact: Compliance/privacy concerns. Provide a redaction/opt-out flag.

Specific Code Corrections (file:line)

Note: Line numbers are approximate; reference by function where lines differ.

Critical fixes

1) Use importlib.resources for models.yaml, and package the file
- fiedler/utils/paths.py:9-16
  - Replace get_config_path with an importlib.resources-based loader.
  - Also support FIEDLER_CONFIG env override.

Suggested replacement:
from importlib.resources import files
import os
from pathlib import Path

def get_config_path() -> Path:
    override = os.getenv("FIEDLER_CONFIG")
    if override:
        return Path(override).expanduser().resolve()
    # Use package resource
    try:
        return Path(files("fiedler.config").joinpath("models.yaml"))
    except Exception:
        # Fallback to repo layout for dev checkouts
        return Path(__file__).resolve().parents[2] / "config" / "models.yaml"

- pyproject.toml: add package data
[tool.setuptools.package-data]
"fiedler.config" = ["models.yaml"]

- Move config/models.yaml into fiedler/config/models.yaml in the repo (or configure MANIFEST.in accordingly).

2) Parameterize external subprocess client paths
- fiedler/providers/gemini.py:28-45
  - Replace hard-coded paths with:
    - Environment variables: FIEDLER_GEMINI_CLIENT_PY, FIEDLER_TOOL_PYTHON
    - Or config fields under providers.google: client.path, client.python
  - Validate existence before run; error with actionable message.

Example snippet:
client_py = os.getenv("FIEDLER_GEMINI_CLIENT_PY") or config.get("client_path")
python_bin = os.getenv("FIEDLER_TOOL_PYTHON") or config.get("python_path") or sys.executable
if not client_py or not Path(client_py).exists():
    raise RuntimeError("Gemini client not configured. Set FIEDLER_GEMINI_CLIENT_PY or providers.google.models.<model>.client_path")

cmd = [python_bin, client_py, "--model", self.model_id, "--timeout", str(self.timeout), "--stdin"]

3) Stop passing prompt via argv; use stdin or file only
- fiedler/providers/xai.py:35-54
  - Remove "prompt" positional argument in cmd construction.
  - Either:
    - Pass all content via --file (already done), and let client read file and model id.
    - Or stream via stdin (like Gemini path).

Replace cmd = [...] with:
cmd = [python_bin, client_py, "--file", temp_path, "--model", self.model_id, "--max-tokens", str(self.max_tokens)]

4) Package and discover config (complements #1)
- Ensure config/models.yaml is included in the wheel and accessible via importlib.resources as above.
- Update README to document FIEDLER_CONFIG override and default discovery.

Major fixes

5) OpenAI/Together request timeouts
- fiedler/providers/openai.py:25-44
- fiedler/providers/together.py:25-44
  - Prefer client-level timeouts:
self.client = OpenAI(api_key=api_key, timeout=self.timeout)
  - Or method-level request_timeout:
response = self.client.chat.completions.create(
    model=self.model_id,
    messages=[{"role":"user","content":full_input}],
    max_tokens=self.max_tokens,
    request_timeout=self.timeout,
)

6) Cap parallelism (and guard zero)
- fiedler/tools/send.py:106-139
  - Define max_workers = min(len(models), max(2, (os.cpu_count() or 4)))
  - Add environment/config override FIEDLER_MAX_WORKERS.

Example:
import os
max_workers = min(len(models), int(os.getenv("FIEDLER_MAX_WORKERS", max(2, (os.cpu_count() or 4)))))

with ThreadPoolExecutor(max_workers=max_workers) as executor:

7) Normalize output_dir
- fiedler/tools/config.py:43-63 (fiedler_set_output)
  - Normalize path before saving:
from pathlib import Path
normalized = str(Path(output_dir).expanduser().resolve())
save_output_dir(normalized)

- fiedler/utils/state.py:58-83
  - In get_output_dir(), return str(Path(state["output_dir"]).expanduser().resolve())

8) Separate context vs completion token budgets
- fiedler/utils/tokens.py and providers using max_tokens
  - Add per-model config fields: context_window, max_completion_tokens
  - Use check_token_budget against context_window
  - Pass max_completion_tokens to API calls.

Change models.yaml entries accordingly:
max_tokens: 32768            # context window
max_completion_tokens: 2048  # new

Then in providers:
self.max_completion_tokens = config.get("max_completion_tokens", min(2048, self.max_tokens))
... create(..., max_tokens=self.max_completion_tokens, ...)

And in check_token_budget calls, use provider.max_tokens (context).

9) Detect alias collisions
- fiedler/tools/models.py:6-21 (build_alias_map)
  - Before overwriting, check if alias already present and raise ValueError with both model ids.

Minor fixes

10) Robust file reading with encoding fallback
- fiedler/utils/package.py:23-65
  - Handle UnicodeDecodeError; optionally detect binary and skip or base64-encode with a header.

Example:
try:
    content = file_path.read_text(encoding="utf-8")
except UnicodeDecodeError:
    logger.log(f"Non-UTF8 content detected in {file_path.name}, using replacement decoding")
    content = file_path.read_text(encoding="utf-8", errors="replace")

11) Structured MCP errors
- fiedler/server.py:86-105 (call_tool)
  - Raise exceptions that MCP can map to tool errors, or wrap in a structured error payload instead of plain text.
  - If staying with TextContent, include a JSON object with {"error": {"type": "...", "message": "...", "tool": name}} for better client parsing.

12) Improve delete_api_key error propagation
- fiedler/utils/secrets.py:128-153
  - Re-raise exceptions or return an explicit error that the caller surfaces.

Option:
except Exception as e:
    raise RuntimeError(f"Failed to delete key for provider={provider}: {e}") from e

Update tools/keys.fiedler_delete_key to surface status="error".

13) Small cleanups and typing
- Remove unused imports:
  - providers/openai.py: remove os
  - providers/together.py: remove os
  - providers/gemini.py: remove tempfile
  - tools/send.py: remove threading
- fiedler/utils/secrets.py: get_backend_info() -> dict[str, object] or dict[str, Any]

14) Privacy option for prompt storage
- fiedler/tools/send.py: After building summary, if os.getenv("FIEDLER_REDACT_PROMPT") in ("1","true"), set summary["prompt"] = "[REDACTED]".

Production Readiness Rating

6.5/10

Rationale: The core architecture, concurrency, retry model, and keyring hardening are strong. The remaining critical issues (config discovery/packaging, hard-coded client paths, prompt leakage in subprocess args) are blockers for production. Fixing these, plus the major operational/compatibility items, would move this to 8.5–9/10.

Recommendations

- Configuration and packaging
  - Package models.yaml inside fiedler/config and load via importlib.resources with an environment-variable override.
  - Provide a single Fiedler config (fiedler.yaml) that can also house external client paths, concurrency caps, and privacy flags.

- Provider robustness and portability
  - Eliminate hard-coded paths; support pure-HTTP implementations where possible to avoid subprocesses.
  - If subprocess wrappers are required, parameterize via config/env and pass sensitive data via stdin or file only.
  - Add graceful 429/5xx backoff/jitter handling (provider-specific rate limits).

- API compatibility and observability
  - Align OpenAI/Together request timeouts to documented client options and capture HTTP errors explicitly with structured details in results (status_code, retry info).
  - Return structured MCP tool errors for client ergonomics.
  - Log provider response IDs when available (OpenAI “id”) for traceability.

- Performance and resilience
  - Cap max_workers, and optionally add per-provider rate limiters.
  - Add cancellation support (e.g., a cancellation flag or future.cancel checks) to abort long runs.
  - Consider asyncio-based HTTP for providers to reduce thread usage if moving away from subprocess wrappers.

- Security and privacy
  - Do not include prompts in summary.json by default, or make it opt-in with a clear flag.
  - Document FIEDLER_REQUIRE_SECURE_KEYRING and how to satisfy secure backends across OSs.
  - Consider encrypting summary/package outputs at rest when configured to do so.

- UX and documentation
  - Expand README/SECURITY with installation steps, config overrides, keyring backend guidance, and example workflows.
  - Add docstrings and type hints consistently in tools and providers to match the quality of utilities.
  - Provide example MCP client snippets demonstrating tool usage.

- Testing
  - Add unit tests for utils (paths, secrets, tokens, package).
  - Add integration tests that mock provider calls and assert parallel execution, retries, and summaries.
  - CI to lint (ruff/flake8), type-check (mypy/pyright), and run tests across platforms (Linux/macOS/Windows) to catch keyring and path issues early.

Correlation ID: FIEDLER-FINAL-R1