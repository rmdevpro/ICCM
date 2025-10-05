# Phase 1: Fiedler Conversation Logging - Code Implementation

**Date:** 2025-10-05
**Purpose:** Add conversation logging to Fiedler per approved architectural realignment plan
**Target:** Log all LLM conversations via Godot MCP `logger_log` tool

---

## Implementation Summary

Add conversation logging to `fiedler_send()` function to capture all LLM request/response pairs and send them to Godot for storage in the `conversations` table.

**Key Design Decisions:**
1. New `conversation_id` per `fiedler_send` call (stateless)
2. Non-blocking logging (LLM calls succeed even if Godot unavailable)
3. Log BEFORE returning result to user
4. Use existing `log_to_godot()` MCP client

---

## Code Changes

### File 1: `/mnt/projects/ICCM/fiedler-blue/fiedler/tools/send.py`

**Change 1.1: Add imports**

**Location:** Lines 1-16 (import section)

```python
# ADD these imports
import asyncio
from ..godot.mcp_logger import log_to_godot
```

**Change 1.2: Add conversation logging helper function**

**Location:** After `send_to_model()` function (after line 95), before `fiedler_send()`

```python
def log_conversation(
    conversation_id: str,
    turn_number: int,
    role: str,
    content: str,
    model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    timing_ms: Optional[int] = None,
    files: Optional[List[str]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """
    Log conversation turn to Godot (non-blocking).

    Args:
        conversation_id: UUID for this conversation
        turn_number: 1 for request, 2 for response
        role: 'user' or 'assistant'
        content: Prompt or response text
        model: LLM model used (for response turn)
        input_tokens: Token count for input (for response turn)
        output_tokens: Token count for output (for response turn)
        timing_ms: Response time in milliseconds (for response turn)
        files: List of files included in request (for request turn)
        correlation_id: Fiedler correlation ID for tracing

    Note:
        Silently fails on error - logging should never break LLM calls
    """
    try:
        # Prepare metadata
        metadata = {}
        if model:
            metadata['model'] = model
        if input_tokens is not None:
            metadata['input_tokens'] = input_tokens
        if output_tokens is not None:
            metadata['output_tokens'] = output_tokens
        if timing_ms is not None:
            metadata['timing_ms'] = timing_ms
        if files:
            metadata['files'] = files
        if correlation_id:
            metadata['correlation_id'] = correlation_id

        # Prepare log data
        log_data = {
            'conversation_id': conversation_id,
            'turn_number': turn_number,
            'role': role,
            'content': content,
            'metadata': metadata
        }

        # Send to Godot via MCP (async, non-blocking)
        asyncio.run(log_to_godot(
            level='INFO',
            message=f'Conversation {conversation_id} turn {turn_number} ({role})',
            component='fiedler-conversations',
            data=log_data,
            trace_id=correlation_id
        ))

    except Exception:
        # Silently fail - logging should never break LLM calls
        pass
```

**Change 1.3: Add conversation logging to fiedler_send()**

**Location:** In `fiedler_send()` function, after line 149 (after output_dir creation)

```python
    # Create conversation ID (one per fiedler_send call)
    conversation_id = str(uuid.uuid4())
```

**Location:** After line 177 (after "Sending to models" log), BEFORE ThreadPoolExecutor

```python
    # Log request (turn 1)
    log_conversation(
        conversation_id=conversation_id,
        turn_number=1,
        role='user',
        content=prompt,
        files=files,
        correlation_id=correlation_id
    )
```

**Location:** Inside the `for future in as_completed(futures):` loop, after `result = future.result()` (line 195)

```python
            result = future.result()
            results.append(result)

            # Log response (turn 2) - only for successful results
            if result["status"] == "success":
                log_conversation(
                    conversation_id=conversation_id,
                    turn_number=2,
                    role='assistant',
                    content=f"Response from {result['model']} (see {result['output_file']})",
                    model=result['model'],
                    input_tokens=result.get('tokens', {}).get('prompt', None),
                    output_tokens=result.get('tokens', {}).get('completion', None),
                    timing_ms=int(result.get('duration', 0) * 1000),
                    correlation_id=correlation_id
                )
```

---

## Modified send.py (Complete with Changes)

```python
"""fiedler_send tool implementation."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import uuid
import json
import asyncio  # ADDED

from ..utils import get_models, get_output_dir
from ..utils.logger import ProgressLogger
from ..utils.package import compile_package
from ..utils.tokens import check_token_budget
from ..providers import GeminiProvider, OpenAIProvider, TogetherProvider, XAIProvider
from ..godot.mcp_logger import log_to_godot  # ADDED


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
    logger: ProgressLogger
) -> Dict[str, Any]:
    """Send to single model (runs in thread)."""
    try:
        # Create provider
        provider = create_provider(model_id, config)

        # Check token budget
        full_input = f"{prompt}\n\n{package}" if package else prompt
        within_budget, estimated, warning = check_token_budget(
            full_input,
            provider.context_window,
            provider.max_completion_tokens,
            model_id
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


# ADDED: Conversation logging helper
def log_conversation(
    conversation_id: str,
    turn_number: int,
    role: str,
    content: str,
    model: Optional[str] = None,
    input_tokens: Optional[int] = None,
    output_tokens: Optional[int] = None,
    timing_ms: Optional[int] = None,
    files: Optional[List[str]] = None,
    correlation_id: Optional[str] = None
) -> None:
    """
    Log conversation turn to Godot (non-blocking).

    Args:
        conversation_id: UUID for this conversation
        turn_number: 1 for request, 2 for response
        role: 'user' or 'assistant'
        content: Prompt or response text
        model: LLM model used (for response turn)
        input_tokens: Token count for input (for response turn)
        output_tokens: Token count for output (for response turn)
        timing_ms: Response time in milliseconds (for response turn)
        files: List of files included in request (for request turn)
        correlation_id: Fiedler correlation ID for tracing

    Note:
        Silently fails on error - logging should never break LLM calls
    """
    try:
        # Prepare metadata
        metadata = {}
        if model:
            metadata['model'] = model
        if input_tokens is not None:
            metadata['input_tokens'] = input_tokens
        if output_tokens is not None:
            metadata['output_tokens'] = output_tokens
        if timing_ms is not None:
            metadata['timing_ms'] = timing_ms
        if files:
            metadata['files'] = files
        if correlation_id:
            metadata['correlation_id'] = correlation_id

        # Prepare log data
        log_data = {
            'conversation_id': conversation_id,
            'turn_number': turn_number,
            'role': role,
            'content': content,
            'metadata': metadata
        }

        # Send to Godot via MCP (async, non-blocking)
        asyncio.run(log_to_godot(
            level='INFO',
            message=f'Conversation {conversation_id} turn {turn_number} ({role})',
            component='fiedler-conversations',
            data=log_data,
            trace_id=correlation_id
        ))

    except Exception:
        # Silently fail - logging should never break LLM calls
        pass


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
    from ..utils.paths import get_config_path
    from .models import build_alias_map

    config_path = get_config_path()
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # Get models (use override or defaults)
    if models is None:
        models = get_models()
        if not models:
            # Fall back to config defaults
            models = config.get("defaults", {}).get("models", ["gemini-2.5-pro"])
    else:
        # Resolve aliases if models provided as override
        alias_map = build_alias_map(config)
        resolved = []
        for m in models:
            if m in alias_map:
                resolved.append(alias_map[m])
            else:
                raise ValueError(f"Unknown model or alias: {m}")
        models = resolved

    # Validate models list
    if not models:
        raise ValueError("No models configured. Use fiedler_set_models or pass models to fiedler_send.")

    # Get output directory
    output_base = get_output_dir()

    # Create correlation ID and timestamped directory
    correlation_id = str(uuid.uuid4())[:8]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(output_base) / f"{timestamp}_{correlation_id}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # ADDED: Create conversation ID (one per fiedler_send call)
    conversation_id = str(uuid.uuid4())

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
    results = []

    # Cap parallelism based on CPU count and environment variable
    max_workers = min(
        len(models),
        int(os.getenv("FIEDLER_MAX_WORKERS", str(max(2, (os.cpu_count() or 4)))))
    )

    logger.log(f"Sending to {len(models)} model(s) in parallel (max_workers={max_workers})")

    # ADDED: Log request (turn 1)
    log_conversation(
        conversation_id=conversation_id,
        turn_number=1,
        role='user',
        content=prompt,
        files=files,
        correlation_id=correlation_id
    )

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                send_to_model,
                model_id,
                package,
                prompt,
                output_dir,
                correlation_id,
                config,
                logger
            ): model_id
            for model_id in models
        }

        for future in as_completed(futures):
            result = future.result()
            results.append(result)

            # ADDED: Log response (turn 2) - only for successful results
            if result["status"] == "success":
                log_conversation(
                    conversation_id=conversation_id,
                    turn_number=2,
                    role='assistant',
                    content=f"Response from {result['model']} (see {result['output_file']})",
                    model=result['model'],
                    input_tokens=result.get('tokens', {}).get('prompt', None),
                    output_tokens=result.get('tokens', {}).get('completion', None),
                    timing_ms=int(result.get('duration', 0) * 1000),
                    correlation_id=correlation_id
                )

    # Create summary (with optional prompt redaction for security)
    save_prompt = os.getenv("FIEDLER_SAVE_PROMPT", "0") == "1"
    summary = {
        "correlation_id": correlation_id,
        "timestamp": timestamp,
        "prompt": prompt if save_prompt else f"<redacted - {len(prompt)} chars>",
        "prompt_length": len(prompt),
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

---

## Database Schema (Godot)

**Note:** This schema is created by Godot, not Fiedler. Included here for reference only.

**File:** `/mnt/projects/ICCM/godot/schema_additions.sql`

```sql
-- Conversations table for Fiedler conversation logging
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,  -- Groups related messages
    turn_number INTEGER NOT NULL,   -- Sequence within conversation
    role VARCHAR(20) NOT NULL,      -- 'user' or 'assistant'
    content TEXT NOT NULL,
    model VARCHAR(100),              -- Which LLM model used
    input_tokens INTEGER,
    output_tokens INTEGER,
    timing_ms INTEGER,
    metadata JSONB,                  -- Extensible metadata
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(conversation_id, turn_number)
);

CREATE INDEX idx_conversations_conv_id ON conversations(conversation_id);
CREATE INDEX idx_conversations_created ON conversations(created_at);
CREATE INDEX idx_conversations_model ON conversations(model);
CREATE INDEX idx_conversations_metadata ON conversations USING GIN(metadata);

-- Grant permissions
GRANT SELECT ON conversations TO dewey_user;  -- READ-only
GRANT INSERT ON conversations TO godot_log_writer;  -- WRITE-only
```

---

## Implementation Notes

1. **Stateless Design:** Each `fiedler_send` call gets a new `conversation_id` - no state management required

2. **Non-Blocking:** `log_conversation()` silently fails if Godot unavailable - LLM calls always succeed

3. **Error Handling:** All exceptions in `log_conversation()` are caught and ignored

4. **Metadata:** All relevant metadata (model, tokens, timing, files) captured in JSONB field

5. **Existing Integration:** Reuses existing `log_to_godot()` MCP client - no new dependencies

6. **Turn Numbering:** Request = turn 1, Response = turn 2 (simple, predictable)

7. **Content Storage:** For responses, we log a reference to the output file (not full response text) to keep database size manageable

---

## Testing Strategy

**Test 1: Basic Logging**
```bash
# Call fiedler_send
mcp__iccm__fiedler_send "Test conversation logging"

# Verify logs in Godot
docker logs godot-mcp | grep fiedler-conversations

# Should see 2 log entries (request + response)
```

**Test 2: Metadata Capture**
```bash
# Send with files
mcp__iccm__fiedler_send "Test with files" --files /path/to/file.txt

# Query Godot logs
mcp__iccm__dewey_query_logs --component fiedler-conversations --limit 10

# Verify metadata includes:
# - files list
# - model name
# - token counts
# - timing
```

**Test 3: Error Handling (Godot Down)**
```bash
# Stop Godot
docker stop godot-mcp

# Send request (should still work)
mcp__iccm__fiedler_send "Test with Godot down"

# Verify LLM call succeeded (response returned)
# Verify no error raised

# Restart Godot
docker start godot-mcp
```

---

## Questions for Triplet Review

1. **Code Correctness:** Is the implementation architecturally sound and bug-free?

2. **Integration:** Does `log_conversation()` correctly use the existing `log_to_godot()` MCP client?

3. **Error Handling:** Is non-blocking logging implemented correctly (silently fails)?

4. **Data Structure:** Is the log_data structure appropriate for Godot → PostgreSQL storage?

5. **Performance:** Will `asyncio.run()` in a ThreadPoolExecutor context cause issues?

6. **Content Storage:** Should we log full response text or just a reference to the output file?

---

## Approval Criteria

- ✅ Code is architecturally correct (logs via Godot MCP)
- ✅ Non-blocking implementation (LLM calls never fail due to logging)
- ✅ Metadata captured correctly
- ✅ Existing code not broken
- ✅ Ready for Blue/Green deployment
