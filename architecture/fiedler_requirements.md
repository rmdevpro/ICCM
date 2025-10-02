# Fiedler MCP Server - Requirements Document

## Document Control

**Version:** 0.1 - Draft for Triplet Validation
**Date:** 2025-10-02
**Status:** Awaiting Triplet Validation
**Purpose:** Define requirements for Fiedler, the Orchestra conductor prototype MCP server

---

## 1. Project Context

### 1.1 Background

**Problem:** Current triplet verifier (`triplet_verifier.py`) is hardcoded for three specific models (Gemini 2.5 Pro, GPT-5, Grok 4) and lacks extensibility.

**Opportunity:** Evolve into Fiedler, an MCP server that serves as the prototype Orchestra conductor for the ICCM ecosystem.

**Design Philosophy:**
- **Unified Interface:** Tools work the same regardless of which LLM provider is used
- **Configure Once, Use Many:** Set models and output directory once, then send multiple requests
- **Internal Complexity Hidden:** Per-model timeouts, retries, rate limits handled transparently
- **Extensible Foundation:** Easy to add new providers and capabilities without changing tools

**Strategic Value:**
- Validates Orchestra conductor architecture early
- Enables dynamic model selection for architecture work
- Supports performance tracking to optimize triplet composition
- Provides foundation for full LLM Orchestra implementation

### 1.2 Relationship to ICCM Architecture

**From scope.md:**
```
LLM Orchestra:
- M5-based infrastructure managing local model deployment
- Dynamic model loading/rotation
- Cloud API coordination
- Intelligent routing (local vs cloud)
- Request batching, response caching, load balancing
```

**Fiedler Role:** Prototype conductor implementing cloud API coordination and intelligent routing. Does NOT handle local models (M5 deployment) - that remains in full Orchestra.

---

## 2. Functional Requirements

### FR-1: MCP Server Protocol

**FR-1.1:** Implement standard MCP server protocol
**FR-1.2:** Support tool invocation via MCP messages
**FR-1.3:** Return responses in MCP-compliant format
**FR-1.4:** Handle concurrent requests (thread-safe)

**Rationale:** Integration with Claude Code and future Republic ecosystem

### FR-2: Multi-Provider Support

**FR-2.1:** Support Gemini API (Google AI Studio)
**FR-2.2:** Support OpenAI API (GPT-4o, GPT-5)
**FR-2.3:** Support Together.AI API (Llama, DeepSeek, Qwen, etc.)
**FR-2.4:** Support xAI API (Grok models)
**FR-2.5:** Extensible provider registry for adding new APIs

**Rationale:** Enable testing of multiple models for architecture work

### FR-3: Configuration Management

**FR-3.1:** Persist active model list across multiple requests
**FR-3.2:** Persist output directory configuration
**FR-3.3:** Support model aliases (e.g., "gemini" → "gemini-2.5-pro")
**FR-3.4:** Allow n-model execution (not limited to 3)
**FR-3.5:** Return current configuration on demand

**Rationale:** Configure once, use many times - reduces repetition in architecture workflow

### FR-4: Parallel Execution

**FR-4.1:** Execute all model requests in parallel (ThreadPoolExecutor)
**FR-4.2:** Real-time progress logging for each model
**FR-4.3:** Independent timeout per model (default: 600s)
**FR-4.4:** Graceful handling of individual model failures

**Rationale:** Speed - essential for architecture workflow

### FR-5: Document Package Compilation

**FR-5.1:** Accept list of file paths
**FR-5.2:** Concatenate files with clear separators
**FR-5.3:** Support markdown, text, and code files
**FR-5.4:** Return compiled package size and metadata

**Rationale:** Replicates current triplet_verifier.py capability

### FR-6: Prompt Management

**FR-6.1:** Accept verification prompt as parameter
**FR-6.2:** Combine prompt + package for model input
**FR-6.3:** Support prompt templates with variables
**FR-6.4:** Store prompts separately from code

**Rationale:** Reusability and maintainability

### FR-7: Output Management

**FR-7.1:** Save each model's response to separate file
**FR-7.2:** Structured output directory (model name + timestamp)
**FR-7.3:** Progress log with timestamps and model labels
**FR-7.4:** Summary report (success/failure, sizes, durations)

**Rationale:** Traceability and debugging

### FR-8: Performance Tracking

**FR-8.1:** Record duration per model
**FR-8.2:** Record output size per model
**FR-8.3:** Record success/failure status
**FR-8.4:** Optional quality scoring (manual post-verification)

**Rationale:** Supports model swap decisions (tracking per triplet_performance_tracking.md)

---

## 3. Non-Functional Requirements

### NFR-1: Performance

**NFR-1.1:** Parallel execution (3+ models simultaneously)
**NFR-1.2:** <5s overhead beyond slowest model
**NFR-1.3:** Support packages up to 2MB (research papers)
**NFR-1.4:** Handle 16K+ token responses

### NFR-2: Reliability

**NFR-2.1:** Retry failed API calls (exponential backoff)
**NFR-2.2:** Graceful degradation (partial results acceptable)
**NFR-2.3:** Thread-safe logging
**NFR-2.4:** Cleanup temp files on crash

### NFR-3: Extensibility

**NFR-3.1:** Plugin architecture for providers
**NFR-3.2:** Configuration-based model registry
**NFR-3.3:** No hardcoded API keys (environment variables)
**NFR-3.4:** Easy to add new providers without modifying core

### NFR-4: Observability

**NFR-4.1:** Real-time progress logging
**NFR-4.2:** Structured log format (timestamp, model, event)
**NFR-4.3:** Error messages include provider and model
**NFR-4.4:** Summary report with all key metrics

### NFR-5: Security

**NFR-5.1:** API keys from environment variables only
**NFR-5.2:** Never log API keys or responses containing secrets
**NFR-5.3:** Temp files cleaned up after use
**NFR-5.4:** Read-only access to source files

---

## 4. MCP Server Tools

### Tool 1: `fiedler_send`

**Description:** Send prompt and optional files to LLMs in parallel

**Parameters:**
```json
{
  "prompt": "Review this requirements document and provide feedback",
  "files": ["path/to/file1.md", "path/to/file2.md"],  // Optional
  "models": ["deepseek-r1", "qwen-72b"]  // Optional override - uses default models if not provided
}
```

**Returns:**
```json
{
  "status": "success",
  "config_used": {
    "models": ["gemini-2.5-pro", "gpt-5", "llama-3.1-70b"],
    "output_dir": "/mnt/projects/ICCM/architecture/triplet_submissions"
  },
  "results": {
    "gemini-2.5-pro": {
      "success": true,
      "duration": 45.2,
      "output_file": "/path/to/output/Gemini_2.5_Pro_Response.md",
      "output_size": 8192,
      "tokens": {"prompt": 1500, "completion": 2000}
    },
    "gpt-5": { ... },
    "llama-3.1-70b": { ... }
  },
  "summary": {
    "total": 3,
    "successful": 3,
    "failed": 0,
    "total_duration": 47.8
  }
}
```

**Rationale:** Main tool for sending content to LLMs. Uses default models unless overridden. Always uses configured output directory.

---

### Tool 2: `fiedler_set_models`

**Description:** Configure default models for `fiedler_send` requests (can be overridden per-call)

**Parameters:**
```json
{
  "models": ["gemini-2.5-pro", "gpt-5", "llama-3.1-70b"]
}
```

**Returns:**
```json
{
  "status": "configured",
  "models": ["gemini-2.5-pro", "gpt-5", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
  "message": "Default models updated (3 models configured)"
}
```

**Rationale:** Set default models once, use for many requests. Can override with `models` parameter in `fiedler_send` for one-off tests. Aliases are resolved to canonical model IDs.

---

### Tool 3: `fiedler_set_output`

**Description:** Configure output directory for subsequent `fiedler_send` requests

**Parameters:**
```json
{
  "output_dir": "/mnt/projects/ICCM/architecture/triplet_submissions/phase1"
}
```

**Returns:**
```json
{
  "status": "configured",
  "output_dir": "/mnt/projects/ICCM/architecture/triplet_submissions/phase1",
  "message": "Output directory updated"
}
```

**Rationale:** Persist output location across multiple sends.

---

### Tool 4: `fiedler_get_config`

**Description:** Get current Fiedler configuration

**Parameters:** None

**Returns:**
```json
{
  "models": ["gemini-2.5-pro", "gpt-5", "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"],
  "output_dir": "/mnt/projects/ICCM/architecture/triplet_submissions/phase1",
  "default_timeout": 600,
  "total_available_models": 12
}
```

**Rationale:** Inspect current configuration before sending.

---

### Tool 5: `fiedler_list_models`

**Description:** List all available models with their capabilities and limits

**Parameters:** None

**Returns:**
```json
{
  "models": [
    {
      "name": "gemini-2.5-pro",
      "provider": "google",
      "aliases": ["gemini"],
      "max_tokens": 32768,
      "capabilities": ["text"]
    },
    {
      "name": "gpt-5",
      "provider": "openai",
      "aliases": ["gpt5"],
      "max_tokens": 32768,
      "capabilities": ["text"]
    },
    {
      "name": "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
      "provider": "together",
      "aliases": ["llama", "llama-3.1-70b"],
      "max_tokens": 8192,
      "capabilities": ["text"]
    }
  ]
}
```

**Rationale:** Discover available models and their properties for configuration.

---

## 5. Provider Specifications

### 5.1 Google AI Studio (Gemini)

**API:** `https://generativelanguage.googleapis.com/v1/models/{model}:generateContent`
**Auth:** `GEMINI_API_KEY` environment variable
**Models:** `gemini-2.5-pro`, `gemini-2.0-flash-exp`
**Max Tokens:** 32K output
**Implementation:** Use existing `/mnt/projects/gemini-tool/gemini_client.py` wrapper

### 5.2 OpenAI

**API:** `https://api.openai.com/v1/chat/completions`
**Auth:** `OPENAI_API_KEY` environment variable
**Models:** `gpt-5`, `gpt-4o`, `gpt-4o-mini`
**Max Tokens:** 32K output (gpt-5)
**Implementation:** Direct `openai` Python SDK

### 5.3 Together.AI

**API:** `https://api.together.xyz/v1/chat/completions` (OpenAI-compatible)
**Auth:** `TOGETHER_API_KEY` environment variable
**Models (Top 6 for architecture work):**
- `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` - Strong reasoning, general-purpose
- `meta-llama/Llama-3.3-70B-Instruct-Turbo` - Latest Llama, improved performance
- `deepseek-ai/DeepSeek-R1` - Excellent reasoning, coding, analysis
- `Qwen/Qwen2.5-72B-Instruct-Turbo` - Strong multilingual, competitive with GPT-4
- `mistralai/Mistral-Large-2411` - European flagship model, strong reasoning
- `nvidia/Llama-3.1-Nemotron-70B-Instruct-HF` - NVIDIA-tuned Llama, excellent for technical tasks

**Max Tokens:** 8K output (most models)
**Implementation:** `openai` SDK with custom base URL

**Rationale for Selection:**
- **Llama 3.1 & 3.3:** Open-weight baseline, widely used
- **DeepSeek-R1:** Best open reasoning model, competitive with o1
- **Qwen 2.5:** Strong alternative perspective, multilingual
- **Mistral Large:** European model, different training approach
- **Nemotron:** NVIDIA fine-tuned, excellent for technical/code review

### 5.4 xAI (Grok)

**API:** `https://api.x.ai/v1/chat/completions` (OpenAI-compatible)
**Auth:** `XAI_API_KEY` environment variable
**Models:** `grok-4-0709`, `grok-2-1212`
**Max Tokens:** 16K output
**Implementation:** Use existing `/mnt/projects/ICCM/tools/grok_client.py` wrapper

---

## 6. Architecture Design

### 6.1 Component Structure

```
fiedler/
├── server.py           # MCP server main entry point
├── providers/
│   ├── __init__.py     # Provider registry
│   ├── base.py         # BaseProvider abstract class
│   ├── gemini.py       # GeminiProvider
│   ├── openai.py       # OpenAIProvider
│   ├── together.py     # TogetherProvider
│   └── xai.py          # XAIProvider
├── tools/
│   ├── verify.py       # fiedler_verify implementation
│   ├── list_models.py  # fiedler_list_models
│   └── test.py         # fiedler_test_provider
├── utils/
│   ├── logger.py       # ProgressLogger
│   ├── package.py      # Document compilation
│   └── config.py       # Model registry and settings
├── config/
│   └── models.yaml     # Model definitions
└── tests/
    └── test_providers.py
```

### 6.2 Provider Interface

```python
class BaseProvider(ABC):
    @abstractmethod
    def verify(self, package: str, prompt: str, model: str,
               timeout: int, logger: ProgressLogger) -> str:
        """Send package+prompt to model, return response"""
        pass

    @abstractmethod
    def list_models(self) -> List[str]:
        """Return list of available models for this provider"""
        pass

    @abstractmethod
    def test_connection(self, model: str) -> Tuple[bool, float]:
        """Test connectivity, return (success, latency)"""
        pass
```

### 6.3 Configuration Format

**File:** `fiedler/config/models.yaml`

```yaml
providers:
  google:
    api_key_env: GEMINI_API_KEY
    models:
      gemini-2.5-pro:
        aliases: [gemini]
        max_tokens: 32768
        timeout: 600  # Per-model timeout (user never configures)
        retry_attempts: 3
        capabilities: [text]  # For future multimodal routing

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

# Default configuration (overridden by fiedler_set_models/fiedler_set_output)
defaults:
  models: [gemini-2.5-pro, gpt-5, grok-4-0709]
  output_dir: ./fiedler_output
```

**Note:** Timeouts, retries, and rate limits are configured per-model but hidden from users. The MCP tools present a clean interface while Fiedler handles complexity internally.

---

## 7. Out of Scope

**Explicitly NOT included in Fiedler v1:**

❌ **Local model deployment** - Handled by full LLM Orchestra on M5
❌ **Model rotation scheduling** - Full Orchestra feature
❌ **Response caching** - Add in v2 if needed
❌ **Request batching** - Single request focus for architecture work
❌ **Cost tracking** - Add later if budget concerns arise (token tracking IS in scope)
❌ **Load balancing** - Single-instance MCP server
❌ **CET integration** - Fiedler works with base LLMs only
❌ **Multimodal support** - Text-only for v1 (images, audio, vision deferred)
❌ **Smart routing** - No capability-based auto-selection (user explicitly picks models)
❌ **User-configurable timeouts** - Handled internally per-model

---

## 8. Success Criteria

**Fiedler v1 is successful if:**

✅ Can send document packages to any combination of 3+ models in parallel
✅ Supports Gemini, OpenAI, Together.AI, xAI providers
✅ Generates structured output (separate files per model + summary)
✅ Integrates with Claude Code as MCP server
✅ Enables testing Llama 3.1 70B and DeepSeek-R1 via Together.AI
✅ Replaces current triplet_verifier.py with zero functionality loss
✅ Provider architecture is extensible (easy to add new APIs)
✅ Performance tracking data supports model swap decisions

---

## 9. Migration Plan

### Phase 1: Build Fiedler v1
- Implement MCP server with provider architecture
- Test with current triplet (Gemini, GPT-5, Grok)
- Validate output matches triplet_verifier.py

### Phase 2: Expand Model Support
- Add Llama 3.1 70B via Together.AI
- Add DeepSeek-R1 via Together.AI
- Test alternative triplets

### Phase 3: Replace triplet_verifier.py
- Update architecture workflow to use Fiedler
- Archive triplet_verifier.py
- Document Fiedler usage in planning_log.md

### Phase 4: Foundation for Orchestra
- Extract learnings for full Orchestra design
- Document conductor pattern
- Inform Phase 2 architecture (when building full Orchestra)

---

## 10. Design Decisions (Post-Triplet Review)

Based on triplet feedback and user clarification, the following decisions were made:

1. **Tool Design:** Five MCP tools (`fiedler_send`, `fiedler_set_models`, `fiedler_set_output`, `fiedler_get_config`, `fiedler_list_models`) provide clean interface
   - **Decision:** Configure defaults (models + output), optional override per-call
   - **Rationale:** Reduces repetition while maintaining flexibility for one-off tests

2. **Provider Priority:** Together.AI is correct priority
   - **Decision:** Enable Llama 3.1 70B and DeepSeek-R1 for testing
   - **Rationale:** Critical to test open-weight models against proprietary

3. **Configuration Format:** YAML for model registry
   - **Decision:** models.yaml with per-model timeouts/retries
   - **Rationale:** Standard, readable, easy to extend

4. **Error Handling:** Partial success as distinct status
   - **Decision:** `status` = `success` | `partial_success` | `failure`
   - **Rationale:** User needs to know if some models failed

5. **Timeout Strategy:** Per-model fixed timeouts (internal)
   - **Decision:** Configure in models.yaml, not exposed to users
   - **Rationale:** Different models have different performance profiles

6. **Output Format:** Separate markdown files per model
   - **Decision:** Keep current approach
   - **Rationale:** Easy to diff, analyze, and integrate with existing workflow

7. **Performance Tracking:** Manual quality scoring
   - **Decision:** Automated metrics (duration, tokens), manual quality scores
   - **Rationale:** Quality assessment is subjective and context-dependent

8. **Missing Provider:** Anthropic (Claude)
   - **Decision:** Design interface for Claude, implement post-v1
   - **Rationale:** High value but not MVP-critical

9. **Interface Design:** Unified tools, provider-agnostic
   - **Decision:** Tools work the same regardless of which LLM is used
   - **Rationale:** Simplifies user experience, enables future smart routing

---

## 11. Validation Questions for Triplets

**Please review this requirements document and provide feedback on:**

1. **Completeness:** Are any critical requirements missing?
2. **Clarity:** Are requirements clear and unambiguous?
3. **Feasibility:** Are all requirements technically feasible?
4. **Architecture:** Is the provider-based architecture appropriate?
5. **Scope:** Is anything in scope that should be deferred?
6. **Priority:** Should any requirements be elevated or demoted in priority?
7. **Risks:** What technical risks do you see?
8. **Improvements:** How could this requirements document be improved?

**Specific concerns:**
- Is MCP server the right architecture for this use case?
- Should Fiedler handle local models (M5) or leave that to full Orchestra?
- Are we missing any providers that would be valuable for architecture work?

---

## 12. References

**ICCM Papers:**
- Paper 10: LLM Orchestra v4 (defines Orchestra architecture)
- Paper 01: ICCM Primary Paper v4.1 (context on progressive training)

**Existing Code:**
- `/mnt/projects/ICCM/tools/triplet_verifier.py` (current implementation)
- `/mnt/projects/gemini-tool/gemini_client.py` (Gemini wrapper)
- `/mnt/projects/ICCM/tools/grok_client.py` (Grok wrapper)

**Architecture Artifacts:**
- `/mnt/projects/ICCM/architecture/scope.md` (ICCM system scope)
- `/mnt/projects/ICCM/architecture/triplet_performance_tracking.md` (tracking methodology)
