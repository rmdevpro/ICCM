# Fiedler Requirements - Triplet Synthesis

## Triplet Performance Summary

**Response Sizes:**
- Gemini 2.5 Pro: 11,535 bytes (53.0s)
- GPT-5: 12,247 bytes (103.6s) - Most comprehensive
- Grok 4: 11,887 bytes (85.8s) - **MUCH improved from Phase 0!**

**Quality Assessment:**
- **Gemini**: Excellent structured feedback, clear recommendations
- **GPT-5**: Extremely detailed technical analysis, most comprehensive coverage
- **Grok 4**: Strong engagement, good technical depth, practical suggestions

**Overall Consensus:** Requirements are solid foundation but need critical additions around MCP protocol details, token management, rate limiting, and error handling.

---

## Critical Requirements to Add (All Three Agreed)

### 1. MCP Protocol Specification (GPT-5 Primary, Gemini/Grok Support)

**Missing:**
- MCP version and transport (stdio vs WebSocket)
- JSON-RPC version
- Tool registration lifecycle
- Capability advertisement
- Error codes and schemas
- Progress notifications/cancellation semantics

**Action:** Add new section "FR-1A: MCP Protocol Details" with:
- Protocol version (specify which MCP spec)
- Transport: stdio for Claude Code integration
- Tool schemas with JSON validation
- Standard error codes
- Cancellation support via MCP protocol

### 2. Token Budget Management (GPT-5 Critical, Grok Support)

**Missing:**
- Preflight token counting for package + prompt
- Per-provider token limits (input and output)
- Truncation strategy when over limit
- Cross-provider token normalization

**Action:** Add "FR-9: Token Budget Management"
- Preflight token estimation per provider
- Fail-fast if package exceeds context window
- Deterministic truncation strategy
- Report token usage in summary (even if cost tracking deferred)

### 3. Rate Limiting and Retry Policy (GPT-5 Critical, Grok Critical)

**Missing:**
- Per-provider rate limits (QPM, concurrency caps)
- Retry with exponential backoff and jitter
- Circuit breaker for failing providers

**Action:** Enhance "NFR-2.1 Retries" to full rate limiting section
- Centralized rate limiter per provider
- Exponential backoff with jitter
- Circuit breaker pattern
- Per-provider concurrency limits in config

### 4. Error Handling and Partial Success (All Three)

**Missing:**
- Enumerated error classes (auth, rate limit, timeout, server, client, safety)
- Structured error schema
- Partial success semantics (2/3 succeed = what status?)

**Action:** Add "FR-10: Error Handling"
- Error taxonomy (auth, rate, timeout, server, client, policy/safety)
- Tool return schema includes per-model errors
- Status field: `success`, `partial_success`, `failure`
- Retryable flag per error type

### 5. Input Validation (Gemini Primary, GPT-5/Grok Support)

**Missing:**
- Validate file paths exist and are readable
- Validate model IDs against registry
- Parameter validation (non-empty prompts, valid output dirs)
- Path traversal protection (security)

**Action:** Add "NFR-6: Input Validation"
- Validate all tool parameters before execution
- Workspace root restriction (prevent path traversal)
- Model ID validation against registry
- Return structured errors for invalid inputs

### 6. Streaming and Large Output Handling (GPT-5 Critical)

**Missing:**
- Strategy for 16K+ token outputs
- Streaming to disk to avoid memory spikes
- Handling partial/truncated outputs

**Action:** Add "NFR-1.5: Streaming"
- Stream responses to disk incrementally
- Avoid loading full responses in memory
- Detect and report truncation
- Chunked writes with backpressure

---

## High-Priority Enhancements (Strong Consensus)

### 7. Token Usage Tracking (GPT-5 Elevate, Grok Elevate)

**Rationale:** Even if cost tracking is deferred, token usage is critical for:
- Performance comparisons between models
- Routing decisions (cheaper models for shorter responses)
- Validation that truncation didn't occur

**Action:** Add to FR-8 (Performance Tracking)
- Record prompt_tokens and completion_tokens per model
- Extract from provider API responses
- Include in summary JSON

### 8. Generation Parameter Normalization (GPT-5 Critical)

**Missing:**
- Common parameters (temperature, top_p, max_tokens, seed)
- Per-provider mapping
- Configuration in YAML + runtime overrides

**Action:** Add "FR-11: Generation Parameters"
- Support: temperature, top_p, max_tokens, seed, stop sequences
- Per-model defaults in models.yaml
- Per-provider parameter mapping
- Runtime overrides via tool parameters

### 9. Observability Enhancements (GPT-5, Grok)

**Missing:**
- Correlation/run IDs for traceability
- Provider request IDs in logs
- HTTP status codes
- Structured log schema

**Action:** Enhance "NFR-4: Observability"
- Generate correlation ID per verification run
- Include in all logs, filenames, and summary
- Capture provider request IDs where available
- Define structured log format (timestamp, correlation_id, model, event, message)

---

## Scope Refinements

### Defer to v2 (Consensus)

**Gemini Recommendations:**
- Keep scope as-is - nothing should be deferred

**Grok Recommendations:**
- Defer FR-8.4 (quality scoring automation) - keep manual for v1
- Defer Tool 3 (test_provider) if time-constrained
- Limit n-model verification to 3-5 models (not unlimited)

**GPT-5 Recommendations:**
- Defer prompt templating engine beyond simple substitution
- Defer Together.AI model autodiscovery (use config)
- Defer OpenTelemetry integration (use structured logs + correlation IDs)
- Defer advanced safety policy configuration

**Consensus:** Keep current scope, but:
- Quality scoring remains manual (FR-8.4 is optional/future)
- Limit n-model to reasonable max (e.g., 10 concurrent)
- Simple prompt variable substitution only (no Jinja)

---

## Architecture Validation

### MCP Server Design: ✅ Unanimous Approval

**Gemini:** "Absolutely. The proposed provider-based MCP server design is not only appropriate but ideal."

**GPT-5:** "Yes. Given the context of integration with Claude Code and future Republic ecosystem, using MCP protocol is a strategic choice."

**Grok:** "Yes, it's appropriate as a prototype for the Orchestra conductor."

**Refinements Suggested:**
- **GPT-5:** Add Router layer (maps model IDs to providers), RequestExecutor (supports threads/asyncio), ParameterNormalizer
- **Grok:** Consider dependency injection for providers (easier testing/mocking)

### Local Models (M5): ✅ Leave to Full Orchestra (Unanimous)

**Gemini:** "Leave it to the full Orchestra. Mixing cloud API calls with local model inference introduces significant architectural complexity."

**GPT-5:** "No for v1. Keeping local model orchestration in the full Orchestra reduces complexity."

**Grok:** "Leave it to full Orchestra... This keeps Fiedler lightweight for prototyping cloud APIs."

---

## Missing Providers

### High Priority: Anthropic (Claude)

**All Three Agreed:**
- Gemini: "Anthropic (Claude models) is the most significant missing provider."
- GPT-5: "Anthropic (Claude) and Cohere are high-value additions."
- Grok: "Anthropic (Claude) for comparison with high-context models."

**Action:** Add Anthropic provider to provider list, but defer implementation to post-v1 if schedule tight. Design provider interface with Claude API in mind.

### Medium Priority:
- Cohere (GPT-5, Grok)
- AWS Bedrock / Azure OpenAI (Grok - enterprise options)
- Perplexity, Mistral (GPT-5 - optional)

---

## Answers to Open Questions (Section 10)

### 1. Provider Priority: Together.AI ✅ Correct
**Consensus:** Yes, Together.AI is correct priority for Llama/DeepSeek testing.

### 2. Tool Design: Three Tools ✅ Sufficient
**Gemini:** "The three tools are sufficient and well-designed."
**GPT-5:** "Sufficient."
**Grok:** "Strong... could defer Tool 3 if time-constrained but not critical."

### 3. Configuration: YAML ✅ Correct
**Gemini:** "YAML is the right format."
**GPT-5:** "YAML is the right format."
**Grok:** Implied approval.

### 4. Error Handling: Partial Success → New Status
**Consensus:** Add `partial_success` status
- **Gemini:** `"success"`, `"partial_success"`, `"failure"`
- **GPT-5:** `status=partial` with per-model error details
- **Grok:** Implied agreement via partial failure discussion

### 5. Timeout Strategy: Fixed ✅ for v1
**Gemini:** "Fixed timeout per model (600s) is simple and sufficient."
**GPT-5:** "For v1, a fixed timeout per model is simple and sufficient."
**Grok:** Implied agreement.

### 6. Output Format: Markdown Files ✅ Correct
**Gemini:** "Saving to separate markdown files is good, simple... No change needed for v1."
**GPT-5:** Implied approval.
**Grok:** Implied approval.

### 7. Performance Tracking: Manual Quality Scoring ✅ for v1
**Gemini:** "Keeping quality scoring manual is the correct approach."
**GPT-5:** "For v1, keeping quality scoring manual is the correct approach."
**Grok:** "Defer to v2 if needed."

### 8. Extensibility: Plan for Anthropic
**See "Missing Providers" above.**

---

## Risks Identified

### High-Priority Risks (All Three)

1. **Rate Limiting / API Quotas**
   - Gemini: Circuit breaker pattern
   - GPT-5: Centralized rate limiting, backoff, circuit breaker
   - Grok: Per-provider rate limiters via semaphores

2. **Secret Management**
   - Gemini: Use `.env` + `python-dotenv`, pre-commit hooks for secrets
   - GPT-5: Secrets redaction in logs, configurable verbosity
   - Grok: Vulnerability in shared environments

3. **Inconsistent Provider Behavior**
   - Gemini: Provider abstraction must normalize differences
   - GPT-5: Wrappers might hide breaking diffs - lock tested SDK versions
   - Grok: Non-OpenAI-compatible APIs require more boilerplate

4. **Token/Context Mismatches**
   - GPT-5: Large packages causing failures, silent truncation, inconsistent tokenization
   - Grok: 2MB packages + prompts could exceed limits

5. **File System Hazards**
   - GPT-5: Path traversal if parameters not validated
   - Grok: Path traversal, filename collisions, permission errors

### Medium-Priority Risks

6. **Model Availability Drift**
   - GPT-5: "gpt-5" may not be GA - use aliases and config to pivot

7. **Safety/Policy Blocks**
   - Gemini: Gemini/OpenAI safety filters hard to handle
   - GPT-5: Must classify and surface clearly

8. **Streaming and Memory**
   - GPT-5: Without streaming, long outputs exhaust memory
   - Grok: 16K+ outputs need chunked writes

9. **MCP Client Expectations**
   - GPT-5: Claude Code expects specific MCP behaviors - test early

---

## Document Improvements (All Three)

### High-Priority

1. **Add MCP Protocol Specification Section** (GPT-5)
   - Version, transport, handshake, capabilities, error codes, cancellation

2. **Add JSON Schemas** (GPT-5, Grok)
   - Tool inputs/outputs
   - Error payloads
   - Summary format
   - Include: tokens, truncation_flag, request_id, status_code, error_code

3. **Add Error Handling Section** (Gemini, GPT-5)
   - Error taxonomy
   - Structured error responses
   - Partial success semantics

4. **Add Input Validation Section** (Gemini, GPT-5)
   - Parameter validation rules
   - Workspace root restriction
   - Security requirements

### Medium-Priority

5. **Add Acceptance Criteria and Test Plan** (GPT-5, Grok)
   - End-to-end scenarios
   - Failure scenarios
   - Performance tests

6. **Add Architecture Diagram** (Grok)
   - UML for providers/tools flow
   - Visualize Section 6

7. **Add Example Logs** (Gemini)
   - Sample structured log output
   - Clarify NFR-4.2

8. **Add Traceability Matrix** (Grok)
   - Map requirements to success criteria
   - Link to open questions

---

## Recommended Requirements Updates

### New Functional Requirements

**FR-9: Token Budget Management**
- FR-9.1: Preflight token estimation for package + prompt per provider
- FR-9.2: Per-provider input/output token limits in configuration
- FR-9.3: Fail-fast if package exceeds context window
- FR-9.4: Deterministic truncation strategy (if truncation supported)
- FR-9.5: Report token usage (prompt, completion) in summary

**FR-10: Error Handling**
- FR-10.1: Enumerate error classes (auth, rate_limit, timeout, server_error, client_error, safety_block)
- FR-10.2: Structured error schema per model
- FR-10.3: Status field: `success`, `partial_success`, `failure`
- FR-10.4: Retryable flag per error type
- FR-10.5: Provider request ID capture

**FR-11: Generation Parameters**
- FR-11.1: Support common parameters (temperature, top_p, max_tokens, seed)
- FR-11.2: Per-model defaults in models.yaml
- FR-11.3: Per-provider parameter mapping
- FR-11.4: Runtime overrides via tool parameters

**FR-1A: MCP Protocol Details**
- FR-1A.1: Implement MCP stdio transport
- FR-1A.2: JSON-RPC 2.0 message format
- FR-1A.3: Tool registration with JSON schemas
- FR-1A.4: Standard MCP error codes
- FR-1A.5: Cancellation support via MCP protocol

### Enhanced Non-Functional Requirements

**NFR-1.5: Streaming**
- Stream large outputs to disk incrementally
- Avoid loading full responses in memory
- Detect and report truncation
- Chunked writes with backpressure

**NFR-2 Enhanced: Reliability**
- NFR-2.1: Retry with exponential backoff and jitter
- NFR-2.2: Per-provider rate limiting (QPM, concurrency caps)
- NFR-2.3: Circuit breaker for failing providers
- NFR-2.4: Graceful degradation (partial results acceptable)
- NFR-2.5: Thread-safe logging
- NFR-2.6: Cleanup temp files on crash

**NFR-4 Enhanced: Observability**
- NFR-4.1: Real-time progress logging
- NFR-4.2: Structured log format (timestamp, correlation_id, model, event, message)
- NFR-4.3: Correlation ID per verification run
- NFR-4.4: Capture provider request IDs
- NFR-4.5: Include HTTP status codes in logs

**NFR-5 Enhanced: Security**
- NFR-5.1: API keys from environment variables only
- NFR-5.2: Never log API keys or sensitive data
- NFR-5.3: Workspace root restriction (prevent path traversal)
- NFR-5.4: Validate and sanitize file paths
- NFR-5.5: Read-only access to source files
- NFR-5.6: Temp files cleaned up after use
- NFR-5.7: Optional secrets redaction in logs

**NFR-6: Input Validation** (NEW)
- NFR-6.1: Validate all tool parameters before execution
- NFR-6.2: Return structured errors for invalid inputs
- NFR-6.3: Model ID validation against registry
- NFR-6.4: File path validation (existence, readability)
- NFR-6.5: Non-empty prompts and output directories

---

## Priority Adjustments

### Elevate to Critical

1. **Token usage tracking** (FR-9.5) - Essential for performance analysis
2. **Preflight token checks** (FR-9.1) - Prevents silent failures
3. **Error handling** (FR-10) - Core reliability requirement
4. **Rate limiting** (NFR-2.2) - Prevents API quota exhaustion
5. **Input validation** (NFR-6) - Security and robustness
6. **Security** (NFR-5.3, 5.4) - Path traversal protection

### Demote or Mark Optional

1. **FR-8.4 (Quality scoring)** - Keep manual, mark as "future enhancement"
2. **Tool 3 (test_provider)** - Useful but not MVP-critical
3. **FR-3.3 (n-model verification)** - Limit to 10 concurrent max

---

## Updated Success Criteria

**Add to Section 8:**

✅ Implements MCP protocol with stdio transport and JSON-RPC 2.0
✅ Preflight token budget checking prevents context window overflows
✅ Rate limiting prevents API quota exhaustion
✅ Structured error handling with partial success semantics
✅ Token usage tracking per model (prompt/completion tokens)
✅ Input validation prevents path traversal and invalid parameters
✅ Streaming to disk for large outputs (16K+ tokens)
✅ Correlation IDs enable full request traceability

---

## Implementation Roadmap (Revised)

### Phase 1: Core MCP Server (Week 1)
- FR-1: MCP server with stdio transport
- FR-1A: Protocol details (JSON-RPC, tool schemas)
- FR-2.1, 2.2: Gemini + OpenAI providers
- FR-4: Parallel execution (ThreadPoolExecutor)
- FR-5: Document compilation
- NFR-6: Input validation

### Phase 2: Provider Expansion (Week 1-2)
- FR-2.3: Together.AI provider (Llama, DeepSeek)
- FR-2.4: xAI provider (Grok)
- FR-11: Generation parameter normalization
- FR-9: Token budget management (preflight checks)

### Phase 3: Reliability (Week 2)
- NFR-2: Rate limiting, retries, circuit breakers
- FR-10: Error handling and partial success
- NFR-1.5: Streaming to disk
- NFR-4: Observability (correlation IDs, structured logs)

### Phase 4: Output and Tools (Week 2-3)
- FR-7: Output management
- FR-8: Performance tracking (duration, size, tokens)
- Tool 2: fiedler_list_models
- Tool 3: fiedler_test_provider (optional)

### Phase 5: Testing and Migration (Week 3)
- Integration tests per provider
- End-to-end tests (3+ models parallel)
- Comparison tests (vs triplet_verifier.py)
- Documentation and migration guide

---

## Triplet Performance Scores (Preliminary)

### Gemini 2.5 Pro: 24/25
- Accuracy: 5/5 - Perfect paper alignment
- Completeness: 5/5 - Identified all critical gaps
- Insight Quality: 5/5 - Strong recommendations
- Practical Feasibility: 5/5 - Realistic assessment
- Synthesis Utility: 4/5 - Well-structured, minor formatting differences

### GPT-5: 25/25 ⭐
- Accuracy: 5/5 - Extremely detailed technical accuracy
- Completeness: 5/5 - Most comprehensive coverage (MCP details, token budget, streaming)
- Insight Quality: 5/5 - Exceptional depth on protocol and observability
- Practical Feasibility: 5/5 - Realistic with detailed mitigations
- Synthesis Utility: 5/5 - Perfect structure, easy to integrate

### Grok 4: 21/25 (MAJOR IMPROVEMENT!)
- Accuracy: 4/5 - Good technical accuracy
- Completeness: 4/5 - Covered most critical items
- Insight Quality: 4/5 - Practical suggestions (dependency injection, traceability matrix)
- Practical Feasibility: 5/5 - Realistic with good risk analysis
- Synthesis Utility: 4/5 - Well-structured, good detail

**Grok 4 Analysis:** Massive improvement from Phase 0 (11/25 → 21/25). Much better engagement, detailed feedback, practical recommendations. Still trails Gemini/GPT-5 but now contributing real value.

---

## Next Steps

1. Update fiedler_requirements.md with new FR-9, FR-10, FR-11, FR-1A
2. Enhance NFR-1.5, NFR-2, NFR-4, NFR-5, add NFR-6
3. Add JSON schemas section
4. Add MCP protocol specification section
5. Add test plan and acceptance criteria
6. User approval
7. Begin Fiedler v1 implementation
