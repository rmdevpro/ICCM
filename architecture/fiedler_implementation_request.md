# Fiedler MCP Server - Implementation Request

## Task

Implement Fiedler, an MCP server that serves as the Orchestra conductor prototype. You are receiving the complete requirements document and the existing `triplet_verifier.py` implementation to evolve.

## Your Deliverables

**Primary:** Complete, working Python implementation of Fiedler MCP server

**Structure:**
```
fiedler/
├── server.py           # MCP server entry point
├── providers/
│   ├── __init__.py
│   ├── base.py        # BaseProvider abstract class
│   ├── gemini.py
│   ├── openai.py
│   ├── together.py
│   └── xai.py
├── tools/
│   ├── send.py        # fiedler_send
│   ├── config.py      # set_models, set_output, get_config
│   └── models.py      # list_models
├── utils/
│   ├── logger.py
│   ├── package.py
│   └── state.py       # Config persistence
├── config/
│   └── models.yaml
└── pyproject.toml     # Dependencies
```

## Requirements Summary

**5 MCP Tools:**
1. `fiedler_send(prompt, files)` - Send to configured LLMs
2. `fiedler_set_models(models)` - Configure active models
3. `fiedler_set_output(output_dir)` - Configure output directory
4. `fiedler_get_config()` - Get current configuration
5. `fiedler_list_models()` - List available models

**Design Principles:**
- Unified interface (works same regardless of provider)
- Configure once, use many times
- Hide internal complexity (timeouts, retries per-model)
- Extensible provider architecture

**Providers (9 models):**
- Google: gemini-2.5-pro
- OpenAI: gpt-5
- Together.AI: llama-3.1-70b, llama-3.3-70b, deepseek-r1, qwen-72b, mistral-large, nemotron-70b
- xAI: grok-4

**Key Features:**
- Parallel execution (ThreadPoolExecutor)
- Per-model timeouts/retries (internal)
- Token usage tracking
- Error handling with partial success
- Alias resolution (e.g., "gemini" → "gemini-2.5-pro")

## Implementation Guidelines

1. **Reuse existing code:** Evolve `/mnt/projects/ICCM/tools/triplet_verifier.py` - don't start from scratch
2. **MCP protocol:** Use stdio transport, JSON-RPC 2.0
3. **Provider abstraction:** BaseProvider with `send()` method
4. **State management:** Persist models/output_dir in memory (YAML config for model specs)
5. **Error handling:** Graceful per-model failures, structured errors
6. **Logging:** Thread-safe progress logging like current triplet_verifier.py

## Specific Requirements

**From Triplet Feedback (synthesized):**
- Token usage tracking (prompt/completion tokens)
- Streaming to disk for large outputs
- Preflight token budget checking
- Rate limiting and retry logic
- Input validation (file paths, model IDs)
- Correlation IDs for traceability

**Critical from GPT-5:**
- MCP protocol specification (stdio, JSON-RPC)
- Structured error schema
- Generation parameter normalization

**Critical from Gemini:**
- Provider-based plugin architecture
- Configuration-driven model registry
- Clean separation of concerns

## Code Style

- Type hints throughout
- Docstrings for all public methods
- Comprehensive error messages
- Async/await if using asyncio (or ThreadPoolExecutor if threaded)
- Follow existing triplet_verifier.py logging pattern

## Testing

Include basic tests:
- Provider connectivity tests
- Tool parameter validation
- Parallel execution tests
- Error handling tests

## Constraints

- Python 3.10+
- Use existing `/mnt/projects/gemini-tool/gemini_client.py` wrapper
- Use existing `/mnt/projects/ICCM/tools/grok_client.py` wrapper
- OpenAI SDK for OpenAI and Together.AI
- No additional heavy dependencies (keep it lightweight)

## Evaluation Criteria

Your implementation will be evaluated on:
1. **Correctness:** Does it work? Does it match requirements?
2. **Code Quality:** Clean, maintainable, well-structured?
3. **Extensibility:** Easy to add new providers?
4. **Performance:** Efficient parallel execution?
5. **Error Handling:** Graceful degradation, clear errors?
6. **Documentation:** Clear README, docstrings, comments?

## Deliverable Format

Provide complete, working code for all files. Include:
- Full source code
- `pyproject.toml` or `requirements.txt`
- `README.md` with setup and usage instructions
- Example usage showing all 5 tools

## Round 2 Note

After all three implementations are submitted, you will receive:
- Your implementation
- The other two implementations
- Feedback on all three

You will then submit a **final best version** incorporating the best ideas from all three approaches.
