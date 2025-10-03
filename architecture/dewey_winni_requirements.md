# Dewey + Winni Requirements Document
## MVP Librarian (Dewey) and Data Lake (Winni) for ICCM

**Version:** 1.0
**Date:** 2025-10-02
**Status:** Requirements - Awaiting Triplet Review

---

## 1. Executive Summary

This document specifies requirements for **Dewey** (MCP Librarian Server) and **Winni** (Data Lake), the MVP implementation of ICCM's Conversation Storage and Retrieval Infrastructure (Paper 12). Dewey is an MCP server that manages Winni, a centralized data lake for conversation histories, startup contexts, and LLM orchestration results.

**Key Goals:**
- Provide persistent conversation storage across Claude Code sessions
- Enable startup context management (write/clear/load workflow)
- Support basic search through conversation history
- Replace Fiedler's file-based output with centralized storage
- Establish foundation for future semantic search and tiered storage

**Out of Scope (Future Phases):**
- Vector embeddings and semantic search (Phase 3)
- Tiered storage (fast/slow) and archival policies (Phase 4)
- Multi-phase training data organization (Phase 5)
- PostgreSQL migration (optional Phase 2)

---

## 2. System Overview

### 2.1 Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                      Claude Code                            │
│                  (MCP Client / User)                        │
└─────────────────────┬───────────────────────────────────────┘
                      │ MCP Protocol (stdio)
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                      Dewey                                  │
│                  (MCP Server)                               │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  MCP Tools:                                          │  │
│  │  - dewey_store_message                               │  │
│  │  - dewey_get_conversation                            │  │
│  │  - dewey_search                                      │  │
│  │  - dewey_get_startup_context                         │  │
│  │  - dewey_set_startup_context                         │  │
│  │  - dewey_store_fiedler_result                        │  │
│  └──────────────────────────────────────────────────────┘  │
│                       │                                     │
│                       │ Storage Interface                   │
│                       ▼                                     │
│  ┌──────────────────────────────────────────────────────┐  │
│  │              Winni (Data Lake)                       │  │
│  │           SQLite Database                            │  │
│  │  - conversations                                     │  │
│  │  - messages                                          │  │
│  │  - startup_contexts                                  │  │
│  │  - fiedler_results                                   │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Integration Points

1. **Claude Code** → Dewey: Store conversation turns, retrieve startup contexts, search history
2. **Fiedler** → Dewey: Store LLM orchestration results (future Phase 2)
3. **Winni**: Centralized storage accessible to all ICCM components

---

## 3. Requirements

### 3.1 Functional Requirements

#### FR-1: Conversation Storage
- **FR-1.1**: Store conversation messages with role (user/assistant/system)
- **FR-1.2**: Store conversation metadata (session_id, timestamps, custom fields)
- **FR-1.3**: Support incremental storage (add messages to ongoing conversation)
- **FR-1.4**: Generate unique conversation IDs (UUID format)
- **FR-1.5**: Store turn numbers to maintain message order

#### FR-2: Conversation Retrieval
- **FR-2.1**: Retrieve complete conversation by ID
- **FR-2.2**: Retrieve conversations by session ID
- **FR-2.3**: Retrieve messages ordered by turn number
- **FR-2.4**: Retrieve conversation metadata

#### FR-3: Startup Context Management
- **FR-3.1**: Store named startup contexts (text content)
- **FR-3.2**: Mark one startup context as "active"
- **FR-3.3**: Retrieve active startup context
- **FR-3.4**: List all available startup contexts
- **FR-3.5**: Update existing startup context
- **FR-3.6**: Support workflow: Write context → Clear chat → Load context

#### FR-4: Search Functionality
- **FR-4.1**: Full-text search across conversation content
- **FR-4.2**: Filter search by date range
- **FR-4.3**: Filter search by session ID
- **FR-4.4**: Return search results with context (surrounding messages)
- **FR-4.5**: Rank search results by relevance

#### FR-5: Fiedler Integration (Phase 2)
- **FR-5.1**: Store Fiedler orchestration results (correlation_id, model, prompt, output)
- **FR-5.2**: Store token counts and duration
- **FR-5.3**: Support prompt redaction (NULL if privacy enabled)
- **FR-5.4**: Retrieve results by correlation ID
- **FR-5.5**: Query results by model, date range, or prompt content

### 3.2 Non-Functional Requirements

#### NFR-1: Performance
- **NFR-1.1**: Store message operation completes in <100ms (p95)
- **NFR-1.2**: Retrieve conversation completes in <200ms (p95)
- **NFR-1.3**: Search operation completes in <500ms for 10K conversations (p95)
- **NFR-1.4**: Support up to 100K conversations without performance degradation
- **NFR-1.5**: Database file size managed via periodic vacuum operations

#### NFR-2: Reliability
- **NFR-2.1**: SQLite database with WAL mode for crash resistance
- **NFR-2.2**: Foreign key constraints enforced
- **NFR-2.3**: Transactions used for multi-operation consistency
- **NFR-2.4**: Automatic database backup before schema migrations
- **NFR-2.5**: Graceful handling of database lock contention

#### NFR-3: Maintainability
- **NFR-3.1**: Schema versioning for future migrations
- **NFR-3.2**: Clear error messages for debugging
- **NFR-3.3**: Logging for all storage operations
- **NFR-3.4**: Configuration file for customization (database path, retention policies)
- **NFR-3.5**: Comprehensive test coverage (>80%)

#### NFR-4: Security
- **NFR-4.1**: File permissions restrict database access to Dewey process
- **NFR-4.2**: Input validation for all MCP tool parameters
- **NFR-4.3**: SQL injection prevention (parameterized queries)
- **NFR-4.4**: Optional prompt redaction for privacy
- **NFR-4.5**: Audit log for sensitive operations (startup context updates)

#### NFR-5: Compatibility
- **NFR-5.1**: MCP protocol compliance (stdio transport)
- **NFR-5.2**: Docker container deployment (like Fiedler)
- **NFR-5.3**: Volume mount for data persistence
- **NFR-5.4**: Python 3.10+ compatibility
- **NFR-5.5**: Cross-platform (Linux, macOS, Windows)

---

## 4. Technical Specifications

### 4.1 Technology Stack

```yaml
dewey_stack:
  language: Python 3.10+
  mcp_framework: mcp (Anthropic SDK)
  database: SQLite 3.35+
  container: Docker + docker-compose
  testing: pytest
  logging: Python logging module

dependencies:
  - mcp[server]
  - sqlite3 (standard library)
  - pytest
  - pytest-asyncio (if async MCP)
```

### 4.2 Database Schema (SQLite)

```sql
-- Schema version tracking
CREATE TABLE schema_version (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Conversations table
CREATE TABLE conversations (
    id TEXT PRIMARY KEY,  -- UUID format
    session_id TEXT,
    storage_type TEXT CHECK (storage_type IN ('conversation', 'startup_context', 'fiedler_result')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON  -- Flexible JSONB-like storage
);

CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_conversations_type ON conversations(storage_type);
CREATE INDEX idx_conversations_created ON conversations(created_at DESC);

-- Messages within conversations
CREATE TABLE messages (
    id TEXT PRIMARY KEY,  -- UUID format
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(conversation_id, turn_number)
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id, turn_number);
CREATE INDEX idx_messages_role ON messages(role);

-- Full-text search index
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=rowid
);

-- Triggers to keep FTS in sync
CREATE TRIGGER messages_fts_insert AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER messages_fts_delete AFTER DELETE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.rowid;
END;

CREATE TRIGGER messages_fts_update AFTER UPDATE ON messages BEGIN
    DELETE FROM messages_fts WHERE rowid = old.rowid;
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- Startup contexts table
CREATE TABLE startup_contexts (
    id TEXT PRIMARY KEY,  -- UUID format
    name TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_startup_contexts_active ON startup_contexts(is_active);

-- Ensure only one active startup context
CREATE TRIGGER enforce_single_active_context
BEFORE UPDATE ON startup_contexts
WHEN NEW.is_active = TRUE
BEGIN
    UPDATE startup_contexts SET is_active = FALSE WHERE is_active = TRUE;
END;

-- Fiedler results table (Phase 2 integration)
CREATE TABLE fiedler_results (
    id TEXT PRIMARY KEY,  -- UUID format
    correlation_id TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT,  -- NULL if redacted
    output TEXT NOT NULL,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    duration_seconds REAL,
    status TEXT CHECK (status IN ('success', 'error')),
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata JSON
);

CREATE INDEX idx_fiedler_correlation ON fiedler_results(correlation_id);
CREATE INDEX idx_fiedler_model ON fiedler_results(model);
CREATE INDEX idx_fiedler_created ON fiedler_results(created_at DESC);
```

### 4.3 MCP Tools Specification

#### Tool: dewey_store_message

**Description:** Store a single message in a conversation

**Parameters:**
```json
{
  "conversation_id": "string (optional, generates new UUID if not provided)",
  "session_id": "string (optional, for grouping conversations)",
  "role": "string (required, one of: user, assistant, system)",
  "content": "string (required, message content)",
  "turn_number": "integer (optional, auto-increments if not provided)",
  "metadata": "object (optional, custom metadata)"
}
```

**Returns:**
```json
{
  "conversation_id": "uuid-string",
  "message_id": "uuid-string",
  "turn_number": 1,
  "created_at": "2025-10-02T17:45:00Z"
}
```

#### Tool: dewey_get_conversation

**Description:** Retrieve complete conversation with all messages

**Parameters:**
```json
{
  "conversation_id": "string (required, UUID)",
  "include_metadata": "boolean (optional, default: true)"
}
```

**Returns:**
```json
{
  "conversation_id": "uuid-string",
  "session_id": "session-123",
  "created_at": "2025-10-02T17:00:00Z",
  "updated_at": "2025-10-02T17:45:00Z",
  "metadata": {},
  "messages": [
    {
      "id": "msg-uuid-1",
      "turn_number": 1,
      "role": "user",
      "content": "Hello",
      "created_at": "2025-10-02T17:00:00Z"
    },
    {
      "id": "msg-uuid-2",
      "turn_number": 2,
      "role": "assistant",
      "content": "Hello! How can I help?",
      "created_at": "2025-10-02T17:00:05Z"
    }
  ]
}
```

#### Tool: dewey_search

**Description:** Full-text search across conversation content

**Parameters:**
```json
{
  "query": "string (required, search query)",
  "session_id": "string (optional, filter by session)",
  "start_date": "string (optional, ISO 8601 timestamp)",
  "end_date": "string (optional, ISO 8601 timestamp)",
  "limit": "integer (optional, default: 20, max: 100)",
  "context_lines": "integer (optional, surrounding messages, default: 2)"
}
```

**Returns:**
```json
{
  "query": "search term",
  "total_results": 15,
  "results": [
    {
      "conversation_id": "uuid-string",
      "message_id": "uuid-string",
      "turn_number": 5,
      "role": "assistant",
      "content": "Here is the search term you asked about...",
      "created_at": "2025-10-02T17:30:00Z",
      "context": [
        {"turn": 4, "role": "user", "content": "What about search term?"},
        {"turn": 6, "role": "user", "content": "Thanks!"}
      ]
    }
  ]
}
```

#### Tool: dewey_get_startup_context

**Description:** Retrieve the active startup context

**Parameters:**
```json
{
  "name": "string (optional, retrieve specific context by name)"
}
```

**Returns:**
```json
{
  "id": "uuid-string",
  "name": "current_context",
  "content": "# Current Project Context\n\n...",
  "is_active": true,
  "created_at": "2025-10-02T10:00:00Z",
  "updated_at": "2025-10-02T15:30:00Z"
}
```

#### Tool: dewey_set_startup_context

**Description:** Create or update a startup context

**Parameters:**
```json
{
  "name": "string (required, unique context name)",
  "content": "string (required, context content)",
  "set_active": "boolean (optional, default: true)"
}
```

**Returns:**
```json
{
  "id": "uuid-string",
  "name": "current_context",
  "is_active": true,
  "created_at": "2025-10-02T17:50:00Z"
}
```

#### Tool: dewey_list_startup_contexts

**Description:** List all available startup contexts

**Parameters:**
```json
{
  "include_content": "boolean (optional, default: false)"
}
```

**Returns:**
```json
{
  "contexts": [
    {
      "id": "uuid-1",
      "name": "project_alpha",
      "is_active": true,
      "created_at": "2025-10-02T10:00:00Z",
      "updated_at": "2025-10-02T15:30:00Z",
      "content": "..." // if include_content=true
    },
    {
      "id": "uuid-2",
      "name": "project_beta",
      "is_active": false,
      "created_at": "2025-10-01T12:00:00Z",
      "updated_at": "2025-10-01T12:00:00Z"
    }
  ]
}
```

#### Tool: dewey_store_fiedler_result (Phase 2)

**Description:** Store LLM orchestration result from Fiedler

**Parameters:**
```json
{
  "correlation_id": "string (required, Fiedler correlation ID)",
  "model": "string (required, model name)",
  "prompt": "string (optional, NULL if redacted)",
  "output": "string (required, LLM output)",
  "tokens_prompt": "integer (optional)",
  "tokens_completion": "integer (optional)",
  "duration_seconds": "number (optional)",
  "status": "string (required, one of: success, error)",
  "error_message": "string (optional, if status=error)",
  "metadata": "object (optional)"
}
```

**Returns:**
```json
{
  "id": "uuid-string",
  "correlation_id": "fiedler-correlation-id",
  "created_at": "2025-10-02T17:55:00Z"
}
```

#### Tool: dewey_get_fiedler_results

**Description:** Retrieve Fiedler results by correlation ID

**Parameters:**
```json
{
  "correlation_id": "string (required, Fiedler correlation ID)"
}
```

**Returns:**
```json
{
  "correlation_id": "fiedler-correlation-id",
  "results": [
    {
      "id": "uuid-1",
      "model": "gemini-2.5-pro",
      "output": "...",
      "tokens_prompt": 163370,
      "tokens_completion": 91,
      "duration_seconds": 25.21,
      "status": "success",
      "created_at": "2025-10-02T17:55:00Z"
    },
    {
      "id": "uuid-2",
      "model": "gpt-5",
      "output": "...",
      "tokens_prompt": 163957,
      "tokens_completion": 1481,
      "duration_seconds": 50.96,
      "status": "success",
      "created_at": "2025-10-02T17:55:00Z"
    }
  ]
}
```

---

## 5. Docker Deployment

### 5.1 Container Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  dewey:
    build:
      context: .
      dockerfile: Dockerfile
    image: dewey-mcp:latest
    container_name: dewey-mcp
    restart: unless-stopped
    stdin_open: true
    tty: true

    environment:
      - DEWEY_DATABASE_PATH=/app/winni/conversations.db
      - DEWEY_LOG_LEVEL=INFO
      - DEWEY_BACKUP_ENABLED=true
      - DEWEY_BACKUP_INTERVAL_HOURS=24

    volumes:
      # Winni data persistence
      - winni_data:/app/winni

      # Read-only project access (for future integration)
      - /mnt/projects:/mnt/projects:ro

    ports:
      - "9020:8080"  # MCP server on port 9020

    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

volumes:
  winni_data:
```

### 5.2 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Dewey server code
COPY dewey/ ./dewey/

# Create Winni data directory
RUN mkdir -p /app/winni

# Run MCP server
CMD ["python", "-m", "dewey.mcp_server"]
```

---

## 6. Integration Requirements

### 6.1 Claude Code Integration

**User Workflow:**
1. During session, Claude stores messages: `dewey_store_message(...)`
2. At end of session, user requests: "Write a new startup context"
3. Claude creates context: `dewey_set_startup_context(name="session_2025-10-02", content="...", set_active=true)`
4. User clears chat manually
5. User starts new session
6. Claude loads context: `dewey_get_startup_context()` and uses content in system prompt

**Search Workflow:**
1. User asks: "What did we discuss about Fiedler last week?"
2. Claude searches: `dewey_search(query="Fiedler", start_date="2025-09-25T00:00:00Z")`
3. Claude presents search results with context

### 6.2 Fiedler Integration (Phase 2)

**Modified Fiedler Flow:**
```python
# Current: Fiedler writes to /app/fiedler_output/
# Future: Fiedler writes to Dewey

# In Fiedler's result handler:
def store_result(correlation_id, model, prompt, output, tokens, duration):
    # Call Dewey via MCP or direct API
    dewey.store_fiedler_result(
        correlation_id=correlation_id,
        model=model,
        prompt=prompt if not redact_prompts else None,
        output=output,
        tokens_prompt=tokens['prompt'],
        tokens_completion=tokens['completion'],
        duration_seconds=duration,
        status='success'
    )
```

**Benefits:**
- Centralized storage for all LLM interactions
- Query Fiedler results by model, date, correlation ID
- Future semantic search across Fiedler outputs
- Integration with conversation history for context

---

## 7. Testing Requirements

### 7.1 Unit Tests

```python
test_cases = [
    # Storage operations
    'test_store_message_creates_conversation',
    'test_store_message_increments_turn_number',
    'test_store_message_validates_role',
    'test_get_conversation_retrieves_all_messages',
    'test_get_conversation_not_found_returns_error',

    # Search functionality
    'test_search_finds_exact_match',
    'test_search_case_insensitive',
    'test_search_with_date_filter',
    'test_search_with_session_filter',
    'test_search_returns_context',

    # Startup contexts
    'test_set_startup_context_creates_new',
    'test_set_startup_context_updates_existing',
    'test_set_active_deactivates_others',
    'test_get_startup_context_returns_active',
    'test_list_startup_contexts',

    # Fiedler integration
    'test_store_fiedler_result',
    'test_get_fiedler_results_by_correlation',
    'test_fiedler_result_prompt_redaction',

    # Database integrity
    'test_foreign_key_cascade_delete',
    'test_unique_constraint_turn_number',
    'test_transaction_rollback_on_error',
]
```

### 7.2 Integration Tests

```python
integration_test_scenarios = [
    # End-to-end conversation storage
    'test_full_conversation_workflow',

    # Search with large dataset
    'test_search_performance_10k_conversations',

    # Startup context workflow
    'test_write_clear_load_context_workflow',

    # MCP protocol compliance
    'test_mcp_tool_discovery',
    'test_mcp_tool_invocation',
    'test_mcp_error_handling',

    # Docker deployment
    'test_container_starts_successfully',
    'test_data_persistence_across_restarts',
]
```

---

## 8. Migration Path

### 8.1 Phase Roadmap

**MVP (Phase 1):**
- SQLite storage
- Basic MCP tools
- Conversation and startup context management
- Simple text search
- Docker deployment

**Phase 2: Fiedler Integration**
- Connect Fiedler to Dewey
- Store orchestration results in Winni
- Query interface for Fiedler data

**Phase 3: Semantic Search**
- Add embedding generation (OpenAI ada-002)
- Store embeddings in SQLite
- Implement vector similarity search (sqlite-vss extension)

**Phase 4: PostgreSQL Migration (Optional)**
- Migration script: SQLite → PostgreSQL
- Add pgvector for semantic search
- Support both SQLite and PostgreSQL backends

**Phase 5: Tiered Storage**
- Implement fast/slow tier logic
- Automatic archival policies
- Compression for old data

**Phase 6: Full Paper 12 Implementation**
- Phase 1-4 training data organization
- RAG source tracking
- Quality metrics and validation
- Advanced analytics

### 8.2 Backward Compatibility

- Schema versioning from day 1
- Migration scripts for each phase
- Support for SQLite throughout (small deployments)
- Clear upgrade path documented in README

---

## 9. Success Criteria

### 9.1 MVP Acceptance Criteria

1. ✅ Dewey MCP server starts successfully in Docker
2. ✅ All MCP tools callable from Claude Code
3. ✅ Conversation storage and retrieval working
4. ✅ Startup context workflow functional
5. ✅ Search returns relevant results within 500ms
6. ✅ Data persists across container restarts
7. ✅ Test suite passes with >80% coverage
8. ✅ Documentation complete (README, API reference)

### 9.2 Performance Targets

- Store message: <100ms (p95)
- Retrieve conversation: <200ms (p95)
- Search (10K conversations): <500ms (p95)
- Startup time: <5 seconds

### 9.3 Quality Targets

- Test coverage: >80%
- Zero data loss on crash (WAL mode)
- MCP protocol compliance: 100%
- Documentation completeness: All tools documented

---

## 10. Open Questions for Triplet Review

1. **Schema Design**: Is the SQLite schema appropriate for MVP? Any missing tables or indices?

2. **MCP Tool Interface**: Are the tool signatures clear and complete? Any missing parameters or return fields?

3. **Search Implementation**: Is SQLite FTS5 sufficient for MVP, or should we use a different approach?

4. **Fiedler Integration**: Should Fiedler integration be in Phase 1 (MVP) or deferred to Phase 2?

5. **Startup Context Workflow**: Does the "write/clear/load" workflow make sense? Better alternatives?

6. **Performance Expectations**: Are the performance targets (100ms store, 500ms search) realistic for SQLite?

7. **Migration Path**: Is the phase roadmap (SQLite → PostgreSQL → tiered storage) reasonable?

8. **Security**: Any security concerns with the proposed file permissions and access controls?

9. **Data Model**: Should we add additional fields to conversations/messages tables for future extensibility?

10. **Testing Strategy**: Are the proposed test cases comprehensive? Any critical scenarios missing?

---

## 11. References

- **Paper 12**: Conversation Storage and Retrieval Infrastructure (`/mnt/projects/ICCM/docs/papers/12_Conversation_Storage_Retrieval_v3.md`)
- **Fiedler Requirements**: Previous requirements document process
- **MCP Protocol**: Anthropic Model Context Protocol specification
- **SQLite Documentation**: https://www.sqlite.org/docs.html
- **SQLite FTS5**: https://www.sqlite.org/fts5.html

---

## 12. Approval

**Requirements Author:** Claude (Sonnet 4.5)
**Date:** 2025-10-02
**Status:** Awaiting triplet review (Gemini 2.5 Pro, GPT-5, Grok-4)

**Next Steps:**
1. Send to triplet for detailed technical review
2. Synthesize feedback into requirements v2
3. Begin implementation after approval
