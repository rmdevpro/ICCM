# Dewey + Winni Requirements Document v2
## MVP Librarian (Dewey) and Data Lake (Winni) for ICCM

**Version:** 2.0
**Date:** 2025-10-02
**Status:** Requirements - Second Triplet Review
**Context:** Single-user development system (Aristotle + Claude)

---

## 1. Executive Summary

**Dewey** is a Docker-based MCP server (like Fiedler) that manages **Winni**, a PostgreSQL data lake for conversation histories, startup contexts, and LLM orchestration results.

**Key Goals:**
- Persistent conversation storage across Claude Code sessions
- Startup context management (write/load workflow)
- Search through conversation history
- Foundation for Fiedler result storage
- Single-user development environment (no auth/multi-user complexity)

**Key Changes from v1:**
- ✅ PostgreSQL from day 1 (already installed on Irina, not SQLite)
- ✅ Docker MCP server like Fiedler (port 9020)
- ✅ Single-user dev environment (removed auth/encryption/multi-user)
- ✅ Fixed schema issues (removed storage_type, added triggers/indices)
- ✅ Added missing tools (delete, list, pagination)
- ✅ **Addressed data capture question explicitly**

---

## 2. System Architecture

### 2.1 Deployment Model

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code                          │
│              (MCP Client on localhost)                  │
└──────────────────────┬──────────────────────────────────┘
                       │ MCP stdio over Docker exec
                       │
┌──────────────────────▼──────────────────────────────────┐
│            dewey-mcp (Docker Container)                 │
│                   Port 9020                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Dewey MCP Server (Python)                 │  │
│  │  - dewey_store_message                            │  │
│  │  - dewey_get_conversation                         │  │
│  │  - dewey_list_conversations                       │  │
│  │  - dewey_delete_conversation                      │  │
│  │  - dewey_search                                   │  │
│  │  - dewey_get_startup_context                      │  │
│  │  - dewey_set_startup_context                      │  │
│  │  - dewey_list_startup_contexts                    │  │
│  │  - dewey_delete_startup_context                   │  │
│  └─────────────────┬─────────────────────────────────┘  │
│                    │ psycopg2                            │
│                    ▼                                     │
│  ┌─────────────────────────────────────────────────┐    │
│  │  PostgreSQL Client Connection                   │    │
│  │  Host: irina (PostgreSQL server)                │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
                       │ TCP 5432
                       ▼
┌─────────────────────────────────────────────────────────┐
│              Irina (PostgreSQL Server)                  │
│                                                         │
│  ┌───────────────────────────────────────────────────┐  │
│  │        Winni Database (PostgreSQL)                │  │
│  │  - conversations                                  │  │
│  │  - messages                                       │  │
│  │  - startup_contexts                               │  │
│  │  - fiedler_results (future)                       │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Key Points:**
- Dewey runs as Docker container (like Fiedler on port 9010)
- Connects to existing PostgreSQL on Irina
- MCP communication via stdio (docker exec)
- Single-user dev environment (no authentication)

---

## 3. Data Capture Strategy

### 3.1 The Data Capture Problem

**Question:** How do we actually capture conversation turns into Dewey?

**Options Evaluated:**

**Option A: Manual Storage (Explicit Calls)**
```python
# After each user message:
dewey_store_message(role="user", content="...")

# After each assistant response:
dewey_store_message(role="assistant", content="...")
```
- ❌ Tedious and error-prone
- ❌ Easy to forget
- ❌ Breaks conversation flow

**Option B: Claude Code Hooks (If Available)**
```yaml
# .claude/hooks.yml (hypothetical)
on_user_message: dewey_store_message(role="user", content="${message}")
on_assistant_message: dewey_store_message(role="assistant", content="${message}")
```
- ✅ Automatic capture
- ✅ No user intervention
- ❓ Does Claude Code support this? (Unknown)

**Option C: Session Wrapper Tool**
```python
# At end of session, Claude calls:
dewey_store_session(messages=[...])  # Bulk store entire session
```
- ❌ No persistent storage during session
- ❌ Data lost if session crashes
- ✅ Simple implementation

**Option D: Periodic Snapshots**
```python
# Claude periodically calls (every 10 turns):
dewey_store_messages_bulk(messages=[...])
```
- ✅ Balances automation and reliability
- ✅ Reduces call overhead
- ✅ Recoverable if session crashes

### 3.2 Recommended Approach (MVP)

**Hybrid: Manual + Bulk Tools**

1. **During development:** Manual `dewey_store_message` calls when needed
2. **For important conversations:** Explicit bulk store at end
3. **Future:** Investigate Claude Code hooks for automatic capture

**MVP Tools:**
- `dewey_store_message` - Single message (manual)
- `dewey_store_messages_bulk` - Batch storage (end of session)
- `dewey_get_conversation` - Retrieve stored conversation

**Open Question for Triplet:** Is there a better automatic capture mechanism we're missing?

---

## 4. Database Schema (PostgreSQL)

### 4.1 Core Tables

```sql
-- Conversations
CREATE TABLE conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT valid_metadata CHECK (metadata IS NULL OR jsonb_typeof(metadata) = 'object')
);

CREATE INDEX idx_conversations_session ON conversations(session_id);
CREATE INDEX idx_conversations_created ON conversations(created_at DESC);
CREATE INDEX idx_conversations_updated ON conversations(updated_at DESC);

-- Auto-update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER conversations_updated_at
BEFORE UPDATE ON conversations
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Messages
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
    content TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(conversation_id, turn_number)
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id, turn_number);
CREATE INDEX idx_messages_created ON messages(created_at DESC);

-- Full-text search (PostgreSQL native)
CREATE INDEX idx_messages_content_fts ON messages USING GIN (to_tsvector('english', content));

-- Startup Contexts
CREATE TABLE startup_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_name CHECK (length(name) > 0)
);

CREATE UNIQUE INDEX idx_startup_contexts_single_active
ON startup_contexts(is_active) WHERE is_active = TRUE;

CREATE TRIGGER startup_contexts_updated_at
BEFORE UPDATE ON startup_contexts
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Fiedler Results (Phase 2)
CREATE TABLE fiedler_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT,  -- NULL if redacted
    output TEXT NOT NULL,
    tokens_prompt INTEGER,
    tokens_completion INTEGER,
    duration_seconds REAL,
    status TEXT CHECK (status IN ('success', 'error')),
    error_message TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT valid_metadata CHECK (metadata IS NULL OR jsonb_typeof(metadata) = 'object')
);

CREATE INDEX idx_fiedler_correlation ON fiedler_results(correlation_id);
CREATE INDEX idx_fiedler_model ON fiedler_results(model);
CREATE INDEX idx_fiedler_created ON fiedler_results(created_at DESC);
```

**Key Changes from v1:**
- ✅ Removed `storage_type` column (separate tables cleaner)
- ✅ Added `updated_at` triggers (automatic timestamps)
- ✅ Added partial unique index for single active context
- ✅ Added JSON validation constraints
- ✅ Added missing indices for performance
- ✅ PostgreSQL-native full-text search (GIN index)

---

## 5. MCP Tools

### Core Conversation Tools

#### dewey_store_message
Store a single message (manual capture)

**Parameters:**
```json
{
  "conversation_id": "uuid (optional, creates new if omitted)",
  "session_id": "string (optional)",
  "role": "user | assistant | system",
  "content": "string",
  "turn_number": "integer (optional, auto-increment)",
  "metadata": "object (optional)"
}
```

**Returns:**
```json
{
  "conversation_id": "uuid",
  "message_id": "uuid",
  "turn_number": 1
}
```

#### dewey_store_messages_bulk
Store multiple messages at once (batch capture)

**Parameters:**
```json
{
  "conversation_id": "uuid (optional, creates new if omitted)",
  "session_id": "string (optional)",
  "messages": [
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ],
  "metadata": "object (optional)"
}
```

**Returns:**
```json
{
  "conversation_id": "uuid",
  "stored": 2,
  "message_ids": ["uuid1", "uuid2"]
}
```

#### dewey_get_conversation
Retrieve complete conversation

**Parameters:**
```json
{
  "conversation_id": "uuid"
}
```

**Returns:**
```json
{
  "conversation_id": "uuid",
  "session_id": "string",
  "created_at": "timestamp",
  "messages": [
    {"turn": 1, "role": "user", "content": "..."},
    {"turn": 2, "role": "assistant", "content": "..."}
  ]
}
```

#### dewey_list_conversations
List conversations with pagination

**Parameters:**
```json
{
  "session_id": "string (optional, filter by session)",
  "limit": "integer (default: 20, max: 100)",
  "offset": "integer (default: 0)",
  "sort_by": "created_at | updated_at (default: updated_at)"
}
```

**Returns:**
```json
{
  "conversations": [
    {
      "id": "uuid",
      "session_id": "string",
      "created_at": "timestamp",
      "updated_at": "timestamp",
      "message_count": 10
    }
  ],
  "total": 150,
  "limit": 20,
  "offset": 0
}
```

#### dewey_delete_conversation
Delete conversation and all messages

**Parameters:**
```json
{
  "conversation_id": "uuid"
}
```

**Returns:**
```json
{
  "deleted": true,
  "messages_deleted": 15
}
```

### Search Tool

#### dewey_search
Full-text search across conversations

**Parameters:**
```json
{
  "query": "string",
  "session_id": "string (optional)",
  "start_date": "timestamp (optional)",
  "end_date": "timestamp (optional)",
  "limit": "integer (default: 20, max: 100)",
  "offset": "integer (default: 0)"
}
```

**Returns:**
```json
{
  "results": [
    {
      "conversation_id": "uuid",
      "message_id": "uuid",
      "turn": 5,
      "role": "assistant",
      "content": "...",
      "rank": 0.85,
      "created_at": "timestamp"
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

### Startup Context Tools

#### dewey_get_startup_context
Get active startup context (or by name)

**Parameters:**
```json
{
  "name": "string (optional, gets active if omitted)"
}
```

**Returns:**
```json
{
  "id": "uuid",
  "name": "current_context",
  "content": "# Context\n...",
  "is_active": true,
  "created_at": "timestamp"
}
```

#### dewey_set_startup_context
Create or update startup context

**Parameters:**
```json
{
  "name": "string",
  "content": "string",
  "set_active": "boolean (default: true)"
}
```

**Returns:**
```json
{
  "id": "uuid",
  "name": "current_context",
  "is_active": true
}
```

#### dewey_list_startup_contexts
List all startup contexts

**Parameters:**
```json
{
  "include_content": "boolean (default: false)"
}
```

**Returns:**
```json
{
  "contexts": [
    {
      "id": "uuid",
      "name": "project_alpha",
      "is_active": true,
      "created_at": "timestamp",
      "content": "..." // if include_content=true
    }
  ]
}
```

#### dewey_delete_startup_context
Delete startup context

**Parameters:**
```json
{
  "name": "string"
}
```

**Returns:**
```json
{
  "deleted": true
}
```

---

## 6. Docker Configuration

### 6.1 docker-compose.yml

```yaml
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
      # PostgreSQL connection
      - DEWEY_DB_HOST=irina
      - DEWEY_DB_PORT=5432
      - DEWEY_DB_NAME=winni
      - DEWEY_DB_USER=dewey
      - DEWEY_DB_PASSWORD=${DEWEY_DB_PASSWORD}

      # Logging
      - DEWEY_LOG_LEVEL=INFO

    ports:
      - "9020:8080"  # MCP server

    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

networks:
  default:
    external:
      name: iccm_network  # Share network with Fiedler
```

### 6.2 Database Setup

```bash
# On Irina, create database and user
sudo -u postgres psql

CREATE DATABASE winni;
CREATE USER dewey WITH PASSWORD 'secure-password-here';
GRANT ALL PRIVILEGES ON DATABASE winni TO dewey;

# Connect and create schema
\c winni
\i schema.sql

# Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dewey;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dewey;
```

---

## 7. Requirements

### 7.1 Functional Requirements

**FR-1: Conversation Storage**
- Store messages with role, content, turn number
- Support incremental and bulk storage
- Generate UUIDs for conversations and messages
- Store optional metadata

**FR-2: Conversation Retrieval**
- Get complete conversation by ID
- List conversations with pagination
- Filter by session ID, date range
- Include message counts

**FR-3: Conversation Management**
- Delete conversation with cascade
- Update conversation metadata
- Auto-increment turn numbers safely

**FR-4: Search**
- Full-text search across message content
- Filter by session, date range
- Return relevance rank (ts_rank)
- Paginated results

**FR-5: Startup Context Management**
- Store named contexts
- Mark one context as active (enforced by unique index)
- Retrieve active context or by name
- List all contexts
- Delete contexts

**FR-6: Fiedler Integration (Phase 2)**
- Store Fiedler orchestration results
- Link to conversations (optional)
- Query by correlation ID, model, date

### 7.2 Non-Functional Requirements

**NFR-1: Performance**
- Store message: <100ms (p95)
- Retrieve conversation: <200ms (p95)
- Search: <500ms for 10K conversations (p95)
- Bulk insert: <50ms per message (p95)

**NFR-2: Reliability**
- PostgreSQL transactions for consistency
- Foreign key constraints enforced
- Automatic timestamps via triggers
- Crash-resistant (PostgreSQL durability)

**NFR-3: Maintainability**
- Schema version tracking
- Migration scripts for future changes
- Comprehensive test coverage (>80%)
- Clear error messages

**NFR-4: Simplicity (Single-User Dev)**
- No authentication (single-user system)
- No encryption (development environment)
- No multi-tenancy
- Minimal operational complexity

---

## 8. Testing Strategy

### 8.1 Unit Tests

- Store/retrieve messages
- Turn number auto-increment (with transactions)
- Single active context enforcement
- JSON metadata validation
- Cascade deletes
- Search relevance ranking

### 8.2 Integration Tests

- Full conversation workflow
- Bulk insert performance
- Pagination correctness
- MCP protocol compliance
- Container startup/health check

### 8.3 Concurrency Tests

- Simultaneous message stores (turn number races)
- Concurrent active context updates
- Search during write operations

---

## 9. Migration Path

**Phase 1 (MVP):** PostgreSQL + basic search
**Phase 2:** Fiedler integration (store orchestration results)
**Phase 3:** Semantic search (pgvector + embeddings)
**Phase 4:** Tiered storage (if data volume grows)
**Phase 5:** Full Paper 12 implementation (multi-phase training data)

---

## 10. Open Questions for Triplet

1. **Data Capture:** Is there a better automatic mechanism than manual/bulk tools? Should we investigate Claude Code hooks?

2. **PostgreSQL from Day 1:** Is starting with PostgreSQL (vs SQLite) the right choice for MVP? (Note: Already installed on Irina)

3. **Docker MCP Pattern:** Should Dewey follow Fiedler's pattern (Docker container, port 9020, MCP stdio)? Any issues?

4. **Simplified Security:** For single-user dev, is it OK to skip authentication/encryption in MVP?

5. **Search Implementation:** Is PostgreSQL GIN full-text search sufficient, or should we add pg_trgm for fuzzy matching?

6. **Schema Completeness:** Are there any critical missing fields or indices for MVP?

7. **Tool Interface:** Are the tool signatures clear and complete? Any missing operations?

8. **Bulk Insert:** Is `dewey_store_messages_bulk` the right abstraction for batch storage?

9. **Fiedler Integration:** Should we implement Phase 2 (Fiedler results) in MVP, or defer?

10. **Migration to pgvector:** When should we add semantic search? After MVP or later?

---

## 11. Changes from v1

**Architecture:**
- ✅ PostgreSQL from day 1 (not SQLite)
- ✅ Docker MCP server like Fiedler (port 9020)
- ✅ Single-user dev environment (no auth/multi-user)

**Schema:**
- ✅ Removed `storage_type` column
- ✅ Added `updated_at` triggers
- ✅ Added partial unique index for active context
- ✅ Added JSON validation constraints
- ✅ Added missing indices (created_at, updated_at)
- ✅ PostgreSQL-native full-text search (GIN)

**Tools:**
- ✅ Added `dewey_store_messages_bulk`
- ✅ Added `dewey_list_conversations`
- ✅ Added `dewey_delete_conversation`
- ✅ Added `dewey_delete_startup_context`
- ✅ Added pagination to search and list

**Requirements:**
- ✅ Simplified NFRs (removed auth/encryption)
- ✅ Added data capture strategy section
- ✅ Clarified Docker deployment model
- ✅ Removed over-engineering for multi-user

---

## 12. Success Criteria

1. ✅ Dewey MCP server runs in Docker (port 9020)
2. ✅ Connects to PostgreSQL on Irina
3. ✅ All MCP tools callable from Claude Code
4. ✅ Conversation storage and retrieval working
5. ✅ Startup context workflow functional
6. ✅ Search returns ranked results
7. ✅ Test suite passes with >80% coverage
8. ✅ Data persists across restarts

---

**Ready for second triplet review with focus on:**
1. Data capture strategy (manual vs automatic)
2. PostgreSQL from day 1 (vs SQLite)
3. Simplified single-user security model
4. Schema and tool completeness
