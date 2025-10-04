# Dewey + Winni Requirements Document v3 (FINAL)
## MVP Librarian (Dewey) and Data Lake (Winni) for ICCM

**Version:** 3.0 (Final)
**Date:** 2025-10-02
**Status:** APPROVED - Ready for Implementation
**Context:** Single-user development system (Aristotle + Claude)

---

## 1. Executive Summary

**Dewey** is a Docker-based MCP server that manages **Winni**, a PostgreSQL data lake for conversation histories, startup contexts, and LLM orchestration results.

**Triplet Review Status:**
- **v1:** All 3 approved with changes (major schema/transport issues)
- **v2:** All 3 approved with minor changes (transport fix, metadata column)
- **v3:** Incorporates all triplet feedback - READY FOR IMPLEMENTATION

**Key Architecture:**
- Docker container on port 9020 (like Fiedler on 9010)
- MCP protocol via WebSocket: `ws://localhost:9020`
- PostgreSQL backend on Irina (already installed)
- Single-user dev environment (no auth complexity)

---

## 2. System Architecture

### 2.1 Deployment Model

```
┌─────────────────────────────────────────────────────────┐
│                    Claude Code                          │
│              (MCP Client on localhost)                  │
└──────────────────────┬──────────────────────────────────┘
                       │ MCP WebSocket: ws://localhost:9020
                       │
┌──────────────────────▼──────────────────────────────────┐
│            dewey-mcp (Docker Container)                 │
│                   Port 9020                             │
│  ┌───────────────────────────────────────────────────┐  │
│  │         Dewey MCP Server (Python)                 │  │
│  │  WebSocket MCP on 0.0.0.0:9020                    │  │
│  │                                                    │  │
│  │  Tools:                                           │  │
│  │  - dewey_begin_conversation                       │  │
│  │  - dewey_store_message                            │  │
│  │  - dewey_store_messages_bulk                      │  │
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
│  │  Host: irina                                    │    │
│  │  Database: winni                                │    │
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
│  │  - fiedler_results (Phase 2)                      │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

**Changed from v2:** MCP transport now correctly specified as **WebSocket** (not stdio)

---

## 3. Data Capture Strategy

### 3.1 Critical Requirement: Real-Time Line-by-Line Storage

**REQUIREMENT:** Conversation turns MUST be captured and stored in real-time as they occur, not post-session.

**User Experience Note:** Previous attempt using CLAUDE.md pinned instructions proved unreliable - Claude only followed instructions intermittently, resulting in data loss.

### 3.2 Proposed Implementation Options

**Option A: Python Session Wrapper**
```python
# dewey_wrapper.py - Launch Claude Code through wrapper
from dewey_client import capture_session

with capture_session():
    # Launch Claude Code subprocess
    # Intercept stdin/stdout to capture messages
    # Automatically log to Dewey in real-time
```

**Pros:**
- Self-contained wrapper script
- No Claude Code modification needed
- Works with stdin/stdout interception

**Cons:**
- May be fragile if Claude Code output format changes
- Requires parsing unstructured text
- Could miss messages if parsing fails

---

**Option B: MCP Proxy/Middleware** (Recommended)
```
┌─────────────┐
│ Claude Code │
└──────┬──────┘
       │ MCP Protocol
       ▼
┌─────────────────┐
│   MCP Proxy     │  (port 9000)
│  (Dewey Logger) │
└────┬────────┬───┘
     │        │
     │        └─→ Log to Dewey (port 9020)
     │
     └─→ Forward to actual MCP servers
         (Fiedler 9010, etc.)
```

**How it works:**
1. Claude Code connects to proxy on port 9000 (instead of direct to MCP servers)
2. Proxy intercepts all MCP tool calls and responses
3. Proxy logs conversations to Dewey automatically
4. Proxy forwards requests to actual MCP servers
5. Completely transparent to user

**Pros:**
- Fully automatic - zero user intervention
- Works with structured MCP protocol (not text parsing)
- Captures ALL conversations (tool calls, responses, everything)
- Transparent to user
- Reliable - uses MCP protocol, not heuristics

**Cons:**
- Additional component to deploy (but lightweight)
- Adds small latency (negligible)

---

**Option C: Claude Code Hook/Extension**
```python
# If Claude Code supports extensions/hooks
class DeweyLoggingHook:
    def on_user_message(self, message):
        dewey_store_message(role="user", content=message)

    def on_assistant_message(self, message):
        dewey_store_message(role="assistant", content=message)
```

**Pros:**
- Native integration
- Minimal overhead
- Reliable if hooks are available

**Cons:**
- Unknown if Claude Code supports this
- Requires Claude Code API knowledge

---

### 3.3 Selected Approach: Option B (MCP Proxy/Middleware)

**DECISION:** Unanimous recommendation from triplet review (Fiedler's default models DeepSeek-R1)

**Why Option B:**
- **Most Reliable:** Captures structured MCP protocol (JSON-RPC), not text parsing
- **Simplest:** WebSocket relay with protocol inspection, no heuristics
- **Most Robust:** Independent of Claude Code UI/output format changes
- **Minimal Overhead:** Transparent proxy, negligible latency (~1-2ms)
- **Protocol-True:** Works as long as MCP WebSocket protocol is used

**Architecture:**
```
Claude Code (MCP Client)
  ↓ ws://localhost:9000
MCP Proxy (Port 9000)
  ├→ Log to Dewey (ws://localhost:9020)
  └→ Forward to upstream MCP servers (Fiedler 9010, etc.)
```

**Implementation Components:**

1. **WebSocket Relay:** Python asyncio proxy listening on port 9000
2. **JSON-RPC Inspector:** Parse MCP messages (tool calls, responses, streams)
3. **Dewey Logger:** Automatically call `dewey_store_message` for each turn
4. **Connection Mapper:** Track client connections → conversation IDs
5. **Message Router:** Forward traffic to correct upstream MCP server

**Key Implementation Details:**

```python
# Proxy flow for each message:
async def handle_message(websocket, message):
    # 1. Parse JSON-RPC message
    msg = json.loads(message)

    # 2. Log to Dewey (parallel)
    asyncio.create_task(log_to_dewey(msg))

    # 3. Forward to upstream server
    await forward_to_upstream(websocket, message)

# Conversation tracking
async def on_client_connect(websocket):
    conversation_id = await dewey_begin_conversation()
    client_sessions[websocket] = conversation_id

# Message logging
async def log_to_dewey(msg):
    if msg['method'] == 'tools/call':
        # User message (tool invocation)
        await dewey_store_message(
            role='user',
            content=extract_user_intent(msg)
        )
    elif msg.get('result'):
        # Assistant response
        await dewey_store_message(
            role='assistant',
            content=msg['result']
        )
```

**Configuration:**

```yaml
# Claude Code MCP config - Point to proxy
{
  "mcpServers": {
    "fiedler": {
      "url": "ws://localhost:9000?upstream=fiedler"
    },
    "dewey": {
      "url": "ws://localhost:9000?upstream=dewey"
    }
  }
}

# Proxy routes upstream by query parameter
# Dewey proxy client connects directly to ws://localhost:9020 (no loop)
```

**Deployment:**
- Docker container alongside Dewey and Fiedler
- Lightweight (Python + websockets library)
- Minimal resource usage (~10MB RAM)
- No persistent state needed

---

## 4. Database Schema (PostgreSQL)

### 4.1 Setup Script

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- For gen_random_uuid()

-- Create database and user (run as postgres)
CREATE DATABASE winni;
CREATE USER dewey WITH PASSWORD 'secure-password-here';
GRANT ALL PRIVILEGES ON DATABASE winni TO dewey;

-- Connect to winni database
\c winni

-- Grant permissions
GRANT ALL PRIVILEGES ON SCHEMA public TO dewey;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dewey;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dewey;
```

### 4.2 Core Tables

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
CREATE INDEX idx_conversations_metadata ON conversations USING GIN(metadata);  -- Optional but helpful

-- Auto-update updated_at trigger
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
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    metadata JSONB,  -- NEW: Added from triplet feedback
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(conversation_id, turn_number),
    CONSTRAINT valid_metadata CHECK (metadata IS NULL OR jsonb_typeof(metadata) = 'object')
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id, turn_number);
CREATE INDEX idx_messages_created ON messages(created_at DESC);
CREATE INDEX idx_messages_conversation_created ON messages(conversation_id, created_at);

-- Full-text search (PostgreSQL GIN)
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

-- Enforce single active context with partial unique index
CREATE UNIQUE INDEX idx_startup_contexts_single_active
ON startup_contexts(is_active) WHERE is_active = TRUE;

CREATE TRIGGER startup_contexts_updated_at
BEFORE UPDATE ON startup_contexts
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Fiedler Results (Phase 2 - create table but defer tools)
CREATE TABLE fiedler_results (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    correlation_id TEXT NOT NULL,
    model TEXT NOT NULL,
    prompt TEXT,
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

**Changes from v2:**
- ✅ Added `messages.metadata JSONB` column
- ✅ Added `CREATE EXTENSION pgcrypto`
- ✅ Added `role = 'tool'` option for tool messages
- ✅ Added optional GIN index on `conversations.metadata`
- ✅ Added JSON validation checks

---

## 5. MCP Tools Specification

### 5.1 Conversation Management

#### dewey_begin_conversation

**NEW:** Helper tool to start a conversation and get ID

**Parameters:**
```json
{
  "session_id": "string (optional)",
  "metadata": "object (optional)"
}
```

**Returns:**
```json
{
  "conversation_id": "uuid",
  "session_id": "string",
  "created_at": "timestamp"
}
```

**Purpose:** Simplifies workflow - call once, use conversation_id for all subsequent messages

#### dewey_store_message

Store a single message

**Parameters:**
```json
{
  "conversation_id": "uuid (required)",
  "role": "user | assistant | system | tool",
  "content": "string (required)",
  "turn_number": "integer (optional, auto-increment if omitted)",
  "metadata": "object (optional)"
}
```

**Returns:**
```json
{
  "message_id": "uuid",
  "turn_number": 1,
  "created_at": "timestamp"
}
```

**Implementation Note:**
- If `turn_number` omitted, calculate with transaction:
  ```sql
  -- Lock conversation row
  SELECT 1 FROM conversations WHERE id = $1 FOR UPDATE;

  -- Calculate next turn number
  SELECT COALESCE(MAX(turn_number) + 1, 1) FROM messages WHERE conversation_id = $1;

  -- Insert with calculated turn_number
  INSERT INTO messages (...) VALUES (...);
  ```

#### dewey_store_messages_bulk

Store multiple messages at once

**Parameters:**
```json
{
  "conversation_id": "uuid (optional, creates new if omitted)",
  "session_id": "string (optional, if creating new conversation)",
  "messages": [
    {"role": "user", "content": "...", "metadata": {}},
    {"role": "assistant", "content": "...", "metadata": {}}
  ],
  "metadata": "object (optional, conversation-level)"
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

**Implementation Note:**
- Server assigns `turn_number` based on array order
- Client does NOT provide turn numbers in bulk mode
- Transaction ensures atomicity

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
  "updated_at": "timestamp",
  "metadata": {},
  "messages": [
    {
      "id": "uuid",
      "turn": 1,
      "role": "user",
      "content": "...",
      "metadata": {},
      "created_at": "timestamp"
    }
  ]
}
```

#### dewey_list_conversations

List conversations with pagination

**Parameters:**
```json
{
  "session_id": "string (optional)",
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
      "message_count": 10,
      "metadata": {}
    }
  ],
  "total": 150,
  "limit": 20,
  "offset": 0
}
```

**Implementation Note:**
- `message_count` computed with subquery: `(SELECT COUNT(*) FROM messages WHERE conversation_id = c.id)`

#### dewey_delete_conversation

Delete conversation and all messages

**Parameters:**
```json
{
  "conversation_id": "uuid",
  "force": "boolean (optional, default: false)"
}
```

**Returns:**
```json
{
  "deleted": true,
  "messages_deleted": 15
}
```

**Implementation Note:**
- `force` parameter for safety (confirm delete in dev)
- Cascade delete handled by `ON DELETE CASCADE`

### 5.2 Search Tool

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
      "session_id": "string",
      "message_id": "uuid",
      "turn": 5,
      "role": "assistant",
      "content": "...",
      "rank": 0.85,
      "created_at": "timestamp",
      "conversation_metadata": {}
    }
  ],
  "total": 42,
  "limit": 20,
  "offset": 0
}
```

**Implementation Note:**
```sql
SELECT
    m.id, m.conversation_id, m.turn_number, m.role, m.content, m.created_at,
    c.session_id, c.metadata as conversation_metadata,
    ts_rank(to_tsvector('english', m.content), plainto_tsquery('english', $1)) as rank
FROM messages m
JOIN conversations c ON m.conversation_id = c.id
WHERE to_tsvector('english', m.content) @@ plainto_tsquery('english', $1)
ORDER BY rank DESC, m.created_at DESC
LIMIT $2 OFFSET $3;
```

### 5.3 Startup Context Tools

#### dewey_get_startup_context

Get startup context (active or by name)

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
  "created_at": "timestamp",
  "updated_at": "timestamp"
}
```

**Returns:** `null` if no active context exists and name not specified

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
  "is_active": true,
  "created_at": "timestamp"
}
```

**Implementation Note:**
- When `set_active = true`, wrap in transaction:
  ```sql
  BEGIN;
  UPDATE startup_contexts SET is_active = false WHERE is_active = true;
  INSERT INTO startup_contexts (...) VALUES (...)
  ON CONFLICT (name) DO UPDATE SET content = EXCLUDED.content, is_active = EXCLUDED.is_active;
  COMMIT;
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
      "updated_at": "timestamp",
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
  "name": "string",
  "force": "boolean (optional, default: false)"
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

      # MCP Server
      - DEWEY_MCP_PORT=9020
      - DEWEY_MCP_HOST=0.0.0.0

      # Logging
      - DEWEY_LOG_LEVEL=INFO

    ports:
      - "127.0.0.1:9020:9020"  # Bind to localhost only

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

**Changes from v2:**
- ✅ Port mapping: `127.0.0.1:9020:9020` (bind to localhost)
- ✅ Added `DEWEY_MCP_PORT` and `DEWEY_MCP_HOST` env vars

### 6.2 Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Dewey server code
COPY dewey/ ./dewey/

# Expose MCP port
EXPOSE 9020

# Run MCP server
CMD ["python", "-m", "dewey.mcp_server"]
```

### 6.3 requirements.txt

```
mcp[server]>=0.9.0
psycopg2-binary>=2.9.9
```

### 6.4 .env

```bash
DEWEY_DB_PASSWORD=secure-password-here
```

---

## 7. Claude Code MCP Configuration

**Add to Claude Code MCP config:**

```json
{
  "mcpServers": {
    "fiedler": {
      "command": "docker",
      "args": ["exec", "-i", "fiedler-mcp", "python", "-m", "fiedler.mcp_server"]
    },
    "dewey": {
      "url": "ws://localhost:9020"
    }
  }
}
```

**Note:** Dewey uses WebSocket transport (not stdio like Fiedler)

---

## 8. Project Structure

```
/mnt/projects/ICCM/dewey/
├── README.md
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env
├── schema.sql                  # Database setup script
├── dewey/
│   ├── __init__.py
│   ├── mcp_server.py          # WebSocket MCP server
│   ├── database.py            # PostgreSQL connection/queries
│   ├── tools.py               # MCP tool implementations
│   └── config.py              # Configuration
└── tests/
    ├── test_database.py
    ├── test_tools.py
    └── test_mcp.py
```

---

## 9. Implementation Requirements

### 9.1 Functional Requirements

**FR-1: Conversation Management**
- Begin conversation (get UUID)
- Store messages (single and bulk)
- Retrieve conversation
- List conversations with pagination
- Delete conversation with cascade

**FR-2: Search**
- Full-text search with relevance ranking
- Filter by session, date range
- Return conversation metadata in results

**FR-3: Startup Context Management**
- Get active context or by name
- Set context with atomic active switch
- List all contexts
- Delete context

**FR-4: Data Integrity**
- Automatic turn number assignment (transaction-safe)
- Single active context enforcement (partial unique index)
- Cascade deletes for messages
- JSON metadata validation

### 9.2 Non-Functional Requirements

**NFR-1: Performance**
- Store message: <100ms (p95)
- Retrieve conversation: <200ms (p95)
- Search: <500ms for 10K conversations (p95)
- Bulk insert: <50ms per message (p95)

**NFR-2: Reliability**
- PostgreSQL transactions
- Foreign key constraints enforced
- Automatic timestamp updates
- Connection pooling

**NFR-3: Testing**
- Unit tests: >80% coverage
- Integration tests: Full workflow coverage
- Concurrency tests: Turn number races, active context
- Performance tests: Validate NFR targets

**NFR-4: Logging**
- Request/response logging
- Query logging (optional for dev)
- Error logging with stack traces

---

## 10. Success Criteria

1. ✅ Dewey MCP server runs in Docker (port 9020, WebSocket)
2. ✅ Connects to PostgreSQL on Irina (database: winni)
3. ✅ All 11 MCP tools callable from Claude Code
4. ✅ Semi-automatic conversation logging via CLAUDE.md instructions
5. ✅ Startup context workflow functional
6. ✅ Search returns ranked results with conversation metadata
7. ✅ Test suite passes (>80% coverage)
8. ✅ Performance targets met

---

## 11. Phase 2 Roadmap

**Defer to Phase 2 (Post-MVP):**
- Fiedler integration (store orchestration results in Winni)
- Fiedler MCP tools: `dewey_store_fiedler_result`, `dewey_get_fiedler_results`
- Link Fiedler results to conversations (optional FK)

**Future Phases:**
- Phase 3: Semantic search (pgvector + embeddings)
- Phase 4: Tiered storage (if data volume grows)
- Phase 5: Full Paper 12 implementation

---

## 12. Changes Summary

### From v1 → v2
- ✅ PostgreSQL from day 1 (not SQLite)
- ✅ Docker MCP server (port 9020)
- ✅ Removed auth/encryption/multi-user
- ✅ Fixed schema (removed storage_type, added triggers)
- ✅ Added missing tools (delete, list, pagination, bulk)

### From v2 → v3 (Final)
- ✅ Fixed MCP transport (WebSocket not stdio)
- ✅ Added `messages.metadata` column
- ✅ Added `CREATE EXTENSION pgcrypto`
- ✅ Added `role = 'tool'` option
- ✅ Added `dewey_begin_conversation` helper tool
- ✅ Clarified turn number transaction logic
- ✅ Specified set_active transaction logic
- ✅ Added semi-automatic data capture strategy (CLAUDE.md)
- ✅ Port binding to localhost only (127.0.0.1:9020:9020)
- ✅ Added force parameter to delete operations
- ✅ Confirmed Fiedler integration is Phase 2

---

## 13. Triplet Approval Summary

**Gemini 2.5 Pro:** APPROVED WITH MINOR CHANGES
- Recommended client-side helper (context manager)
- PostgreSQL choice validated
- No critical issues

**GPT-5:** APPROVED WITH MINOR CHANGES
- Identified transport/metadata issues (fixed in v3)
- Suggested CLAUDE.md automatic logging (adopted)
- Recommended `dewey_begin_conversation` (adopted)

**Grok-4:** APPROVED WITH MINOR CHANGES
- Validated simplification approach
- Confirmed PostgreSQL choice
- No critical issues

---

**STATUS: APPROVED FOR IMPLEMENTATION**

All triplet feedback incorporated. Ready for code implementation.
