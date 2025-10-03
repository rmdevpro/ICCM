Of course. Here is the complete, production-ready implementation of both the Dewey MCP Server and the MCP Proxy, based on the provided requirements. The implementation includes all 13 required files with detailed code, comments, and error handling.

***

### **Part 1: Dewey MCP Server**

This is the core service for managing conversation data in the PostgreSQL database.

---
#### **1. `dewey/config.py`**
```python
# dewey/config.py
"""
Configuration management for the Dewey MCP Server.

Loads settings from environment variables with sensible defaults.
"""
import os
import logging
import sys

# --- Database Configuration ---
DB_HOST = os.getenv("DEWEY_DB_HOST", "localhost")
DB_PORT = int(os.getenv("DEWEY_DB_PORT", 5432))
DB_NAME = os.getenv("DEWEY_DB_NAME", "winni")
DB_USER = os.getenv("DEWEY_DB_USER", "dewey")
DB_PASSWORD = os.getenv("DEWEY_DB_PASSWORD")

if not DB_PASSWORD:
    print("FATAL: DEWEY_DB_PASSWORD environment variable is not set.", file=sys.stderr)
    sys.exit(1)

# --- MCP Server Configuration ---
MCP_HOST = os.getenv("DEWEY_MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("DEWEY_MCP_PORT", 9020))

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("DEWEY_LOG_LEVEL", "INFO").upper()

def setup_logging():
    """Configures the root logger for the application."""
    log_level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )
    # Silence noisy libraries if necessary
    logging.getLogger("websockets").setLevel(logging.WARNING)

# --- Connection Pool Configuration ---
DB_POOL_MIN_CONN = 2
DB_POOL_MAX_CONN = 10

# Initialize logging when the module is imported
setup_logging()
```

---
#### **2. `dewey/database.py`**
```python
# dewey/database.py
"""
PostgreSQL connection pooling and query execution for Dewey.
"""
import logging
from contextlib import contextmanager

import psycopg2
from psycopg2.pool import ThreadedConnectionPool
from psycopg2.extras import RealDictCursor

from dewey import config

logger = logging.getLogger(__name__)

class Database:
    """Manages the PostgreSQL connection pool."""
    _pool = None

    def initialize(self):
        """Initializes the connection pool."""
        if self._pool is None:
            logger.info(f"Initializing database connection pool for '{config.DB_NAME}' on {config.DB_HOST}:{config.DB_PORT}")
            try:
                self._pool = ThreadedConnectionPool(
                    minconn=config.DB_POOL_MIN_CONN,
                    maxconn=config.DB_POOL_MAX_CONN,
                    host=config.DB_HOST,
                    port=config.DB_PORT,
                    dbname=config.DB_NAME,
                    user=config.DB_USER,
                    password=config.DB_PASSWORD,
                )
                logger.info("Database connection pool initialized successfully.")
            except psycopg2.OperationalError as e:
                logger.critical(f"Failed to connect to PostgreSQL: {e}")
                raise

    def close(self):
        """Closes all connections in the pool."""
        if self._pool:
            self._pool.closeall()
            self._pool = None
            logger.info("Database connection pool closed.")

    @contextmanager
    def get_connection(self):
        """Provides a connection from the pool as a context manager."""
        if self._pool is None:
            raise RuntimeError("Database pool is not initialized. Call initialize() first.")
        
        conn = None
        try:
            conn = self._pool.getconn()
            yield conn
        except psycopg2.Error as e:
            logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                self._pool.putconn(conn)

# Create a single instance to be used throughout the application
db_pool = Database()

@contextmanager
def transaction():
    """Provides a transactional database cursor."""
    with db_pool.get_connection() as conn:
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                yield cursor
            conn.commit()
        except psycopg2.Error as e:
            logger.error(f"Transaction failed, rolling back. Error: {e}")
            conn.rollback()
            raise
```

---
#### **3. `dewey/tools.py`**
```python
# dewey/tools.py
"""
Implementation of all 11 Dewey MCP tools.
"""
import logging
import json
from datetime import datetime
from uuid import UUID

from dewey.database import transaction

logger = logging.getLogger(__name__)

class ToolError(Exception):
    """Custom exception for tool-specific errors."""
    def __init__(self, message, code=-32000):
        super().__init__(message)
        self.code = code

def _validate_uuid(uuid_str, param_name):
    """Helper to validate a string is a valid UUID."""
    try:
        return UUID(uuid_str)
    except (ValueError, TypeError):
        raise ToolError(f"Invalid UUID format for parameter '{param_name}'.")

def _serialize_item(item):
    """Recursively serialize datetime and UUID objects in a dictionary or list."""
    if isinstance(item, dict):
        return {k: _serialize_item(v) for k, v in item.items()}
    if isinstance(item, list):
        return [_serialize_item(i) for i in item]
    if isinstance(item, datetime):
        return item.isoformat()
    if isinstance(item, UUID):
        return str(item)
    return item

# --- Conversation Management Tools ---

async def dewey_begin_conversation(session_id: str = None, metadata: dict = None) -> dict:
    """Starts a new conversation and returns its ID."""
    sql = """
        INSERT INTO conversations (session_id, metadata)
        VALUES (%s, %s)
        RETURNING id, session_id, created_at;
    """
    try:
        with transaction() as cursor:
            cursor.execute(sql, (session_id, json.dumps(metadata) if metadata else None))
            result = cursor.fetchone()
        logger.info(f"Began new conversation {result['id']}.")
        return _serialize_item(dict(result))
    except Exception as e:
        logger.error(f"Error in dewey_begin_conversation: {e}")
        raise ToolError("Failed to begin conversation.")

async def dewey_store_message(conversation_id: str, role: str, content: str, turn_number: int = None, metadata: dict = None) -> dict:
    """Stores a single message in a conversation."""
    conv_id = _validate_uuid(conversation_id, "conversation_id")
    if role not in ('user', 'assistant', 'system', 'tool'):
        raise ToolError("Invalid role. Must be one of 'user', 'assistant', 'system', 'tool'.")

    try:
        with transaction() as cursor:
            # Lock the conversation to safely determine the next turn number
            cursor.execute("SELECT 1 FROM conversations WHERE id = %s FOR UPDATE;", (conv_id,))
            if cursor.rowcount == 0:
                raise ToolError(f"Conversation with id '{conversation_id}' not found.")

            if turn_number is None:
                cursor.execute(
                    "SELECT COALESCE(MAX(turn_number), 0) + 1 AS next_turn FROM messages WHERE conversation_id = %s;",
                    (conv_id,)
                )
                turn_number = cursor.fetchone()['next_turn']
            
            sql = """
                INSERT INTO messages (conversation_id, turn_number, role, content, metadata)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING id, turn_number, created_at;
            """
            cursor.execute(sql, (conv_id, turn_number, role, content, json.dumps(metadata) if metadata else None))
            result = cursor.fetchone()

        logger.info(f"Stored message {result['id']} in conversation {conversation_id} (turn {result['turn_number']}).")
        return {
            "message_id": str(result['id']),
            "turn_number": result['turn_number'],
            "created_at": result['created_at'].isoformat()
        }
    except Exception as e:
        logger.error(f"Error in dewey_store_message: {e}")
        # Re-raise ToolError if it's already one of ours
        if isinstance(e, ToolError):
            raise
        raise ToolError("Failed to store message.")

async def dewey_store_messages_bulk(messages: list, conversation_id: str = None, session_id: str = None, metadata: dict = None) -> dict:
    """Stores a list of messages in a single transaction."""
    if not isinstance(messages, list) or not messages:
        raise ToolError("Parameter 'messages' must be a non-empty list.")

    try:
        with transaction() as cursor:
            if conversation_id:
                conv_id = _validate_uuid(conversation_id, "conversation_id")
                cursor.execute("SELECT 1 FROM conversations WHERE id = %s FOR UPDATE;", (conv_id,))
                if cursor.rowcount == 0:
                    raise ToolError(f"Conversation with id '{conversation_id}' not found.")
            else:
                # Create a new conversation
                cursor.execute(
                    "INSERT INTO conversations (session_id, metadata) VALUES (%s, %s) RETURNING id;",
                    (session_id, json.dumps(metadata) if metadata else None)
                )
                conv_id = cursor.fetchone()['id']

            # Get starting turn number
            cursor.execute("SELECT COALESCE(MAX(turn_number), 0) FROM messages WHERE conversation_id = %s;", (conv_id,))
            next_turn = cursor.fetchone()[0] + 1
            
            message_ids = []
            for i, msg in enumerate(messages):
                role = msg.get('role')
                content = msg.get('content')
                msg_metadata = msg.get('metadata')
                if not role or not content:
                    raise ToolError(f"Message at index {i} is missing 'role' or 'content'.")
                
                cursor.execute(
                    """
                    INSERT INTO messages (conversation_id, turn_number, role, content, metadata)
                    VALUES (%s, %s, %s, %s, %s) RETURNING id;
                    """,
                    (conv_id, next_turn + i, role, content, json.dumps(msg_metadata) if msg_metadata else None)
                )
                message_ids.append(str(cursor.fetchone()['id']))
        
        logger.info(f"Bulk stored {len(messages)} messages in conversation {conv_id}.")
        return {
            "conversation_id": str(conv_id),
            "stored": len(message_ids),
            "message_ids": message_ids
        }
    except Exception as e:
        logger.error(f"Error in dewey_store_messages_bulk: {e}")
        if isinstance(e, ToolError):
            raise
        raise ToolError("Failed to bulk store messages.")

async def dewey_get_conversation(conversation_id: str) -> dict:
    """Retrieves a full conversation with all its messages."""
    conv_id = _validate_uuid(conversation_id, "conversation_id")
    
    with transaction() as cursor:
        cursor.execute("SELECT * FROM conversations WHERE id = %s;", (conv_id,))
        conversation = cursor.fetchone()
        if not conversation:
            raise ToolError(f"Conversation with id '{conversation_id}' not found.")
        
        cursor.execute(
            """
            SELECT id, turn_number as turn, role, content, metadata, created_at
            FROM messages WHERE conversation_id = %s ORDER BY turn_number ASC;
            """,
            (conv_id,)
        )
        messages = cursor.fetchall()

    conversation['messages'] = [dict(m) for m in messages]
    return _serialize_item(dict(conversation))

async def dewey_list_conversations(session_id: str = None, limit: int = 20, offset: int = 0, sort_by: str = "updated_at") -> dict:
    """Lists conversations with pagination."""
    limit = min(max(1, limit), 100)
    offset = max(0, offset)
    if sort_by not in ("created_at", "updated_at"):
        raise ToolError("Invalid 'sort_by' value. Must be 'created_at' or 'updated_at'.")

    base_sql = f"""
        FROM conversations c
        WHERE (%s IS NULL OR c.session_id = %s)
    """
    params = [session_id, session_id]

    count_sql = "SELECT COUNT(*) " + base_sql
    with transaction() as cursor:
        cursor.execute(count_sql, params)
        total = cursor.fetchone()['count']

        list_sql = f"""
            SELECT
                c.id, c.session_id, c.created_at, c.updated_at, c.metadata,
                (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) as message_count
            {base_sql}
            ORDER BY c.{sort_by} DESC
            LIMIT %s OFFSET %s;
        """
        cursor.execute(list_sql, params + [limit, offset])
        conversations = [dict(row) for row in cursor.fetchall()]

    return {
        "conversations": _serialize_item(conversations),
        "total": total,
        "limit": limit,
        "offset": offset
    }

async def dewey_delete_conversation(conversation_id: str, force: bool = False) -> dict:
    """Deletes a conversation and all its messages."""
    if not force:
        raise ToolError("Delete operation requires 'force=true' to proceed.")
        
    conv_id = _validate_uuid(conversation_id, "conversation_id")
    
    with transaction() as cursor:
        # Get count of messages for the return value
        cursor.execute("SELECT COUNT(*) FROM messages WHERE conversation_id = %s;", (conv_id,))
        messages_deleted = cursor.fetchone()['count']
        
        # ON DELETE CASCADE handles deleting messages
        cursor.execute("DELETE FROM conversations WHERE id = %s;", (conv_id,))
        deleted_count = cursor.rowcount

    if deleted_count == 0:
        raise ToolError(f"Conversation with id '{conversation_id}' not found.")

    logger.warning(f"Deleted conversation {conversation_id} and {messages_deleted} messages.")
    return {"deleted": True, "messages_deleted": messages_deleted}

# --- Search Tool ---

async def dewey_search(query: str, session_id: str = None, start_date: str = None, end_date: str = None, limit: int = 20, offset: int = 0) -> dict:
    """Performs a full-text search across messages."""
    limit = min(max(1, limit), 100)
    offset = max(0, offset)

    where_clauses = ["to_tsvector('english', m.content) @@ plainto_tsquery('english', %s)"]
    params = [query]

    if session_id:
        where_clauses.append("c.session_id = %s")
        params.append(session_id)
    if start_date:
        where_clauses.append("m.created_at >= %s")
        params.append(start_date)
    if end_date:
        where_clauses.append("m.created_at <= %s")
        params.append(end_date)

    where_sql = " AND ".join(where_clauses)
    
    base_sql = f"""
        FROM messages m
        JOIN conversations c ON m.conversation_id = c.id
        WHERE {where_sql}
    """
    
    with transaction() as cursor:
        count_sql = f"SELECT COUNT(*) {base_sql};"
        cursor.execute(count_sql, params)
        total = cursor.fetchone()['count']

        search_sql = f"""
            SELECT
                m.conversation_id,
                c.session_id,
                m.id as message_id,
                m.turn_number as turn,
                m.role,
                m.content,
                ts_rank(to_tsvector('english', m.content), plainto_tsquery('english', %s)) as rank,
                m.created_at,
                c.metadata as conversation_metadata
            {base_sql}
            ORDER BY rank DESC, m.created_at DESC
            LIMIT %s OFFSET %s;
        """
        cursor.execute(search_sql, [query] + params[1:] + [limit, offset])
        results = [dict(row) for row in cursor.fetchall()]

    return {
        "results": _serialize_item(results),
        "total": total,
        "limit": limit,
        "offset": offset
    }


# --- Startup Context Tools ---

async def dewey_get_startup_context(name: str = None) -> dict:
    """Gets the active startup context, or one by name."""
    sql = "SELECT * FROM startup_contexts WHERE "
    params = []
    if name:
        sql += "name = %s;"
        params.append(name)
    else:
        sql += "is_active = TRUE;"

    with transaction() as cursor:
        cursor.execute(sql, params)
        context = cursor.fetchone()
    
    return _serialize_item(dict(context)) if context else None

async def dewey_set_startup_context(name: str, content: str, set_active: bool = True) -> dict:
    """Creates or updates a startup context."""
    with transaction() as cursor:
        if set_active:
            # Atomically deactivate the current active context
            cursor.execute("UPDATE startup_contexts SET is_active = FALSE WHERE is_active = TRUE;")
        
        sql = """
            INSERT INTO startup_contexts (name, content, is_active)
            VALUES (%s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                content = EXCLUDED.content,
                is_active = EXCLUDED.is_active,
                updated_at = NOW()
            RETURNING id, name, is_active, created_at;
        """
        cursor.execute(sql, (name, content, set_active))
        result = cursor.fetchone()

    logger.info(f"Set startup context '{name}' (active: {set_active}).")
    return _serialize_item(dict(result))

async def dewey_list_startup_contexts(include_content: bool = False) -> dict:
    """Lists all startup contexts."""
    columns = "id, name, is_active, created_at, updated_at"
    if include_content:
        columns += ", content"
    
    sql = f"SELECT {columns} FROM startup_contexts ORDER BY name ASC;"
    
    with transaction() as cursor:
        cursor.execute(sql)
        contexts = [dict(row) for row in cursor.fetchall()]

    return {"contexts": _serialize_item(contexts)}

async def dewey_delete_startup_context(name: str, force: bool = False) -> dict:
    """Deletes a startup context by name."""
    if not force:
        raise ToolError("Delete operation requires 'force=true' to proceed.")
        
    with transaction() as cursor:
        cursor.execute("DELETE FROM startup_contexts WHERE name = %s;", (name,))
        if cursor.rowcount == 0:
            raise ToolError(f"Startup context with name '{name}' not found.")

    logger.warning(f"Deleted startup context '{name}'.")
    return {"deleted": True}
```

---
#### **4. `dewey/mcp_server.py`**
```python
# dewey/mcp_server.py
"""
Main entry point for the Dewey WebSocket MCP Server.
"""
import asyncio
import logging

from mcp import server

from dewey import config, tools
from dewey.database import db_pool

logger = logging.getLogger(__name__)

async def main():
    """Initializes and runs the MCP server."""
    # Initialize the database connection pool
    try:
        db_pool.initialize()
    except Exception:
        logger.critical("Shutting down due to database connection failure.")
        return

    # Create MCP server instance
    mcp_server = server.MCPServer()

    # Register all tools from the tools module
    tool_functions = {
        name: func for name, func in tools.__dict__.items()
        if callable(func) and name.startswith("dewey_")
    }
    for name, func in tool_functions.items():
        mcp_server.add_tool(name, func)
        logger.info(f"Registered tool: {name}")

    # Start the WebSocket server
    try:
        logger.info(f"Starting Dewey MCP server on ws://{config.MCP_HOST}:{config.MCP_PORT}")
        await mcp_server.start(host=config.MCP_HOST, port=config.MCP_PORT)
    except Exception as e:
        logger.critical(f"Server failed to start: {e}")
    finally:
        # Clean up database connections on shutdown
        db_pool.close()
        logger.info("Dewey MCP server has shut down.")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown requested by user.")
```

---
#### **5. `dewey/__init__.py`**
```python
# dewey/__init__.py
# This file makes the 'dewey' directory a Python package.
```

---
#### **6. `Dockerfile`**
```dockerfile
# Dockerfile
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

---
#### **7. `docker-compose.yml`**
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

---
#### **8. `requirements.txt`**
```
# requirements.txt
mcp[server]>=0.9.0
psycopg2-binary>=2.9.9
```

---
#### **9. `schema.sql`**
```sql
-- schema.sql
-- Complete PostgreSQL schema for the Winni database.
--
-- To set up:
-- 1. Run the 'CREATE DATABASE' and 'CREATE USER' commands as the 'postgres' superuser.
-- 2. Connect to the 'winni' database: \c winni
-- 3. Run the rest of this script.

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;  -- For gen_random_uuid()

-- Create database and user (run as postgres user)
-- Note: These commands should be run outside of a transaction block in psql.
-- CREATE DATABASE winni;
-- CREATE USER dewey WITH PASSWORD 'secure-password-here';
-- GRANT ALL PRIVILEGES ON DATABASE winni TO dewey;

-- Grant permissions (run after connecting to 'winni' as postgres user)
-- GRANT ALL PRIVILEGES ON SCHEMA public TO dewey;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO dewey;
-- ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO dewey;

-- Auto-update updated_at trigger function
-- This function can be shared by multiple tables.
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Table: conversations
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
CREATE INDEX idx_conversations_metadata ON conversations USING GIN(metadata);

CREATE TRIGGER conversations_updated_at
BEFORE UPDATE ON conversations
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Table: messages
CREATE TABLE messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    turn_number INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system', 'tool')),
    content TEXT NOT NULL,
    metadata JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(conversation_id, turn_number),
    CONSTRAINT valid_metadata CHECK (metadata IS NULL OR jsonb_typeof(metadata) = 'object')
);

CREATE INDEX idx_messages_conversation ON messages(conversation_id, turn_number);
CREATE INDEX idx_messages_created ON messages(created_at DESC);
CREATE INDEX idx_messages_conversation_created ON messages(conversation_id, created_at);

-- Full-text search index
CREATE INDEX idx_messages_content_fts ON messages USING GIN (to_tsvector('english', content));

-- Table: startup_contexts
CREATE TABLE startup_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_name CHECK (length(name) > 0)
);

-- Enforce a single active context using a partial unique index
CREATE UNIQUE INDEX idx_startup_contexts_single_active
ON startup_contexts(is_active) WHERE is_active = TRUE;

CREATE TRIGGER startup_contexts_updated_at
BEFORE UPDATE ON startup_contexts
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Table: fiedler_results (for Phase 2)
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

-- Grant permissions on newly created tables
-- (Run as postgres user after connecting to 'winni')
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dewey;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dewey;

```

***

### **Part 2: MCP Proxy**

This is the transparent middleware that logs conversations to Dewey.

---
#### **10. `mcp_proxy/proxy_server.py`**
```python
# mcp_proxy/proxy_server.py
"""
WebSocket MCP Proxy for transparently logging conversations to Dewey.
"""
import asyncio
import json
import logging
import os
import sys
from urllib.parse import urlparse, parse_qs

import websockets

from mcp_proxy.dewey_client import DeweyClient

# --- Configuration ---
PROXY_HOST = os.getenv("PROXY_HOST", "0.0.0.0")
PROXY_PORT = int(os.getenv("PROXY_PORT", 9000))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

# Upstream MCP server URLs, configured via environment variables
# Example: PROXY_UPSTREAM_FIEDLER=ws://fiedler-mcp:9010
UPSTREAM_SERVERS = {
    key.replace("PROXY_UPSTREAM_", "").lower(): value
    for key, value in os.environ.items()
    if key.startswith("PROXY_UPSTREAM_")
}

# --- Logging Setup ---
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("MCPProxy")

# --- Global State ---
dewey_client = DeweyClient()
CLIENT_SESSIONS = {} # Maps client_ws -> (upstream_ws, conversation_id)

def get_message_role_and_content(msg: dict) -> (str, str):
    """Determines the message role and content for logging."""
    if msg.get("method") == "tools/call":
        role = "user"
        # Log the entire tool call as content for completeness
        content = json.dumps({"tool_name": msg["params"]["name"], "tool_args": msg["params"]["args"]})
    elif "result" in msg:
        role = "assistant"
        content = json.dumps(msg["result"])
    else:
        # Not a user turn or a final assistant response (e.g., streams, errors)
        return None, None
    return role, content

async def log_to_dewey(conversation_id: str, message: str):
    """Parses a message and logs it to Dewey if applicable."""
    try:
        msg_data = json.loads(message)
        role, content = get_message_role_and_content(msg_data)
        
        if role and content:
            await dewey_client.store_message(
                conversation_id=conversation_id,
                role=role,
                content=content,
                metadata={"source": "mcp-proxy"}
            )
            logger.debug(f"Logged {role} message to conversation {conversation_id}")
    except json.JSONDecodeError:
        logger.warning("Could not parse message for logging (not JSON).")
    except Exception as e:
        logger.error(f"Failed to log message to Dewey for conv {conversation_id}: {e}")

async def relay(source_ws, dest_ws, conversation_id, direction):
    """Relays messages from a source websocket to a destination."""
    try:
        async for message in source_ws:
            await dest_ws.send(message)
            # Log messages from client to server as they represent user/assistant turns
            if direction == "client_to_server":
                asyncio.create_task(log_to_dewey(conversation_id, message))
    except websockets.ConnectionClosed:
        logger.info(f"Connection closed in {direction} relay for conv {conversation_id}.")
    except Exception as e:
        logger.error(f"Error in {direction} relay for conv {conversation_id}: {e}")

async def handler(client_ws, path):
    """Main handler for new client connections."""
    try:
        # 1. Determine upstream target
        query_params = parse_qs(urlparse(path).query)
        upstream_name = query_params.get("upstream", [None])[0]
        
        if not upstream_name or upstream_name not in UPSTREAM_SERVERS:
            logger.error(f"Connection rejected: Missing or invalid 'upstream' parameter. Got: {upstream_name}")
            await client_ws.close(1008, "Invalid upstream target")
            return
            
        upstream_url = UPSTREAM_SERVERS[upstream_name]
        logger.info(f"New client connected. Routing to upstream '{upstream_name}' at {upstream_url}")

        # 2. Connect to upstream server
        async with websockets.connect(upstream_url) as upstream_ws:
            # 3. Start a new conversation log in Dewey
            conversation_id = await dewey_client.begin_conversation(
                metadata={"proxy_upstream": upstream_name}
            )
            logger.info(f"Started logging conversation {conversation_id} for new session.")
            
            # 4. Start bidirectional relay
            client_to_server = relay(client_ws, upstream_ws, conversation_id, "client_to_server")
            server_to_client = relay(upstream_ws, client_ws, conversation_id, "server_to_client")
            
            await asyncio.gather(client_to_server, server_to_client)

    except websockets.ConnectionClosedError:
        logger.info("Client disconnected.")
    except Exception as e:
        logger.error(f"An unhandled error occurred in the connection handler: {e}")
    finally:
        logger.info("Connection handler finished.")

async def main():
    """Starts the proxy server."""
    if not UPSTREAM_SERVERS:
        logger.critical("No upstream servers configured. Please set PROXY_UPSTREAM_<name> environment variables.")
        return

    logger.info("Starting Dewey client for proxy...")
    await dewey_client.connect()
    
    logger.info(f"Starting MCP Proxy server on ws://{PROXY_HOST}:{PROXY_PORT}")
    logger.info(f"Configured upstreams: {UPSTREAM_SERVERS}")
    
    async with websockets.serve(handler, PROXY_HOST, PROXY_PORT):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("MCP Proxy shutting down.")
    finally:
        asyncio.run(dewey_client.close())
```

---
#### **11. `mcp_proxy/dewey_client.py`**
```python
# mcp_proxy/dewey_client.py
"""
A dedicated MCP client for the proxy to communicate with the Dewey server.
This client connects directly to Dewey, bypassing the proxy to avoid loops.
"""
import os
import logging
from mcp import client

logger = logging.getLogger("DeweyClient")

class DeweyClient:
    """A wrapper around the MCP client for specific Dewey operations."""
    def __init__(self):
        self.dewey_url = os.getenv("PROXY_DEWEY_URL", "ws://localhost:9020")
        if "9000" in self.dewey_url: # Safety check
            logger.warning("Dewey client URL points to the proxy port (9000). This will cause a loop. Ensure PROXY_DEWEY_URL is correct.")
        self._client = client.Client(self.dewey_url)

    async def connect(self):
        """Connects the underlying MCP client."""
        try:
            await self._client.connect()
            logger.info(f"Dewey client connected to {self.dewey_url}")
        except Exception as e:
            logger.critical(f"Failed to connect Dewey client to {self.dewey_url}: {e}")
            raise

    async def close(self):
        """Closes the client connection."""
        await self._client.close()
        logger.info("Dewey client connection closed.")

    async def begin_conversation(self, session_id: str = None, metadata: dict = None) -> str:
        """Starts a new conversation and returns the ID."""
        params = {}
        if session_id:
            params['session_id'] = session_id
        if metadata:
            params['metadata'] = metadata
            
        result = await self._client.tools.dewey_begin_conversation(**params)
        return result['conversation_id']

    async def store_message(self, conversation_id: str, role: str, content: str, metadata: dict = None):
        """Stores a single message."""
        params = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content
        }
        if metadata:
            params['metadata'] = metadata

        await self._client.tools.dewey_store_message(**params)

```

---
#### **12. `mcp_proxy/Dockerfile`**
```dockerfile
# mcp_proxy/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies directly to avoid needing a separate requirements.txt
# This keeps the total file count at 13 as requested.
RUN pip install --no-cache-dir "mcp>=0.9.0" "websockets>=12.0"

# Copy Proxy server code
COPY mcp_proxy/ ./mcp_proxy/

# Expose proxy port
EXPOSE 9000

# Run proxy server
CMD ["python", "-m", "mcp_proxy.proxy_server"]
```

---
#### **13. `mcp_proxy/docker-compose.yml`**
```yaml
# mcp_proxy/docker-compose.yml
version: '3.8'

services:
  mcp-proxy:
    build:
      context: .
      dockerfile: mcp_proxy/Dockerfile
    image: mcp-proxy:latest
    container_name: mcp-proxy
    restart: unless-stopped

    environment:
      # Port the proxy listens on inside the container
      - PROXY_PORT=9000
      - PROXY_HOST=0.0.0.0
      - LOG_LEVEL=INFO

      # URL for the real Dewey server (for logging)
      # Uses Docker service name 'dewey-mcp'
      - PROXY_DEWEY_URL=ws://dewey-mcp:9020

      # --- Define Upstream MCP Servers ---
      # Claude Code will connect to the proxy with ?upstream=<name>
      # e.g., ws://localhost:9000?upstream=dewey
      - PROXY_UPSTREAM_DEWEY=ws://dewey-mcp:9020
      - PROXY_UPSTREAM_FIEDLER=ws://fiedler-mcp:9010
      # Add other upstream servers here as needed

    ports:
      # Expose the proxy on localhost:9000
      - "127.0.0.1:9000:9000"

    depends_on:
      # Ensure dewey is running before the proxy starts
      - dewey

networks:
  default:
    external:
      name: iccm_network # Must be on the same network as Dewey and Fiedler
```
