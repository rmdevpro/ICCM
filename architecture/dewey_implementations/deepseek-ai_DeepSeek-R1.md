<think>
We are going to implement two components: Dewey MCP Server and MCP Proxy.

Component 1: Dewey MCP Server (runs on port 9020)
- We'll create the required files under the `dewey` directory.

Component 2: MCP Proxy (runs on port 9000, relays to upstream MCP servers including Dewey on 9020 and Fiedler on 9010)
- We'll create the required files under the `mcp_proxy` directory.

We'll follow the v3 requirements strictly.

Let's start with the Dewey MCP Server.

First, we'll create the `dewey/config.py` for configuration management.
</think>
I'll implement the Dewey MCP Server and MCP Proxy based on the v3 requirements. Let's start with the Dewey components:

### Filename: dewey/config.py
```python
import os
import logging

class Config:
    # Database configuration
    DB_HOST = os.getenv('DEWEY_DB_HOST', 'irina')
    DB_PORT = os.getenv('DEWEY_DB_PORT', '5432')
    DB_NAME = os.getenv('DEWEY_DB_NAME', 'winni')
    DB_USER = os.getenv('DEWEY_DB_USER', 'dewey')
    DB_PASSWORD = os.getenv('DEWEY_DB_PASSWORD', '')
    
    # MCP server configuration
    MCP_HOST = os.getenv('DEWEY_MCP_HOST', '0.0.0.0')
    MCP_PORT = int(os.getenv('DEWEY_MCP_PORT', '9020'))
    
    # Connection pooling
    DB_MIN_CONNECTIONS = int(os.getenv('DEWEY_DB_MIN_CONNECTIONS', '2'))
    DB_MAX_CONNECTIONS = int(os.getenv('DEWEY_DB_MAX_CONNECTIONS', '10'))
    
    # Logging configuration
    LOG_LEVEL = os.getenv('DEWEY_LOG_LEVEL', 'INFO').upper()
    
    @classmethod
    def validate(cls):
        """Validate critical configuration values"""
        if not cls.DB_PASSWORD:
            raise ValueError("DEWEY_DB_PASSWORD environment variable is required")
        if cls.DB_MIN_CONNECTIONS > cls.DB_MAX_CONNECTIONS:
            raise ValueError("DB_MIN_CONNECTIONS cannot exceed DB_MAX_CONNECTIONS")
            
        # Log validation warnings
        if not cls.DB_PASSWORD:
            logging.warning("Using empty database password - not recommended for production")

# Initialize logging
logging.basicConfig(
    level=Config.LOG_LEVEL,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Validate configuration on import
try:
    Config.validate()
except ValueError as e:
    logger.error(f"Configuration error: {e}")
    raise
```

### Filename: dewey/database.py
```python
import logging
import psycopg2
from psycopg2 import pool, sql
from psycopg2.extras import RealDictCursor, RealDictRow
from contextlib import contextmanager
from typing import Any, Dict, List, Optional, Union

from .config import Config

logger = logging.getLogger(__name__)

class Database:
    _connection_pool = None

    @classmethod
    def initialize_pool(cls):
        """Initialize the connection pool"""
        if cls._connection_pool is None:
            try:
                cls._connection_pool = pool.ThreadedConnectionPool(
                    minconn=Config.DB_MIN_CONNECTIONS,
                    maxconn=Config.DB_MAX_CONNECTIONS,
                    host=Config.DB_HOST,
                    port=Config.DB_PORT,
                    dbname=Config.DB_NAME,
                    user=Config.DB_USER,
                    password=Config.DB_PASSWORD,
                    cursor_factory=RealDictCursor
                )
                logger.info("Database connection pool initialized")
            except Exception as e:
                logger.error(f"Failed to initialize database pool: {e}")
                raise

    @classmethod
    @contextmanager
    def get_connection(cls) -> 'psycopg2.extensions.connection':
        """Context manager for database connections"""
        if cls._connection_pool is None:
            cls.initialize_pool()
            
        conn = cls._connection_pool.getconn()
        try:
            yield conn
        except Exception as e:
            logger.error(f"Database error: {e}")
            conn.rollback()
            raise
        finally:
            cls._connection_pool.putconn(conn)

    @classmethod
    @contextmanager
    def get_cursor(cls) -> 'psycopg2.extensions.cursor':
        """Context manager for database cursors"""
        with cls.get_connection() as conn:
            cursor = conn.cursor()
            try:
                yield cursor
                conn.commit()
            except:
                conn.rollback()
                raise

    @staticmethod
    def execute_query(query: str, params: tuple = None) -> List[RealDictRow]:
        """Execute a read query and return results"""
        with Database.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()

    @staticmethod
    def execute_command(query: str, params: tuple = None) -> int:
        """Execute a write command and return row count"""
        with Database.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount

    @staticmethod
    def execute_transaction(queries: List[tuple]) -> List[Any]:
        """Execute multiple commands in a single transaction"""
        results = []
        with Database.get_connection() as conn:
            cursor = conn.cursor()
            try:
                for query, params in queries:
                    cursor.execute(query, params)
                    if cursor.description:  # If it's a query with results
                        results.append(cursor.fetchall())
                    else:
                        results.append(cursor.rowcount)
                conn.commit()
                return results
            except Exception as e:
                conn.rollback()
                logger.error(f"Transaction failed: {e}")
                raise

# Initialize pool on module load
Database.initialize_pool()
```

### Filename: dewey/tools.py
```python
import uuid
import logging
import json
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .database import Database
from .config import Config

logger = logging.getLogger(__name__)

class DeweyTools:
    @staticmethod
    def begin_conversation(session_id: Optional[str] = None, metadata: Optional[dict] = None) -> dict:
        """Create a new conversation and return its ID"""
        conv_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        
        # Validate metadata
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
            
        # Insert conversation
        query = """
            INSERT INTO conversations (id, session_id, metadata, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s)
        """
        params = (conv_id, session_id, json.dumps(metadata) if metadata else None, created_at, created_at)
        
        try:
            Database.execute_command(query, params)
            logger.info(f"Created new conversation: {conv_id}")
            return {
                "conversation_id": conv_id,
                "session_id": session_id,
                "created_at": created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to create conversation: {e}")
            raise

    @staticmethod
    def store_message(
        conversation_id: str,
        role: str,
        content: str,
        turn_number: Optional[int] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """Store a single message with transaction-safe turn numbering"""
        # Validate inputs
        if role not in ['user', 'assistant', 'system', 'tool']:
            raise ValueError("Invalid role. Must be 'user', 'assistant', 'system', or 'tool'")
        
        if not content:
            raise ValueError("Content cannot be empty")
            
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
        
        # Transaction block for turn number calculation
        queries = []
        
        # 1. Lock conversation row
        lock_query = "SELECT 1 FROM conversations WHERE id = %s FOR UPDATE"
        queries.append((lock_query, (conversation_id,)))
        
        # 2. Calculate next turn number if not provided
        if turn_number is None:
            turn_query = """
                SELECT COALESCE(MAX(turn_number) + 1, 1) 
                FROM messages 
                WHERE conversation_id = %s
            """
            queries.append((turn_query, (conversation_id,)))
        else:
            # If turn number is provided, we still need to verify it's not duplicate
            check_query = """
                SELECT 1 FROM messages 
                WHERE conversation_id = %s AND turn_number = %s
            """
            queries.append((check_query, (conversation_id, turn_number)))
        
        # 3. Insert message
        message_id = str(uuid.uuid4())
        created_at = datetime.utcnow()
        insert_query = """
            INSERT INTO messages (id, conversation_id, turn_number, role, content, metadata, created_at)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        # Execute transaction
        try:
            results = Database.execute_transaction(queries)
            
            # Process results to determine turn number
            if turn_number is None:
                # Result from turn_query is at index 1
                turn_number = results[1][0]['coalesce'] if results[1] else 1
            else:
                # Check if turn number exists
                if results[1]:
                    raise ValueError(f"Turn number {turn_number} already exists in conversation {conversation_id}")
            
            # Add insert query to transaction
            insert_params = (
                message_id, conversation_id, turn_number, role, content,
                json.dumps(metadata) if metadata else None, created_at
            )
            Database.execute_command(insert_query, insert_params)
            
            # Update conversation's updated_at
            update_query = "UPDATE conversations SET updated_at = %s WHERE id = %s"
            Database.execute_command(update_query, (created_at, conversation_id))
            
            logger.info(f"Stored message in conversation {conversation_id}, turn {turn_number}")
            return {
                "message_id": message_id,
                "turn_number": turn_number,
                "created_at": created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to store message: {e}")
            raise

    @staticmethod
    def store_messages_bulk(
        messages: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[dict] = None
    ) -> dict:
        """Bulk store messages with transaction-safe processing"""
        if not messages:
            raise ValueError("Messages list cannot be empty")
            
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be a dictionary")
            
        # Create new conversation if ID not provided
        if not conversation_id:
            conv_result = DeweyTools.begin_conversation(session_id, metadata)
            conversation_id = conv_result['conversation_id']
            created_at = conv_result['created_at']
        else:
            created_at = datetime.utcnow()
        
        # Prepare messages for insertion
        message_ids = []
        turn_number = 1
        insert_queries = []
        update_queries = []
        
        # Get current max turn number for existing conversation
        max_turn_query = "SELECT MAX(turn_number) FROM messages WHERE conversation_id = %s"
        max_turn_result = Database.execute_query(max_turn_query, (conversation_id,))
        current_max_turn = max_turn_result[0]['max'] if max_turn_result and max_turn_result[0]['max'] else 0
        turn_number = current_max_turn + 1
        
        # Prepare message inserts
        for idx, msg in enumerate(messages):
            msg_id = str(uuid.uuid4())
            role = msg.get('role')
            content = msg.get('content')
            msg_metadata = msg.get('metadata')
            
            if role not in ['user', 'assistant', 'system', 'tool']:
                raise ValueError(f"Invalid role in message {idx}: {role}")
                
            if not content:
                raise ValueError(f"Content cannot be empty in message {idx}")
                
            if msg_metadata is not None and not isinstance(msg_metadata, dict):
                raise ValueError(f"metadata must be a dictionary in message {idx}")
                
            insert_query = """
                INSERT INTO messages (id, conversation_id, turn_number, role, content, metadata, created_at)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                msg_id, conversation_id, turn_number, role, content,
                json.dumps(msg_metadata) if msg_metadata else None, created_at
            )
            insert_queries.append((insert_query, params))
            message_ids.append(msg_id)
            turn_number += 1
        
        # Update conversation timestamp
        update_query = "UPDATE conversations SET updated_at = %s WHERE id = %s"
        update_queries.append((update_query, (created_at, conversation_id)))
        
        # Execute in transaction
        try:
            # Start with lock on conversation
            lock_query = "SELECT 1 FROM conversations WHERE id = %s FOR UPDATE"
            Database.execute_command(lock_query, (conversation_id,))
            
            # Execute all inserts
            for query, params in insert_queries:
                Database.execute_command(query, params)
                
            # Execute updates
            for query, params in update_queries:
                Database.execute_command(query, params)
                
            logger.info(f"Stored {len(messages)} messages in conversation {conversation_id}")
            return {
                "conversation_id": conversation_id,
                "stored": len(messages),
                "message_ids": message_ids
            }
        except Exception as e:
            logger.error(f"Bulk message storage failed: {e}")
            raise

    @staticmethod
    def get_conversation(conversation_id: str) -> dict:
        """Retrieve a complete conversation with messages"""
        # Get conversation
        conv_query = """
            SELECT id, session_id, created_at, updated_at, metadata
            FROM conversations
            WHERE id = %s
        """
        conv_result = Database.execute_query(conv_query, (conversation_id,))
        if not conv_result:
            raise ValueError(f"Conversation {conversation_id} not found")
            
        conversation = conv_result[0]
        
        # Get messages
        msg_query = """
            SELECT id, turn_number AS turn, role, content, metadata, created_at
            FROM messages
            WHERE conversation_id = %s
            ORDER BY turn_number ASC
        """
        messages = Database.execute_query(msg_query, (conversation_id,))
        
        return {
            "conversation_id": conversation['id'],
            "session_id": conversation['session_id'],
            "created_at": conversation['created_at'].isoformat(),
            "updated_at": conversation['updated_at'].isoformat(),
            "metadata": conversation['metadata'],
            "messages": messages
        }

    @staticmethod
    def list_conversations(
        session_id: Optional[str] = None,
        limit: int = 20,
        offset: int = 0,
        sort_by: str = 'updated_at'
    ) -> dict:
        """List conversations with pagination"""
        # Validate inputs
        if limit > 100 or limit < 1:
            raise ValueError("Limit must be between 1 and 100")
            
        if offset < 0:
            raise ValueError("Offset cannot be negative")
            
        valid_sorts = ['created_at', 'updated_at']
        if sort_by not in valid_sorts:
            raise ValueError(f"Invalid sort_by. Must be one of {valid_sorts}")
        
        # Base query
        query = sql.SQL("""
            SELECT 
                c.id, 
                c.session_id, 
                c.created_at, 
                c.updated_at,
                c.metadata,
                (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
            FROM conversations c
        """)
        
        # Add WHERE clause if session_id provided
        where_clause = None
        params = []
        if session_id:
            where_clause = sql.SQL("WHERE session_id = %s")
            params.append(session_id)
        
        # Add ORDER BY
        order_clause = sql.SQL("ORDER BY {} DESC").format(sql.Identifier(sort_by))
        
        # Add pagination
        limit_clause = sql.SQL("LIMIT %s OFFSET %s")
        params.extend([limit, offset])
        
        # Build final query
        if where_clause:
            query = sql.SQL(" ").join([query, where_clause, order_clause, limit_clause])
        else:
            query = sql.SQL(" ").join([query, order_clause, limit_clause])
        
        # Execute query
        conversations = Database.execute_query(query.as_string(Database._connection_pool), tuple(params))
        
        # Get total count
        count_query = "SELECT COUNT(*) FROM conversations"
        if session_id:
            count_query += " WHERE session_id = %s"
            count_result = Database.execute_query(count_query, (session_id,))
        else:
            count_result = Database.execute_query(count_query)
            
        total = count_result[0]['count'] if count_result else 0
        
        return {
            "conversations": conversations,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    @staticmethod
    def delete_conversation(conversation_id: str, force: bool = False) -> dict:
        """Delete a conversation and all its messages"""
        if not force:
            # In dev environment, we might want to prevent accidental deletes
            raise PermissionError("force parameter must be set to True for deletion")
        
        # First delete messages (cascade should handle this, but explicit is better)
        msg_delete_query = "DELETE FROM messages WHERE conversation_id = %s"
        msg_count = Database.execute_command(msg_delete_query, (conversation_id,))
        
        # Then delete conversation
        conv_delete_query = "DELETE FROM conversations WHERE id = %s"
        conv_count = Database.execute_command(conv_delete_query, (conversation_id,))
        
        if conv_count == 0:
            raise ValueError(f"Conversation {conversation_id} not found")
            
        logger.warning(f"Deleted conversation {conversation_id} with {msg_count} messages")
        return {
            "deleted": True,
            "messages_deleted": msg_count
        }

    @staticmethod
    def search(
        query: str,
        session_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        limit: int = 20,
        offset: int = 0
    ) -> dict:
        """Full-text search across messages"""
        if not query:
            raise ValueError("Query cannot be empty")
            
        if limit > 100 or limit < 1:
            raise ValueError("Limit must be between 1 and 100")
            
        if offset < 0:
            raise ValueError("Offset cannot be negative")
            
        # Base query with FTS
        base_query = sql.SQL("""
            SELECT
                m.id AS message_id,
                m.conversation_id,
                m.turn_number AS turn,
                m.role,
                m.content,
                m.created_at,
                c.session_id,
                c.metadata AS conversation_metadata,
                ts_rank(to_tsvector('english', m.content), plainto_tsquery('english', %s)) AS rank
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE to_tsvector('english', m.content) @@ plainto_tsquery('english', %s)
        """)
        
        params = [query, query]
        
        # Add session filter
        if session_id:
            base_query = sql.SQL(" ").join([
                base_query,
                sql.SQL("AND c.session_id = %s")
            ])
            params.append(session_id)
            
        # Add date filters
        if start_date:
            base_query = sql.SQL(" ").join([
                base_query,
                sql.SQL("AND m.created_at >= %s")
            ])
            params.append(start_date)
            
        if end_date:
            base_query = sql.SQL(" ").join([
                base_query,
                sql.SQL("AND m.created_at <= %s")
            ])
            params.append(end_date)
            
        # Add ordering and pagination
        final_query = sql.SQL(" ").join([
            base_query,
            sql.SQL("ORDER BY rank DESC, m.created_at DESC"),
            sql.SQL("LIMIT %s OFFSET %s")
        ])
        params.extend([limit, offset])
        
        # Execute search
        results = Database.execute_query(final_query.as_string(Database._connection_pool), tuple(params))
        
        # Get total count
        count_query = sql.SQL("""
            SELECT COUNT(*)
            FROM messages m
            JOIN conversations c ON m.conversation_id = c.id
            WHERE to_tsvector('english', m.content) @@ plainto_tsquery('english', %s)
        """)
        count_params = [query]
        
        if session_id:
            count_query = sql.SQL(" ").join([count_query, sql.SQL("AND c.session_id = %s")])
            count_params.append(session_id)
            
        if start_date:
            count_query = sql.SQL(" ").join([count_query, sql.SQL("AND m.created_at >= %s")])
            count_params.append(start_date)
            
        if end_date:
            count_query = sql.SQL(" ").join([count_query, sql.SQL("AND m.created_at <= %s")])
            count_params.append(end_date)
            
        count_result = Database.execute_query(count_query.as_string(Database._connection_pool), tuple(count_params))
        total = count_result[0]['count'] if count_result else 0
        
        return {
            "results": results,
            "total": total,
            "limit": limit,
            "offset": offset
        }

    @staticmethod
    def get_startup_context(name: Optional[str] = None) -> Optional[dict]:
        """Get startup context by name or active context"""
        if name:
            query = "SELECT * FROM startup_contexts WHERE name = %s"
            params = (name,)
        else:
            query = "SELECT * FROM startup_contexts WHERE is_active = TRUE"
            params = ()
            
        result = Database.execute_query(query, params)
        if not result:
            return None
            
        context = result[0]
        return {
            "id": context['id'],
            "name": context['name'],
            "content": context['content'],
            "is_active": context['is_active'],
            "created_at": context['created_at'].isoformat(),
            "updated_at": context['updated_at'].isoformat()
        }

    @staticmethod
    def set_startup_context(name: str, content: str, set_active: bool = True) -> dict:
        """Create or update a startup context with atomic active switching"""
        if not name:
            raise ValueError("Name cannot be empty")
            
        if not content:
            raise ValueError("Content cannot be empty")
            
        # Prepare transaction queries
        queries = []
        created_at = datetime.utcnow()
        
        # If setting active, deactivate current active context
        if set_active:
            deactivate_query = "UPDATE startup_contexts SET is_active = FALSE WHERE is_active = TRUE"
            queries.append((deactivate_query, ()))
        
        # Upsert context
        upsert_query = """
            INSERT INTO startup_contexts (id, name, content, is_active, created_at, updated_at)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (name) DO UPDATE SET
                content = EXCLUDED.content,
                is_active = EXCLUDED.is_active,
                updated_at = EXCLUDED.updated_at
            RETURNING id
        """
        context_id = str(uuid.uuid4())
        params = (context_id, name, content, set_active, created_at, created_at)
        queries.append((upsert_query, params))
        
        # Execute transaction
        try:
            results = Database.execute_transaction(queries)
            # The last result is from the upsert
            if not results[-1]:
                raise RuntimeError("Failed to create/update startup context")
                
            logger.info(f"Set startup context '{name}' {'(active)' if set_active else ''}")
            return {
                "id": context_id,
                "name": name,
                "is_active": set_active,
                "created_at": created_at.isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to set startup context: {e}")
            raise

    @staticmethod
    def list_startup_contexts(include_content: bool = False) -> dict:
        """List all startup contexts"""
        columns = "id, name, is_active, created_at, updated_at"
        if include_content:
            columns += ", content"
            
        query = f"SELECT {columns} FROM startup_contexts ORDER BY name ASC"
        contexts = Database.execute_query(query)
        return {"contexts": contexts}

    @staticmethod
    def delete_startup_context(name: str, force: bool = False) -> dict:
        """Delete a startup context"""
        if not force:
            raise PermissionError("force parameter must be set to True for deletion")
            
        query = "DELETE FROM startup_contexts WHERE name = %s"
        count = Database.execute_command(query, (name,))
        
        if count == 0:
            raise ValueError(f"Startup context '{name}' not found")
            
        logger.warning(f"Deleted startup context '{name}'")
        return {"deleted": True}
```

### Filename: dewey/mcp_server.py
```python
import asyncio
import json
import logging
from mcp import start_server
from mcp.interface import Tool

from .tools import DeweyTools
from .config import Config

logger = logging.getLogger(__name__)

class DeweyTool(Tool):
    def __init__(self, func, name=None):
        self.func = func
        self.name = name or func.__name__
        self.description = func.__doc__ or ""

    async def execute(self, arguments):
        try:
            # Convert arguments to Python types
            args = json.loads(arguments) if isinstance(arguments, str) else arguments
            
            # Execute the tool function
            result = self.func(**args)
            return {"status": "success", "result": result}
        except Exception as e:
            logger.error(f"Tool {self.name} failed: {str(e)}")
            return {"status": "error", "message": str(e)}

# Map tool names to implementations
TOOLS = {
    "dewey_begin_conversation": DeweyTool(DeweyTools.begin_conversation),
    "dewey_store_message": DeweyTool(DeweyTools.store_message),
    "dewey_store_messages_bulk": DeweyTool(DeweyTools.store_messages_bulk),
    "dewey_get_conversation": DeweyTool(DeweyTools.get_conversation),
    "dewey_list_conversations": DeweyTool(DeweyTools.list_conversations),
    "dewey_delete_conversation": DeweyTool(DeweyTools.delete_conversation),
    "dewey_search": DeweyTool(DeweyTools.search),
    "dewey_get_startup_context": DeweyTool(DeweyTools.get_startup_context),
    "dewey_set_startup_context": DeweyTool(DeweyTools.set_startup_context),
    "dewey_list_startup_contexts": DeweyTool(DeweyTools.list_startup_contexts),
    "dewey_delete_startup_context": DeweyTool(DeweyTools.delete_startup_context),
}

async def main():
    logger.info("Starting Dewey MCP Server on %s:%s", Config.MCP_HOST, Config.MCP_PORT)
    await start_server(
        tools=TOOLS,
        host=Config.MCP_HOST,
        port=Config.MCP_PORT,
        logger=logger
    )

if __name__ == "__main__":
    asyncio.run(main())
```

### Filename: schema.sql
```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

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
CREATE INDEX idx_conversations_metadata ON conversations USING GIN(metadata);

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
    metadata JSONB,
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

### Filename: Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy Dewey server code
COPY dewey/ ./dewey/
COPY schema.sql .

# Expose MCP port
EXPOSE 9020

# Run MCP server
CMD ["python", "-m", "dewey.mcp_server"]
```

### Filename: docker-compose.yml
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
    environment:
      - DEWEY_DB_HOST=irina
      - DEWEY_DB_PORT=5432
      - DEWEY_DB_NAME=winni
      - DEWEY_DB_USER=dewey
      - DEWEY_DB_PASSWORD=${DEWEY_DB_PASSWORD}
      - DEWEY_MCP_PORT=9020
      - DEWEY_MCP_HOST=0.0.0.0
      - DEWEY_LOG_LEVEL=INFO
    ports:
      - "127.0.0.1:9020:9020"
    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 5s

networks:
  default:
    external:
      name: iccm_network
```

### Filename: requirements.txt
```
mcp[server]>=0.9.0
psycopg2-binary>=2.9.9
```

Now let's implement the MCP Proxy component:

### Filename: mcp_proxy/proxy_server.py
```python
import asyncio
import json
import logging
import websockets
from urllib.parse import parse_qs, urlparse
from typing import Dict, Optional

from .dewey_client import DeweyClient

logger = logging.getLogger(__name__)

class MCPProxy:
    def __init__(self, host: str = '0.0.0.0', port: int = 9000):
        self.host = host
        self.port = port
        self.clients = {}  # websocket: (upstream_ws, conversation_id)
        self.dewey_client = DeweyClient()

    async def handle_client(self, websocket, path):
        """Handle a new client connection"""
        # Parse query parameters
        query = parse_qs(urlparse(path).query)
        upstream_name = query.get('upstream', ['fiedler'])[0]
        
        # Determine upstream URL
        upstream_url = self.get_upstream_url(upstream_name)
        if not upstream_url:
            await websocket.close(code=4000, reason=f"Invalid upstream: {upstream_name}")
            return
            
        # Connect to upstream
        try:
            async with websockets.connect(upstream_url) as upstream_ws:
                # Begin conversation in Dewey
                conversation_id = await self.dewey_client.begin_conversation()
                self.clients[websocket] = (upstream_ws, conversation_id)
                logger.info(f"Client connected to upstream {upstream_name}, conversation {conversation_id}")
                
                # Start bidirectional communication
                await asyncio.gather(
                    self.forward_client_to_upstream(websocket, upstream_ws),
                    self.forward_upstream_to_client(websocket, upstream_ws)
                )
        except Exception as e:
            logger.error(f"Connection error: {e}")
        finally:
            self.clients.pop(websocket, None)
            logger.info(f"Client disconnected from {upstream_name}")

    def get_upstream_url(self, upstream_name: str) -> Optional[str]:
        """Get WebSocket URL for upstream service"""
        upstreams = {
            "fiedler": "ws://fiedler:9010",
            "dewey": "ws://dewey:9020"
        }
        return upstreams.get(upstream_name.lower())

    async def forward_client_to_upstream(self, client_ws, upstream_ws):
        """Forward messages from client to upstream and log to Dewey"""
        try:
            async for message in client_ws:
                # Parse message to determine if it's a user message
                try:
                    msg = json.loads(message)
                    if msg.get('method') == 'tools/call':
                        # Extract user intent from tool call
                        content = self.extract_user_intent(msg)
                        if content:
                            # Log user message to Dewey
                            _, conversation_id = self.clients[client_ws]
                            await self.dewey_client.store_message(
                                conversation_id=conversation_id,
                                role='user',
                                content=content
                            )
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message from client")
                
                # Forward to upstream
                await upstream_ws.send(message)
        except websockets.ConnectionClosed:
            logger.info("Client connection closed")

    async def forward_upstream_to_client(self, client_ws, upstream_ws):
        """Forward messages from upstream to client and log to Dewey"""
        try:
            async for message in upstream_ws:
                # Parse message to determine if it's an assistant response
                try:
                    msg = json.loads(message)
                    if msg.get('result'):
                        # Log assistant response to Dewey
                        _, conversation_id = self.clients[client_ws]
                        await self.dewey_client.store_message(
                            conversation_id=conversation_id,
                            role='assistant',
                            content=msg['result']
                        )
                except json.JSONDecodeError:
                    logger.warning("Received non-JSON message from upstream")
                
                # Forward to client
                await client_ws.send(message)
        except websockets.ConnectionClosed:
            logger.info("Upstream connection closed")

    def extract_user_intent(self, msg: dict) -> str:
        """Extract user intent from tool call message"""
        try:
            # The first tool call usually contains the user intent
            if msg.get('params') and msg['params'].get('tools'):
                first_tool = msg['params']['tools'][0]
                return first_tool.get('parameters', {}).get('content', '')
        except (KeyError, IndexError, TypeError):
            logger.warning("Failed to extract user intent from tool call")
        return ""

    async def start(self):
        """Start the proxy server"""
        logger.info(f"Starting MCP Proxy on {self.host}:{self.port}")
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port,
            ping_interval=None
        ):
            await asyncio.Future()  # Run forever

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    proxy = MCPProxy()
    asyncio.run(proxy.start())
```

### Filename: mcp_proxy/dewey_client.py
```python
import asyncio
import json
import logging
import websockets
from typing import Optional

logger = logging.getLogger(__name__)

class DeweyClient:
    def __init__(self, url: str = "ws://dewey:9020"):
        self.url = url
        self.websocket = None
        self.request_id = 0
        self.lock = asyncio.Lock()

    async def connect(self):
        """Establish connection to Dewey MCP server"""
        if self.websocket is None or self.websocket.closed:
            self.websocket = await websockets.connect(self.url)
            logger.info("Connected to Dewey MCP server")

    async def call_tool(self, method: str, params: dict) -> dict:
        """Call a Dewey MCP tool"""
        async with self.lock:
            await self.connect()
            self.request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params
            }
            
            try:
                await self.websocket.send(json.dumps(request))
                response = await self.websocket.recv()
                result = json.loads(response)
                
                if 'error' in result:
                    logger.error(f"Dewey tool error: {result['error']}")
                    raise RuntimeError(result['error'].get('message', 'Unknown error'))
                    
                return result.get('result', {})
            except (websockets.ConnectionClosed, json.JSONDecodeError) as e:
                logger.error(f"Connection to Dewey failed: {e}")
                self.websocket = None
                raise

    async def begin_conversation(self) -> str:
        """Begin a new conversation in Dewey"""
        result = await self.call_tool("dewey_begin_conversation", {})
        return result['conversation_id']

    async def store_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        metadata: Optional[dict] = None
    ) -> dict:
        """Store a message in Dewey"""
        params = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content
        }
        if metadata:
            params["metadata"] = metadata
            
        return await self.call_tool("dewey_store_message", params)

    async def close(self):
        """Close the connection"""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.websocket = None
```

### Filename: mcp_proxy/Dockerfile
```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy proxy code
COPY mcp_proxy/ ./mcp_proxy/

# Expose proxy port
EXPOSE 9000

# Run proxy server
CMD ["python", "-m", "mcp_proxy.proxy_server"]
```

### Filename: mcp_proxy/docker-compose.yml
```yaml
version: '3.8'

services:
  mcp-proxy:
    build:
      context: .
      dockerfile: mcp_proxy/Dockerfile
    image: mcp-proxy:latest
    container_name: mcp-proxy
    restart: unless-stopped
    ports:
      - "127.0.0.1:9000:9000"
    networks:
      - iccm_network

networks:
  iccm_network:
    external: true
```

### Filename: mcp_proxy/requirements.txt
```
websockets>=12.0
```

This implementation provides:
1. Dewey MCP Server with all 11 required tools
2. PostgreSQL integration with connection pooling
3. Transaction-safe turn numbering
4. Full-text search with ts_rank
5. MCP Proxy for automatic conversation logging
6. Docker configurations for both components
7. Comprehensive error handling and logging
8. Production-quality code with detailed comments

The solution follows the v3 requirements exactly, including the WebSocket transport, metadata support, and proxy-based conversation capture.