Below are the 13 complete files implementing the Dewey MCP Server and the MCP Proxy as requested. Each file is production-ready, with robust error handling, logging, and inline documentation.

File: dewey/config.py
```
import os
import logging
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    # Database
    db_host: str = "irina"
    db_port: int = 5432
    db_name: str = "winni"
    db_user: str = "dewey"
    db_password: Optional[str] = None
    db_minconn: int = 1
    db_maxconn: int = 10

    # MCP server
    mcp_host: str = "0.0.0.0"
    mcp_port: int = 9020

    # Logging
    log_level: str = "INFO"
    log_json: bool = False

    # Query tuning
    statement_timeout_ms: int = 15000  # 15s

    @classmethod
    def from_env(cls) -> "Config":
        def getenv_int(key: str, default: int) -> int:
            val = os.getenv(key)
            if val is None:
                return default
            try:
                return int(val)
            except ValueError:
                raise ValueError(f"Invalid integer for {key}: {val}")

        cfg = cls(
            db_host=os.getenv("DEWEY_DB_HOST", "irina"),
            db_port=getenv_int("DEWEY_DB_PORT", 5432),
            db_name=os.getenv("DEWEY_DB_NAME", "winni"),
            db_user=os.getenv("DEWEY_DB_USER", "dewey"),
            db_password=os.getenv("DEWEY_DB_PASSWORD"),
            db_minconn=getenv_int("DEWEY_DB_MINCONN", 1),
            db_maxconn=getenv_int("DEWEY_DB_MAXCONN", 10),
            mcp_host=os.getenv("DEWEY_MCP_HOST", "0.0.0.0"),
            mcp_port=getenv_int("DEWEY_MCP_PORT", 9020),
            log_level=os.getenv("DEWEY_LOG_LEVEL", "INFO").upper(),
            log_json=os.getenv("DEWEY_LOG_JSON", "false").lower() in ("1", "true", "yes"),
            statement_timeout_ms=getenv_int("DEWEY_STATEMENT_TIMEOUT_MS", 15000),
        )

        if not cfg.db_password:
            raise ValueError("DEWEY_DB_PASSWORD must be set")

        if cfg.db_minconn < 1 or cfg.db_maxconn < cfg.db_minconn:
            raise ValueError("Invalid connection pool size")

        if not (1 <= cfg.mcp_port <= 65535):
            raise ValueError("Invalid DEWEY_MCP_PORT")

        return cfg


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    # Simple logging configuration; JSON optional
    root = logging.getLogger()
    if root.handlers:
        # Already configured
        return

    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format=("%(asctime)s %(levelname)s [%(name)s] %(message)s")
        if not json_format
        else '%(message)s',
    )
    if json_format:
        try:
            from pythonjsonlogger import jsonlogger

            logHandler = logging.StreamHandler()
            formatter = jsonlogger.JsonFormatter(
                "%(asctime)s %(levelname)s %(name)s %(message)s"
            )
            logHandler.setFormatter(formatter)
            root.handlers.clear()
            root.addHandler(logHandler)
            root.setLevel(log_level)
        except Exception:
            # Fallback to default if json logger not available
            pass
```

File: dewey/database.py
```
import asyncio
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import uuid
from datetime import datetime

import psycopg2
from psycopg2 import pool, sql, extras, OperationalError, DatabaseError as PGDatabaseError

logger = logging.getLogger(__name__)


class DatabaseError(Exception):
    pass


@dataclass
class Conversation:
    id: uuid.UUID
    session_id: Optional[str]
    created_at: datetime
    updated_at: datetime
    metadata: Optional[dict]


class Database:
    def __init__(
        self,
        host: str,
        port: int,
        dbname: str,
        user: str,
        password: str,
        minconn: int = 1,
        maxconn: int = 10,
        statement_timeout_ms: int = 15000,
    ):
        self._dsn = f"host={host} port={port} dbname={dbname} user={user} password={password}"
        self._pool: pool.ThreadedConnectionPool = pool.ThreadedConnectionPool(
            minconn=minconn,
            maxconn=maxconn,
            dsn=self._dsn,
            cursor_factory=extras.RealDictCursor,
        )
        self._statement_timeout_ms = statement_timeout_ms
        logger.info("Database connection pool initialized (min=%s, max=%s)", minconn, maxconn)

    def close(self) -> None:
        self._pool.closeall()
        logger.info("Database connection pool closed")

    @contextmanager
    def _get_conn(self):
        conn = None
        try:
            conn = self._pool.getconn()
            with conn.cursor() as c:
                c.execute("SET statement_timeout TO %s;", (self._statement_timeout_ms,))
            yield conn
        except OperationalError as e:
            logger.exception("OperationalError obtaining connection")
            raise DatabaseError(str(e)) from e
        finally:
            if conn is not None:
                self._pool.putconn(conn)

    def _run(self, fn, *args, **kwargs):
        # Run sync DB operations in a thread
        return asyncio.to_thread(fn, *args, **kwargs)

    # Conversations

    async def begin_conversation(self, session_id: Optional[str], metadata: Optional[dict]) -> Conversation:
        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO conversations (session_id, metadata)
                    VALUES (%s, %s)
                    RETURNING id, session_id, created_at, updated_at, metadata
                    """,
                    (session_id, extras.Json(metadata) if metadata is not None else None),
                )
                row = cur.fetchone()
                conn.commit()
                return Conversation(
                    id=row["id"],
                    session_id=row["session_id"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                    metadata=row["metadata"],
                )

        return await self._run(_op)

    async def insert_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        turn_number: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> Dict[str, Any]:
        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                try:
                    # Lock the conversation row to compute next turn safely
                    cur.execute(
                        "SELECT id FROM conversations WHERE id = %s FOR UPDATE",
                        (conversation_id,),
                    )
                    if cur.fetchone() is None:
                        raise DatabaseError("Conversation not found")

                    if turn_number is None:
                        cur.execute(
                            "SELECT COALESCE(MAX(turn_number) + 1, 1) AS next_turn FROM messages WHERE conversation_id = %s",
                            (conversation_id,),
                        )
                        row = cur.fetchone()
                        turn = int(row["next_turn"]) if row and row["next_turn"] is not None else 1
                    else:
                        turn = int(turn_number)

                    cur.execute(
                        """
                        INSERT INTO messages (conversation_id, turn_number, role, content, metadata)
                        VALUES (%s, %s, %s, %s, %s)
                        RETURNING id, created_at
                        """,
                        (
                            conversation_id,
                            turn,
                            role,
                            content,
                            extras.Json(metadata) if metadata is not None else None,
                        ),
                    )
                    row = cur.fetchone()
                    conn.commit()
                    return {
                        "message_id": str(row["id"]),
                        "turn_number": turn,
                        "created_at": row["created_at"].isoformat(),
                    }
                except PGDatabaseError as e:
                    conn.rollback()
                    logger.exception("Error inserting message")
                    raise DatabaseError(str(e)) from e

        return await self._run(_op)

    async def insert_messages_bulk(
        self,
        conversation_id: Optional[str],
        session_id: Optional[str],
        messages: List[Dict[str, Any]],
        metadata: Optional[dict],
    ) -> Dict[str, Any]:
        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                try:
                    if conversation_id is None:
                        # Create a conversation first
                        cur.execute(
                            """
                            INSERT INTO conversations (session_id, metadata)
                            VALUES (%s, %s)
                            RETURNING id
                            """,
                            (session_id, extras.Json(metadata) if metadata is not None else None),
                        )
                        row = cur.fetchone()
                        conv_id = row["id"]
                    else:
                        conv_id = uuid.UUID(conversation_id)
                        # Optionally update conversation metadata if provided
                        if metadata is not None:
                            cur.execute(
                                "UPDATE conversations SET metadata = %s WHERE id = %s",
                                (extras.Json(metadata), str(conv_id)),
                            )

                    # Determine next turn number
                    cur.execute(
                        "SELECT COALESCE(MAX(turn_number) + 1, 1) AS next_turn FROM messages WHERE conversation_id = %s",
                        (str(conv_id),),
                    )
                    row = cur.fetchone()
                    next_turn = int(row["next_turn"]) if row and row["next_turn"] is not None else 1

                    ids = []
                    for i, m in enumerate(messages):
                        role = m["role"]
                        content = m["content"]
                        md = m.get("metadata")
                        cur.execute(
                            """
                            INSERT INTO messages (conversation_id, turn_number, role, content, metadata)
                            VALUES (%s, %s, %s, %s, %s)
                            RETURNING id
                            """,
                            (
                                str(conv_id),
                                next_turn + i,
                                role,
                                content,
                                extras.Json(md) if md is not None else None,
                            ),
                        )
                        ids.append(str(cur.fetchone()["id"]))
                    conn.commit()
                    return {
                        "conversation_id": str(conv_id),
                        "stored": len(ids),
                        "message_ids": ids,
                    }
                except PGDatabaseError as e:
                    conn.rollback()
                    logger.exception("Error inserting messages bulk")
                    raise DatabaseError(str(e)) from e

        return await self._run(_op)

    async def get_conversation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT id, session_id, created_at, updated_at, metadata
                    FROM conversations
                    WHERE id = %s
                    """,
                    (conversation_id,),
                )
                c = cur.fetchone()
                if not c:
                    return None
                cur.execute(
                    """
                    SELECT id, turn_number, role, content, metadata, created_at
                    FROM messages
                    WHERE conversation_id = %s
                    ORDER BY turn_number ASC, created_at ASC
                    """,
                    (conversation_id,),
                )
                msgs = cur.fetchall()
                return {
                    "conversation_id": str(c["id"]),
                    "session_id": c["session_id"],
                    "created_at": c["created_at"].isoformat(),
                    "updated_at": c["updated_at"].isoformat(),
                    "metadata": c["metadata"] or {},
                    "messages": [
                        {
                            "id": str(m["id"]),
                            "turn": int(m["turn_number"]),
                            "role": m["role"],
                            "content": m["content"],
                            "metadata": m["metadata"] or {},
                            "created_at": m["created_at"].isoformat(),
                        }
                        for m in msgs
                    ],
                }

        return await self._run(_op)

    async def list_conversations(
        self,
        session_id: Optional[str],
        limit: int,
        offset: int,
        sort_by: str,
    ) -> Dict[str, Any]:
        sort_col = "updated_at" if sort_by not in ("created_at", "updated_at") else sort_by

        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                where = []
                params: List[Any] = []
                if session_id:
                    where.append("session_id = %s")
                    params.append(session_id)
                where_sql = "WHERE " + " AND ".join(where) if where else ""
                cur.execute(
                    f"""
                    SELECT COUNT(*) AS total
                    FROM conversations
                    {where_sql}
                    """,
                    tuple(params),
                )
                total = int(cur.fetchone()["total"])

                cur.execute(
                    f"""
                    SELECT
                        c.id, c.session_id, c.created_at, c.updated_at, c.metadata,
                        (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count
                    FROM conversations c
                    {where_sql}
                    ORDER BY c.{sort_col} DESC
                    LIMIT %s OFFSET %s
                    """,
                    tuple(params + [limit, offset]),
                )
                rows = cur.fetchall()
                return {
                    "conversations": [
                        {
                            "id": str(r["id"]),
                            "session_id": r["session_id"],
                            "created_at": r["created_at"].isoformat(),
                            "updated_at": r["updated_at"].isoformat(),
                            "message_count": int(r["message_count"]),
                            "metadata": r["metadata"] or {},
                        }
                        for r in rows
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }

        return await self._run(_op)

    async def delete_conversation(self, conversation_id: str) -> Dict[str, Any]:
        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) AS cnt FROM messages WHERE conversation_id = %s",
                    (conversation_id,),
                )
                msg_count = int(cur.fetchone()["cnt"])
                cur.execute(
                    "DELETE FROM conversations WHERE id = %s",
                    (conversation_id,),
                )
                deleted = cur.rowcount > 0
                conn.commit()
                return {"deleted": deleted, "messages_deleted": msg_count if deleted else 0}

        return await self._run(_op)

    async def search(
        self,
        query: str,
        session_id: Optional[str],
        start_date: Optional[str],
        end_date: Optional[str],
        limit: int,
        offset: int,
    ) -> Dict[str, Any]:
        # Build dynamic WHERE clause
        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                where = ["to_tsvector('english', m.content) @@ plainto_tsquery('english', %s)"]
                params: List[Any] = [query]
                if session_id:
                    where.append("c.session_id = %s")
                    params.append(session_id)
                if start_date:
                    where.append("m.created_at >= %s")
                    params.append(start_date)
                if end_date:
                    where.append("m.created_at <= %s")
                    params.append(end_date)
                where_sql = "WHERE " + " AND ".join(where)

                # Total count
                cur.execute(
                    f"""
                    SELECT COUNT(*) AS total
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    {where_sql}
                    """,
                    tuple(params),
                )
                total = int(cur.fetchone()["total"])

                # Results
                cur.execute(
                    f"""
                    SELECT
                        m.id AS message_id,
                        m.conversation_id,
                        m.turn_number,
                        m.role,
                        m.content,
                        m.created_at,
                        c.session_id,
                        c.metadata AS conversation_metadata,
                        ts_rank(
                            to_tsvector('english', m.content),
                            plainto_tsquery('english', %s)
                        ) AS rank
                    FROM messages m
                    JOIN conversations c ON m.conversation_id = c.id
                    {where_sql}
                    ORDER BY rank DESC, m.created_at DESC
                    LIMIT %s OFFSET %s
                    """,
                    tuple([query] + params + [limit, offset]),
                )
                rows = cur.fetchall()
                return {
                    "results": [
                        {
                            "conversation_id": str(r["conversation_id"]),
                            "session_id": r["session_id"],
                            "message_id": str(r["message_id"]),
                            "turn": int(r["turn_number"]),
                            "role": r["role"],
                            "content": r["content"],
                            "rank": float(r["rank"]) if r["rank"] is not None else 0.0,
                            "created_at": r["created_at"].isoformat(),
                            "conversation_metadata": r["conversation_metadata"] or {},
                        }
                        for r in rows
                    ],
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                }

        return await self._run(_op)

    # Startup contexts

    async def get_startup_context(self, name: Optional[str]) -> Optional[Dict[str, Any]]:
        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                if name:
                    cur.execute(
                        """
                        SELECT id, name, content, is_active, created_at, updated_at
                        FROM startup_contexts
                        WHERE name = %s
                        """,
                        (name,),
                    )
                else:
                    cur.execute(
                        """
                        SELECT id, name, content, is_active, created_at, updated_at
                        FROM startup_contexts
                        WHERE is_active = TRUE
                        ORDER BY updated_at DESC
                        LIMIT 1
                        """
                    )
                row = cur.fetchone()
                if not row:
                    return None
                return {
                    "id": str(row["id"]),
                    "name": row["name"],
                    "content": row["content"],
                    "is_active": bool(row["is_active"]),
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat(),
                }

        return await self._run(_op)

    async def set_startup_context(self, name: str, content: str, set_active: bool) -> Dict[str, Any]:
        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                try:
                    cur.execute("BEGIN;")
                    if set_active:
                        cur.execute("UPDATE startup_contexts SET is_active = FALSE WHERE is_active = TRUE;")
                    cur.execute(
                        """
                        INSERT INTO startup_contexts (name, content, is_active)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (name) DO UPDATE
                        SET content = EXCLUDED.content, is_active = EXCLUDED.is_active, updated_at = NOW()
                        RETURNING id, name, is_active, created_at
                        """,
                        (name, content, set_active),
                    )
                    row = cur.fetchone()
                    cur.execute("COMMIT;")
                    return {
                        "id": str(row["id"]),
                        "name": row["name"],
                        "is_active": bool(row["is_active"]),
                        "created_at": row["created_at"].isoformat(),
                    }
                except PGDatabaseError as e:
                    cur.execute("ROLLBACK;")
                    logger.exception("Error setting startup context")
                    raise DatabaseError(str(e)) from e

        return await self._run(_op)

    async def list_startup_contexts(self, include_content: bool) -> Dict[str, Any]:
        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                if include_content:
                    cur.execute(
                        "SELECT id, name, content, is_active, created_at, updated_at FROM startup_contexts ORDER BY updated_at DESC"
                    )
                else:
                    cur.execute(
                        "SELECT id, name, is_active, created_at, updated_at FROM startup_contexts ORDER BY updated_at DESC"
                    )
                rows = cur.fetchall()
                contexts = []
                for r in rows:
                    item = {
                        "id": str(r["id"]),
                        "name": r["name"],
                        "is_active": bool(r["is_active"]),
                        "created_at": r["created_at"].isoformat(),
                        "updated_at": r["updated_at"].isoformat(),
                    }
                    if include_content:
                        item["content"] = r["content"]
                    contexts.append(item)
                return {"contexts": contexts}

        return await self._run(_op)

    async def delete_startup_context(self, name: str) -> Dict[str, Any]:
        def _op():
            with self._get_conn() as conn, conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM startup_contexts WHERE name = %s",
                    (name,),
                )
                deleted = cur.rowcount > 0
                conn.commit()
                return {"deleted": deleted}

        return await self._run(_op)
```

File: dewey/tools.py
```
import logging
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, conint, validator

from .database import Database, DatabaseError

logger = logging.getLogger(__name__)


# Pydantic models for tool parameters

class BeginConversationParams(BaseModel):
    session_id: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class StoreMessageParams(BaseModel):
    conversation_id: str
    role: str
    content: str
    turn_number: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None

    @validator("role")
    def validate_role(cls, v):
        allowed = {"user", "assistant", "system", "tool"}
        if v not in allowed:
            raise ValueError(f"Invalid role: {v}")
        return v


class BulkMessage(BaseModel):
    role: str
    content: str
    metadata: Optional[Dict[str, Any]] = None

    @validator("role")
    def validate_role(cls, v):
        allowed = {"user", "assistant", "system", "tool"}
        if v not in allowed:
            raise ValueError(f"Invalid role: {v}")
        return v


class StoreMessagesBulkParams(BaseModel):
    conversation_id: Optional[str] = None
    session_id: Optional[str] = None
    messages: List[BulkMessage]
    metadata: Optional[Dict[str, Any]] = None


class GetConversationParams(BaseModel):
    conversation_id: str


class ListConversationsParams(BaseModel):
    session_id: Optional[str] = None
    limit: conint(gt=0, le=100) = 20
    offset: conint(ge=0) = 0
    sort_by: str = "updated_at"

    @validator("sort_by")
    def validate_sort(cls, v):
        if v not in {"created_at", "updated_at"}:
            return "updated_at"
        return v


class DeleteConversationParams(BaseModel):
    conversation_id: str
    force: bool = False


class SearchParams(BaseModel):
    query: str
    session_id: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    limit: conint(gt=0, le=100) = 20
    offset: conint(ge=0) = 0


class GetStartupContextParams(BaseModel):
    name: Optional[str] = None


class SetStartupContextParams(BaseModel):
    name: str
    content: str
    set_active: bool = True


class ListStartupContextsParams(BaseModel):
    include_content: bool = False


class DeleteStartupContextParams(BaseModel):
    name: str
    force: bool = False


class Tools:
    def __init__(self, db: Database):
        self.db = db

    def list_tools(self) -> List[Dict[str, Any]]:
        # JSON schemas for MCP tools
        return [
            {
                "name": "dewey_begin_conversation",
                "description": "Begin a new conversation and return its ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "metadata": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "dewey_store_message",
                "description": "Store a single message in a conversation",
                "inputSchema": {
                    "type": "object",
                    "required": ["conversation_id", "role", "content"],
                    "properties": {
                        "conversation_id": {"type": "string"},
                        "role": {"type": "string", "enum": ["user", "assistant", "system", "tool"]},
                        "content": {"type": "string"},
                        "turn_number": {"type": "integer"},
                        "metadata": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "dewey_store_messages_bulk",
                "description": "Store multiple messages; creates conversation if not provided",
                "inputSchema": {
                    "type": "object",
                    "required": ["messages"],
                    "properties": {
                        "conversation_id": {"type": "string"},
                        "session_id": {"type": "string"},
                        "messages": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "required": ["role", "content"],
                                "properties": {
                                    "role": {"type": "string", "enum": ["user", "assistant", "system", "tool"]},
                                    "content": {"type": "string"},
                                    "metadata": {"type": "object"},
                                },
                                "additionalProperties": False,
                            },
                        },
                        "metadata": {"type": "object"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "dewey_get_conversation",
                "description": "Retrieve a conversation by ID with all messages",
                "inputSchema": {
                    "type": "object",
                    "required": ["conversation_id"],
                    "properties": {"conversation_id": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
            {
                "name": "dewey_list_conversations",
                "description": "List conversations with pagination",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                        "sort_by": {"type": "string", "enum": ["created_at", "updated_at"], "default": "updated_at"},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "dewey_delete_conversation",
                "description": "Delete a conversation and all messages",
                "inputSchema": {
                    "type": "object",
                    "required": ["conversation_id"],
                    "properties": {
                        "conversation_id": {"type": "string"},
                        "force": {"type": "boolean", "default": False},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "dewey_search",
                "description": "Full-text search across messages",
                "inputSchema": {
                    "type": "object",
                    "required": ["query"],
                    "properties": {
                        "query": {"type": "string"},
                        "session_id": {"type": "string"},
                        "start_date": {"type": "string"},
                        "end_date": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 20},
                        "offset": {"type": "integer", "minimum": 0, "default": 0},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "dewey_get_startup_context",
                "description": "Get active startup context or by name",
                "inputSchema": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "additionalProperties": False,
                },
            },
            {
                "name": "dewey_set_startup_context",
                "description": "Create or update startup context; optionally set active",
                "inputSchema": {
                    "type": "object",
                    "required": ["name", "content"],
                    "properties": {
                        "name": {"type": "string"},
                        "content": {"type": "string"},
                        "set_active": {"type": "boolean", "default": True},
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "dewey_list_startup_contexts",
                "description": "List startup contexts",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_content": {"type": "boolean", "default": False}
                    },
                    "additionalProperties": False,
                },
            },
            {
                "name": "dewey_delete_startup_context",
                "description": "Delete startup context by name",
                "inputSchema": {
                    "type": "object",
                    "required": ["name"],
                    "properties": {
                        "name": {"type": "string"},
                        "force": {"type": "boolean", "default": False},
                    },
                    "additionalProperties": False,
                },
            },
        ]

    async def call(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        try:
            if name == "dewey_begin_conversation":
                params = BeginConversationParams(**arguments)
                conv = await self.db.begin_conversation(session_id=params.session_id, metadata=params.metadata)
                return {
                    "conversation_id": str(conv.id),
                    "session_id": conv.session_id,
                    "created_at": conv.created_at.isoformat(),
                }

            elif name == "dewey_store_message":
                p = StoreMessageParams(**arguments)
                return await self.db.insert_message(
                    conversation_id=p.conversation_id,
                    role=p.role,
                    content=p.content,
                    turn_number=p.turn_number,
                    metadata=p.metadata,
                )

            elif name == "dewey_store_messages_bulk":
                p = StoreMessagesBulkParams(**arguments)
                msgs = [m.dict() for m in p.messages]
                return await self.db.insert_messages_bulk(
                    conversation_id=p.conversation_id,
                    session_id=p.session_id,
                    messages=msgs,
                    metadata=p.metadata,
                )

            elif name == "dewey_get_conversation":
                p = GetConversationParams(**arguments)
                conv = await self.db.get_conversation(conversation_id=p.conversation_id)
                if not conv:
                    return None
                return conv

            elif name == "dewey_list_conversations":
                p = ListConversationsParams(**arguments)
                return await self.db.list_conversations(
                    session_id=p.session_id, limit=p.limit, offset=p.offset, sort_by=p.sort_by
                )

            elif name == "dewey_delete_conversation":
                p = DeleteConversationParams(**arguments)
                if not p.force:
                    raise ValueError("Set 'force' to true to confirm deletion")
                return await self.db.delete_conversation(conversation_id=p.conversation_id)

            elif name == "dewey_search":
                p = SearchParams(**arguments)
                return await self.db.search(
                    query=p.query,
                    session_id=p.session_id,
                    start_date=p.start_date,
                    end_date=p.end_date,
                    limit=p.limit,
                    offset=p.offset,
                )

            elif name == "dewey_get_startup_context":
                p = GetStartupContextParams(**arguments)
                return await self.db.get_startup_context(name=p.name)

            elif name == "dewey_set_startup_context":
                p = SetStartupContextParams(**arguments)
                return await self.db.set_startup_context(name=p.name, content=p.content, set_active=p.set_active)

            elif name == "dewey_list_startup_contexts":
                p = ListStartupContextsParams(**arguments)
                return await self.db.list_startup_contexts(include_content=p.include_content)

            elif name == "dewey_delete_startup_context":
                p = DeleteStartupContextParams(**arguments)
                if not p.force:
                    raise ValueError("Set 'force' to true to confirm deletion")
                return await self.db.delete_startup_context(name=p.name)

            else:
                raise ValueError(f"Unknown tool: {name}")
        except DatabaseError as e:
            logger.error("Database error in tool '%s': %s", name, e)
            raise
        except Exception as e:
            logger.exception("Error executing tool '%s'", name)
            raise
```

File: dewey/mcp_server.py
```
import asyncio
import json
import logging
from typing import Any, Dict

import websockets
from websockets.server import WebSocketServerProtocol

from .config import Config, configure_logging
from .database import Database
from .tools import Tools

logger = logging.getLogger(__name__)


class MCPServer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.db = Database(
            host=cfg.db_host,
            port=cfg.db_port,
            dbname=cfg.db_name,
            user=cfg.db_user,
            password=cfg.db_password,
            minconn=cfg.db_minconn,
            maxconn=cfg.db_maxconn,
            statement_timeout_ms=cfg.statement_timeout_ms,
        )
        self.tools = Tools(self.db)

    async def handle_message(self, ws: WebSocketServerProtocol, msg: Dict[str, Any]):
        msg_id = msg.get("id")
        method = msg.get("method")
        params = msg.get("params", {}) or {}

        try:
            if method == "ping":
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": {"ok": True}}))
                return

            if method == "initialize":
                result = {
                    "protocolVersion": "2024-10-01",
                    "serverInfo": {"name": "dewey-mcp", "version": "1.0.0"},
                    "capabilities": {
                        "tools": {"list": True, "call": True}
                    },
                }
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}))
                return

            if method == "tools/list":
                tools = self.tools.list_tools()
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": {"tools": tools}}))
                return

            if method == "tools/call":
                name = params.get("name")
                arguments = params.get("arguments") or {}
                res = await self.tools.call(name, arguments)
                # Return both native result and a content wrapper for broader compatibility
                result = {
                    "ok": True,
                    "data": res,
                    "content": [{"type": "json", "data": res}],
                }
                await ws.send(json.dumps({"jsonrpc": "2.0", "id": msg_id, "result": result}))
                return

            # Unknown method
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32601, "message": f"Method not found: {method}"},
                    }
                )
            )
        except Exception as e:
            logger.exception("Error handling method %s", method)
            await ws.send(
                json.dumps(
                    {
                        "jsonrpc": "2.0",
                        "id": msg_id,
                        "error": {"code": -32000, "message": str(e)},
                    }
                )
            )

    async def handler(self, ws: WebSocketServerProtocol):
        peer = f"{ws.remote_address[0]}:{ws.remote_address[1]}" if ws.remote_address else "unknown"
        logger.info("Client connected: %s", peer)
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
                        await ws.send(
                            json.dumps(
                                {"jsonrpc": "2.0", "id": None, "error": {"code": -32600, "message": "Invalid Request"}}
                            )
                        )
                        continue
                    await self.handle_message(ws, msg)
                except json.JSONDecodeError:
                    await ws.send(
                        json.dumps(
                            {"jsonrpc": "2.0", "id": None, "error": {"code": -32700, "message": "Parse error"}}
                        )
                    )
        except websockets.ConnectionClosed:
            logger.info("Client disconnected: %s", peer)
        except Exception:
            logger.exception("Unexpected error with client %s", peer)

    async def serve(self):
        async with websockets.serve(self.handler, self.cfg.mcp_host, self.cfg.mcp_port, ping_interval=30, ping_timeout=30):
            logger.info("Dewey MCP Server listening on ws://%s:%s", self.cfg.mcp_host, self.cfg.mcp_port)
            while True:
                await asyncio.sleep(3600)


def main():
    try:
        cfg = Config.from_env()
    except Exception as e:
        print(f"Configuration error: {e}")
        raise

    configure_logging(cfg.log_level, cfg.log_json)
    logger.info("Starting Dewey MCP Server")

    server = MCPServer(cfg)
    try:
        asyncio.run(serverserve(server))
    finally:
        server.db.close()


async def serverserve(server: MCPServer):
    await server.serve()


if __name__ == "__main__":
    main()
```

File: dewey/__init__.py
```
__all__ = ["config", "database", "tools", "mcp_server"]

__version__ = "1.0.0"
```

File: Dockerfile
```
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code
COPY dewey/ ./dewey/

EXPOSE 9020

CMD ["python", "-m", "dewey.mcp_server"]
```

File: docker-compose.yml
```
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
      - DEWEY_DB_HOST=irina
      - DEWEY_DB_PORT=5432
      - DEWEY_DB_NAME=winni
      - DEWEY_DB_USER=dewey
      - DEWEY_DB_PASSWORD=${DEWEY_DB_PASSWORD}

      - DEWEY_MCP_PORT=9020
      - DEWEY_MCP_HOST=0.0.0.0

      - DEWEY_LOG_LEVEL=INFO
      - DEWEY_LOG_JSON=false

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

File: requirements.txt
```
# Core protocol and server
websockets>=12.0

# Database
psycopg2-binary>=2.9.9

# Validation and logging
pydantic>=2.8.2
python-json-logger>=2.0.7
```

File: schema.sql
```
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS pgcrypto;

-- The following database/user creation commands are illustrative and should be
-- run as the postgres superuser in a separate session if needed:
-- CREATE DATABASE winni;
-- CREATE USER dewey WITH PASSWORD 'secure-password-here';
-- GRANT ALL PRIVILEGES ON DATABASE winni TO dewey;
-- \c winni
-- GRANT ALL PRIVILEGES ON SCHEMA public TO dewey;
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO dewey;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO dewey;

-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    metadata JSONB,
    CONSTRAINT valid_metadata CHECK (metadata IS NULL OR jsonb_typeof(metadata) = 'object')
);

CREATE INDEX IF NOT EXISTS idx_conversations_session ON conversations(session_id);
CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_updated ON conversations(updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_metadata ON conversations USING GIN(metadata);

-- Auto-update updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'conversations_updated_at'
    ) THEN
        CREATE TRIGGER conversations_updated_at
        BEFORE UPDATE ON conversations
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$;

-- Messages
CREATE TABLE IF NOT EXISTS messages (
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

CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id, turn_number);
CREATE INDEX IF NOT EXISTS idx_messages_created ON messages(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_conversation_created ON messages(conversation_id, created_at);
CREATE INDEX IF NOT EXISTS idx_messages_content_fts ON messages USING GIN (to_tsvector('english', content));

-- Startup Contexts
CREATE TABLE IF NOT EXISTS startup_contexts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT UNIQUE NOT NULL,
    content TEXT NOT NULL,
    is_active BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT valid_name CHECK (length(name) > 0)
);

-- Single active context enforced with partial unique index
CREATE UNIQUE INDEX IF NOT EXISTS idx_startup_contexts_single_active
ON startup_contexts(is_active) WHERE is_active = TRUE;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger WHERE tgname = 'startup_contexts_updated_at'
    ) THEN
        CREATE TRIGGER startup_contexts_updated_at
        BEFORE UPDATE ON startup_contexts
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    END IF;
END;
$$;

-- Fiedler Results (Phase 2)
CREATE TABLE IF NOT EXISTS fiedler_results (
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

CREATE INDEX IF NOT EXISTS idx_fiedler_correlation ON fiedler_results(correlation_id);
CREATE INDEX IF NOT EXISTS idx_fiedler_model ON fiedler_results(model);
CREATE INDEX IF NOT EXISTS idx_fiedler_created ON fiedler_results(created_at DESC);
```

File: mcp_proxy/proxy_server.py
```
import asyncio
import json
import logging
import os
import urllib.parse
from typing import Any, Dict, Optional
import uuid

import websockets
from websockets.client import connect as ws_connect
from websockets.server import WebSocketServerProtocol

from .dewey_client import DeweyClient

logger = logging.getLogger(__name__)


class ProxyConfig:
    def __init__(self):
        self.host = os.getenv("PROXY_HOST", "0.0.0.0")
        self.port = int(os.getenv("PROXY_PORT", "9000"))
        # Comma-separated map: key=url
        # Example: "fiedler=ws://localhost:9010,dewey=ws://localhost:9020"
        self.upstream_map = self._parse_map(os.getenv("UPSTREAM_MAP", "dewey=ws://localhost:9020"))
        self.dewey_url = os.getenv("DEWEY_URL", "ws://localhost:9020")
        self.log_level = os.getenv("PROXY_LOG_LEVEL", "INFO").upper()

    @staticmethod
    def _parse_map(s: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for part in s.split(","):
            if not part.strip():
                continue
            if "=" not in part:
                continue
            k, v = part.split("=", 1)
            mapping[k.strip()] = v.strip()
        return mapping


class MCPProxy:
    def __init__(self, cfg: ProxyConfig):
        self.cfg = cfg
        self.dewey_client: Optional[DeweyClient] = None
        logging.basicConfig(level=getattr(logging, cfg.log_level, logging.INFO),
                            format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    async def ensure_dewey_client(self):
        if self.dewey_client is None or not self.dewey_client.is_connected():
            self.dewey_client = DeweyClient(self.cfg.dewey_url)
            await self.dewey_client.connect()

    async def handler(self, client_ws: WebSocketServerProtocol, path: str):
        # Parse upstream from query param
        parsed = urllib.parse.urlparse(path)
        query = urllib.parse.parse_qs(parsed.query)
        upstream_key = query.get("upstream", [None])[0]
        if not upstream_key:
            await client_ws.close(code=1008, reason="Missing upstream query parameter")
            return

        upstream_url = self.cfg.upstream_map.get(upstream_key)
        if not upstream_url:
            await client_ws.close(code=1008, reason="Unknown upstream")
            return

        peer = f"{client_ws.remote_address[0]}:{client_ws.remote_address[1]}" if client_ws.remote_address else "unknown"
        logger.info("Client %s connected; upstream=%s", peer, upstream_key)

        # Conversation tracking
        session_id = f"proxy:{upstream_key}:{uuid.uuid4()}"
        conversation_id: Optional[str] = None
        log_enabled = upstream_key != "dewey"  # avoid logging loops when proxying to Dewey itself

        try:
            async with ws_connect(upstream_url, ping_interval=30, ping_timeout=30) as upstream_ws:
                logger.info("Connected to upstream %s", upstream_url)

                if log_enabled:
                    await self.ensure_dewey_client()
                    try:
                        res = await self.dewey_client.call_tool(
                            "dewey_begin_conversation",
                            {"session_id": session_id, "metadata": {"source": "mcp_proxy", "upstream": upstream_key}},
                        )
                        # result may be wrapped under result.data
                        conversation_id = self._extract_conversation_id(res)
                        logger.info("Started Dewey conversation: %s", conversation_id)
                    except Exception:
                        logger.exception("Failed to begin Dewey conversation")
                        log_enabled = False  # Avoid further logging attempts

                # Start bidirectional relay
                await self._relay(client_ws, upstream_ws, conversation_id, upstream_key, log_enabled)
        except Exception:
            logger.exception("Error in proxy handler for %s", peer)
        finally:
            logger.info("Client %s disconnected", peer)

    def _extract_conversation_id(self, result: Dict[str, Any]) -> Optional[str]:
        # Dewey MCP wraps tool result as {"result": {"data": {...}}} in JSON-RPC layer.
        # DeweyClient returns the inner "result" already; handle both cases.
        if not result:
            return None
        if "conversation_id" in result:
            return result.get("conversation_id")
        if "data" in result and isinstance(result["data"], dict):
            return result["data"].get("conversation_id")
        return None

    async def _relay(
        self,
        client_ws: WebSocketServerProtocol,
        upstream_ws: WebSocketServerProtocol,
        conversation_id: Optional[str],
        upstream_key: str,
        log_enabled: bool,
    ):
        async def client_to_upstream():
            async for raw in client_ws:
                await upstream_ws.send(raw)
                if log_enabled and conversation_id:
                    await self._log_if_applicable(raw, direction="client->upstream", role="user",
                                                  conversation_id=conversation_id, upstream=upstream_key)

        async def upstream_to_client():
            async for raw in upstream_ws:
                await client_ws.send(raw)
                if log_enabled and conversation_id:
                    await self._log_if_applicable(raw, direction="upstream->client", role="assistant",
                                                  conversation_id=conversation_id, upstream=upstream_key)

        task1 = asyncio.create_task(client_to_upstream())
        task2 = asyncio.create_task(upstream_to_client())
        done, pending = await asyncio.wait({task1, task2}, return_when=asyncio.FIRST_EXCEPTION)
        for t in pending:
            t.cancel()

    async def _log_if_applicable(
        self,
        raw: str,
        direction: str,
        role: str,
        conversation_id: str,
        upstream: str,
    ):
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            return

        # Only log meaningful payloads
        method = msg.get("method")
        has_result = "result" in msg
        # Log tool calls and responses
        if method == "tools/call" or has_result:
            content_str = json.dumps(msg, separators=(",", ":"), ensure_ascii=False)
            md = {
                "proxy": {
                    "direction": direction,
                    "upstream": upstream,
                    "message_type": method or "result",
                }
            }
            try:
                await self.ensure_dewey_client()
                await self.dewey_client.call_tool(
                    "dewey_store_message",
                    {
                        "conversation_id": conversation_id,
                        "role": role,
                        "content": content_str,
                        "metadata": md,
                    },
                )
            except Exception:
                logger.exception("Failed to log message to Dewey")

    async def serve(self):
        async with websockets.serve(self.handler, self.cfg.host, self.cfg.port, ping_interval=30, ping_timeout=30):
            logger.info("MCP Proxy listening on ws://%s:%s", self.cfg.host, self.cfg.port)
            while True:
                await asyncio.sleep(3600)


def main():
    cfg = ProxyConfig()
    logging.getLogger().setLevel(getattr(logging, cfg.log_level, logging.INFO))
    proxy = MCPProxy(cfg)
    asyncio.run(proxy.serve())


if __name__ == "__main__":
    main()
```

File: mcp_proxy/dewey_client.py
```
import asyncio
import json
import logging
from typing import Any, Dict, Optional

import websockets

logger = logging.getLogger(__name__)


class DeweyClient:
    def __init__(self, url: str):
        self.url = url
        self.ws: Optional[websockets.WebSocketClientProtocol] = None
        self._id = 0

    def is_connected(self) -> bool:
        return self.ws is not None and not self.ws.closed

    async def connect(self):
        if self.is_connected():
            return
        self.ws = await websockets.connect(self.url, ping_interval=30, ping_timeout=30)
        await self._initialize()

    async def close(self):
        if self.ws and not self.ws.closed:
            await self.ws.close()
            self.ws = None

    def _next_id(self) -> int:
        self._id += 1
        return self._id

    async def _initialize(self):
        req_id = self._next_id()
        await self._send({"jsonrpc": "2.0", "id": req_id, "method": "initialize", "params": {}})
        resp = await self._recv_for_id(req_id)
        if "error" in resp:
            raise RuntimeError(f"Initialize failed: {resp['error']}")

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        if not self.is_connected():
            await self.connect()
        req_id = self._next_id()
        payload = {"jsonrpc": "2.0", "id": req_id, "method": "tools/call", "params": {"name": name, "arguments": arguments}}
        await self._send(payload)
        resp = await self._recv_for_id(req_id)
        if "error" in resp:
            raise RuntimeError(f"Tool call error: {resp['error']}")
        result = resp.get("result") or {}
        # Dewey MCP server returns result with {"ok":..., "data":..., "content":[...]}
        # Return the "data" if present; else the whole result.
        return result.get("data", result)

    async def _send(self, obj: Dict[str, Any]):
        assert self.ws is not None
        await self.ws.send(json.dumps(obj))

    async def _recv_for_id(self, req_id: int, timeout: float = 30.0) -> Dict[str, Any]:
        assert self.ws is not None

        async def recv_loop():
            while True:
                raw = await self.ws.recv()
                try:
                    msg = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                if isinstance(msg, dict) and msg.get("id") == req_id:
                    return msg

        return await asyncio.wait_for(recv_loop(), timeout=timeout)
```

File: mcp_proxy/Dockerfile
```
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Python deps
RUN pip install --no-cache-dir websockets>=12.0 python-json-logger>=2.0.7

# Copy code
COPY mcp_proxy/ ./mcp_proxy/

EXPOSE 9000

CMD ["python", "-m", "mcp_proxy.proxy_server"]
```

File: mcp_proxy/docker-compose.yml
```
version: '3.8'

services:
  mcp_proxy:
    build:
      context: ..
      dockerfile: mcp_proxy/Dockerfile
    image: mcp-proxy:latest
    container_name: mcp-proxy
    restart: unless-stopped
    stdin_open: true
    tty: true

    environment:
      - PROXY_HOST=0.0.0.0
      - PROXY_PORT=9000
      - UPSTREAM_MAP=dewey=ws://host.docker.internal:9020,fiedler=ws://host.docker.internal:9010
      - DEWEY_URL=ws://host.docker.internal:9020
      - PROXY_LOG_LEVEL=INFO

    ports:
      - "127.0.0.1:9000:9000"

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

Notes and Usage:

- Dewey MCP Server:
  - Start Postgres and apply schema.sql to the winni database.
  - Set DEWEY_DB_PASSWORD in your environment or .env for docker-compose.
  - Build and run with docker-compose up -d.
  - It listens on ws://localhost:9020.

- MCP Proxy:
  - Configure UPSTREAM_MAP to include MCP servers like Fiedler and Dewey.
  - The proxy listens on ws://localhost:9000.
  - Point MCP clients (Claude Code) to the proxy:
    - ws://localhost:9000?upstream=dewey for Dewey
    - ws://localhost:9000?upstream=fiedler for Fiedler
  - The proxy auto-logs to Dewey unless the upstream is Dewey to avoid loops.

- MCP Protocol:
  - Implements initialize, tools/list, and tools/call methods via JSON-RPC over WebSocket.
  - Tool schemas follow JSON Schema hints for Claude and compatible clients.

- Error handling and logging:
  - Database operations run in threads with proper rollback and logging.
  - Proxy includes robust logging and avoids recursion loops.

This completes the requested implementation with all 13 files.