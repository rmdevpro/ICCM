"""
Godot MCP Facade Server - WebSocket MCP server for logging and conversation storage tools
Based on Dewey MCP server pattern + Gemini-2.5-Pro recommendation (correlation_id: b5afd3b0)
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from uuid import UUID

import redis.asyncio as redis
import websockets
from websockets.server import WebSocketServerProtocol

import config
from mcp_client import MCPClient
from database import db_pool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [MCP_SERVER] %(message)s')
logger = logging.getLogger(__name__)

# Redis client for local log queue
redis_client = redis.Redis(host=config.REDIS_HOST, port=config.REDIS_PORT, decode_responses=True)

# MCP client to Dewey for proxying query/clear operations
dewey_client = MCPClient(config.DEWEY_MCP_URL, timeout=config.DEWEY_CONNECT_TIMEOUT)


# Tool Implementations

async def logger_log(component: str, level: str, message: str, trace_id: str = None, data: dict = None):
    """Facade tool to push a single log entry to the Redis queue (REQ-GOD-001)"""
    log_entry = {
        "component": component,
        "level": level.upper(),
        "message": message,
        "trace_id": trace_id,
        "data": data,
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    # REQ-GOD-001: Push to Redis queue
    await redis_client.lpush(config.LOG_QUEUE_NAME, json.dumps(log_entry))

    # REQ-GOD-005: Enforce queue size limit (simple approach, worker uses Lua for atomicity)
    await redis_client.ltrim(config.LOG_QUEUE_NAME, 0, config.MAX_QUEUE_SIZE - 1)

    return {"status": "ok", "message": "Log entry queued."}


async def logger_query(trace_id: str = None, component: str = None, level: str = None,
                       start_time: str = None, end_time: str = None, limit: int = 100):
    """Facade tool that proxies to dewey_query_logs"""
    logger.info(f"Forwarding query to Dewey: trace_id={trace_id}, component={component}")

    # Build arguments dict, only include non-None values
    args = {}
    if trace_id: args['trace_id'] = trace_id
    if component: args['component'] = component
    if level: args['level'] = level
    if start_time: args['start_time'] = start_time
    if end_time: args['end_time'] = end_time
    args['limit'] = limit

    return await dewey_client.call_tool("dewey_query_logs", args)


async def logger_clear(component: str = None, before_time: str = None):
    """Facade tool that proxies to dewey_clear_logs"""
    logger.info(f"Forwarding clear command to Dewey: component={component}, before={before_time}")

    args = {}
    if component: args['component'] = component
    if before_time: args['before_time'] = before_time

    return await dewey_client.call_tool("dewey_clear_logs", args)


async def logger_set_level(component: str, level: str):
    """
    Sets the log level for a component (REQ-LIB-006).
    Stores config in Redis hash for central management.
    """
    valid_levels = {'ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE'}
    level_upper = level.upper()

    if level_upper not in valid_levels:
        raise ValueError(f"Invalid log level '{level}'. Must be one of {valid_levels}")

    await redis_client.hset("logger_config:levels", component, level_upper)
    logger.info(f"Set log level for {component} to {level_upper}")

    return {"status": "ok", "component": component, "level": level_upper}


async def conversation_begin(session_id: str = None, metadata: dict = None):
    """Starts a new conversation and returns its ID"""
    sql = """
        INSERT INTO conversations (session_id, metadata)
        VALUES ($1, $2)
        RETURNING id, session_id, created_at;
    """
    try:
        async with db_pool.transaction() as conn:
            result = await conn.fetchrow(sql, session_id, json.dumps(metadata) if metadata else None)
        logger.info(f"Began new conversation {result['id']}")
        return {
            "conversation_id": str(result['id']),
            "session_id": result['session_id'],
            "created_at": result['created_at'].isoformat()
        }
    except Exception as e:
        logger.error(f"Error in conversation_begin: {e}")
        raise ValueError("Failed to begin conversation")


async def conversation_store_message(conversation_id: str, role: str, content: str, turn_number: int = None, metadata: dict = None):
    """Stores a single message in a conversation"""
    # Validate UUID
    try:
        conv_id = UUID(conversation_id)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid UUID format for conversation_id: {conversation_id}")

    # Validate role
    if role not in ('user', 'assistant', 'system', 'tool'):
        raise ValueError("Invalid role. Must be one of 'user', 'assistant', 'system', 'tool'")

    try:
        async with db_pool.transaction() as conn:
            # Lock the conversation to safely determine the next turn number
            check = await conn.fetchval("SELECT 1 FROM conversations WHERE id = $1 FOR UPDATE;", conv_id)
            if check is None:
                raise ValueError(f"Conversation with id '{conversation_id}' not found")

            if turn_number is None:
                turn_number = await conn.fetchval(
                    "SELECT COALESCE(MAX(turn_number), 0) + 1 FROM messages WHERE conversation_id = $1;",
                    conv_id
                )

            sql = """
                INSERT INTO messages (conversation_id, turn_number, role, content, metadata)
                VALUES ($1, $2, $3, $4, $5)
                RETURNING id, turn_number, created_at;
            """
            result = await conn.fetchrow(sql, conv_id, turn_number, role, content, json.dumps(metadata) if metadata else None)

            # Update conversation's updated_at timestamp
            await conn.execute("UPDATE conversations SET updated_at = NOW() WHERE id = $1;", conv_id)

        logger.info(f"Stored message {result['id']} in conversation {conversation_id} (turn {result['turn_number']})")
        return {
            "message_id": str(result['id']),
            "conversation_id": conversation_id,
            "turn_number": result['turn_number'],
            "created_at": result['created_at'].isoformat()
        }
    except Exception as e:
        logger.error(f"Error in conversation_store_message: {e}")
        raise ValueError(f"Failed to store message: {e}")


async def conversation_store_messages_bulk(messages: list = None, messages_file: str = None, conversation_id: str = None, session_id: str = None, metadata: dict = None) -> dict:
    """Stores a list of messages in a single transaction using optimized multi-row INSERT.

    Args:
        messages: List of message objects (inline)
        messages_file: Path to JSON file containing message array (industry-standard file reference)
        conversation_id: Existing conversation ID
        session_id: Session ID for new conversation
        metadata: Metadata for new conversation
    """
    # Support industry-standard file reference pattern
    if messages_file:
        import os
        if not os.path.exists(messages_file):
            raise ValueError(f"File not found: {messages_file}")
        try:
            with open(messages_file, 'r') as f:
                # Detect format: JSON array or JSONL (Claude Code session format)
                first_char = f.read(1)
                f.seek(0)

                if first_char == '[':
                    # Standard JSON array
                    messages = json.load(f)
                elif first_char == '{':
                    # JSONL format (newline-delimited JSON objects)
                    messages = []
                    for line in f:
                        line = line.strip()
                        if line:
                            messages.append(json.loads(line))
                else:
                    raise ValueError(f"Unrecognized file format (must start with '[' or '{{')")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in messages_file: {e}")

    if not isinstance(messages, list) or not messages:
        raise ValueError("Parameter 'messages' (or messages_file content) must be a non-empty list.")

    # Normalize messages to fit Godot schema
    for i, msg in enumerate(messages):
        # Handle Claude Code session format
        if 'message' in msg and isinstance(msg['message'], dict):
            # Extract role/content from nested message
            role = msg['message'].get('role', 'NA')
            content = msg['message'].get('content', 'NA')
            # Store original full entry in metadata
            messages[i] = {
                'role': role,
                'content': json.dumps(content) if isinstance(content, (list, dict)) else str(content),
                'metadata': msg  # Full original entry
            }
        elif 'role' in msg and 'content' in msg:
            # Already in correct format
            content = msg.get('content')
            if isinstance(content, (list, dict)):
                messages[i]['content'] = json.dumps(content)
        else:
            # Non-message entry (snapshot, etc.) - store as system message with full entry in metadata
            messages[i] = {
                'role': 'system',
                'content': msg.get('type', 'unknown'),  # Use type as content placeholder
                'metadata': msg  # Full original entry
            }

    try:
        async with db_pool.transaction() as conn:
            if conversation_id:
                conv_id = UUID(conversation_id)
                check = await conn.fetchval("SELECT 1 FROM conversations WHERE id = $1 FOR UPDATE;", conv_id)
                if check is None:
                    raise ValueError(f"Conversation with id '{conversation_id}' not found.")
            else:
                # Create a new conversation
                conv_id = await conn.fetchval(
                    "INSERT INTO conversations (session_id, metadata) VALUES ($1, $2) RETURNING id;",
                    session_id, json.dumps(metadata) if metadata else None
                )

            # Get the current max turn_number
            base_turn_number = await conn.fetchval(
                "SELECT COALESCE(MAX(turn_number), 0) FROM messages WHERE conversation_id = $1;",
                conv_id
            ) or 0

            # Prepare bulk insert data
            insert_values = []
            for idx, msg in enumerate(messages, start=1):
                insert_values.append((
                    conv_id,
                    base_turn_number + idx,
                    msg.get('role', 'user'),
                    msg.get('content', ''),
                    json.dumps(msg.get('metadata')) if msg.get('metadata') else None
                ))

            # Multi-row INSERT (PostgreSQL optimized)
            await conn.executemany(
                "INSERT INTO messages (conversation_id, turn_number, role, content, metadata) VALUES ($1, $2, $3, $4, $5);",
                insert_values
            )

            # Update conversation's updated_at timestamp
            await conn.execute("UPDATE conversations SET updated_at = NOW() WHERE id = $1;", conv_id)

            stored_count = len(insert_values)

        logger.info(f"Stored {stored_count} messages in conversation {conv_id}")
        return {
            "conversation_id": str(conv_id),
            "messages_stored": stored_count,
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Error in conversation_store_messages_bulk: {e}")
        raise ValueError(f"Failed to store messages in bulk: {e}")


# MCP Protocol Handlers

TOOLS = [
    {
        "name": "logger_log",
        "description": "Push a log entry to the logging queue",
        "inputSchema": {
            "type": "object",
            "properties": {
                "component": {"type": "string", "description": "Component name"},
                "level": {"type": "string", "enum": ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"]},
                "message": {"type": "string", "description": "Log message"},
                "trace_id": {"type": "string", "description": "Optional trace ID for correlation"},
                "data": {"type": "object", "description": "Optional structured data"}
            },
            "required": ["component", "level", "message"]
        }
    },
    {
        "name": "logger_query",
        "description": "Query logs from storage (proxies to Dewey)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "trace_id": {"type": "string"},
                "component": {"type": "string"},
                "level": {"type": "string"},
                "start_time": {"type": "string"},
                "end_time": {"type": "string"},
                "limit": {"type": "integer", "default": 100}
            }
        }
    },
    {
        "name": "logger_clear",
        "description": "Clear logs from storage (proxies to Dewey)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "component": {"type": "string"},
                "before_time": {"type": "string"}
            }
        }
    },
    {
        "name": "logger_set_level",
        "description": "Set log level for a component",
        "inputSchema": {
            "type": "object",
            "properties": {
                "component": {"type": "string"},
                "level": {"type": "string", "enum": ["ERROR", "WARN", "INFO", "DEBUG", "TRACE"]}
            },
            "required": ["component", "level"]
        }
    },
    {
        "name": "conversation_begin",
        "description": "Start a new conversation and return its ID",
        "inputSchema": {
            "type": "object",
            "properties": {
                "session_id": {"type": "string", "description": "Optional session ID to group conversations"},
                "metadata": {"type": "object", "description": "Optional metadata as key-value pairs"}
            }
        }
    },
    {
        "name": "conversation_store_message",
        "description": "Store a single message in a conversation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "conversation_id": {"type": "string", "description": "UUID of the conversation"},
                "role": {"type": "string", "enum": ["user", "assistant", "system", "tool"], "description": "Message role"},
                "content": {"type": "string", "description": "Message content"},
                "turn_number": {"type": "integer", "description": "Optional turn number (auto-increments if not provided)"},
                "metadata": {"type": "object", "description": "Optional metadata (e.g., model, tokens, timing)"}
            },
            "required": ["conversation_id", "role", "content"]
        }
    },
    {
        "name": "conversation_store_messages_bulk",
        "description": "Store multiple messages at once (up to 1000). Supports file references for large payloads.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "messages": {
                    "type": "array",
                    "description": "Array of message objects with role and content (inline)",
                    "items": {"type": "object"}
                },
                "messages_file": {
                    "type": "string",
                    "description": "Path to JSON file containing message array (file reference - industry standard for large payloads)"
                },
                "conversation_id": {"type": "string", "description": "Optional existing conversation ID"},
                "session_id": {"type": "string", "description": "Optional session ID for new conversation"},
                "metadata": {"type": "object", "description": "Optional metadata for new conversation"}
            }
        }
    }
]


async def handle_initialize(request_id):
    """Handle MCP initialize request"""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": "godot",
                "version": "1.0.0"
            }
        }
    }


async def handle_tools_list(request_id):
    """Handle MCP tools/list request"""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": TOOLS
        }
    }


async def handle_tools_call(request_id, params):
    """Handle MCP tools/call request"""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    logger.info(f"Calling tool: {tool_name}")

    try:
        # Route to appropriate tool function
        if tool_name == "logger_log":
            result = await logger_log(**arguments)
        elif tool_name == "logger_query":
            result = await logger_query(**arguments)
        elif tool_name == "logger_clear":
            result = await logger_clear(**arguments)
        elif tool_name == "logger_set_level":
            result = await logger_set_level(**arguments)
        elif tool_name == "conversation_begin":
            result = await conversation_begin(**arguments)
        elif tool_name == "conversation_store_message":
            result = await conversation_store_message(**arguments)
        elif tool_name == "conversation_store_messages_bulk":
            result = await conversation_store_messages_bulk(**arguments)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "content": [{
                    "type": "text",
                    "text": json.dumps(result)
                }]
            }
        }

    except Exception as e:
        logger.error(f"Tool {tool_name} failed: {e}")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32000,
                "message": str(e)
            }
        }


async def handle_client(websocket: WebSocketServerProtocol):
    """Handle incoming WebSocket client connection"""
    client_id = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
    logger.info(f"Client connected: {client_id}")

    try:
        async for message in websocket:
            try:
                request = json.loads(message)
                method = request.get("method")
                request_id = request.get("id")
                params = request.get("params", {})

                logger.debug(f"Received: {method} (id={request_id})")

                # Route to appropriate handler
                if method == "initialize":
                    response = await handle_initialize(request_id)
                elif method == "tools/list":
                    response = await handle_tools_list(request_id)
                elif method == "tools/call":
                    response = await handle_tools_call(request_id, params)
                else:
                    response = {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {
                            "code": -32601,
                            "message": f"Method not found: {method}"
                        }
                    }

                await websocket.send(json.dumps(response))

            except json.JSONDecodeError as e:
                logger.error(f"Invalid JSON from {client_id}: {e}")
            except Exception as e:
                logger.error(f"Error handling message from {client_id}: {e}")

    except websockets.exceptions.ConnectionClosed:
        logger.info(f"Client disconnected: {client_id}")
    except Exception as e:
        logger.error(f"Error with client {client_id}: {e}")


async def main():
    """Start the Godot MCP facade server"""
    logger.info(f"Starting Godot MCP server on port {config.MCP_PORT}")

    # Initialize database pool for conversation storage
    await db_pool.initialize()

    async with websockets.serve(handle_client, "0.0.0.0", config.MCP_PORT):
        logger.info(f"Godot MCP server listening on port {config.MCP_PORT}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
