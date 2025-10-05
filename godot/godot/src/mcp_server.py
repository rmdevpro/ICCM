"""
Godot MCP Facade Server - WebSocket MCP server for logging tools
Based on Dewey MCP server pattern + Gemini-2.5-Pro recommendation (correlation_id: b5afd3b0)
"""
import asyncio
import json
import logging
import time
from datetime import datetime, timezone

import redis.asyncio as redis
import websockets
from websockets.server import WebSocketServerProtocol

import config
from mcp_client import MCPClient

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


async def handle_client(websocket: WebSocketServerProtocol, path: str):
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

    async with websockets.serve(handle_client, "0.0.0.0", config.MCP_PORT):
        logger.info(f"Godot MCP server listening on port {config.MCP_PORT}")
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    asyncio.run(main())
