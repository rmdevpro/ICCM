# Request: Design iccm-network Standard Library

## Context

The ICCM system has 7+ microservices that communicate via WebSocket MCP protocol. **Every component reimplements networking from scratch**, causing:

- **10+ hours wasted** debugging connection timeouts, handshake failures, IP vs hostname issues
- Inconsistent behavior (some work, some timeout)
- Same bugs fixed multiple times across components
- No standardized error handling or retry logic

## Current State

Each component uses raw `websockets.connect()` and `websockets.serve()` with different:
- Timeout configurations
- Host/IP resolution strategies
- Error handling approaches
- Retry mechanisms

**Components analyzed:**
1. Dewey - Most reliable, works consistently
2. Fiedler - Works with HTTP proxy integration
3. Godot - Has connection issues (timeouts from network)
4. Horace - Has connection issues (timeouts from network)

## Relay Constraint

**CRITICAL: The MCP relay works perfectly and will NOT change without VERY good justification.**

The relay successfully:
- Connects to all components
- Routes tool calls correctly
- Manages multiple backends
- Has proven stability over dozens of tool discoveries

**The problem is NOT the relay - it's how components implement their MCP servers.**

## Your Task

Design a **`iccm-network`** Python library that:

### 1. Server Side
Standardizes MCP server creation:

```python
from iccm_network import MCPServer

server = MCPServer(
    name="horace",
    version="1.0.0", 
    port=8070,
    tools={...}  # Tool definitions
)

await server.start()
```

**Must handle:**
- Binding to `0.0.0.0` for network access
- JSON-RPC 2.0 protocol (initialize, tools/list, tools/call)
- Connection from both localhost AND remote containers
- Graceful shutdown
- Error responses

### 2. Client Side (Optional, Lower Priority)
Standardizes calling other MCP services:

```python
from iccm_network import MCPClient

client = MCPClient("horace-mcp", port=8070)
result = await client.call("horace_register_file", {"file_path": "/tmp/test"})
```

### 3. Key Features Required

- **Works from localhost AND network:** Test from inside container and from other containers
- **Standard error handling:** Consistent JSON-RPC error codes
- **WebSocket best practices:** Proper handshake, no timeouts
- **Logging integration:** Auto-log connection events to Godot
- **Zero configuration:** Components should "just work"

### 4. Implementation Details Needed

Your response should include:

1. **Library API design** - Classes, methods, parameters
2. **Server wrapper implementation** - How to wrap websockets.serve()
3. **Protocol handler** - JSON-RPC 2.0 message handling
4. **Error handling** - Standard error types and responses
5. **Configuration** - What can be customized vs defaults
6. **Migration guide** - How to convert existing components

### 5. Success Criteria

- Single library that ALL components use
- Eliminates custom WebSocket code in each component
- Works reliably from localhost AND network
- Relay can connect without issues
- Reduces connection debugging time by 90%+

## Code Examples Attached

I'm providing the current MCP server implementations from working components. Analyze these patterns and extract the BEST practices into a single reusable library.

**Focus on:**
- What makes Dewey reliable?
- Why do Godot/Horace timeout from network but work on localhost?
- What's the minimal correct implementation?

Design a library that makes networking invisible and bulletproof.

---

## RELAY IMPLEMENTATION (DO NOT CHANGE - Reference Only)

The relay works perfectly. Study how it connects to backends:

```python
#!/usr/bin/env python3
"""
Ultra-Stable WebSocket Relay - Never needs restarts.

Design: Forward all bytes without parsing. Auto-reconnect to backend.
Claude stays connected even when backend is down.
"""
import asyncio
import logging
import sys
from pathlib import Path

import websockets
import yaml

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class StableRelay:
    """Ultra-stable WebSocket relay - keeps Claude connection alive."""

    def __init__(self, config_path: str = "/app/config.yaml"):
        self.config_path = config_path
        self.backend_url = None
        self.reconnect_delay = 5
        self.load_config()

    def load_config(self):
        """Load backend URL from config file."""
        try:
            config = yaml.safe_load(Path(self.config_path).read_text())
            self.backend_url = config['backend']
            logger.info(f"Backend: {self.backend_url}")
        except Exception as e:
            logger.error(f"Config load failed: {e}")
            sys.exit(1)

    async def connect_backend(self, path: str = ""):
        """Connect to backend with infinite retry, forwarding client path."""
        backend_url = self.backend_url + path
        while True:
            try:
                ws = await websockets.connect(backend_url)
                logger.info(f"Backend connected: {backend_url}")
                return ws
            except Exception as e:
                logger.warning(f"Backend unavailable: {e} (retry in {self.reconnect_delay}s)")
                await asyncio.sleep(self.reconnect_delay)

    async def handle_client(self, client_ws):
        """Handle client connection - keep alive even if backend dies."""
        # Extract path from request URI (websockets 14.x compatibility)
        path = client_ws.request.path if hasattr(client_ws, 'request') else "/"
        client_id = id(client_ws)
        logger.info(f"Client {client_id} connected (path: {path})")
        backend_ws = await self.connect_backend(path)

        async def forward_client_to_backend():
            """Forward client → backend, reconnect backend if needed."""
            nonlocal backend_ws
            try:
                async for message in client_ws:
                    while True:
                        try:
                            await backend_ws.send(message)
                            break
                        except:
                            logger.warning(f"Client {client_id}: backend lost, reconnecting...")
                            backend_ws = await self.connect_backend(path)
            except websockets.exceptions.ConnectionClosed:
                logger.info(f"Client {client_id} disconnected")

        async def forward_backend_to_client():
            """Forward backend → client, reconnect backend if needed."""
            nonlocal backend_ws
            while True:
                try:
                    async for message in backend_ws:
                        await client_ws.send(message)
                except websockets.exceptions.ConnectionClosed:
                    if client_ws.closed:
                        break
                    logger.warning(f"Client {client_id}: backend lost, reconnecting...")
                    backend_ws = await self.connect_backend(path)
                except Exception as e:
                    logger.error(f"Client {client_id}: relay error: {e}")
                    break

        try:
            await asyncio.gather(
                forward_client_to_backend(),
                forward_backend_to_client()
            )
        finally:
            if not backend_ws.closed:
                await backend_ws.close()
            logger.info(f"Client {client_id} session ended")

    async def run(self, host: str = "0.0.0.0", port: int = 8000):
        """Start relay server."""
        logger.info(f"Stable Relay starting on {host}:{port}")
        async with websockets.serve(self.handle_client, host, port):
            logger.info(f"Stable Relay listening on ws://{host}:{port}")
            await asyncio.Future()


if __name__ == "__main__":
    relay = StableRelay()
    asyncio.run(relay.run())
```

**Key insight:** The relay successfully connects to all backends. The issue is in how backends implement their servers, NOT in how the relay connects.

# ICCM Network Code Extraction for iccm-network Library

## Task: Extract WebSocket networking patterns from working components

### Components to analyze:
1. **Dewey** - Most reliable, works consistently
2. **Fiedler** - Works with HTTP proxy integration
3. **Gates** - Document generation MCP
4. **Playfair** - Diagram generation MCP
5. **Marco** - Browser automation MCP
6. **Godot** - Logging MCP (has issues)
7. **Horace** - File storage MCP (has issues)

---

## 1. DEWEY - MCP Server Implementation

```python
# dewey/mcp_server.py
"""
WebSocket MCP server for Dewey.
"""
import asyncio
import json
import logging
import os
from typing import Any, Dict

import websockets
from websockets.server import WebSocketServerProtocol

from dewey import config
from dewey.database import db_pool
from dewey import tools
from dewey.godot import log_to_godot

logger = logging.getLogger(__name__)

# Godot logging configuration
GODOT_URL = os.getenv('GODOT_URL', 'ws://godot-mcp:9060')
LOGGING_ENABLED = os.getenv('LOGGING_ENABLED', 'true').lower() == 'true'

class DeweyMCPServer:
    """WebSocket MCP server for Dewey."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server = None

        # MCP tool definitions
        self.tools = {
            "dewey_begin_conversation": {
                "description": "Start a new conversation and return its ID",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Optional session ID to group conversations"},
                        "metadata": {"type": "object", "description": "Optional metadata as key-value pairs"}
                    }
                }
            },
            "dewey_store_message": {
                "description": "Store a single message in a conversation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "conversation_id": {"type": "string", "description": "UUID of the conversation"},
                        "role": {"type": "string", "description": "Role: user, assistant, or system"},
                        "content": {"type": "string", "description": "Message content"},
                        "turn_number": {"type": "integer", "description": "Optional turn number"},
                        "metadata": {"type": "object", "description": "Optional metadata"}
                    },
                    "required": ["conversation_id", "role", "content"]
                }
            },
            "dewey_store_messages_bulk": {
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
            },
            "dewey_get_conversation": {
                "description": "Retrieve all messages from a conversation",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "conversation_id": {"type": "string", "description": "UUID of the conversation"}
                    },
                    "required": ["conversation_id"]
                }
            },
            "dewey_list_conversations": {
                "description": "List conversations with pagination and filtering",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_id": {"type": "string", "description": "Optional session ID filter"},
                        "limit": {"type": "integer", "description": "Maximum results (default 20)"},
                        "offset": {"type": "integer", "description": "Number to skip (default 0)"},
                        "sort_by": {"type": "string", "description": "Sort field: created_at or updated_at"}
                    }
                }
            },
            "dewey_delete_conversation": {
                "description": "Delete a conversation and all its messages",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "conversation_id": {"type": "string", "description": "UUID of the conversation"},
                        "force": {"type": "boolean", "description": "Force delete without confirmation"}
                    },
                    "required": ["conversation_id"]
                }
            },
            "dewey_search": {
                "description": "Full-text search across all messages",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query string"},
                        "session_id": {"type": "string", "description": "Optional session ID filter"},
                        "start_date": {"type": "string", "description": "Optional start date (ISO format)"},
                        "end_date": {"type": "string", "description": "Optional end date (ISO format)"},
                        "limit": {"type": "integer", "description": "Maximum results (default 20)"},
                        "offset": {"type": "integer", "description": "Number to skip (default 0)"}
                    },
                    "required": ["query"]
                }
            },
            "dewey_get_startup_context": {
                "description": "Get the active startup context or a specific one by name",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Optional context name (defaults to active)"}
                    }
                }
            },
            "dewey_set_startup_context": {
                "description": "Create or update a startup context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Context name"},
                        "content": {"type": "string", "description": "Context content"},
                        "set_active": {"type": "boolean", "description": "Make this the active context"}
                    },
                    "required": ["name", "content"]
                }
            },
            "dewey_list_startup_contexts": {
                "description": "List all available startup contexts",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_content": {"type": "boolean", "description": "Include full content (default false)"}
                    }
                }
            },
            "dewey_delete_startup_context": {
                "description": "Delete a startup context",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string", "description": "Context name"},
                        "force": {"type": "boolean", "description": "Force delete without confirmation"}
                    },
                    "required": ["name"]
                }
            },
            "dewey_query_logs": {
                "description": "Query logs with various filters",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "trace_id": {"type": "string", "description": "Filter by trace ID"},
                        "component": {"type": "string", "description": "Filter by component name"},
                        "level": {"type": "string", "description": "Minimum log level (TRACE, DEBUG, INFO, WARN, ERROR)"},
                        "start_time": {"type": "string", "description": "Start time filter (ISO format)"},
                        "end_time": {"type": "string", "description": "End time filter (ISO format)"},
                        "search": {"type": "string", "description": "Full-text search in message"},
                        "limit": {"type": "integer", "description": "Maximum results (1-1000, default 100)"}
                    }
                }
            },
            "dewey_get_log_stats": {
                "description": "Get statistics about the logs table",
                "inputSchema": {
                    "type": "object",
                    "properties": {}
                }
            }
        }

    async def handle_request(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connection."""
        logger.info(f"Client connected from {websocket.remote_address}")

        if LOGGING_ENABLED:
            await log_to_godot('INFO', 'Client connected', data={'client': str(websocket.remote_address)})

        try:
            async for message in websocket:
                try:
                    # Parse JSON-RPC request
                    request = json.loads(message)
                    response = await self.process_request(request)

                    # Send response
                    await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        },
                        "id": None
                    }
                    await websocket.send(json.dumps(error_response))

                except Exception as e:
                    logger.exception("Error processing request")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": str(e)
                        },
                        "id": request.get("id") if isinstance(request, dict) else None
                    }
                    await websocket.send(json.dumps(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a JSON-RPC request."""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        logger.debug(f"Processing request: {method}")

        if LOGGING_ENABLED:
            await log_to_godot('TRACE', 'MCP request received', data={'method': method, 'id': request_id})

        # Handle MCP protocol methods
        if method == "initialize":
            logger.info("Client initializing MCP connection")
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "dewey-mcp-server",
                        "version": "1.0.0"
                    }
                },
                "id": request_id
            }

        elif method == "tools/list":
            logger.info(f"Listing {len(self.tools)} available tools")
            return {
                "jsonrpc": "2.0",
                "result": {
                    "tools": [
                        {"name": name, **spec}
                        for name, spec in self.tools.items()
                    ]
                },
                "id": request_id
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            logger.info(f"Calling tool: {tool_name}")

            if LOGGING_ENABLED:
                await log_to_godot('TRACE', 'Tool call started', data={'tool_name': tool_name, 'has_args': bool(arguments)})

            # Get the tool function
            tool_func = getattr(tools, tool_name, None)
            if not tool_func or not callable(tool_func):
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Tool not found: {tool_name}"
                    },
                    "id": request_id
                }

            try:
                # Execute the tool
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(**arguments)
                else:
                    result = tool_func(**arguments)

                if LOGGING_ENABLED:
                    await log_to_godot('TRACE', 'Tool call completed', data={'tool_name': tool_name, 'success': True})

                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    },
                    "id": request_id
                }

            except tools.ToolError as e:
                logger.error(f"Tool {tool_name} error: {str(e)}")

                if LOGGING_ENABLED:
                    await log_to_godot('ERROR', 'Tool error', data={'tool_name': tool_name, 'error': str(e), 'code': e.code})

                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": e.code,
                        "message": str(e)
                    },
                    "id": request_id
                }

            except Exception as e:
                logger.exception(f"Error executing tool {tool_name}")

                if LOGGING_ENABLED:
                    await log_to_godot('ERROR', 'Tool execution error', data={'tool_name': tool_name, 'error': str(e), 'type': type(e).__name__})

                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": f"Internal error: {str(e)}"
                    },
                    "id": request_id
                }

        # Legacy direct tool call (for backward compatibility)
        else:
            # Get the tool function
            tool_func = getattr(tools, method, None)
            if not tool_func or not callable(tool_func):
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    },
                    "id": request_id
                }

        try:
            # Execute the tool
            if asyncio.iscoroutinefunction(tool_func):
                result = await tool_func(**params)
            else:
                result = tool_func(**params)

            return {
                "jsonrpc": "2.0",
                "result": result,
                "id": request_id
            }

        except tools.ToolError as e:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": e.code,
                    "message": str(e)
                },
                "id": request_id
            }

        except Exception as e:
            logger.exception(f"Error executing tool {method}")
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": f"Internal error: {str(e)}"
                },
                "id": request_id
            }

    async def start(self):
        """Start the MCP server."""
        logger.info(f"Starting Dewey MCP Server on {self.host}:{self.port}")

        if LOGGING_ENABLED:
            await log_to_godot('INFO', 'Dewey MCP server starting', data={'host': self.host, 'port': self.port, 'godot_url': GODOT_URL})

        self.server = await websockets.serve(
            self.handle_request,
            self.host,
            self.port
        )

        logger.info("Dewey MCP Server is running")

        if LOGGING_ENABLED:
            await log_to_godot('INFO', 'Dewey MCP server running', data={'status': 'operational'})

    async def stop(self):
        """Stop the MCP server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Dewey MCP Server stopped")


async def main():
    """Main entry point for Dewey MCP Server."""
    # Initialize async database
    await db_pool.initialize()

    # Create and start server
    server = DeweyMCPServer(config.MCP_HOST, config.MCP_PORT)
    await server.start()

    # Run forever
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await server.stop()
        await db_pool.close()


if __name__ == "__main__":
    asyncio.run(main())
```

## 2. FIEDLER - MCP Server Implementation

```python
"""Fiedler MCP Server - Orchestra Conductor Prototype."""
import asyncio
import os
import sys
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from .tools import (
    fiedler_list_models,
    fiedler_set_models,
    fiedler_set_output,
    fiedler_get_config,
    fiedler_send,
    fiedler_set_key,
    fiedler_delete_key,
    fiedler_list_keys,
)

# Initialize Godot MCP logger
from .godot import log_to_godot

GODOT_URL = os.getenv('GODOT_URL', 'ws://godot-mcp:9060')
LOGGING_ENABLED = os.getenv('LOGGING_ENABLED', 'true').lower() == 'true'


# Create server instance
app = Server("fiedler")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available Fiedler tools."""
    return [
        Tool(
            name="fiedler_list_models",
            description="List all available LLM models with their properties (name, provider, aliases, max_tokens, capabilities).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="fiedler_set_models",
            description="Configure default models for fiedler_send. Accepts list of model IDs or aliases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model IDs or aliases to use as defaults",
                    },
                },
                "required": ["models"],
            },
        ),
        Tool(
            name="fiedler_set_output",
            description="Configure output directory for fiedler_send results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_dir": {
                        "type": "string",
                        "description": "Path to output directory",
                    },
                },
                "required": ["output_dir"],
            },
        ),
        Tool(
            name="fiedler_get_config",
            description="Get current Fiedler configuration (models, output_dir, total_available_models).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="fiedler_send",
            description="Send prompt and optional package files to configured LLMs. Returns results from all models in parallel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "User prompt or question to send to models",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of file paths to compile into package",
                    },
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional override of default models (use model IDs or aliases)",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="fiedler_set_key",
            description="Store API key securely in system keyring (encrypted). Replaces need for environment variables.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name: google, openai, together, or xai",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key to store (will be encrypted by OS keyring)",
                        "format": "password",
                        "writeOnly": True,
                    },
                },
                "required": ["provider", "api_key"],
            },
        ),
        Tool(
            name="fiedler_delete_key",
            description="Delete stored API key from system keyring.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name: google, openai, together, or xai",
                    },
                },
                "required": ["provider"],
            },
        ),
        Tool(
            name="fiedler_list_keys",
            description="List providers that have API keys stored in system keyring.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# Tool dispatch table for sync tools
SYNC_TOOL_DISPATCH = {
    "fiedler_list_models": lambda args: fiedler_list_models(),
    "fiedler_set_models": lambda args: fiedler_set_models(args["models"]),
    "fiedler_set_output": lambda args: fiedler_set_output(args["output_dir"]),
    "fiedler_get_config": lambda args: fiedler_get_config(),
    "fiedler_set_key": lambda args: fiedler_set_key(args["provider"], args["api_key"]),
    "fiedler_delete_key": lambda args: fiedler_delete_key(args["provider"]),
    "fiedler_list_keys": lambda args: fiedler_list_keys(),
}


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls with dictionary-based dispatch and structured errors."""
    import json

    try:
        # Special handling for async fiedler_send
        if name == "fiedler_send":
            result = await asyncio.to_thread(
                fiedler_send,
                prompt=arguments["prompt"],
                files=arguments.get("files"),
                models=arguments.get("models"),
            )
        # Dispatch sync tools
        elif name in SYNC_TOOL_DISPATCH:
            result = SYNC_TOOL_DISPATCH[name](arguments)
        else:
            # Unknown tool
            error_response = {
                "error": {
                    "code": "UNKNOWN_TOOL",
                    "message": f"Unknown tool: {name}",
                    "tool_name": name
                }
            }
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

        # Format successful result
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except KeyError as e:
        # Missing required argument
        error_response = {
            "error": {
                "code": "MISSING_ARGUMENT",
                "message": f"Missing required argument: {e}",
                "argument": str(e)
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    except ValueError as e:
        # Invalid input (e.g., unknown model, file too large)
        error_response = {
            "error": {
                "code": "INVALID_INPUT",
                "message": str(e)
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    except PermissionError as e:
        # File access denied
        error_response = {
            "error": {
                "code": "ACCESS_DENIED",
                "message": str(e)
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    except FileNotFoundError as e:
        # File not found
        error_response = {
            "error": {
                "code": "FILE_NOT_FOUND",
                "message": str(e)
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    except RuntimeError as e:
        # Provider/configuration errors
        error_response = {
            "error": {
                "code": "PROVIDER_ERROR",
                "message": str(e)
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    except Exception as e:
        # Unexpected internal error
        error_response = {
            "error": {
                "code": "INTERNAL_ERROR",
                "message": f"Internal error: {str(e)}",
                "type": type(e).__name__
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]


async def _amain():
    """Async entry point for MCP server - WebSocket mode + HTTP Streaming Proxy."""
    import sys
    import websockets
    import json
    import logging
    from websockets.server import WebSocketServerProtocol
    from .proxy_server import start_proxy_server

    print("=== FIEDLER: _amain() ENTRY POINT ===", flush=True, file=sys.stderr)

    # Force logging to stderr with explicit configuration
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        stream=sys.stderr,
        force=True
    )
    logger = logging.getLogger(__name__)
    print("=== FIEDLER: Logger configured ===", flush=True, file=sys.stderr)
    logger.info("=== FIEDLER STARTUP: _amain() called ===")

    if LOGGING_ENABLED:
        await log_to_godot('INFO', 'Fiedler MCP server starting', data={'godot_url': GODOT_URL})

    async def handle_client(websocket: WebSocketServerProtocol):
        """Handle WebSocket client connection."""
        logger.info(f"=== FIEDLER: Client connected from {websocket.remote_address} ===")
        if LOGGING_ENABLED:
            await log_to_godot('INFO', 'Client connected', data={'client': str(websocket.remote_address)})

        try:
            async for message in websocket:
                try:
                    # Parse MCP request
                    request = json.loads(message)
                    method = request.get("method")
                    params = request.get("params", {})
                    request_id = request.get("id")

                    logger.info(f"Received request: {method}")
                    if LOGGING_ENABLED:
                        await log_to_godot('TRACE', 'MCP request received', data={'method': method, 'id': request_id})

                    # Handle MCP protocol methods
                    if method == "initialize":
                        response = {
                            "jsonrpc": "2.0",
                            "result": {
                                "protocolVersion": "2024-11-05",
                                "capabilities": {"tools": {}},
                                "serverInfo": {
                                    "name": "fiedler",
                                    "version": "1.0.0"
                                }
                            },
                            "id": request_id
                        }

                    elif method == "tools/list":
                        tools_list = await list_tools()
                        response = {
                            "jsonrpc": "2.0",
                            "result": {
                                "tools": [
                                    {
                                        "name": tool.name,
                                        "description": tool.description,
                                        "inputSchema": tool.inputSchema
                                    }
                                    for tool in tools_list
                                ]
                            },
                            "id": request_id
                        }

                    elif method == "tools/call":
                        tool_name = params.get("name")
                        arguments = params.get("arguments", {})

                        logger.info(f"Calling tool: {tool_name}")
                        if LOGGING_ENABLED:
                            await log_to_godot('TRACE', 'Tool call started', data={'tool_name': tool_name, 'has_args': bool(arguments)})

                        # Call the tool via the app's handler
                        result = await call_tool(tool_name, arguments)

                        if LOGGING_ENABLED:
                            await log_to_godot('TRACE', 'Tool call completed', data={'tool_name': tool_name, 'success': True})

                        response = {
                            "jsonrpc": "2.0",
                            "result": {
                                "content": [
                                    {
                                        "type": "text",
                                        "text": item.text
                                    }
                                    for item in result
                                ]
                            },
                            "id": request_id
                        }

                    else:
                        response = {
                            "jsonrpc": "2.0",
                            "error": {
                                "code": -32601,
                                "message": f"Method not found: {method}"
                            },
                            "id": request_id
                        }

                    # Send response
                    await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        },
                        "id": None
                    }
                    await websocket.send(json.dumps(error_response))

                except Exception as e:
                    logger.exception(f"Error processing request: {e}")
                    if LOGGING_ENABLED:
                        await log_to_godot('ERROR', 'Request processing error', data={'error': str(e), 'type': type(e).__name__})
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": str(e)
                        },
                        "id": request_id if 'request_id' in locals() else None
                    }
                    await websocket.send(json.dumps(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")

    # Start HTTP Streaming Proxy on port 8081
    print("=== FIEDLER: Starting HTTP Streaming Proxy ===", flush=True, file=sys.stderr)
    proxy = await start_proxy_server(host="0.0.0.0", port=8081)
    print("=== FIEDLER: HTTP Streaming Proxy STARTED on port 8081 ===", flush=True, file=sys.stderr)
    logger.info("=== FIEDLER: HTTP Streaming Proxy RUNNING on port 8081 ===")

    # Start WebSocket MCP server on port 8080
    host = "0.0.0.0"
    port = 8080
    print(f"=== FIEDLER: About to start WebSocket server on {host}:{port} ===", flush=True, file=sys.stderr)
    logger.info(f"=== FIEDLER STARTUP: Starting WebSocket MCP server on {host}:{port} ===")

    try:
        print(f"=== FIEDLER: Calling websockets.serve ===", flush=True, file=sys.stderr)
        async with websockets.serve(handle_client, host, port):
            print(f"=== FIEDLER: WebSocket server STARTED on port {port} ===", flush=True, file=sys.stderr)
            logger.info(f"=== FIEDLER STARTUP: WebSocket MCP server RUNNING on ws://{host}:{port} ===")
            logger.info(f"=== FIEDLER STARTUP: Both servers operational (MCP: 8080, Proxy: 8081) ===")
            await asyncio.Future()  # Run forever
    except Exception as e:
        print(f"=== FIEDLER ERROR: {e} ===", flush=True, file=sys.stderr)
        logger.error(f"=== FIEDLER ERROR: Failed to start WebSocket server: {e} ===", exc_info=True)
        raise


def main():
    """Synchronous entry point for console script."""
    import sys
    print("=== FIEDLER: main() ENTRY POINT ===", flush=True, file=sys.stderr)
    print("=== FIEDLER: About to call asyncio.run(_amain()) ===", flush=True, file=sys.stderr)
    try:
        asyncio.run(_amain())
    except Exception as e:
        print(f"=== FIEDLER FATAL ERROR: {e} ===", flush=True, file=sys.stderr)
        import traceback
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
```

## 3. GATES - MCP Server Implementation

```python
```

## 4. PLAYFAIR - MCP Server Implementation

```python
```

## 5. GODOT - MCP Server Implementation (has connection issues)

```python
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
```

## 6. HORACE - MCP Server Implementation (has connection issues)

```python
# src/mcp_server.py
import asyncio
import json
import signal
from typing import Any, Dict

import websockets
from websockets.server import serve as websockets_serve

from .config import config, Config
from .database import Database
from .tools import Tools
from .utils.logger import logger, godot_info, godot_error
from .exceptions import HoraceError, DuplicateFileError

db = Database()
tools_instance: Tools = None


class HoraceMCPServer:
    """WebSocket MCP server for Horace."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server = None

        # MCP tool definitions
        self.tools = {
            "horace_register_file": {
                "description": "Register a file with Horace catalog",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_path": {
                            "type": "string",
                            "description": "Absolute path to file in /mnt/irina_storage/"
                        },
                        "metadata": {
                            "type": "object",
                            "description": "File metadata",
                            "properties": {
                                "owner": {"type": "string"},
                                "purpose": {"type": "string"},
                                "tags": {"type": "array", "items": {"type": "string"}},
                                "collection": {"type": "string"},
                                "correlation_id": {"type": "string"}
                            }
                        }
                    },
                    "required": ["file_path", "metadata"]
                }
            },
            "horace_search_files": {
                "description": "Search file catalog by metadata",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "tags": {"type": "array", "items": {"type": "string"}},
                        "owner": {"type": "string"},
                        "created_after": {"type": "string"},
                        "created_before": {"type": "string"},
                        "file_type": {"type": "string"},
                        "min_size": {"type": "integer"},
                        "max_size": {"type": "integer"},
                        "collection": {"type": "string"},
                        "status": {"type": "string"},
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"}
                    }
                }
            },
            "horace_get_file_info": {
                "description": "Get detailed metadata for specific file",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string"},
                        "include_versions": {"type": "boolean"}
                    },
                    "required": ["file_id"]
                }
            },
            "horace_create_collection": {
                "description": "Create or update a file collection",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "file_ids": {"type": "array", "items": {"type": "string"}},
                        "metadata": {"type": "object"}
                    },
                    "required": ["name"]
                }
            },
            "horace_list_collections": {
                "description": "List all collections",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "limit": {"type": "integer"},
                        "offset": {"type": "integer"}
                    }
                }
            },
            "horace_update_file": {
                "description": "Update file metadata or trigger re-versioning",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string"},
                        "metadata": {"type": "object"},
                        "check_for_changes": {"type": "boolean"}
                    },
                    "required": ["file_id"]
                }
            },
            "horace_restore_version": {
                "description": "Restore a file to a previous version",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "file_id": {"type": "string"},
                        "version": {"type": "integer"}
                    },
                    "required": ["file_id", "version"]
                }
            }
        }

    async def handle_request(self, websocket, path: str):
        """Handle incoming WebSocket connection."""
        logger.info(f"Client connected from {websocket.remote_address}")
        await godot_info('Client connected to Horace', context={'client': str(websocket.remote_address)})

        try:
            async for message in websocket:
                try:
                    # Parse JSON-RPC request
                    request = json.loads(message)
                    response = await self.process_request(request)

                    # Send response
                    await websocket.send(json.dumps(response))

                except json.JSONDecodeError:
                    logger.error("Invalid JSON received")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32700,
                            "message": "Parse error"
                        },
                        "id": None
                    }
                    await websocket.send(json.dumps(error_response))

                except Exception as e:
                    logger.exception("Error processing request")
                    error_response = {
                        "jsonrpc": "2.0",
                        "error": {
                            "code": -32603,
                            "message": str(e)
                        },
                        "id": request.get("id") if isinstance(request, dict) else None
                    }
                    await websocket.send(json.dumps(error_response))

        except websockets.exceptions.ConnectionClosed:
            logger.info("Client disconnected")

    async def process_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a JSON-RPC request."""
        request_id = request.get("id")
        method = request.get("method")
        params = request.get("params", {})

        logger.debug(f"Processing request: {method}")

        # Handle MCP protocol methods
        if method == "initialize":
            logger.info("Client initializing MCP connection")
            return {
                "jsonrpc": "2.0",
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {
                        "tools": {}
                    },
                    "serverInfo": {
                        "name": "horace-mcp-server",
                        "version": "1.0.0"
                    }
                },
                "id": request_id
            }

        elif method == "tools/list":
            logger.info(f"Listing {len(self.tools)} available tools")
            return {
                "jsonrpc": "2.0",
                "result": {
                    "tools": [
                        {"name": name, **spec}
                        for name, spec in self.tools.items()
                    ]
                },
                "id": request_id
            }

        elif method == "tools/call":
            tool_name = params.get("name")
            arguments = params.get("arguments", {})
            logger.info(f"Calling tool: {tool_name}")

            await godot_info(f'Tool call: {tool_name}', context={'has_args': bool(arguments)})

            # Get the tool function
            tool_func = getattr(tools_instance, tool_name, None)
            if not tool_func:
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Tool not found: {tool_name}"
                    },
                    "id": request_id
                }

            try:
                # Execute tool
                result = await tool_func(arguments)

                # Handle duplicate file special case
                if isinstance(result, dict) and "file_id" in result:
                    return {
                        "jsonrpc": "2.0",
                        "result": {
                            "content": [
                                {
                                    "type": "text",
                                    "text": json.dumps(result, indent=2, default=str)
                                }
                            ]
                        },
                        "id": request_id
                    }

                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2, default=str)
                            }
                        ]
                    },
                    "id": request_id
                }

            except DuplicateFileError as e:
                # Per spec, return existing file_id with success
                result = {"file_id": e.file_id, "message": e.message}
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(result, indent=2)
                            }
                        ]
                    },
                    "id": request_id
                }

            except HoraceError as e:
                await godot_error(f"Horace Error in {tool_name}: {e.message}", context={"error_code": e.error_code})
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32000,
                        "message": e.message,
                        "data": {"error_code": e.error_code}
                    },
                    "id": request_id
                }

            except Exception as e:
                await godot_error(f"Unhandled exception in {tool_name}: {e}")
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32603,
                        "message": str(e)
                    },
                    "id": request_id
                }

        else:
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32601,
                    "message": f"Unknown method: {method}"
                },
                "id": request_id
            }

    async def start(self):
        """Start the WebSocket MCP server."""
        self.server = await websockets_serve(
            self.handle_request,
            self.host,
            self.port
        )
        logger.info(f"Horace MCP Server started at ws://{self.host}:{self.port}")
        await godot_info('Horace MCP server running', context={'status': 'operational'})

    async def stop(self):
        """Stop the MCP server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("Horace MCP Server stopped")


async def main():
    """Main entry point for Horace MCP Server."""
    global tools_instance

    # Initialize storage directories
    Config.initialize_storage_dirs()
    logger.info("Storage directories initialized.")

    # Initialize database connection
    await db.connect()
    tools_instance = Tools(db)

    # Create and start server
    server = HoraceMCPServer(config.MCP_HOST, config.MCP_PORT)
    await server.start()

    # Run forever
    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await server.stop()
        await db.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
```
