# Review Request: iccm-network Library Synthesis

## Context

You previously provided designs for a standardized MCP networking library to solve recurring WebSocket connection issues in the ICCM system. I have now synthesized all three responses (GPT-4o-mini, Gemini-2.5-pro, DeepSeek-R1) into a final implementation.

**Your task:** Review the synthesized library and provide feedback on:
1. **Correctness** - Does it correctly implement JSON-RPC 2.0 and MCP protocol?
2. **Completeness** - Is anything missing from your original design?
3. **Best Practices** - Any improvements to make it more robust?
4. **API Design** - Is the API intuitive and easy to use?
5. **Migration** - Will this work for all ICCM components (Dewey, Fiedler, Godot, Horace)?

**Approval Criteria:**
- If you approve without changes, say: **"APPROVED - Ready for deployment"**
- If you have suggestions, provide specific code changes
- If you find critical issues, explain what must be fixed

---

## Original Problem Statement (Reminder)

10+ hours wasted debugging WebSocket connection issues:
- Components binding to `127.0.0.1` instead of `0.0.0.0` (unreachable from network)
- Inconsistent JSON-RPC 2.0 implementations
- Different error handling across components
- Same bugs fixed multiple times

**Goal:** Single library that makes networking invisible and bulletproof.

**Constraints:**
- MCP relay will NOT change (works perfectly)
- Must work from localhost AND network (container-to-container)
- Zero configuration required

---

## Synthesized Implementation

### File 1: iccm_network/server.py

```python
"""
iccm-network: Standard MCP Server Implementation

This module provides a standardized WebSocket MCP server that eliminates
connection issues caused by inconsistent binding and configuration.

Key Design Principles:
- ALWAYS bind to 0.0.0.0 for container-to-container communication
- Implement JSON-RPC 2.0 protocol correctly
- Standardized error handling with MCPToolError
- Zero configuration required - just works

Based on synthesis of:
- Dewey (most reliable reference implementation)
- Fiedler (HTTP proxy integration patterns)
- Gemini-2.5-pro analysis (root cause identification)
"""

import asyncio
import json
import logging
from typing import Dict, Any, Callable, Awaitable, Optional
import traceback

import websockets
from websockets.server import WebSocketServerProtocol


class MCPToolError(Exception):
    """
    Custom exception for tool-specific errors.

    Provides standardized error responses following JSON-RPC 2.0 spec.

    Args:
        message: Human-readable error description
        code: JSON-RPC error code (default -32000 for server error)
        data: Optional additional error data
    """
    def __init__(self, message: str, code: int = -32000, data: Any = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.data = data


class MCPServer:
    """
    Standardized MCP WebSocket server.

    Handles all WebSocket connection, JSON-RPC protocol, and tool routing.
    Components only need to provide tool definitions and handlers.

    Example:
        ```python
        server = MCPServer(
            name="horace",
            version="1.0.0",
            port=8070,
            tool_definitions={
                "horace_register_file": {
                    "description": "Register a file",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "file_path": {"type": "string"}
                        },
                        "required": ["file_path"]
                    }
                }
            },
            tool_handlers={
                "horace_register_file": register_file_handler
            }
        )
        await server.start()
        ```

    Args:
        name: Service name (e.g., "horace", "dewey")
        version: Service version (e.g., "1.0.0")
        port: Port to listen on
        tool_definitions: Dict mapping tool names to MCP tool schemas
        tool_handlers: Dict mapping tool names to async handler functions
        logger: Optional custom logger (creates default if None)
    """

    def __init__(
        self,
        name: str,
        version: str,
        port: int,
        tool_definitions: Dict[str, Any],
        tool_handlers: Dict[str, Callable[..., Awaitable[Any]]],
        logger: Optional[logging.Logger] = None
    ):
        self.name = name
        self.version = version
        self.port = port
        self.tool_definitions = tool_definitions
        self.tool_handlers = tool_handlers

        # CRITICAL: Always bind to 0.0.0.0 for container networking
        # This is NOT configurable - it must be 0.0.0.0 for relay to connect
        self.host = "0.0.0.0"

        # Setup logging
        self.logger = logger or self._create_default_logger()

        # Server instance (set during start())
        self._server = None

    def _create_default_logger(self) -> logging.Logger:
        """Create default logger with standard format."""
        logger = logging.getLogger(f"iccm_network.{self.name}")
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                f'%(asctime)s - {self.name} - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP 'initialize' method.

        Returns server capabilities and metadata.
        """
        self.logger.info(f"Initialize request from client: {params.get('clientInfo', {})}")

        return {
            "protocolVersion": "2024-11-05",
            "capabilities": {
                "tools": {}
            },
            "serverInfo": {
                "name": self.name,
                "version": self.version
            }
        }

    async def _handle_tools_list(self) -> Dict[str, Any]:
        """
        Handle MCP 'tools/list' method.

        Returns list of available tools with their schemas.
        """
        tools = [
            {
                "name": name,
                **definition
            }
            for name, definition in self.tool_definitions.items()
        ]

        self.logger.info(f"Tools list requested: {len(tools)} tools available")
        return {"tools": tools}

    async def _handle_tools_call(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle MCP 'tools/call' method.

        Routes to appropriate tool handler and handles errors.
        """
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name not in self.tool_handlers:
            raise MCPToolError(
                f"Tool '{tool_name}' not found",
                code=-32601  # Method not found
            )

        self.logger.info(f"Calling tool: {tool_name}")

        try:
            handler = self.tool_handlers[tool_name]
            result = await handler(**arguments)

            return {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(result, indent=2)
                    }
                ]
            }

        except MCPToolError:
            # Re-raise MCPToolError as-is
            raise

        except Exception as e:
            # Wrap unexpected errors
            self.logger.error(f"Tool {tool_name} failed: {e}\n{traceback.format_exc()}")
            raise MCPToolError(
                f"Tool execution failed: {str(e)}",
                code=-32603,  # Internal error
                data={"traceback": traceback.format_exc()}
            )

    async def _handle_message(self, message_str: str) -> str:
        """
        Handle incoming JSON-RPC message.

        Parses, routes to handler, and formats response.
        """
        try:
            message = json.loads(message_str)
        except json.JSONDecodeError as e:
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32700,  # Parse error
                    "message": f"Invalid JSON: {str(e)}"
                },
                "id": None
            })

        method = message.get("method")
        params = message.get("params", {})
        msg_id = message.get("id")

        try:
            # Route to appropriate handler
            if method == "initialize":
                result = await self._handle_initialize(params)
            elif method == "tools/list":
                result = await self._handle_tools_list()
            elif method == "tools/call":
                result = await self._handle_tools_call(params)
            else:
                raise MCPToolError(
                    f"Method '{method}' not supported",
                    code=-32601  # Method not found
                )

            # Success response
            return json.dumps({
                "jsonrpc": "2.0",
                "result": result,
                "id": msg_id
            })

        except MCPToolError as e:
            # Standardized error response
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {
                    "code": e.code,
                    "message": e.message,
                    "data": e.data
                },
                "id": msg_id
            })

        except Exception as e:
            # Unexpected error
            self.logger.error(f"Unexpected error handling {method}: {e}\n{traceback.format_exc()}")
            return json.dumps({
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,  # Internal error
                    "message": f"Internal server error: {str(e)}",
                    "data": {"traceback": traceback.format_exc()}
                },
                "id": msg_id
            })

    async def _connection_handler(self, websocket: WebSocketServerProtocol):
        """
        Handle WebSocket connection lifecycle.

        Processes messages until client disconnects.
        """
        client_id = id(websocket)
        self.logger.info(f"Client {client_id} connected")

        try:
            async for message in websocket:
                response = await self._handle_message(message)
                await websocket.send(response)

        except websockets.exceptions.ConnectionClosed:
            self.logger.info(f"Client {client_id} disconnected")
        except Exception as e:
            self.logger.error(f"Connection error for client {client_id}: {e}")
        finally:
            self.logger.info(f"Client {client_id} session ended")

    async def start(self):
        """
        Start the MCP server and run forever.

        Binds to 0.0.0.0:{port} and handles connections.
        """
        self.logger.info(f"Starting {self.name} MCP server v{self.version}")
        self.logger.info(f"Binding to {self.host}:{self.port}")
        self.logger.info(f"Tools available: {list(self.tool_definitions.keys())}")

        self._server = await websockets.serve(
            self._connection_handler,
            self.host,
            self.port
        )

        self.logger.info(f"✓ {self.name} MCP server listening on ws://{self.host}:{self.port}")

        # Run forever
        await asyncio.Future()

    async def stop(self):
        """
        Gracefully stop the server.

        Useful for testing and controlled shutdown.
        """
        if self._server:
            self.logger.info(f"Stopping {self.name} MCP server")
            self._server.close()
            await self._server.wait_closed()
            self.logger.info(f"✓ {self.name} MCP server stopped")
```

### File 2: iccm_network/__init__.py

```python
"""
iccm-network: Standardized MCP networking library for ICCM components

This library eliminates the recurring WebSocket connection issues that have
plagued ICCM development by providing a battle-tested, zero-configuration
MCP server implementation.

Problem Solved:
- Components binding to 127.0.0.1 instead of 0.0.0.0 (can't connect from network)
- Inconsistent JSON-RPC 2.0 protocol implementation
- Different error handling approaches across components
- Hours wasted debugging timeouts and connection failures

Usage:
    ```python
    from iccm_network import MCPServer, MCPToolError

    # Define your tools
    async def my_tool_handler(arg1: str, arg2: int) -> dict:
        if not arg1:
            raise MCPToolError("arg1 is required", code=-32602)
        return {"result": f"Processed {arg1} with {arg2}"}

    # Create server
    server = MCPServer(
        name="my_service",
        version="1.0.0",
        port=9000,
        tool_definitions={
            "my_tool": {
                "description": "Does something useful",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "arg1": {"type": "string"},
                        "arg2": {"type": "integer"}
                    },
                    "required": ["arg1", "arg2"]
                }
            }
        },
        tool_handlers={
            "my_tool": my_tool_handler
        }
    )

    # Start server (runs forever)
    await server.start()
    ```

Components using this library:
- Horace (testbed - first implementation)
- Dewey (migration planned)
- Fiedler (migration planned)
- Godot (migration planned)

Design based on synthesis of:
- Dewey's reliable implementation patterns
- Fiedler's HTTP proxy integration
- Gemini-2.5-pro's root cause analysis
- GPT-4o-mini's simplified API design
- DeepSeek's decorator pattern considerations

Version: 1.0.0
"""

from .server import MCPServer, MCPToolError

__version__ = "1.0.0"
__all__ = ["MCPServer", "MCPToolError"]
```

### File 3: pyproject.toml

```toml
[build-system]
requires = ["setuptools>=61.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "iccm-network"
version = "1.0.0"
description = "Standardized MCP networking library for ICCM microservices"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "ICCM Project", email = "iccm@example.com"}
]
keywords = ["mcp", "websocket", "microservices", "iccm"]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Developers",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
]

dependencies = [
    "websockets>=11.0,<15.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-asyncio>=0.21",
    "black>=23.0",
    "mypy>=1.0",
]

[project.urls]
Homepage = "https://github.com/rmdevpro/ICCM"
Issues = "https://github.com/rmdevpro/ICCM/issues"
Repository = "https://github.com/rmdevpro/ICCM/tree/main/iccm-network"

[tool.setuptools.packages.find]
where = ["."]
include = ["iccm_network*"]

[tool.black]
line-length = 100
target-version = ['py310', 'py311', 'py312']

[tool.mypy]
python_version = "3.10"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = false
```

---

## Questions for Review

1. **Protocol Compliance:** Does the implementation correctly follow JSON-RPC 2.0 and MCP protocol specifications?

2. **Error Handling:** Are the error codes and error structure correct? Should we handle any additional edge cases?

3. **API Simplicity:** Is the API as simple as possible while still being flexible?

4. **0.0.0.0 Binding:** Is the forced `0.0.0.0` binding implemented correctly? Should this EVER be configurable?

5. **Migration Path:** Will this work for all existing ICCM components?
   - Dewey (most reliable, currently works)
   - Fiedler (works, has HTTP proxy integration)
   - Godot (has network connection issues)
   - Horace (has network issues, uses non-standard protocol)

6. **Missing Features:** Is anything critical missing? (e.g., client-side library, health checks, metrics, etc.)

7. **Best Practices:** Any WebSocket or asyncio best practices we should add?

---

## Example Migration (Horace)

**Before (custom protocol):**
```python
# Old: 80+ lines of custom WebSocket handling
async def mcp_handler(websocket):
    async for message in websocket:
        data = json.loads(message)
        tool_name = data.get("tool")  # Non-standard protocol
        params = data.get("params", {})
        # ... custom error handling
        # ... custom response format

async def main():
    async with serve(mcp_handler, config.MCP_HOST, config.MCP_PORT):  # BUG: Often 127.0.0.1
        await stop
```

**After (iccm-network):**
```python
from iccm_network import MCPServer, MCPToolError

TOOLS = {
    "horace_register_file": {
        "description": "Register a file with metadata",
        "inputSchema": {...}
    },
    # ... other tools
}

HANDLERS = {
    "horace_register_file": register_file_handler,
    # ... other handlers
}

async def main():
    server = MCPServer(
        name="horace",
        version="1.0.0",
        port=8070,
        tool_definitions=TOOLS,
        tool_handlers=HANDLERS
    )
    await server.start()
```

---

## Your Verdict

Please provide one of the following responses:

1. **"APPROVED - Ready for deployment"** - No changes needed, library is production-ready
2. **"APPROVED WITH SUGGESTIONS"** - Library is good but here are optional improvements: [list]
3. **"CHANGES REQUIRED"** - Critical issues that must be fixed: [detailed list with code snippets]

If you suggest changes, please provide specific code snippets showing exactly what to change.

Thank you for your review!
