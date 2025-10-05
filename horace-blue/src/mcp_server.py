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
