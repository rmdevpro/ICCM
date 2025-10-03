# dewey/mcp_server.py
"""
WebSocket MCP server for Dewey.
"""
import asyncio
import json
import logging
from typing import Any, Dict

import websockets
from websockets.server import WebSocketServerProtocol

from dewey import config
from dewey.database import db_pool
from dewey import tools

logger = logging.getLogger(__name__)

class DeweyMCPServer:
    """WebSocket MCP server for Dewey."""

    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.server = None

    async def handle_request(self, websocket: WebSocketServerProtocol, path: str):
        """Handle incoming WebSocket connection."""
        logger.info(f"Client connected from {websocket.remote_address}")

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

        self.server = await websockets.serve(
            self.handle_request,
            self.host,
            self.port
        )

        logger.info("Dewey MCP Server is running")

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
