#!/mnt/projects/ICCM/mcp-relay/.venv/bin/python3
"""
MCP Relay - stdio to WebSocket multiplexer for Claude Code.

Acts as an MCP server on stdio, connects to multiple WebSocket MCP backends,
aggregates their tools, and routes tool calls to the appropriate backend.

Design:
- Claude Code connects via stdio (officially supported)
- Relay connects to backends via WebSocket (direct in bare metal, through KGB in containerized)
- Auto-reconnects to backends if they restart
- Exposes all backend tools as a unified interface
"""
import asyncio
import json
import logging
import sys
from typing import Dict, List, Any, Optional
from pathlib import Path

import websockets
import yaml
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """Watches backends.yaml for changes and triggers reload."""

    def __init__(self, relay, config_path: str):
        self.relay = relay
        self.config_path = str(Path(config_path).absolute())

    def on_modified(self, event):
        # Check if the modified file matches our config file
        if not event.is_directory and str(Path(event.src_path).absolute()) == self.config_path:
            logger.info(f"Config file changed: {self.config_path}")
            # Schedule reload in the event loop
            loop = asyncio.get_event_loop()
            loop.create_task(self.relay.reload_config())


class MCPRelay:
    """MCP server that multiplexes stdio to multiple WebSocket backends."""

    def __init__(self, config_path: str = "/mnt/projects/ICCM/mcp-relay/backends.yaml"):
        self.config_path = config_path
        self.backends: Dict[str, dict] = {}  # backend_name -> {url, ws, tools}
        self.tool_routing: Dict[str, str] = {}  # tool_name -> backend_name
        self.reconnect_delay = 5
        self.initialized = False
        self.load_config()

    def load_config(self):
        """Load backend configuration."""
        try:
            config = yaml.safe_load(Path(self.config_path).read_text())
            for backend in config['backends']:
                name = backend['name']
                self.backends[name] = {
                    'url': backend['url'],
                    'ws': None,
                    'tools': []
                }
            logger.info(f"Loaded {len(self.backends)} backends: {list(self.backends.keys())}")
        except Exception as e:
            logger.error(f"Config load failed: {e}")
            sys.exit(1)

    async def reload_config(self):
        """Reload configuration and reconnect to changed backends."""
        logger.info("Reloading configuration...")

        try:
            # Load new config
            config = yaml.safe_load(Path(self.config_path).read_text())
            new_backends = {}

            for backend in config['backends']:
                name = backend['name']
                new_url = backend['url']
                new_backends[name] = new_url

            # Compare with current backends and reconnect if URL changed
            for name, new_url in new_backends.items():
                if name in self.backends:
                    old_url = self.backends[name]['url']
                    if old_url != new_url:
                        logger.info(f"Backend {name} URL changed: {old_url} → {new_url}")
                        # Close old connection
                        if self.backends[name]['ws']:
                            await self.backends[name]['ws'].close()
                        # Update URL and reconnect
                        self.backends[name]['url'] = new_url
                        self.backends[name]['ws'] = None
                        self.backends[name]['tools'] = []
                        await self.reconnect_backend(name)
                else:
                    # New backend added
                    logger.info(f"New backend added: {name}")
                    self.backends[name] = {
                        'url': new_url,
                        'ws': None,
                        'tools': []
                    }
                    await self.reconnect_backend(name)

            # Remove backends that no longer exist in config
            removed = set(self.backends.keys()) - set(new_backends.keys())
            for name in removed:
                logger.info(f"Backend removed: {name}")
                if self.backends[name]['ws']:
                    await self.backends[name]['ws'].close()
                del self.backends[name]
                # Remove tools from routing table
                self.tool_routing = {k: v for k, v in self.tool_routing.items() if v != name}

            logger.info("Configuration reloaded successfully")

        except Exception as e:
            logger.error(f"Config reload failed: {e}", exc_info=True)

    async def connect_backend(self, backend_name: str) -> bool:
        """Connect to a backend MCP server."""
        backend = self.backends[backend_name]
        url = backend['url']

        try:
            ws = await websockets.connect(url)
            backend['ws'] = ws
            logger.info(f"Connected to {backend_name}: {url}")

            # Send initialize to backend
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "mcp-relay",
                        "version": "1.0.0"
                    }
                },
                "id": 1
            }
            await ws.send(json.dumps(init_request))
            response = await ws.recv()
            logger.info(f"{backend_name} initialized")

            # Send initialized notification
            await ws.send(json.dumps({
                "jsonrpc": "2.0",
                "method": "notifications/initialized"
            }))

            # Consume any response to the notification (some servers send errors for unknown notifications)
            try:
                notification_response = await asyncio.wait_for(ws.recv(), timeout=0.5)
                logger.debug(f"{backend_name} notification response: {notification_response}")
            except asyncio.TimeoutError:
                pass  # No response is fine for notifications

            # Discover tools
            await self.discover_tools(backend_name)
            return True

        except Exception as e:
            logger.warning(f"Failed to connect to {backend_name}: {e}")
            backend['ws'] = None
            return False

    async def discover_tools(self, backend_name: str):
        """Discover tools from a backend."""
        backend = self.backends[backend_name]
        ws = backend['ws']

        if not ws:
            return

        try:
            # Request tools list
            request = {
                "jsonrpc": "2.0",
                "method": "tools/list",
                "id": 2
            }
            await ws.send(json.dumps(request))
            response = json.loads(await ws.recv())
            logger.info(f"Tool discovery response from {backend_name}: {json.dumps(response)[:200]}")

            if 'result' in response and 'tools' in response['result']:
                tools = response['result']['tools']
                backend['tools'] = tools

                # Update routing table
                for tool in tools:
                    tool_name = tool['name']
                    self.tool_routing[tool_name] = backend_name
                    logger.info(f"Registered tool: {tool_name} → {backend_name}")

        except Exception as e:
            logger.error(f"Failed to discover tools from {backend_name}: {e}")

    async def connect_all_backends(self):
        """Connect to all configured backends."""
        tasks = [self.connect_backend(name) for name in self.backends.keys()]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        connected = sum(1 for r in results if r is True)
        logger.info(f"Connected to {connected}/{len(self.backends)} backends")

    async def reconnect_backend(self, backend_name: str):
        """Reconnect to a backend with retry logic."""
        while True:
            if await self.connect_backend(backend_name):
                return
            logger.info(f"Retrying {backend_name} in {self.reconnect_delay}s...")
            await asyncio.sleep(self.reconnect_delay)

    def get_all_tools(self) -> List[dict]:
        """Aggregate tools from all backends."""
        tools = []
        for backend in self.backends.values():
            tools.extend(backend['tools'])
        return tools

    async def call_tool(self, tool_name: str, arguments: dict) -> dict:
        """Route tool call to appropriate backend."""
        backend_name = self.tool_routing.get(tool_name)

        if not backend_name:
            raise ValueError(f"Unknown tool: {tool_name}")

        backend = self.backends[backend_name]
        ws = backend['ws']

        if not ws:
            # Try to reconnect
            logger.warning(f"Backend {backend_name} disconnected, reconnecting...")
            await self.reconnect_backend(backend_name)
            ws = backend['ws']

        # Forward tool call to backend
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": 3
        }

        try:
            await ws.send(json.dumps(request))
            response = json.loads(await ws.recv())
            return response
        except (websockets.exceptions.ConnectionClosed,
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK) as e:
            # Connection lost - reconnect and retry
            logger.warning(f"Backend {backend_name} connection lost: {e}, reconnecting...")
            backend['ws'] = None  # Mark as disconnected
            await self.reconnect_backend(backend_name)
            ws = backend['ws']

            # Retry the tool call once
            logger.info(f"Retrying tool call: {tool_name}")
            await ws.send(json.dumps(request))
            response = json.loads(await ws.recv())
            return response
        except Exception as e:
            logger.error(f"Tool call failed for {tool_name}: {e}")
            raise

    async def handle_request(self, request: dict) -> Optional[dict]:
        """Handle MCP request from Claude."""
        method = request.get("method")
        request_id = request.get("id")
        params = request.get("params", {})

        try:
            if method == "initialize":
                # Initialize relay
                if not self.initialized:
                    await self.connect_all_backends()
                    self.initialized = True

                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "protocolVersion": "2024-11-05",
                        "capabilities": {
                            "tools": {}
                        },
                        "serverInfo": {
                            "name": "mcp-relay",
                            "version": "1.0.0"
                        }
                    },
                    "id": request_id
                }

            elif method == "notifications/initialized":
                # No response needed for notifications
                return None

            elif method == "tools/list":
                tools = self.get_all_tools()
                return {
                    "jsonrpc": "2.0",
                    "result": {
                        "tools": tools
                    },
                    "id": request_id
                }

            elif method == "tools/call":
                tool_name = params.get("name")
                arguments = params.get("arguments", {})

                logger.info(f"Calling tool: {tool_name}")
                response = await self.call_tool(tool_name, arguments)

                # Forward backend response (preserve id)
                if "result" in response:
                    return {
                        "jsonrpc": "2.0",
                        "result": response["result"],
                        "id": request_id
                    }
                elif "error" in response:
                    return {
                        "jsonrpc": "2.0",
                        "error": response["error"],
                        "id": request_id
                    }
                else:
                    return response

            else:
                logger.warning(f"Unknown method: {method}")
                return {
                    "jsonrpc": "2.0",
                    "error": {
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    },
                    "id": request_id
                }

        except Exception as e:
            logger.error(f"Error handling {method}: {e}", exc_info=True)
            return {
                "jsonrpc": "2.0",
                "error": {
                    "code": -32603,
                    "message": str(e)
                },
                "id": request_id
            }

    async def run(self):
        """Main stdio loop."""
        logger.info("MCP Relay starting on stdio")

        # Start file watcher for config changes
        config_dir = str(Path(self.config_path).parent)
        event_handler = ConfigFileHandler(self, self.config_path)
        observer = Observer()
        observer.schedule(event_handler, config_dir, recursive=False)
        observer.start()
        logger.info(f"Watching config file: {self.config_path}")

        # Read from stdin, write to stdout
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, None, asyncio.get_event_loop())

        logger.info("MCP Relay ready")

        try:
            while True:
                # Read JSON-RPC request (line-delimited)
                line = await reader.readline()
                if not line:
                    break

                try:
                    request = json.loads(line.decode('utf-8'))
                    logger.info(f"Received request: {request.get('method')}")

                    response = await self.handle_request(request)

                    if response:
                        writer.write(json.dumps(response).encode('utf-8') + b'\n')
                        await writer.drain()

                except json.JSONDecodeError as e:
                    logger.error(f"Invalid JSON: {e}")
                except Exception as e:
                    logger.error(f"Request handling error: {e}", exc_info=True)

        finally:
            # Cleanup
            observer.stop()
            observer.join()
            for backend in self.backends.values():
                if backend['ws']:
                    await backend['ws'].close()
            logger.info("MCP Relay shutdown")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MCP Relay - stdio to WebSocket multiplexer")
    parser.add_argument("--config", default="/mnt/projects/ICCM/mcp-relay/backends.yaml",
                       help="Path to backends configuration file")
    args = parser.parse_args()

    relay = MCPRelay(config_path=args.config)
    asyncio.run(relay.run())
