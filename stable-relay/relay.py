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
