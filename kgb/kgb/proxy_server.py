# kgb/proxy_server.py
"""
KGB: WebSocket spy coordinator with automatic conversation logging.
Each client connection spawns a dedicated spy worker with its own Dewey client.
"""
import asyncio
import json
import logging
import uuid
from urllib.parse import urlparse, parse_qs
from typing import Dict, Optional

import websockets
from websockets.server import WebSocketServerProtocol

from kgb.dewey_client import DeweyClient, MAX_LOG_CONTENT

logger = logging.getLogger(__name__)

class SpyWorker:
    """
    Individual spy worker for a single client connection.
    Each spy has its own dedicated Dewey client to avoid race conditions.
    """
    def __init__(self, client_id: str, websocket: WebSocketServerProtocol, upstream: str):
        self.client_id = client_id
        self.websocket = websocket
        self.upstream = upstream
        self.dewey_client = DeweyClient()  # Dedicated client per spy
        self.conversation_id = None
        self.upstream_ws = None
        logger.info(f"Spy {client_id} recruited for upstream: {upstream}")

    async def start(self):
        """Start spy operations."""
        try:
            # Connect to Dewey and begin conversation
            conv = await self.dewey_client.begin_conversation(
                metadata={"upstream": self.upstream, "spy_id": self.client_id}
            )
            self.conversation_id = conv["conversation_id"]
            logger.info(f"Spy {self.client_id} logging to conversation: {self.conversation_id}")

            # Get upstream URL (validates against whitelist)
            upstream_url = self.get_upstream_url(self.upstream)
            if not upstream_url:
                allowed = ", ".join(self.ALLOWED_UPSTREAMS.keys())
                error_msg = f"Invalid upstream '{self.upstream}'. Allowed: {allowed}"
                logger.error(f"Spy {self.client_id}: {error_msg}")
                await self.websocket.close(code=4001, reason=error_msg)
                return

            # Connect to upstream
            self.upstream_ws = await websockets.connect(upstream_url)
            logger.info(f"Spy {self.client_id} connected to upstream: {upstream_url}")

            # Start bidirectional relay
            await asyncio.gather(
                self.relay_client_to_upstream(),
                self.relay_upstream_to_client()
            )

        except Exception as e:
            logger.error(f"Spy {self.client_id} error: {e}", exc_info=True)

        finally:
            await self.shutdown()

    async def shutdown(self):
        """Clean shutdown of spy operations."""
        if self.upstream_ws and not self.upstream_ws.closed:
            await self.upstream_ws.close()
        await self.dewey_client.close()
        logger.info(f"Spy {self.client_id} retired")

    # Allowed upstream services (whitelist for security)
    ALLOWED_UPSTREAMS = {
        "fiedler": "ws://fiedler-mcp:8080",
        "dewey": "ws://dewey-mcp:9020"
    }

    def get_upstream_url(self, upstream: str) -> Optional[str]:
        """Get WebSocket URL for upstream service (validates against whitelist)."""
        upstream_lower = upstream.lower()
        if upstream_lower not in self.ALLOWED_UPSTREAMS:
            logger.warning(f"Spy {self.client_id} rejected invalid upstream: {upstream}")
            return None
        return self.ALLOWED_UPSTREAMS[upstream_lower]

    def truncate_content(self, content: str) -> str:
        """Truncate content to max size with indicator."""
        if len(content) > MAX_LOG_CONTENT:
            return content[:MAX_LOG_CONTENT] + f"\n... [truncated {len(content) - MAX_LOG_CONTENT} bytes]"
        return content

    async def relay_client_to_upstream(self):
        """Relay and log all messages from client to upstream."""
        try:
            async for raw_message in self.websocket:
                try:
                    # Convert to string for logging (handle binary data safely)
                    if isinstance(raw_message, bytes):
                        try:
                            content = raw_message.decode('utf-8')
                        except UnicodeDecodeError:
                            content = f"[Binary data: {len(raw_message)} bytes - cannot decode as UTF-8]"
                    else:
                        content = raw_message

                    # Truncate and log
                    content = self.truncate_content(content)
                    await self.dewey_client.store_message(
                        conversation_id=self.conversation_id,
                        role="user",
                        content=content
                    )

                    # Forward to upstream (raw message as-is)
                    await self.upstream_ws.send(raw_message)

                except Exception as e:
                    logger.error(f"Spy {self.client_id} error logging user message: {e}")
                    # Still forward the message even if logging fails
                    try:
                        await self.upstream_ws.send(raw_message)
                    except:
                        pass

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Spy {self.client_id} client connection closed")

    async def relay_upstream_to_client(self):
        """Relay and log all messages from upstream to client."""
        try:
            async for raw_message in self.upstream_ws:
                try:
                    # Convert to string for logging (handle binary data safely)
                    if isinstance(raw_message, bytes):
                        try:
                            content = raw_message.decode('utf-8')
                        except UnicodeDecodeError:
                            content = f"[Binary data: {len(raw_message)} bytes - cannot decode as UTF-8]"
                    else:
                        content = raw_message

                    # Truncate and log
                    content = self.truncate_content(content)
                    await self.dewey_client.store_message(
                        conversation_id=self.conversation_id,
                        role="assistant",
                        content=content
                    )

                    # Forward to client (raw message as-is)
                    await self.websocket.send(raw_message)

                except Exception as e:
                    logger.error(f"Spy {self.client_id} error logging assistant message: {e}")
                    # Still forward the message even if logging fails
                    try:
                        await self.websocket.send(raw_message)
                    except:
                        pass

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Spy {self.client_id} upstream connection closed")


class KGBProxy:
    """KGB Proxy: Spy coordinator for transparent conversation logging."""

    def __init__(self, host: str = "0.0.0.0", port: int = 9000):
        self.host = host
        self.port = port
        self.active_spies = {}  # client_id -> SpyWorker
        logger.info(f"KGB Proxy initialized on {host}:{port}")

    async def start(self):
        """Start the KGB proxy server."""
        self.server = await websockets.serve(
            self.handle_client,
            self.host,
            self.port
        )
        logger.info("KGB Proxy started - recruiting spies...")

    async def stop(self):
        """Stop the KGB proxy server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()

        # Retire all active spies
        for spy in list(self.active_spies.values()):
            await spy.shutdown()

        logger.info("KGB Proxy stopped - all spies retired")

    async def handle_client(self, websocket: WebSocketServerProtocol, path: str):
        """
        Handle new client connection by recruiting a dedicated spy worker.
        Each spy gets its own Dewey client to avoid race conditions.
        """
        client_id = str(uuid.uuid4())
        query = parse_qs(urlparse(path).query)
        upstream = query.get("upstream", ["fiedler"])[0]

        # Recruit a new spy for this connection
        spy = SpyWorker(client_id, websocket, upstream)
        self.active_spies[client_id] = spy

        try:
            # Spy begins operations
            await spy.start()
        finally:
            # Spy retires after connection closes
            if client_id in self.active_spies:
                del self.active_spies[client_id]


async def main():
    """Main entry point for KGB Proxy - runs both WebSocket spy and HTTP gateway."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Start WebSocket spy (existing)
    ws_proxy = KGBProxy()
    await ws_proxy.start()

    # Start HTTP gateway (new)
    from kgb.http_gateway import start_http_gateway
    http_gateway = await start_http_gateway(host="0.0.0.0", port=8089)

    logger.info("KGB fully operational: WebSocket spy (9000) + HTTP gateway (8089)")

    try:
        await asyncio.Future()
    except KeyboardInterrupt:
        logger.info("Shutting down KGB...")
    finally:
        await ws_proxy.stop()
        await http_gateway.stop()


if __name__ == "__main__":
    asyncio.run(main())
