# kgb/http_gateway.py
"""
KGB HTTP Gateway: Reverse proxy for Anthropic API with conversation logging.
Runs alongside WebSocket spy to provide complete coverage.
"""
import asyncio
import json
import logging
import uuid
from typing import Optional
from urllib.parse import urljoin

import aiohttp
from aiohttp import web

from kgb.dewey_client import DeweyClient, MAX_LOG_CONTENT

logger = logging.getLogger(__name__)

class HTTPGateway:
    """HTTP reverse proxy gateway with Dewey logging."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8089, upstream: str = "https://api.anthropic.com"):
        self.host = host
        self.port = port
        self.upstream = upstream.rstrip('/')
        self.app = None
        self.runner = None
        logger.info(f"HTTP Gateway initialized on {host}:{port} -> {upstream}")

    async def start(self):
        """Start the HTTP gateway server."""
        self.app = web.Application()

        # Routes
        self.app.router.add_get('/health', self.health_check)
        self.app.router.add_route('*', '/{path:.*}', self.proxy_request)

        # Start server
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()
        site = web.TCPSite(self.runner, self.host, self.port)
        await site.start()
        logger.info(f"HTTP Gateway started on {self.host}:{self.port}")

    async def stop(self):
        """Stop the HTTP gateway server."""
        if self.runner:
            await self.runner.cleanup()
        logger.info("HTTP Gateway stopped")

    async def health_check(self, request: web.Request) -> web.Response:
        """Health check endpoint."""
        return web.json_response({
            "status": "healthy",
            "upstream": self.upstream,
            "gateway": "kgb-http"
        })

    def truncate_content(self, content: str) -> str:
        """Truncate content to max size with indicator."""
        if len(content) > MAX_LOG_CONTENT:
            return content[:MAX_LOG_CONTENT] + f"\n... [truncated {len(content) - MAX_LOG_CONTENT} bytes]"
        return content

    def sanitize_headers(self, headers: dict) -> dict:
        """Remove sensitive headers before logging."""
        sanitized = dict(headers)
        sensitive = ['x-api-key', 'authorization', 'cookie', 'set-cookie']

        for key in sensitive:
            if key in sanitized:
                sanitized[key] = '[REDACTED]'
            # Case-insensitive check
            for header_key in list(sanitized.keys()):
                if header_key.lower() == key:
                    sanitized[header_key] = '[REDACTED]'

        return sanitized

    async def proxy_request(self, request: web.Request) -> web.Response:
        """Proxy request to upstream and log to Dewey."""
        request_id = str(uuid.uuid4())
        path = request.match_info.get('path', '')

        # Create Dewey client for this request
        dewey_client = DeweyClient()
        conversation_id = None

        try:
            # Begin conversation in Dewey
            conv = await dewey_client.begin_conversation(
                metadata={
                    "source": "kgb-http-gateway",
                    "request_id": request_id,
                    "client": "claude-code-container"
                }
            )
            conversation_id = conv["conversation_id"]
            logger.info(f"Gateway {request_id} -> {request.method} /{path} (conv: {conversation_id})")

            # Build upstream URL
            upstream_url = urljoin(self.upstream, f"/{path}")
            if request.query_string:
                upstream_url += f"?{request.query_string}"

            # Read request body
            request_body = await request.read()
            request_body_str = request_body.decode('utf-8') if request_body else ""

            # Log request to Dewey
            request_log = {
                "method": request.method,
                "path": f"/{path}",
                "headers": self.sanitize_headers(dict(request.headers)),
                "body": json.loads(request_body_str) if request_body_str else None
            }
            await dewey_client.store_message(
                conversation_id=conversation_id,
                role="user",
                content=self.truncate_content(json.dumps(request_log))
            )

            # Forward request to upstream
            async with aiohttp.ClientSession() as session:
                async with session.request(
                    method=request.method,
                    url=upstream_url,
                    headers=request.headers,
                    data=request_body
                ) as upstream_response:

                    # Read response body
                    response_body = await upstream_response.read()
                    response_body_str = response_body.decode('utf-8') if response_body else ""

                    # Log response to Dewey
                    response_log = {
                        "status": upstream_response.status,
                        "headers": self.sanitize_headers(dict(upstream_response.headers)),
                        "body": json.loads(response_body_str) if response_body_str else None
                    }
                    await dewey_client.store_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=self.truncate_content(json.dumps(response_log))
                    )

                    logger.info(f"Gateway {request_id} <- {upstream_response.status}")

                    # Return response to client
                    return web.Response(
                        body=response_body,
                        status=upstream_response.status,
                        headers=upstream_response.headers
                    )

        except Exception as e:
            logger.error(f"Gateway {request_id} error: {e}", exc_info=True)
            return web.json_response({
                "error": "Gateway Error",
                "message": str(e),
                "request_id": request_id
            }, status=500)

        finally:
            await dewey_client.close()


async def start_http_gateway(host: str = "0.0.0.0", port: int = 8089):
    """Start HTTP gateway as standalone or alongside WebSocket spy."""
    gateway = HTTPGateway(host, port)
    await gateway.start()
    return gateway
