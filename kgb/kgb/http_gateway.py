# kgb/http_gateway.py
"""
KGB HTTP Gateway: Reverse proxy for Anthropic API with conversation logging.
Runs alongside WebSocket spy to provide complete coverage.
"""
import asyncio
import json
import logging
import os
import ssl
import uuid
from typing import Optional
from urllib.parse import urljoin

import aiohttp
from aiohttp import web

from kgb.dewey_client import DeweyClient, MAX_LOG_CONTENT

logger = logging.getLogger(__name__)

class HTTPGateway:
    """HTTP reverse proxy gateway with Dewey logging."""

    def __init__(self, host: str = "0.0.0.0", port: int = 8089, upstream: str = None):
        self.host = host
        self.port = port
        # Use environment variable KGB_TARGET_URL, fallback to direct Anthropic API
        self.upstream = (upstream or os.getenv("KGB_TARGET_URL", "https://api.anthropic.com")).rstrip('/')
        self.app = None
        self.runner = None
        logger.info(f"HTTP Gateway initialized on {host}:{port} -> {self.upstream}")

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
            # Filter headers - remove Host and other proxy-specific headers
            # Make headers case-insensitive by converting to regular dict
            forward_headers = {}
            for key, value in request.headers.items():
                forward_headers[key] = value

            # Remove headers that should not be forwarded (case-insensitive)
            headers_to_remove = ['host', 'connection', 'keep-alive', 'proxy-connection']
            forward_headers = {k: v for k, v in forward_headers.items() if k.lower() not in headers_to_remove}

            # Ensure User-Agent is set (some APIs block requests without it)
            if 'User-Agent' not in forward_headers and 'user-agent' not in forward_headers:
                forward_headers['User-Agent'] = 'KGB-HTTP-Gateway/1.0'

            # Force identity encoding to prevent gzip compression (critical for streaming)
            forward_headers['Accept-Encoding'] = 'identity'

            # Debug logging
            logger.info(f"Gateway {request_id} forwarding to: {upstream_url}")
            logger.info(f"Gateway {request_id} forward_headers: {json.dumps(forward_headers, indent=2)}")

            # Create SSL context with proper SNI configuration
            ssl_context = ssl.create_default_context()

            # Create TCPConnector with proper SSL/TLS settings
            connector = aiohttp.TCPConnector(
                ssl=ssl_context,
                force_close=False,
                enable_cleanup_closed=True
            )

            async with aiohttp.ClientSession(connector=connector) as session:
                async with session.request(
                    method=request.method,
                    url=upstream_url,
                    headers=forward_headers,
                    data=request_body,
                    allow_redirects=True,
                    ssl=ssl_context  # Explicit SSL context
                ) as upstream_response:

                    # Create streaming response to client
                    # Filter out headers that shouldn't be forwarded when re-streaming
                    response_headers = dict(upstream_response.headers)
                    headers_to_remove = ['Content-Length', 'Content-Encoding', 'Transfer-Encoding']
                    for header in headers_to_remove:
                        response_headers.pop(header, None)
                        # Case-insensitive removal
                        for key in list(response_headers.keys()):
                            if key.lower() == header.lower():
                                response_headers.pop(key, None)

                    # Add SSE-specific headers to prevent downstream buffering
                    response_headers['Cache-Control'] = 'no-cache'
                    response_headers['X-Accel-Buffering'] = 'no'

                    response = web.StreamResponse(
                        status=upstream_response.status,
                        headers=response_headers
                    )
                    await response.prepare(request)

                    # Stream response chunks to client and accumulate for logging
                    # Use iter_any() for immediate, unbuffered streaming (critical fix from triplet)
                    response_chunks = []

                    async for chunk in upstream_response.content.iter_any():
                        if chunk:  # iter_any() can yield empty bytes
                            # Forward chunk to client immediately
                            await response.write(chunk)
                            # Accumulate for logging
                            response_chunks.append(chunk)

                    # Complete the streaming response
                    await response.write_eof()

                    # Reconstruct full response body for logging
                    response_body = b''.join(response_chunks)
                    response_body_str = response_body.decode('utf-8') if response_body else ""

                    # Log response to Dewey
                    # Try to parse as JSON, but fall back to raw string if it fails
                    try:
                        response_body_parsed = json.loads(response_body_str) if response_body_str else None
                    except json.JSONDecodeError:
                        response_body_parsed = response_body_str or None

                    response_log = {
                        "status": upstream_response.status,
                        "headers": self.sanitize_headers(dict(upstream_response.headers)),
                        "body": response_body_parsed
                    }
                    await dewey_client.store_message(
                        conversation_id=conversation_id,
                        role="assistant",
                        content=self.truncate_content(json.dumps(response_log))
                    )

                    logger.info(f"Gateway {request_id} <- {upstream_response.status}")

                    return response

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
