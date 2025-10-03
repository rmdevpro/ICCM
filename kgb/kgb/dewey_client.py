# kgb/dewey_client.py
"""
Dewey client for KGB spy workers - each spy gets their own dedicated client.
"""
import asyncio
import json
import logging
from typing import Any, Dict, Optional

import websockets

logger = logging.getLogger(__name__)

# Content size limit for logging
MAX_LOG_CONTENT = 10000  # 10KB per message

class DeweyClient:
    """Dedicated Dewey MCP client for a single spy worker."""

    def __init__(self, url: str = "ws://dewey-mcp:9020"):
        self.url = url
        self.websocket = None
        self.request_id = 0
        self.lock = asyncio.Lock()  # Ensure thread-safe operations
        logger.debug(f"Dewey client initialized for {url}")

    async def connect(self):
        """Connect to Dewey MCP server with persistent connection."""
        if not self.websocket or self.websocket.closed:
            try:
                self.websocket = await websockets.connect(self.url)
                logger.debug("Connected to Dewey MCP server")
            except Exception as e:
                logger.error(f"Failed to connect to Dewey: {e}")
                raise

    async def close(self):
        """Close connection to Dewey."""
        if self.websocket and not self.websocket.closed:
            await self.websocket.close()
            self.websocket = None
            logger.debug("Disconnected from Dewey MCP server")

    async def _call_tool(self, method: str, params: Dict[str, Any]) -> Any:
        """Call a Dewey MCP tool with lock protection."""
        async with self.lock:  # Protect concurrent calls
            await self.connect()

            self.request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self.request_id,
                "method": method,
                "params": params
            }

            try:
                await self.websocket.send(json.dumps(request))
                response_text = await self.websocket.recv()
                response = json.loads(response_text)

                if "error" in response:
                    error = response["error"]
                    raise Exception(f"Dewey error {error.get('code')}: {error.get('message')}")

                return response.get("result")

            except Exception as e:
                logger.error(f"Tool call failed: {e}")
                raise

    async def begin_conversation(self, session_id: Optional[str] = None, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Begin a new conversation in Dewey."""
        params = {}
        if session_id:
            params["session_id"] = session_id
        if metadata:
            params["metadata"] = metadata
        return await self._call_tool("dewey_begin_conversation", params)

    async def store_message(self, conversation_id: str, role: str, content: str, metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """Store a message in Dewey."""
        params = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content
        }
        if metadata:
            params["metadata"] = metadata
        return await self._call_tool("dewey_store_message", params)

    async def store_messages_bulk(self, messages: list, conversation_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Store multiple messages in Dewey."""
        params = {"messages": messages}
        if conversation_id:
            params["conversation_id"] = conversation_id
        if session_id:
            params["session_id"] = session_id
        return await self._call_tool("dewey_store_messages_bulk", params)

    async def get_conversation(self, conversation_id: str) -> Dict[str, Any]:
        """Retrieve a conversation from Dewey."""
        return await self._call_tool("dewey_get_conversation", {"conversation_id": conversation_id})

    async def search(self, query: str, session_id: Optional[str] = None) -> Dict[str, Any]:
        """Search conversations in Dewey."""
        params = {"query": query}
        if session_id:
            params["session_id"] = session_id
        return await self._call_tool("dewey_search", params)
