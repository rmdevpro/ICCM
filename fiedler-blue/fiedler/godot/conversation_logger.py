"""Conversation logging client for Godot (via MCP)."""
import json
import asyncio
from typing import Optional, List
import websockets


async def begin_conversation_godot(
    session_id: Optional[str] = None,
    metadata: Optional[dict] = None,
    godot_url: str = 'ws://godot-mcp:9060',
    timeout: float = 2.0
) -> Optional[str]:
    """
    Begin a new conversation in Godot.

    Returns:
        conversation_id (str) if successful, None if failed
    """
    try:
        async with websockets.connect(godot_url, open_timeout=timeout) as ws:
            # MCP Initialize
            init_request = {
                'jsonrpc': '2.0',
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {},
                    'clientInfo': {'name': 'fiedler-conversation-logger', 'version': '1.0'}
                },
                'id': 1
            }

            await ws.send(json.dumps(init_request))
            await asyncio.wait_for(ws.recv(), timeout=timeout)

            # Call conversation_begin
            begin_request = {
                'jsonrpc': '2.0',
                'method': 'tools/call',
                'params': {
                    'name': 'conversation_begin',
                    'arguments': {
                        'session_id': session_id,
                        'metadata': metadata
                    }
                },
                'id': 2
            }

            await ws.send(json.dumps(begin_request))
            response_str = await asyncio.wait_for(ws.recv(), timeout=timeout)
            response = json.loads(response_str)

            if 'result' in response:
                result_text = response['result']['content'][0]['text']
                result_data = json.loads(result_text)
                return result_data['conversation_id']

            return None

    except Exception:
        return None


async def store_message_godot(
    conversation_id: str,
    role: str,
    content: str,
    turn_number: Optional[int] = None,
    metadata: Optional[dict] = None,
    godot_url: str = 'ws://godot-mcp:9060',
    timeout: float = 2.0
) -> bool:
    """
    Store a message in Godot conversation.

    Returns:
        True if successful, False if failed
    """
    try:
        async with websockets.connect(godot_url, open_timeout=timeout) as ws:
            # MCP Initialize
            init_request = {
                'jsonrpc': '2.0',
                'method': 'initialize',
                'params': {
                    'protocolVersion': '2024-11-05',
                    'capabilities': {},
                    'clientInfo': {'name': 'fiedler-conversation-logger', 'version': '1.0'}
                },
                'id': 1
            }

            await ws.send(json.dumps(init_request))
            await asyncio.wait_for(ws.recv(), timeout=timeout)

            # Call conversation_store_message
            store_request = {
                'jsonrpc': '2.0',
                'method': 'tools/call',
                'params': {
                    'name': 'conversation_store_message',
                    'arguments': {
                        'conversation_id': conversation_id,
                        'role': role,
                        'content': content,
                        'turn_number': turn_number,
                        'metadata': metadata
                    }
                },
                'id': 2
            }

            await ws.send(json.dumps(store_request))
            response_str = await asyncio.wait_for(ws.recv(), timeout=timeout)
            response = json.loads(response_str)

            return 'result' in response

    except Exception:
        return False
