"""Fiedler MCP Server - Orchestra Conductor Prototype."""
import asyncio
from typing import Any

from mcp.server import Server
from mcp.types import Tool, TextContent

from .tools import (
    fiedler_list_models,
    fiedler_set_models,
    fiedler_set_output,
    fiedler_get_config,
    fiedler_send,
    fiedler_set_key,
    fiedler_delete_key,
    fiedler_list_keys,
)


# Create server instance
app = Server("fiedler")


@app.list_tools()
async def list_tools() -> list[Tool]:
    """List all available Fiedler tools."""
    return [
        Tool(
            name="fiedler_list_models",
            description="List all available LLM models with their properties (name, provider, aliases, max_tokens, capabilities).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="fiedler_set_models",
            description="Configure default models for fiedler_send. Accepts list of model IDs or aliases.",
            inputSchema={
                "type": "object",
                "properties": {
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of model IDs or aliases to use as defaults",
                    },
                },
                "required": ["models"],
            },
        ),
        Tool(
            name="fiedler_set_output",
            description="Configure output directory for fiedler_send results.",
            inputSchema={
                "type": "object",
                "properties": {
                    "output_dir": {
                        "type": "string",
                        "description": "Path to output directory",
                    },
                },
                "required": ["output_dir"],
            },
        ),
        Tool(
            name="fiedler_get_config",
            description="Get current Fiedler configuration (models, output_dir, total_available_models).",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="fiedler_send",
            description="Send prompt and optional package files to configured LLMs. Returns results from all models in parallel.",
            inputSchema={
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "User prompt or question to send to models",
                    },
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of file paths to compile into package",
                    },
                    "models": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional override of default models (use model IDs or aliases)",
                    },
                },
                "required": ["prompt"],
            },
        ),
        Tool(
            name="fiedler_set_key",
            description="Store API key securely in system keyring (encrypted). Replaces need for environment variables.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name: google, openai, together, or xai",
                    },
                    "api_key": {
                        "type": "string",
                        "description": "API key to store (will be encrypted by OS keyring)",
                        "format": "password",
                        "writeOnly": True,
                    },
                },
                "required": ["provider", "api_key"],
            },
        ),
        Tool(
            name="fiedler_delete_key",
            description="Delete stored API key from system keyring.",
            inputSchema={
                "type": "object",
                "properties": {
                    "provider": {
                        "type": "string",
                        "description": "Provider name: google, openai, together, or xai",
                    },
                },
                "required": ["provider"],
            },
        ),
        Tool(
            name="fiedler_list_keys",
            description="List providers that have API keys stored in system keyring.",
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
    ]


# Tool dispatch table for sync tools
SYNC_TOOL_DISPATCH = {
    "fiedler_list_models": lambda args: fiedler_list_models(),
    "fiedler_set_models": lambda args: fiedler_set_models(args["models"]),
    "fiedler_set_output": lambda args: fiedler_set_output(args["output_dir"]),
    "fiedler_get_config": lambda args: fiedler_get_config(),
    "fiedler_set_key": lambda args: fiedler_set_key(args["provider"], args["api_key"]),
    "fiedler_delete_key": lambda args: fiedler_delete_key(args["provider"]),
    "fiedler_list_keys": lambda args: fiedler_list_keys(),
}


@app.call_tool()
async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
    """Handle tool calls with dictionary-based dispatch and structured errors."""
    import json

    try:
        # Special handling for async fiedler_send
        if name == "fiedler_send":
            result = await asyncio.to_thread(
                fiedler_send,
                prompt=arguments["prompt"],
                files=arguments.get("files"),
                models=arguments.get("models"),
            )
        # Dispatch sync tools
        elif name in SYNC_TOOL_DISPATCH:
            result = SYNC_TOOL_DISPATCH[name](arguments)
        else:
            # Unknown tool
            error_response = {
                "error": {
                    "code": "UNKNOWN_TOOL",
                    "message": f"Unknown tool: {name}",
                    "tool_name": name
                }
            }
            return [TextContent(type="text", text=json.dumps(error_response, indent=2))]

        # Format successful result
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    except KeyError as e:
        # Missing required argument
        error_response = {
            "error": {
                "code": "MISSING_ARGUMENT",
                "message": f"Missing required argument: {e}",
                "argument": str(e)
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    except ValueError as e:
        # Invalid input (e.g., unknown model, file too large)
        error_response = {
            "error": {
                "code": "INVALID_INPUT",
                "message": str(e)
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    except PermissionError as e:
        # File access denied
        error_response = {
            "error": {
                "code": "ACCESS_DENIED",
                "message": str(e)
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    except FileNotFoundError as e:
        # File not found
        error_response = {
            "error": {
                "code": "FILE_NOT_FOUND",
                "message": str(e)
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    except RuntimeError as e:
        # Provider/configuration errors
        error_response = {
            "error": {
                "code": "PROVIDER_ERROR",
                "message": str(e)
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]
    except Exception as e:
        # Unexpected internal error
        error_response = {
            "error": {
                "code": "INTERNAL_ERROR",
                "message": f"Internal error: {str(e)}",
                "type": type(e).__name__
            }
        }
        return [TextContent(type="text", text=json.dumps(error_response, indent=2))]


async def _amain():
    """Async entry point for MCP server."""
    from mcp.server.stdio import stdio_server

    async with stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options(),
        )


def main():
    """Synchronous entry point for console script."""
    asyncio.run(_amain())


if __name__ == "__main__":
    main()
