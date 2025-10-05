import os

# --- Godot General Config ---
REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
REDIS_PORT = int(os.environ.get("REDIS_PORT", 6379))
LOG_QUEUE_NAME = os.environ.get("LOG_QUEUE_NAME", "logs:queue")
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", 100000))  # REQ-GOD-004, REQ-PERF-003

# --- Worker Config ---
DEWEY_MCP_URL = os.environ.get("DEWEY_MCP_URL", "ws://localhost:8080")
DEWEY_CONNECT_TIMEOUT = int(os.environ.get("DEWEY_CONNECT_TIMEOUT", 30))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 100))
BATCH_TIMEOUT_MS = int(os.environ.get("BATCH_TIMEOUT_MS", 100))
RETRY_MAX_ATTEMPTS = int(os.environ.get("RETRY_MAX_ATTEMPTS", 5))
RETRY_INITIAL_DELAY = int(os.environ.get("RETRY_INITIAL_DELAY", 1))
RETRY_MAX_DELAY = int(os.environ.get("RETRY_MAX_DELAY", 30))

# --- MCP Server Config ---
MCP_PORT = int(os.environ.get("MCP_PORT", 9060))
