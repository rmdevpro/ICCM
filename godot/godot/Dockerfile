# Godot Unified Logging Infrastructure - Synthesized Implementation
# Base: Gemini-2.5-Pro with refinements from DeepSeek-R1
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y redis-server supervisor procps && rm -rf /var/lib/apt/lists/*

# Copy application code
COPY src/ /app/src/
COPY redis.conf /etc/redis/redis.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf

# Install Python dependencies
RUN pip install --no-cache-dir "redis>=5.0" "mcp-tools-py>=0.4.0" "httpx>=0.25.0"

# Expose ports
EXPOSE 6379 # Redis
EXPOSE 9060 # MCP Server

# Start via supervisord
CMD ["/usr/bin/supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
