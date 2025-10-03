# KGB (Knowledge Gateway Broker)

Transparent WebSocket spy coordinator with automatic conversation logging to Dewey.
Each client connection spawns a dedicated spy worker with its own Dewey client.

## Architecture

```
Claude Code → KGB Proxy (port 9000) → Upstream MCP Servers (Fiedler, Dewey)
                    ↓
            Spy Workers (1 per connection)
                    ↓
                 Dewey (logs conversations)
```

## Features

- **Spy Worker Pattern**: Each connection gets dedicated spy with isolated Dewey client
- **Transparent Relay**: Forwards all traffic to upstream MCP servers
- **Complete Logging**: Captures ALL messages (not just tool calls) with truncation
- **Race-Free**: No shared WebSocket connections between clients
- **Multi-Upstream**: Supports routing to different MCP servers
- **Security Limits**: 10KB message size limit with truncation indicators

## Quick Start

### 1. Start Dewey First

```bash
cd /mnt/projects/ICCM/dewey
docker-compose up -d
```

### 2. Build and Start KGB

```bash
cd /mnt/projects/ICCM/kgb
docker-compose build
docker-compose up -d
```

### 3. Configure Claude Code

Update your MCP configuration to use KGB:

```json
{
  "mcpServers": {
    "fiedler": {
      "url": "ws://localhost:9000?upstream=fiedler"
    },
    "dewey": {
      "url": "ws://localhost:9000?upstream=dewey"
    }
  }
}
```

## How It Works

1. **Client Connects**: Claude Code connects to KGB with upstream parameter
2. **Spy Recruited**: KGB spawns dedicated spy worker with own Dewey client
3. **Conversation Begins**: Spy creates conversation in Dewey
4. **Bidirectional Relay**: Spy forwards ALL messages both ways
5. **Complete Logging**: Every message logged with truncation if > 10KB
6. **Spy Retires**: On disconnect, spy closes Dewey connection and terminates
7. **Transparent**: Client and upstream server unaware of logging

## Configuration

### Upstream Routes

Configured in `proxy_server.py`:

```python
upstreams = {
    "fiedler": "ws://fiedler-mcp:9010",
    "dewey": "ws://dewey-mcp:9020"
}
```

### Environment Variables

- `KGB_HOST`: Bind address (default: 0.0.0.0)
- `KGB_PORT`: Listen port (default: 9000)
- `DEWEY_URL`: Dewey server URL (default: ws://dewey-mcp:9020)

## Development

### Project Structure
```
kgb/
├── kgb/
│   ├── __init__.py
│   ├── dewey_client.py    # Dewey MCP client for spy workers
│   └── proxy_server.py    # KGB coordinator + SpyWorker class
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

### Viewing Logs

```bash
# Follow logs
docker logs -f kgb-proxy

# Last 100 lines
docker logs --tail 100 kgb-proxy
```

## Troubleshooting

### KGB Won't Connect to Dewey
- Ensure Dewey is running: `docker ps | grep dewey-mcp`
- Check network connectivity: `docker exec kgb-proxy ping dewey-mcp`
- Verify Dewey logs: `docker logs dewey-mcp`

### Messages Not Being Logged
- Check KGB logs for errors: `docker logs kgb-proxy`
- Verify spy worker created conversation in Dewey
- Check for truncation if messages > 10KB
- Ensure Dewey has async database (asyncpg) configured

### Connection Drops
- Check upstream server health
- Verify network stability
- Review proxy logs for WebSocket errors

## Future Enhancements

- **Message Filtering**: Configurable logging rules
- **Caching**: Cache frequent queries
- **Metrics**: Performance monitoring
- **Authentication**: API key validation
- **Load Balancing**: Multiple upstream instances

## Contributing

This is part of the ICCM (Integrated Cognitive-Cybernetic Methodology) project.

## License

Internal use only
