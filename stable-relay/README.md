# Stable Relay & MCP Relay

**Two complementary components for resilient MCP connectivity:**

1. **Stable Relay** (`relay.py`) - WebSocket-to-WebSocket relay with auto-reconnect
2. **MCP Relay** (`mcp_relay.py`) - stdio-to-WebSocket multiplexer for Claude Code

**Combined Purpose**: Unified access to all ICCM MCP tools with backend restart resilience.

## The Problem

**Two challenges:**
1. Claude Code MCP only supports stdio transport (not WebSocket)
2. All ICCM MCP servers use WebSocket (Fiedler, Dewey)
3. Restarting backend servers requires Claude restart (loses context)

## The Solution

**MCP Relay + Stable Relay architecture:**

```
Claude Code (stdio, NEVER RESTARTS)
    ↓
MCP Relay (stdio ↔ WebSocket multiplexer)
    ↓
Stable Relay (port 8000, auto-reconnects backends)
    ↓
KGB (port 9000, automatic logging)
    ├→ Fiedler (WebSocket MCP server)
    └→ Dewey (WebSocket MCP server)
```

**Benefits:**
- Claude speaks stdio (officially supported)
- MCP Relay aggregates tools from all backends
- Backends can restart freely without affecting Claude
- Single MCP entry exposes all ICCM tools
- Network-wide access to any WebSocket MCP server

## Design Principles

1. **<100 lines of code** - So simple there's nothing to break
2. **Zero parsing** - Just forward bytes, don't understand MCP/JSON-RPC
3. **Auto-reconnect** - Backend dies? Keep trying every 5s until it's back
4. **Keep Claude alive** - Even if backend offline for minutes, Claude connection stays open
5. **Single config file** - Change backend targets without code changes

## Architecture

```
relay.py (98 lines)
├── Load config.yaml (backend URL)
├── Listen on port 8000 for Claude
├── For each Claude connection:
│   ├── Connect to backend (with auto-reconnect)
│   ├── Forward client → backend (bytes only)
│   └── Forward backend → client (bytes only)
└── If backend dies: reconnect transparently
```

**Files:**
- `relay.py` - The relay (98 lines total)
- `config.yaml` - Backend URL configuration
- `Dockerfile` - Container build
- `docker-compose.yml` - Deployment config
- `requirements.txt` - Python dependencies (websockets, pyyaml)

## Configuration

**config.yaml:**
```yaml
backend: "ws://kgb-proxy:9000"
```

To change backend, edit this file and restart relay (Claude stays up).

## Deployment

```bash
cd /mnt/projects/ICCM/stable-relay

# Build container
docker compose build

# Start relay
docker compose up -d

# Check logs
docker logs -f stable-relay

# Check status
docker ps --filter "name=stable-relay"
```

## MCP Configuration

**Claude Code config** (`~/.config/claude-code/mcp.json`):
```json
{
  "mcpServers": {
    "fiedler": {
      "transport": {
        "type": "ws",
        "url": "ws://localhost:8000?upstream=fiedler"
      }
    },
    "dewey": {
      "transport": {
        "type": "ws",
        "url": "ws://localhost:8000?upstream=dewey"
      }
    }
  }
}
```

**Note**: Point ALL MCP servers to relay port 8000. Relay forwards to KGB (9000).

## How It Works

**Normal Operation:**
1. Claude connects to relay (port 8000)
2. Relay connects to KGB (port 9000)
3. All messages forwarded as raw bytes (no parsing)
4. KGB routes to Fiedler/Dewey and logs conversations

**When Backend Restarts:**
1. Relay detects KGB connection lost
2. Relay keeps trying to reconnect to KGB every 5s
3. **Claude connection to relay stays alive**
4. When KGB comes back, relay reconnects
5. Messages flow again
6. **Claude never knew anything happened**

## Benefits

- **No more conversation loss** - Claude never needs to restart
- **Safe backend updates** - Restart KGB/Fiedler/Dewey anytime
- **Simple design** - Too simple to have bugs
- **Transparent** - Claude and backends don't know relay exists
- **Auto-recovery** - Backend restarts are seamless

## Testing

**Test 1: Basic connection**
```bash
# Check relay running
docker logs stable-relay | tail

# Should see: "Stable Relay listening on ws://0.0.0.0:8000"
```

**Test 2: Backend restart survival**
```bash
# While using Claude:
docker restart kgb-proxy

# Relay logs: "Backend lost, reconnecting..."
# Relay logs: "Backend connected"
# Claude: Never disconnected, tools still work
```

**Test 3: Connection from host**
```python
import asyncio
import websockets

async def test():
    ws = await websockets.connect('ws://localhost:8000')
    print("Connected to relay")
    await ws.close()

asyncio.run(test())
```

## Maintenance

**This relay is designed to NEVER need updates.**

The only time you restart relay:
1. Changing backend URL in config.yaml
2. Actual Python bugs (extremely unlikely due to simplicity)

**If you need to restart relay:**
```bash
docker restart stable-relay
# Claude will reconnect automatically (5s downtime max)
```

## Troubleshooting

**Relay won't connect to backend:**
- Check config.yaml has correct backend URL
- Check backend is running: `docker ps | grep kgb`
- Check Docker network: `docker inspect stable-relay | grep -A 10 Networks`

**Claude won't connect to relay:**
- Check relay running: `docker ps | grep stable-relay`
- Check port 8000 bound: `ss -tlnp | grep 8000`
- Check mcp.json points to `ws://localhost:8000`

## Current Backend Chain

```
Claude → Relay (8000) → KGB (9000) → Fiedler (8080) → 7 LLMs
                         ↓
                      Dewey (9020) → PostgreSQL (automatic logging)
```

**All components can restart except Claude and Relay.**

---

**Implementation**: 98 lines of Python
**Dependencies**: websockets, pyyaml
**Purpose**: Never restart Claude Code again
**Status**: Production ready
