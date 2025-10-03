# ICCM Development Status - Current Session

**Last Updated:** 2025-10-03 22:35 EDT
**Session:** MCP Relay bug fix - websockets 15.x compatibility
**Status:** ⏸️ **Bug fixed, awaiting Claude Code restart for verification**

---

## 🎯 Current Objective

**COMPLETED:** Built MCP Relay for direct WebSocket MCP backend access

**Problem Solved:**
- Claude Code only supports stdio transport (not WebSocket)
- ICCM MCP servers (Fiedler, Dewey) use WebSocket
- Need unified interface to multiple backends
- Need backend restart resilience

**Solution Implemented:**
MCP Relay (`/mnt/projects/ICCM/mcp-relay/mcp_relay.py`) bridges stdio → WebSocket:

```
Claude Code (stdio subprocess)
    ↓
MCP Relay (stdio ↔ WebSocket multiplexer)
    ↓ Direct WebSocket connections
Fiedler (ws://localhost:9010) - 8 LLM models
Dewey (ws://localhost:9020) - Conversation storage
```

**Key Benefits:**
1. ✅ **Single MCP entry** - One "iccm" server exposes ALL backend tools
2. ✅ **stdio transport** - Claude Code officially supported protocol
3. ✅ **Dynamic tool discovery** - Aggregates tools from all backends automatically
4. ✅ **Backend restart resilience** - Relay auto-reconnects transparently
5. ✅ **Runtime server management** - Add/remove servers via MCP tools (no file editing)
6. ✅ **Network extensible** - Can connect to any WebSocket MCP server
7. ✅ **Config file watching** - Automatically reconnects when backends.yaml changes
8. ✅ **Status monitoring** - Query connection status through relay tools

---

## 📋 Implementation Status

### ✅ Phase 1: MCP Relay Implementation (COMPLETED)
- ✅ Created `/mnt/projects/ICCM/mcp-relay/mcp_relay.py` (371 lines)
- ✅ Created `/mnt/projects/ICCM/mcp-relay/backends.yaml` configuration
- ✅ Installed dependencies (websockets, pyyaml) in `.venv`
- ✅ Fixed Fiedler MCP server bugs (lines 298, 321)
- ✅ Rebuilt and restarted Fiedler container with fixes
- ✅ Fixed MCP relay to consume notification responses
- ✅ Verified all 8 Fiedler tools discovered successfully
- ✅ Updated `~/.claude.json` to use relay
- ✅ Archived old stable-relay code
- ✅ Updated architecture documentation

### ✅ Phase 2: Verification (COMPLETED)
**Results:**
- ✅ Both MCP servers connected (sequential-thinking, iccm)
- ✅ All Fiedler tools available via `mcp__iccm__fiedler_*` prefix
- ✅ 10 LLM models accessible: Gemini 2.5 Pro, GPT-5, GPT-4o, GPT-4o-mini, GPT-4-turbo, Llama 3.1-70B, Llama 3.3-70B, DeepSeek R1, Qwen 2.5-72B, Grok-4
- ✅ Auto-reconnection implemented and tested (2025-10-03 21:34)
- ✅ **Production verification:** Fiedler container restart → immediate reconnection, zero manual intervention

### ✅ Phase 3: Runtime Management Tools (COMPLETED)
**New Relay Tools:**
- ✅ `relay_add_server(name, url)` - Add and connect to new MCP server
- ✅ `relay_remove_server(name)` - Remove MCP server
- ✅ `relay_list_servers()` - Show all servers with connection status
- ✅ `relay_reconnect_server(name)` - Force reconnect to a server
- ✅ `relay_get_status()` - Detailed status with tool lists

**Design:**
- backends.yaml = startup config only (not source of truth)
- Runtime server management through MCP tools
- No file editing or restarts required for server changes
- Switch between direct/KGB routing via tool calls

---

## 🏗️ System Architecture

### Current Deployment: Bare Metal Claude + MCP Relay

```
Claude Code (bare metal)
    ├→ Claude Max (Anthropic API) - Direct HTTPS
    ├→ Sequential Thinking (NPM package) - stdio
    └→ MCP Relay (stdio subprocess)
         ├→ ws://localhost:9010 → Fiedler MCP (8 LLM models)
         └→ ws://localhost:9020 → Dewey MCP (conversation storage)
```

**Characteristics:**
- **No logging** - Direct connections bypass KGB
- **Minimal latency** - No intermediary proxies
- **Maximum stability** - Simple, direct architecture
- **Emergency fallback** - Always available

### Future: Containerized Claude (Logged Mode)

```
Claude Code (container)
    └→ MCP Relay (stdio subprocess inside container)
         └→ ws://kgb-proxy:9000 → KGB Proxy
              ├→ Fiedler (automatic logging to Winni)
              └→ Dewey (automatic logging to Winni)
```

**Benefits:**
- **Automatic logging** - All conversations captured via KGB
- **Same relay code** - Just different backends.yaml configuration
- **Production mode** - When logging/auditing required

---

## 🔧 Current Configuration

### MCP Server Config
**File:** `~/.claude.json`
```json
{
  "mcpServers": {
    "sequential-thinking": {
      "type": "stdio",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-sequential-thinking"]
    },
    "iccm": {
      "type": "stdio",
      "command": "/mnt/projects/ICCM/mcp-relay/mcp_relay.py",
      "args": []
    }
  }
}
```

### Backend Config
**File:** `/mnt/projects/ICCM/mcp-relay/backends.yaml`
```yaml
backends:
  - name: fiedler
    url: ws://localhost:9010
  - name: dewey
    url: ws://localhost:9020
```

### Trust Status
```json
"hasTrustDialogAccepted": true
```

---

## 🧪 Test Plan (After Claude Code Restart)

### 1. Verify Relay Management Tools
```
# List all connected servers
relay_list_servers()

# Get detailed status
relay_get_status()
```

### 2. Test Dynamic Server Management
```
# Switch Fiedler to KGB routing (enables logging)
relay_remove_server(name="fiedler")
relay_add_server(name="fiedler", url="ws://localhost:9000?upstream=fiedler")

# Verify tools still work
fiedler_list_models()

# Switch back to direct (disable logging)
relay_remove_server(name="fiedler")
relay_add_server(name="fiedler", url="ws://localhost:9010")
```

### 3. Test Status Monitoring
```
# Check connection status
relay_list_servers()

# Force reconnect if needed
relay_reconnect_server(name="fiedler")
```

### 4. Test Backend Restart Resilience
```
# In terminal: docker restart fiedler-mcp
# Relay should auto-reconnect, no manual intervention needed
fiedler_send(models=["gemini-2.5-pro"], prompt="Test after restart")
```

---

## 📁 Key Files

### MCP Relay
- `/mnt/projects/ICCM/mcp-relay/mcp_relay.py` - Main relay implementation
- `/mnt/projects/ICCM/mcp-relay/backends.yaml` - Backend configuration
- `/mnt/projects/ICCM/mcp-relay/.venv/` - Python virtual environment

### Architecture Documentation
- `/mnt/projects/ICCM/architecture/General Architecture.PNG` - System diagram
- `/mnt/projects/ICCM/architecture/CURRENT_ARCHITECTURE_OVERVIEW.md` - Detailed architecture
- `/mnt/projects/ICCM/CURRENT_STATUS.md` - This file
- `/mnt/projects/ICCM/BUG_TRACKING.md` - Bug investigation log

### Archived (Reference Only)
- `/mnt/projects/General Tools and Docs/archive/stable-relay_archived_2025-10-03/` - Old standalone relay

---

## 🚀 Quick Commands

### Check MCP Status
```bash
claude mcp list
```

### Check Backend Health
```bash
docker ps --filter "name=fiedler|dewey|kgb"
```

### View Logs
```bash
docker logs fiedler-mcp --tail 30
docker logs dewey-mcp --tail 30
```

### Restart Backends (if needed)
```bash
cd /mnt/projects/ICCM/fiedler && docker compose restart
```

---

## 🐛 Known Issues

### BUG #4: Relay Management Tools - websockets 15.x API Incompatibility

**Status:** ✅ Fixed, awaiting restart
**Discovered:** 2025-10-03 22:20 EDT
**Fixed:** 2025-10-03 22:30 EDT

**Problem:**
- `relay_list_servers()` and `relay_get_status()` fail with error: `'ClientConnection' object has no attribute 'closed'`
- Fiedler tools work fine, only relay management tools affected

**Root Cause:**
- websockets 15.x changed API: `ClientConnection` objects don't have `.closed` or `.open` attributes
- Must use `ws.state == State.OPEN` instead
- Code was checking `ws.closed` (doesn't exist)

**Fix Applied:**
- Added `from websockets.protocol import State` import
- Changed all connection checks from `ws.closed` to `ws.state == State.OPEN`
- Lines affected: 22, 388, 456-457

**Next Step:**
- Restart Claude Code to reload relay subprocess with fixed code
- Then verify all relay management tools work

**Previous bugs (all resolved):**
- ✅ Fiedler `_list_tools_handler` bug fixed (lines 298, 321)
- ✅ MCP relay notification response handling fixed
- ✅ Direct WebSocket connections working
- ✅ All 10 tools discovered successfully
- ✅ Auto-reconnection implemented and verified working (2025-10-03 21:34)

---

## 🔄 Next Steps

### Completed
1. ✅ **MCP Relay implemented** - All code complete
2. ✅ **Configuration updated** - ~/.claude.json points to new location
3. ✅ **Verified tools** - All 10 Fiedler tools accessible
4. ✅ **Auto-reconnection added** - Backend restart resilience implemented
5. ✅ **Auto-reconnection tested** - Fiedler restart verified (2025-10-03 21:34)
6. ✅ **BUG #3 marked RESOLVED** - All documentation updated

### Current Work (2025-10-03 22:35)
1. ✅ **BUG #4 Fixed** - websockets 15.x API compatibility issue resolved
2. 🔄 **Restart Claude Code** - Load relay with fixed code
3. ⏸️ **Comprehensive relay testing** - All tools, KGB routing, reconnection, Dewey integration

### Future Work
1. Investigate Dewey MCP tools (currently not implementing tools/list)
2. Plan containerized Claude implementation (optional)

---

## 📝 Session Notes

### What We Built (2025-10-03)
- **MCP Relay**: stdio-to-WebSocket bridge for Claude Code
- **Direct connections**: Removed unnecessary Stable Relay layer
- **Dynamic discovery**: Relay queries backends for tools automatically
- **Auto-reconnect**: Backend restarts handled transparently with automatic retry
- **Configuration-driven**: Easy to add new WebSocket MCP backends
- **Connection resilience**: WebSocket error detection and transparent reconnection

### Key Architectural Decisions
1. **MCP Relay as subprocess** - Lives inside Claude Code, not standalone service
2. **Direct WebSocket** - Bare metal bypasses KGB for maximum simplicity
3. **Containerized routes through KGB** - Future mode enables automatic logging
4. **Same relay code, different configs** - backends.yaml determines routing

### Critical Fixes Applied
1. Fiedler line 298: `app._list_tools_handler()` → `await list_tools()`
2. Fiedler line 321: `app._call_tool_handler(...)` → `await call_tool(...)`
3. MCP relay: Added notification response consumption (asyncio.wait_for)
4. Backends config: Direct WebSocket to backends (`ws://localhost:9010`, `ws://localhost:9020`)
5. MCP relay: Added WebSocket connection error handling with automatic reconnection and retry

---

**DEBUGGING JOURNEY SUMMARY:**
1. ❌ Design mistake - Tried to make Claude speak WebSocket (not supported)
2. ❌ Config crash - Mixed config formats broke MCP parser
3. ❌ Red herring - Chased "corrupted state", forced unnecessary Claude reinstall
4. ✅ Real culprit - Config format incompatibility discovered
5. ✅ Fundamental limit - Confirmed Claude Code doesn't support WebSocket
6. ✅ MCP Relay built - stdio-to-WebSocket bridge created
7. ✅ Bugs fixed - Fiedler (lines 298, 321) and relay notification handling
8. ✅ Production-hardened - Auto-reconnection tested and verified
9. ✅ **Management tools added** - Dynamic server control via MCP tools (no file editing)

**KEY INSIGHT:** Relay needed file watching + management tools
- Initial design: backends.yaml as source of truth (required restarts)
- Improved design: Runtime management through MCP tools
- Result: Add/remove/monitor servers without file editing or restarts

**CURRENT ACTION:** Restart Claude Code to load enhanced relay

**Next Test:** Use relay_add_server to switch to KGB routing, verify Dewey logging
