# Current Architecture Overview - ICCM System

**Last Updated:** 2025-10-03 22:10 EDT
**Purpose:** Explain the immutable architecture (PNG) and document current protocol configuration

---

## 🎯 Immutable Architecture (from PNG)

The architecture PNG shows the **component relationships and data flows** that define the ICCM system. These relationships are immutable and can only be changed in an architecture planning session.

### Component Diagram Overview

```
LEFT SIDE (Bare Metal Claude - Current Active):
- Claude Code (bare metal) with MCP Relay extension
  - MCP Relay connects directly to WebSocket MCP servers:
    - Fiedler (ws://localhost:9010) - 8 LLM models
    - Dewey (ws://localhost:9020) - Conversation storage
  - Direct connections to:
    - Claude Max (Anthropic API) - Yellow dotted line
    - Local LLMs - Blue solid line (future)

RIGHT SIDE (Future - Containerized with Logging):
- Containerized Claude Code with MCP Relay inside container
  - MCP Relay → KGB (9000) → Fiedler (8080)
  - MCP Relay → KGB (9000) → Dewey (9020) → Winni Database
  - KGB provides automatic conversation logging

LEGEND:
- Yellow dotted lines: Direct Claude Max connections
- Yellow solid lines: Claude Max Official connections
- Blue solid lines: LLM API / WebSocket connections
- Red lines: Logging data flow
- Black lines: Database queries
- Gray lines: Generic connections
```

**Key Architecture Principles:**
1. **MCP Relay as Claude extension**: Acts as stdio-to-WebSocket bridge, runs as subprocess of Claude Code
2. **Two deployment modes**: Bare metal (direct, no logging) and containerized (logged through KGB)
3. **Bare metal connects directly**: MCP Relay → WebSocket MCP servers (no intermediary relay)
4. **Containerized routes through KGB**: MCP Relay → KGB → backends (automatic logging)
5. **Runtime server management**: Switch modes via MCP tools (no file editing or restarts)
6. **Architecture changes** require formal architecture planning session

---

## 📋 Current Protocol Implementation

**Note:** The protocols and connection methods below implement the architecture shown in the PNG. These can be updated through bug fixes and feature implementations without changing the underlying architecture.

### Bare Metal Claude (CURRENT ACTIVE)

**Status:** Production, stable, primary deployment

**Component Connections:**

1. **Claude Code → Claude Max**
   - Protocol: HTTPS REST API
   - Authentication: User login (primary) or API key (sparse use)
   - Purpose: AI assistant conversations

2. **Claude Code → ICCM Services (Fiedler, Dewey)**
   - **Protocol:** stdio (MCP Relay extension)
   - **Current Config:** MCP relay at `/mnt/projects/ICCM/mcp-relay/mcp_relay.py`
   - **Status:** ✅ Working - All 8 Fiedler tools registered
   - **Configuration Location:** `~/.claude.json` "iccm" mcpServer entry
   - **Architecture:** Claude Code → MCP Relay (stdio subprocess) → Direct WebSocket to backends
     - Fiedler: `ws://localhost:9010` (8 LLM models)
     - Dewey: `ws://localhost:9020` (conversation storage)
   - **Trust Status:** ✅ Enabled (`hasTrustDialogAccepted: true`)
   - Purpose: Unified access to all ICCM MCP tools through single stdio interface

3. **Claude Code → Sequential Thinking**
   - **Protocol:** stdio (NPM package)
   - **Current Config:** `npx @modelcontextprotocol/server-sequential-thinking`
   - **Status:** ✅ Working
   - **Configuration Location:** `~/.claude.json` lines 210-217
   - Purpose: Extended thinking capability for complex reasoning

4. **Claude Code → Local LLMs**
   - Protocol: Not yet implemented
   - Purpose: Future local model integration

**Configuration Location:** `~/.claude.json` (project-specific mcpServers)

**Running Infrastructure:**
- `fiedler-mcp` - Port 8080 (container), 9010 (host) - 8 LLM models via WebSocket
- `dewey-mcp` - Port 9020 (host) - Conversation storage/retrieval via WebSocket
- `kgb-proxy` - Port 9000 (container) - Automatic logging (containerized mode only)
- `winni` - PostgreSQL on Irina (192.168.1.210) - Persistent storage

**Characteristics:**
- **MCP Relay runs as Claude subprocess** - Lives inside Claude Code process, not a separate service
- **Direct WebSocket connections** - No intermediary relay in bare metal mode
- **Tool aggregation** - Single "iccm" MCP entry exposes all backend tools
- **Dynamic discovery** - Relay queries backends for tools on startup
- **Zero-restart tool updates** - MCP `notifications/tools/list_changed` protocol
- **Network extensible** - Can connect to any WebSocket MCP server by updating backends.yaml
- **No logging in bare metal** - Connections go directly to Fiedler/Dewey, bypassing KGB
- **Trust must be accepted** - `hasTrustDialogAccepted: true` required for MCP servers to load

---

### Containerized Claude (FUTURE)

**Status:** Not yet built - MCP Relay code ready, containerized Claude not created yet

**Component Connections:**

1. **Claude Code (container) with MCP Relay extension**
   - MCP Relay runs as stdio subprocess inside container
   - Configured to connect through KGB for automatic logging

2. **MCP Relay → KGB**
   - Protocol: WebSocket
   - URL: `ws://kgb-proxy:9000?upstream=<target>`
   - Purpose: Route to upstream with automatic conversation logging

3. **KGB → Fiedler**
   - Protocol: WebSocket
   - URL: `ws://fiedler-mcp:8080`
   - Purpose: LLM orchestration tools

4. **KGB → Dewey**
   - Protocol: WebSocket
   - URL: `ws://dewey-mcp:9020`
   - Purpose: Conversation storage/retrieval tools

5. **KGB Internal → Dewey** (for logging)
   - Protocol: Direct MCP client calls
   - Purpose: Automatic conversation logging
   - Note: Separate code path from user's Dewey MCP tools

6. **Dewey → Winni**
   - Protocol: PostgreSQL (asyncpg)
   - Host: 192.168.1.210 (Irina)
   - Database: winni
   - Purpose: Persistent conversation storage

**Characteristics:**
- **Same MCP Relay code** - Just different backends.yaml configuration
- **All conversations automatically logged** - Via KGB interception
- **Backend restart resilience** - MCP Relay auto-reconnects to KGB
- **More complex, but logged** - Trade-off for automatic conversation capture

---

## 🔧 Component Details

### MCP Relay
- **Location:** `/mnt/projects/ICCM/mcp-relay/mcp_relay.py`
- **Type:** Python subprocess (stdio transport)
- **Purpose:** Bridge Claude Code (stdio) to WebSocket MCP servers
- **Startup Config:** `/mnt/projects/ICCM/mcp-relay/backends.yaml` (initial servers only)
- **Features:**
  - Dynamic tool discovery and aggregation
  - **Zero-restart tool updates** via MCP notifications
  - Auto-reconnect on backend failures
  - Config file watching (hot-reload)
  - Runtime server management via MCP tools
- **MCP Protocol Support:**
  - `initialize` - Declares `"tools": { "listChanged": true }` capability
  - `tools/list` - Returns aggregated tools from all backends
  - `tools/call` - Routes to appropriate backend
  - `notifications/tools/list_changed` - Notifies client when tools change
- **Management Tools:**
  - `relay_add_server(name, url)` - Add new MCP server
  - `relay_remove_server(name)` - Remove MCP server
  - `relay_list_servers()` - List all servers with status
  - `relay_reconnect_server(name)` - Force reconnect
  - `relay_get_status()` - Detailed status report
- **Default Backends:** Fiedler (ws://localhost:9010), Dewey (ws://localhost:9020)

### Fiedler MCP Server
- **Container:** `fiedler-mcp`
- **Internal Port:** 8080 (WebSocket)
- **Host Port:** 9010 (mapped)
- **Purpose:** Orchestra conductor - unified interface to 8 LLM providers
- **Models:** Gemini 2.5 Pro, GPT-5, GPT-4o-mini, Grok 2, Llama 3.3, DeepSeek Chat, Qwen 2.5, Claude 3.7
- **Protocol:** WebSocket MCP
- **Tools:** 8 tools (list_models, send, set_models, get_config, set_output, keyring management)

### Dewey MCP Server
- **Container:** `dewey-mcp`
- **Port:** 9020 (host)
- **Purpose:** Conversation storage, search, and startup context management
- **Backend:** PostgreSQL on Irina (192.168.1.210)
- **Tools:** 11 MCP tools for conversation management
- **Protocol:** WebSocket MCP

### KGB Proxy (Knowledge Gateway Broker)
- **Container:** `kgb-proxy`
- **Port:** 9000 (container)
- **Purpose:** Transparent WebSocket proxy with automatic conversation logging
- **Pattern:** Spy worker per connection
- **Logs to:** Dewey via internal client
- **Usage:** Optional - enable via relay_add_server tool
  - Direct mode: `ws://localhost:9010` (no logging)
  - Logged mode: `ws://localhost:9000?upstream=fiedler` (automatic logging)

### Winni Database
- **Type:** PostgreSQL
- **Host:** Irina (192.168.1.210)
- **Database:** winni
- **Purpose:** Data lake for conversations, contexts, LLM results
- **Current Protocol:** PostgreSQL wire protocol

---

## 🔄 Connection Flows

### Bare Metal Flow (Current)
```
User types command
  ↓
Claude Code processes
  ↓
Needs LLM orchestration → Calls mcp__fiedler__fiedler_send
  ↓
MCP Relay (stdio subprocess) receives request
  ↓
Relay routes to Fiedler backend (ws://localhost:9010)
  ↓
Fiedler MCP container executes (WebSocket MCP protocol)
  ↓
Response flows back: Fiedler → MCP Relay → Claude Code
  ↓
Claude presents result to user
```

**Key Points:**
- No logging - Direct connection bypasses KGB
- MCP Relay lives as subprocess of Claude Code
- Backend restart → Relay auto-reconnects transparently

### Containerized Flow (Future)
```
User types command
  ↓
Claude Code (container) processes
  ↓
Calls MCP tool
  ↓
MCP Relay (stdio subprocess inside container)
  ↓
WebSocket to KGB (ws://kgb-proxy:9000?upstream=fiedler)
  ↓
KGB spawns spy worker (logs conversation to Dewey)
  ↓
Spy forwards to Fiedler or Dewey
  ↓
Backend executes and returns
  ↓
Response flows back: Backend → KGB → MCP Relay → Claude Code
  ↓
User sees result (conversation logged in Winni)
```

**Key Points:**
- Automatic logging - KGB intercepts all traffic
- Same MCP Relay code, different backends.yaml
- Logging separate from user-facing Dewey tools

---

## 📝 Configuration Files

### Bare Metal Claude (ACTUAL CURRENT CONFIG)
**File:** `~/.claude.json`
**Section:** `projects["/home/aristotle9"].mcpServers`
**Lines:** 129-142

**Current Configuration (WORKING - Direct WebSocket Connections):**
```json
{
  "mcpServers": {
    "sequential-thinking": {
      "type": "stdio",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-sequential-thinking"],
      "env": {}
    },
    "iccm": {
      "type": "stdio",
      "command": "/mnt/projects/ICCM/mcp-relay/mcp_relay.py",
      "args": []
    }
  }
}
```

**Backend Configuration:** `/mnt/projects/ICCM/mcp-relay/backends.yaml`
```yaml
backends:
  - name: fiedler
    url: ws://localhost:9010
    # Fiedler MCP server on host port 9010 (container port 8080)

  - name: dewey
    url: ws://localhost:9020
    # Dewey MCP server on port 9020
```

**Trust Configuration:**
```json
"hasTrustDialogAccepted": true
```

**Status:** ✅ MCP Relay working - All 8 Fiedler tools registered successfully

**Critical Notes:**
- **Claude Code MCP limitation:** Only supports stdio, SSE, HTTP (NOT WebSocket directly)
- **MCP Relay solution:** Bridges Claude Code (stdio) ↔ WebSocket MCP backends
- **Direct connections:** Bare metal goes straight to Fiedler/Dewey (no intermediary relay)
- **Single "iccm" entry:** Exposes all backend tools through unified interface
- **Dynamic tool discovery:** Relay queries backends on startup
- **Auto-reconnect:** Backend restarts handled transparently
- **Network extensible:** Add any WebSocket MCP server via backends.yaml
- **Trust required:** `hasTrustDialogAccepted: true` must be set for MCP servers to load

### Containerized Claude (future configuration)
**File:** Container's `~/.claude.json` (inside containerized Claude)

**MCP Server Config (same as bare metal):**
```json
{
  "mcpServers": {
    "iccm": {
      "type": "stdio",
      "command": "/app/mcp-relay/mcp_relay.py",
      "args": []
    }
  }
}
```

**Backend Configuration (different from bare metal):**
`/app/mcp-relay/backends.yaml` inside container:
```yaml
backends:
  - name: fiedler
    url: ws://kgb-proxy:9000?upstream=fiedler
    # Routes through KGB for automatic logging

  - name: dewey
    url: ws://kgb-proxy:9000?upstream=dewey
    # Routes through KGB for automatic logging
```

**Key Difference:** Containerized mode routes through KGB, bare metal connects directly

---

## 🎯 Design Rationale

### Why Two Deployment Modes?

1. **Bare Metal (Left side of PNG)**
   - Emergency fallback
   - Maximum stability
   - No dependencies on logging infrastructure
   - Simple troubleshooting

2. **Containerized (Right side of PNG)**
   - Automatic conversation logging
   - Can fix/restart backends without losing context
   - Production use when logging is required
   - More resilient to component failures

### Why Route Both Through KGB?

**Question:** Why route Dewey through KGB when KGB logs to Dewey?

**Answer:** Different code paths prevent circular logging:
- **User's Dewey tools** → Relay → KGB → Dewey MCP server (logged)
- **KGB's logging** → Internal Dewey client → Dewey database directly (not logged)

No circular dependency because the internal client bypasses the MCP layer.

---

## 🚧 Current Deployment Status

**Active:** Bare Metal Claude
**Configuration:** stdio adapter to Fiedler (WORKING)
**Logging:** None (by design)
**Status:** Both MCP servers connected, awaiting restart for tool access

**Future:** Containerized Claude
**Components:** Relay, KGB, Fiedler, Dewey containers exist
**Containerized Claude:** Not yet built
**Status:** Infrastructure ready, container not created yet

---

## 📚 Related Documentation

- `/mnt/projects/ICCM/architecture/General Architecture.PNG` - Immutable architecture diagram
- `/mnt/projects/ICCM/CURRENT_STATUS.md` - Current work and session status
- `/mnt/projects/ICCM/BUG_TRACKING.md` - Active bug investigations
- `/mnt/projects/ICCM/fiedler/README.md` - Fiedler MCP server details
- `/mnt/projects/ICCM/kgb/README.md` - KGB proxy details
- `/mnt/projects/ICCM/architecture/dewey_winni_requirements_v3.md` - Dewey specifications
- `/mnt/projects/ICCM/architecture/STABLE_RELAY_DEPLOYMENT.md` - Relay deployment details

---

**Note:** This document describes the current protocol configuration. The architecture itself (component relationships and data flows) is defined in the PNG diagram and can only be changed in architecture planning sessions.

---

## ⚠️ Important Notes

1. **Architecture changes** (component relationships in PNG) require formal planning session
2. **Protocol configuration** (transport types, adapters, ports) documented here - can be updated in regular sessions
3. **Bare metal deployment** uses stdio adapter to WebSocket (Claude Code limitation)
4. **This document** reflects current configuration state and should be updated when protocols change
5. **Configuration bugs** should be tracked in BUG_TRACKING.md with attempted solutions
6. **stdio adapter** is the bridge solution for Claude Code ↔ WebSocket-based services
