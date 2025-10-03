# Current Architecture Overview - ICCM System

**Last Updated:** 2025-10-03
**Purpose:** Explain the immutable architecture (PNG) and document current protocol configuration

---

## 🎯 Immutable Architecture (from PNG)

The architecture PNG shows the **component relationships and data flows** that define the ICCM system. These relationships are immutable and can only be changed in an architecture planning session.

### Component Diagram Overview

```
LEFT SIDE (Bare Metal Claude - Current Active):
- Claude Code (bare metal) connects to:
  - Claude Max (Anthropic API) - Yellow dotted line
  - Fiedler - Blue solid line
  - Local LLMs - Blue solid line

RIGHT SIDE (Future - Containerized with Logging):
- Containerized Claude Code connects through:
  - Relay Container → KGB Container → Fiedler Container
  - Relay Container → KGB Container → Dewey Container → Winni Database

LEGEND:
- Yellow dotted lines: Direct Claude Max connections
- Yellow solid lines: Claude Max Official connections
- Blue solid lines: LLM API connections
- Red lines: Logging data flow
- Black lines: Database queries
- Gray lines: Generic connections
```

**Key Architecture Principles:**
1. **Two deployment modes**: Bare metal (simple, direct) and containerized (logged, resilient)
2. **Bare metal left unaltered** for emergency/fallback purposes
3. **Containerized path routes through KGB** for automatic conversation logging
4. **All containers optional** except when logging is required
5. **Architecture changes** require formal architecture planning session

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

2. **Claude Code → Fiedler**
   - **Protocol:** stdio (via adapter to WebSocket)
   - **Current Config:** stdio adapter at `/mnt/projects/ICCM/fiedler/stdio_adapter.py`
   - **Status:** ✅ Working - Both MCP servers connected (awaiting Claude restart for tool access)
   - **Configuration Location:** `~/.claude.json` lines 269-282
   - **Architecture:** Claude Code (stdio) → stdio_adapter.py → ws://localhost:9010 → Fiedler
   - **Trust Status:** ✅ Enabled (`hasTrustDialogAccepted: true`)
   - Purpose: LLM orchestration via MCP tools (8 models: Gemini 2.5 Pro, GPT-5, etc.)

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
- `stable-relay` - Port 8000 (localhost) - For containerized Claude (future)
- `kgb-proxy` - Port 9000 (localhost) - For containerized Claude (future)
- `fiedler-mcp` - Port 9010 (0.0.0.0) - Available for direct connection
- `dewey-mcp` - Port 9020 (localhost) - For containerized Claude (future)

**Characteristics:**
- Bare metal Claude uses stdio adapter for Fiedler (bridges stdio ↔ WebSocket)
- Sequential-thinking uses stdio (NPM package execution)
- stdio adapter allows Claude Code compatibility while preserving Fiedler's WebSocket for AutoGen
- MCP subsystem operational (both servers show "Connected")
- **Trust must be accepted** - `hasTrustDialogAccepted: true` required for MCP servers to load
- Direct connection to Fiedler container via adapter (bypasses Relay/KGB for simplicity)

---

### Containerized Claude (FUTURE)

**Status:** Not yet built - infrastructure exists but containerized Claude itself doesn't exist yet

**Component Connections:**

1. **Claude Code (container) → Relay**
   - Protocol: WebSocket
   - URL: `ws://localhost:8000`
   - Purpose: Stable connection that survives backend restarts

2. **Relay → KGB**
   - Protocol: WebSocket
   - URL: `ws://kgb-proxy:9000?upstream=<target>`
   - Purpose: Route to upstream with automatic logging

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
- All conversations automatically logged
- Can restart backends without restarting Claude
- Relay auto-reconnects within 5 seconds
- More complex, but more resilient

---

## 🔧 Component Details

### Fiedler MCP Server
- **Container:** `fiedler-mcp`
- **Internal Port:** 8080
- **Host Port:** 9010 (mapped)
- **Purpose:** Orchestra conductor - unified interface to 7+ LLM providers
- **Models:** Gemini 2.5 Pro, GPT-5, GPT-4o, Grok 4, Llama 3.3, DeepSeek-R1, Qwen 2.5
- **Current Protocol:** WebSocket (containerized) or stdio (bare metal)

### KGB Proxy (Knowledge Gateway Broker)
- **Container:** `kgb-proxy`
- **Port:** 9000
- **Purpose:** Transparent proxy with automatic conversation logging
- **Pattern:** Spy worker per connection
- **Logs to:** Dewey via internal client
- **Current Protocol:** WebSocket

### Stable Relay
- **Container:** `stable-relay`
- **Port:** 8000 (localhost only)
- **Purpose:** Keep Claude alive during backend restarts
- **Size:** 111 lines, zero message parsing
- **Current Protocol:** WebSocket

### Dewey MCP Server
- **Container:** `dewey-mcp`
- **Port:** 9020
- **Purpose:** Conversation storage, search, and startup context management
- **Backend:** PostgreSQL on Irina
- **Tools:** 11 MCP tools for conversation management
- **Current Protocol:** WebSocket

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
Needs LLM orchestration
  ↓
Calls mcp__fiedler__* tool
  ↓
docker exec -i fiedler-mcp fiedler
  ↓
Fiedler executes (stdio communication)
  ↓
Returns result to Claude
  ↓
Claude presents to user
```

### Containerized Flow (Future)
```
User types command
  ↓
Claude Code (container) processes
  ↓
Calls MCP tool
  ↓
WebSocket to Stable Relay (port 8000)
  ↓
Relay forwards to KGB (port 9000)
  ↓
KGB spawns spy worker
  ↓
Spy logs to Dewey (internal client)
  ↓
Spy forwards to upstream (Fiedler or Dewey)
  ↓
Upstream executes and returns
  ↓
Response flows back through chain
  ↓
User sees result
```

---

## 📝 Configuration Files

### Bare Metal Claude (ACTUAL CURRENT CONFIG)
**File:** `~/.claude.json`
**Section:** `projects["/home/aristotle9"].mcpServers`
**Lines:** 129-142

**Current Configuration (WORKING - stdio adapter solution):**
```json
{
  "mcpServers": {
    "sequential-thinking": {
      "type": "stdio",
      "command": "npx",
      "args": ["@modelcontextprotocol/server-sequential-thinking"],
      "env": {}
    },
    "fiedler": {
      "type": "stdio",
      "command": "/mnt/projects/ICCM/fiedler/stdio_adapter.py",
      "args": []
    }
  }
}
```

**Trust Configuration (Line 286):**
```json
"hasTrustDialogAccepted": true
```

**Status:** ✅ Both MCP servers connected - awaiting Claude Code restart for Fiedler tool access

**Critical Notes:**
- **IMPORTANT:** Claude Code MCP only supports stdio, SSE, HTTP (NOT WebSocket)
- stdio adapter bridges Claude Code (stdio) ↔ Fiedler (WebSocket)
- Fiedler keeps WebSocket for AutoGen/agent ecosystem compatibility
- **Trust must be accepted** (`hasTrustDialogAccepted: true`) for MCP servers to load
- Containerized Claude (future) can use stdio adapter or SSE/HTTP transport

### Containerized Claude (when active)
**File:** `~/.config/claude-code/mcp.json` or `~/.claude.json`

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
