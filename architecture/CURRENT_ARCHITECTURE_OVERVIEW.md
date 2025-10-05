# Current Architecture Overview - ICCM System

**Last Updated:** 2025-10-05 16:30 EDT
**Purpose:** Explain the immutable architecture (PNGs) and document current protocol configuration

---

## 🎯 Immutable Architecture (from PNGs)

The architecture PNGs show the **component relationships and data flows** that define the ICCM system. These relationships are immutable and can only be changed in an architecture planning session.

### Architecture Diagrams

**Three complementary views of ICCM architecture:**

1. **Diagram_1_MCP_Traffic.png** - MCP connections and LLM access
2. **Diagram_2_Data_Writes.png** - Write-only logging and conversation storage flow
3. **Diagram_3_Data_Reads.png** - Read-only query and retrieval flow

### Component Diagram Overview

**OUTSIDE ECOSYSTEM:**
- **Claude Code (Bare Metal)** - Yellow box, outside ecosystem
  - Can bypass to Claude API directly (emergency access - red dashed line)
  - Connects to MCP Relay via MCP protocol (bidirectional)
  - Logs directly to Godot (cannot fail if ecosystem broken)

**INSIDE ECOSYSTEM:**
- **Claudette (Containerized Claude)** - Blue box, inside ecosystem
  - Must follow all architectural rules
  - Connects to MCP Relay via MCP protocol (bidirectional)
  - Routes through Fiedler for ALL LLM access (no direct API access)
  - Logs to Godot via MCP tools

- **MCP Relay (9000)** - Central hub
  - Bidirectional MCP connections to all servers (9022, 9030, 9041, 9050, 9060, 9031)
  - Logs to Godot

- **Fiedler (9030)** - LLM Gateway
  - Routes ALL LLM calls to Cloud LLMs (OpenAI, Google, xAI, Anthropic)
  - Logs conversations AND operational logs to Godot

- **Godot (9060)** - WRITE Specialist
  - Receives logs from ALL components (blue arrows)
  - Receives conversation logs from Fiedler
  - Single source of truth for ALL writes to Winni
  - Writes to PostgreSQL Winni (44TB RAID 5)

- **Dewey (9022)** - READ Specialist
  - Query-only access to Winni (bidirectional request/reply - green arrows)
  - NO write capabilities
  - Logs operational activity to Godot

- **Support Services:**
  - Playfair (9041) - Diagram generation
  - Gates (9050) - Document generation
  - Marco (9031) - Browser automation

- **Cloud LLMs** - Right side, gray box
  - Claude API (pink oval) - Anthropic
  - Other LLMs (yellow oval) - OpenAI, Google, xAI, etc.

**Key Architecture Principles:**
1. **Option 4: Write/Read Separation**
   - **Godot = WRITE specialist**: ALL database writes flow through Godot
   - **Dewey = READ specialist**: Query-only, no write capabilities

2. **Two-tier access control:**
   - **Claude Code (Bare Metal)**: Outside ecosystem, can bypass for emergency repairs
   - **Claudette (Containerized)**: Inside ecosystem, must follow all rules

3. **Centralized logging**: ALL components log to Godot via MCP tools

4. **LLM Gateway**: Fiedler is the ONLY path to Cloud LLMs for ecosystem components

5. **Conversation logging**: Fiedler logs ALL LLM conversations to Godot

6. **KGB elimination**: No HTTP proxy needed - Claudette uses MCP Relay directly

7. **Architecture changes** require formal architecture planning session

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

2. **Claude Code → ICCM Services (Fiedler, Dewey, Marco)**
   - **Protocol:** stdio (MCP Relay extension)
   - **Current Config:** MCP relay at `/mnt/projects/ICCM/mcp-relay/mcp_relay.py`
   - **Status:** ✅ Working - All 10 Fiedler tools registered
   - **Configuration Location:** `~/.claude.json` "iccm" mcpServer entry
   - **Architecture:** Claude Code → MCP Relay (stdio subprocess) → Direct WebSocket to backends
     - Fiedler: `ws://localhost:9010` (10 LLM models)
     - Dewey: `ws://localhost:9020` (conversation storage)
     - Marco: `ws://localhost:9030` (browser automation - planned)
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
- `fiedler-mcp` - Port 8080 (container), 9010 (host WebSocket), 9011 (host HTTP proxy) - 10 LLM models
- `dewey-mcp` - Port 9020 (host) - Conversation storage/retrieval via WebSocket
- `marco-mcp` - Port 8030 (container), 9030 (host) - Browser automation via WebSocket (planned)
- `kgb-proxy` - Port 8089 (HTTP/SSE), 9000 (WebSocket) - Logging proxy and gateway
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

### Claudette: Containerized Claude - ✅ OPERATIONAL (2025-10-04)

**Status:** ✅ Production ready and verified

**Component Connections:**

1. **Claudette (claude-code-container) with MCP Relay extension**
   - MCP Relay runs as stdio subprocess inside container
   - Configured to connect through KGB for automatic logging
   - Container: `iccm/claude-code:latest`
   - Config: `/mnt/projects/ICCM/claude-container/config/`

2. **Claudette → Anthropic API (via KGB HTTP Gateway)**
   - Protocol: HTTPS (proxied through KGB)
   - URL: `http://kgb-proxy:8089` → `https://api.anthropic.com`
   - Purpose: All Anthropic API conversations with automatic logging
   - Status: ✅ Verified working (200 OK responses)

3. **MCP Relay → KGB WebSocket Spy**
   - Protocol: WebSocket
   - URL: `ws://kgb-proxy:9000?upstream=<target>`
   - Purpose: Route MCP tool calls with automatic conversation logging

4. **KGB → Fiedler**
   - Protocol: WebSocket
   - URL: `ws://fiedler-mcp:8080`
   - Purpose: LLM orchestration tools

5. **KGB → Dewey**
   - Protocol: WebSocket
   - URL: `ws://dewey-mcp:9020`
   - Purpose: Conversation storage/retrieval tools

6. **KGB Internal → Dewey** (for logging)
   - Protocol: Direct MCP client calls
   - Purpose: Automatic conversation logging
   - Note: Separate code path from user's Dewey MCP tools
   - Status: ✅ Verified (conversations logged successfully)

7. **Dewey → Winni**
   - Protocol: PostgreSQL (asyncpg)
   - Host: 192.168.1.210 (Irina)
   - Database: winni
   - Purpose: Persistent conversation storage
   - Status: ✅ Verified (messages stored in database)

**Characteristics:**
- **Same MCP Relay code** - Just different backends.yaml configuration
- **All conversations automatically logged** - Via KGB interception
- **Backend restart resilience** - MCP Relay auto-reconnects to KGB
- **Complete audit trail** - Both API calls and MCP tool usage logged
- **Cloudflare 403 resolved** - SSL/TLS connector fix applied to KGB gateway

**Verification (2025-10-04):**
- ✅ Claudette → KGB Gateway: 200 OK responses
- ✅ KGB → Anthropic: SSL/TLS handshake successful
- ✅ Conversations logged: `b02ea596-74fe-4919-b2a5-d8630751fd6d`, etc.
- ✅ Messages stored: Turn 1 (request), Turn 2 (response)
- ✅ Complete pipeline operational

**Documentation:** `/mnt/projects/ICCM/claude-container/README.md`

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

### Fiedler MCP Server (LLM Orchestra Gateway)
- **Container:** `fiedler-mcp`
- **Internal Ports:** 8080 (WebSocket MCP), 8081 (HTTP streaming proxy)
- **Host Ports:** 9010 (WebSocket), 9011 (HTTP proxy)
- **Purpose:** Orchestra conductor - unified interface to 10 LLM providers
- **Models:** Gemini 2.5 Pro, GPT-5, GPT-4o-mini, Grok 2, Llama 3.3, DeepSeek Chat, DeepSeek-R1, Qwen 2.5, Claude 3.7, Together
- **Protocol:** WebSocket MCP + HTTP streaming proxy
- **Tools:** 10 tools (list_models, send, set_models, get_config, set_output, keyring management)
- **Dual Role:** MCP tool server + HTTP streaming proxy for KGB routing

### Dewey MCP Server (Conversation Storage Gateway)
- **Container:** `dewey-mcp-blue` (production), `dewey-mcp` (stopped)
- **Port:** 9022 (blue deployment, mapped to 9020 internally)
- **Purpose:** Conversation storage, search, and startup context management
- **Backend:** PostgreSQL (Winni) on Irina (192.168.1.210) - 44TB RAID 5 storage
- **Architecture:** Option 4 - Write/Read Separation (Dewey = READ specialist, Godot = WRITE specialist for logs)
- **Tools:** 13 MCP tools (conversation management + log READ tools: dewey_query_logs, dewey_get_log_stats)
- **Protocol:** WebSocket MCP
- **Logging Integration:** Uses Godot's logger_log tool via MCP (ws://godot-mcp:9060)

### Marco MCP Server (Internet Gateway) - 📋 PLANNED
- **Container:** `marco-mcp` (to be implemented)
- **Internal Port:** 8030 (WebSocket MCP + HTTP health check)
- **Host Port:** 9030
- **Purpose:** Internet/browser automation gateway via Playwright
- **Backend:** Playwright MCP subprocess (Chromium headless)
- **Tools:** ~7 Playwright tools + `marco_reset_browser`
- **Protocol:** WebSocket MCP
- **Status:** Requirements approved v1.2, implementation pending
- **Architecture:** Marco WebSocket Server → stdio-WebSocket Bridge → Playwright MCP → Chromium
- **Phase 1 Limitations:**
  - Single browser instance with FIFO request queuing
  - No authentication (network isolation only)
  - Internal use only - NEVER expose to public internet
- **Resource Limits:** 2GB memory, headless mode
- **Documentation:** `/mnt/projects/ICCM/marco/REQUIREMENTS.md`

### KGB Proxy (Logging Proxy Gateway)
- **Container:** `kgb-proxy`
- **Ports:** 8089 (HTTP/SSE gateway), 9000 (WebSocket spy - deprecated)
- **Purpose:** Transparent WebSocket proxy with automatic conversation logging
- **Pattern:** Spy worker per connection
- **Logs to:** Dewey via internal client
- **Usage:** Optional - enable via relay_add_server tool
  - Direct mode: `ws://localhost:9010` (no logging)
  - Logged mode: `ws://localhost:9000?upstream=fiedler` (automatic logging)

### Godot Centralized Logging (Production)
- **Container:** `godot-mcp`
- **Port:** 9060 (WebSocket MCP)
- **Purpose:** Centralized logging infrastructure for all ICCM components
- **Architecture:** Option 4 - Write/Read Separation (Godot = WRITE specialist for logs)
  - MCP Server + Redis Queue (internal) + Worker with direct PostgreSQL INSERT
  - Godot writes logs directly to PostgreSQL (bypassing Dewey for writes)
  - Dewey provides READ-only tools for log queries
- **Redis:** Port 6379 internal only (bind: 127.0.0.1) - NOT exposed on network
- **Integration:** ALL MCP servers MUST use MCP-based logging via `logger_log` tool
- **Data Flow:** Component → logger_log (WS MCP) → Redis (internal) → Worker → PostgreSQL (direct INSERT)
- **Database Access:** Godot has INSERT-only permission on logs table via godot_log_writer user
- **FORBIDDEN:** Direct Redis connections - violates MCP protocol architecture
- **Documentation:** `/mnt/projects/ICCM/godot/REQUIREMENTS.md`

### Winni Database
- **Type:** PostgreSQL 16
- **Host:** Irina (192.168.1.210)
- **Database:** winni
- **Storage:** 44TB RAID 5 array (4x 14.6TB drives) at /mnt/storage/postgresql/16/main
- **Purpose:** Data lake for conversations, contexts, LLM results, centralized logs
- **Access Patterns:**
  - Dewey: Full read/write for conversations, READ-only for logs
  - Godot: INSERT-only for logs table via dedicated godot_log_writer user
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
    url: ws://localhost:9012
    # Fiedler Blue MCP server on host port 9012 (container port 8080)

  - name: dewey
    url: ws://localhost:9022
    # Dewey Blue MCP server on port 9022 (container internal 9020)

  - name: gates
    url: ws://localhost:9051
    # Gates Blue MCP server on port 9051

  - name: playfair
    url: ws://localhost:9041
    # Playfair Blue MCP server on port 9041

  - name: marco
    url: ws://localhost:9031
    # Marco Blue MCP server on port 9031
```

**Trust Configuration:**
```json
"hasTrustDialogAccepted": true
```

**Status:** ✅ MCP Relay working - 30 tools across 5 backends (Fiedler, Dewey, Gates, Playfair, Marco)

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

**Active Deployments:**

1. **Bare Metal Claude** (Development/Emergency)
   - **Configuration:** Direct WebSocket to MCP servers
   - **Logging:** None (by design)
   - **Status:** ✅ Operational
   - **Use Case:** Development, debugging, emergency fallback

2. **Claudette** (Production/Logged) - ✅ OPERATIONAL
   - **Configuration:** KGB HTTP Gateway + WebSocket Spy
   - **Logging:** Complete (Anthropic API + MCP tools)
   - **Status:** ✅ Verified and operational (2025-10-04)
   - **Use Case:** Production sessions requiring audit trail
   - **Container:** `claude-code-container` on `iccm_network`

**Infrastructure Status:**
- ✅ KGB Proxy (dual-protocol: HTTP 8089 + WebSocket 9000)
- ✅ Fiedler MCP (10 LLM models)
- ✅ Dewey MCP (conversation storage)
- ✅ Winni Database (PostgreSQL on Irina)
- ✅ MCP Relay (stdio-to-WebSocket bridge)

---

## 📚 Related Documentation

- `/mnt/projects/ICCM/architecture/General Architecture.PNG` - Immutable architecture diagram
- `/mnt/projects/ICCM/CURRENT_STATUS.md` - Current work and session status
- GitHub Issues (https://github.com/rmdevpro/ICCM/issues) - Active bugs and tasks
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
5. **Configuration bugs** should be tracked in GitHub Issues with attempted solutions
6. **stdio adapter** is the bridge solution for Claude Code ↔ WebSocket-based services
