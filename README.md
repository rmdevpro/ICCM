# ICCM - Integrated Cognitive Computing Machine

**Last Updated:** 2025-10-06
**Status:** Production - Option 4 Architecture Fully Deployed
**GitHub:** https://github.com/rmdevpro/ICCM

---

## Overview

ICCM is a multi-agent AI system featuring:
- **Option 4 Architecture**: Write/Read separation (Godot=WRITE, Dewey=READ)
- **MCP Protocol**: Model Context Protocol for all component communication
- **Triplet-Driven Development**: Gemini 2.5 Pro, GPT-5, Grok-4 consensus-based design
- **Blue/Green Deployment**: Zero-downtime component updates
- **Unified Logging**: All operational logs flow through Godot to PostgreSQL

---

## Quick Start for Developers

### Before Writing ANY Code

**CRITICAL - Read these documents first:**

1. **`CONTRIBUTING.md`** - Development standards and required libraries
2. **`architecture/CURRENT_ARCHITECTURE_OVERVIEW.md`** - System architecture
3. **`CURRENT_STATUS.md`** - Current session status and recent changes
4. **`Development Cyle.PNG`** - Triplet-driven development process

### Required Standard Libraries

**MANDATORY for all components - Never reimplement these:**

#### 1. iccm-network (Python MCP Servers)
- **Location:** `/mnt/projects/ICCM/iccm-network/`
- **Docs:** `iccm-network/README.md`
- **Usage:** WebSocket MCP server (eliminates 10+ hours of debugging)

```python
from iccm_network import MCPServer, MCPToolError

server = MCPServer(
    name="myservice",
    version="1.0.0",
    port=8070,
    tool_definitions=TOOLS,
    tool_handlers=HANDLERS
)
await server.start()
```

#### 2. Godot MCP Logger (All Components)
- **Location:** Copy `godot/mcp_logger.py` from any Blue component
- **Docs:** `/mnt/projects/ICCM/godot/README.md`
- **Usage:** Operational logging to Godot → Dewey → PostgreSQL

```python
from .godot.mcp_logger import log_to_godot

await log_to_godot('INFO', 'Server started', component='myservice')
```

### Starting a New Component

```bash
# 1. Copy component template (when available - see Issue #12)
cp -r component-template/ your-component/

# 2. Customize the template
cd your-component/
# Update name, port, tools

# 3. Build and deploy
docker compose build your-component-blue
docker compose up -d your-component-blue

# 4. Test
# Via MCP Relay: relay_add_server(name="yourcomponent", url="ws://...")
```

---

## System Architecture

### Core Components

| Component | Port | Purpose | Status |
|-----------|------|---------|--------|
| **Fiedler** | 9010 | LLM Gateway (10 models) | ✅ Production |
| **Dewey** | 9020 | READ Specialist (queries only) | ✅ Production |
| **Godot** | 9060 | WRITE Specialist (logging, conversations) | ✅ Production |
| **Gates** | 9050 | Document Generation | ✅ Production |
| **Playfair** | 9040 | Diagram Generation | ✅ Production |
| **Marco** | 9030 | Browser Automation | ✅ Production |
| **Horace** | 9070 | File Storage Gateway | ✅ Production |
| **MCP Relay** | stdio | Tool Aggregation Hub | ✅ Production |

### Architecture Diagrams

**Three complementary views:**
1. `architecture/Diagram_1_MCP_Traffic.png` - MCP connections and LLM access
2. `architecture/Diagram_2_Data_Writes.png` - Write-only logging flow
3. `architecture/Diagram_3_Data_Reads.png` - Read-only query flow

### Database

- **Winni PostgreSQL** - 44TB RAID 5 on Irina (192.168.1.210)
- **Godot Tables:** logs, conversations, messages
- **Horace Tables:** horace_files, horace_collections, horace_versions

---

## Development Process

### Triplet-Driven Development

**All major decisions require consensus from the LLM Triplet:**

1. **Gemini 2.5 Pro** - Root cause analysis, PostgreSQL expertise
2. **GPT-5** - Architecture patterns, code synthesis
3. **Grok-4** - Alternative perspectives, edge cases

**Process:**
1. Draft requirements
2. Send to triplet via Fiedler (`fiedler_send`)
3. Synthesize responses (consensus or majority)
4. Implement with Blue/Green deployment
5. Test and validate

See `architecture/TRIPLET_CONSULTATION_PROCESS.md` for details.

### Blue/Green Deployment

**All components use Blue/Green for zero-downtime updates:**

```bash
# 1. Build new version (Green)
docker compose build mycomponent-green

# 2. Deploy Green alongside Blue
docker compose up -d mycomponent-green

# 3. Test Green
# Use relay tools or direct testing

# 4. Cutover (switch relay to Green)
relay_reconnect_server("mycomponent")

# 5. Stop Blue after validation
docker compose stop mycomponent-blue
```

See `Code Deployment Cycle.PNG` for full process.

---

## Documentation Structure

### Essential Reading
- **`CURRENT_STATUS.md`** - What we're working on NOW
- **`architecture/CURRENT_ARCHITECTURE_OVERVIEW.md`** - How it's configured
- **GitHub Issues** - Active bugs and tasks (`gh issue list`)

### Reference Documentation
- **`MILESTONES.md`** - Major accomplishments and timeline
- **`architecture/planning_log.md`** - Detailed development history
- **Component READMEs** - Per-component documentation

### Process Documentation
- **`Development Cyle.PNG`** - Triplet-driven development
- **`Code Deployment Cycle.PNG`** - Blue/Green deployment
- **`architecture/TRIPLET_CONSULTATION_PROCESS.md`** - Triplet workflow

---

## GitHub Issues

**View open issues:**
```bash
gh issue list --state open
```

**Current priorities** (as of 2025-10-06):
- Issue #12: Developer onboarding infrastructure
- Issue #13: Component audit and migration to standard libraries

**Closed recently:**
- ✅ Issue #1: Dewey write tools removed (Option 4 compliance)
- ✅ Issue #2: Fiedler conversation logging fixed
- ✅ Issue #3: KGB eliminated from architecture
- ✅ Issue #11: iccm-network library created and documented

---

## Key Principles

### 1. Option 4 Architecture (Write/Read Separation)
- **Godot = WRITE specialist** - ALL database writes
- **Dewey = READ specialist** - Query-only, NO writes
- **Single source of truth** - Godot is the ONLY path to PostgreSQL writes

### 2. Standard Libraries (No Custom Code)
- ❌ Never use raw `websockets.serve()` - Use `iccm-network`
- ❌ Never use `print()` for logs - Use `log_to_godot()`
- ❌ Never bind to `127.0.0.1` - Always `0.0.0.0` (automatic in iccm-network)

### 3. MCP Protocol Everywhere
- All component communication via MCP (Model Context Protocol)
- WebSocket-based (not HTTP REST)
- JSON-RPC 2.0 (initialize, tools/list, tools/call)

### 4. Centralized Logging
- All components log to Godot (port 9060)
- Godot writes to Dewey → PostgreSQL
- Queryable via `dewey_query_logs` tool

### 5. Blue/Green Deployment Only
- Never modify production containers in-place
- Always build Green, test, then cutover
- Keep Blue running until Green validated

---

## Environment

### Host Machine
- **OS:** Ubuntu 24.04 LTS
- **Node.js:** v22.19.0 (via NVM)
- **Docker:** Compose v2
- **Network:** `iccm_network` (Docker bridge)

### Key Paths
- **Projects:** `/mnt/projects/ICCM/`
- **MCP Relay:** `/mnt/projects/ICCM/mcp-relay/`
- **Standard Libraries:** `/mnt/projects/ICCM/iccm-network/`
- **Architecture Docs:** `/mnt/projects/ICCM/architecture/`

---

## Contributing

See **`CONTRIBUTING.md`** for:
- Development standards
- Code review process
- Testing requirements
- PR templates

**Pull Request Requirements:**
- [ ] Uses iccm-network (if Python MCP server)
- [ ] Uses Godot MCP logger (if needs logging)
- [ ] Follows Blue/Green deployment
- [ ] Updates architecture docs
- [ ] Tests pass in Blue deployment

---

## Support

- **GitHub Issues:** https://github.com/rmdevpro/ICCM/issues
- **Documentation:** `/mnt/projects/ICCM/architecture/`
- **Current Status:** `CURRENT_STATUS.md`

---

## License

MIT License - See LICENSE file for details.

---

**Made with 🤖 by the ICCM Project**

*Triplet-driven development: Gemini 2.5 Pro + GPT-5 + Grok-4*
