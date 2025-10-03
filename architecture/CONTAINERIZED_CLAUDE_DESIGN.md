# Containerized Claude Code - Design Document

**Created:** 2025-10-03 01:05 EDT
**Purpose:** Safe isolated environment for debugging Claude Code infrastructure
**Status:** 🔵 DESIGN PHASE

---

## 1. Executive Summary

### The Problem

**Current Dangerous Situation:**
```
Native Claude Code → Stable Relay → KGB → Fiedler/Dewey
        ↑
        └── Using THIS Claude to debug THIS infrastructure
            Breaking it = LOCKED OUT with no recovery tool
```

**Critical Risk:** Any bug in relay/KGB/MCP configuration bricks our only development tool.

### The Solution

**Containerized Test Environment:**
```
┌─────────────────────────────────────────────────────┐
│ HOST SYSTEM (Safe Zone)                             │
│                                                      │
│  ┌────────────────────────────────────┐            │
│  │ Native Claude Code (PRODUCTION)    │            │
│  │  - Used to develop/debug containers │            │
│  │  - Always stays functional          │            │
│  │  - Never touched during experiments │            │
│  └────────────────────────────────────┘            │
│                    ↓ (manages)                       │
│  ┌────────────────────────────────────┐            │
│  │ TEST STACK (Disposable)            │            │
│  │                                     │            │
│  │  Container: claude-code-test       │            │
│  │      ↓                              │            │
│  │  Container: stable-relay-test      │            │
│  │      ↓                              │            │
│  │  Shared: KGB → Fiedler/Dewey       │            │
│  └────────────────────────────────────┘            │
│                                                      │
│ Can destroy and rebuild test stack                  │
│ without affecting production Claude                 │
└─────────────────────────────────────────────────────┘
```

---

## 2. Architecture Design

### 2.1 Component Overview

**Production (Native - Never Touch):**
- Native Claude Code on host
- Connected to host Docker daemon
- No experimental configs
- Used only to build/monitor test stack

**Test Stack (Isolated - Safe to Break):**
- `claude-code-test` container - Isolated Claude Code instance
- `stable-relay-test` container - Test relay (port 8001)
- Shared backend: Existing KGB, Fiedler, Dewey (unchanged)

### 2.2 Network Architecture

```
┌─────────────────────────────────────────────────────────┐
│ HOST                                                     │
│                                                          │
│  Native Claude Code (port: none, local only)            │
│      ↓ Docker socket                                     │
│  Docker Daemon                                           │
│      ↓ Manages                                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │ iccm_network (Bridge)                            │   │
│  │                                                   │   │
│  │  claude-code-test:3000 ←─┐                       │   │
│  │      ↓ ws://relay-test:8001                      │   │
│  │  stable-relay-test:8001   │ (Test Traffic)       │   │
│  │      ↓ ws://kgb-proxy:9000                       │   │
│  │  kgb-proxy:9000           │                       │   │
│  │      ↓                     │                       │   │
│  │  fiedler-mcp:8080         │                       │   │
│  │  dewey-mcp:9020           │                       │   │
│  │                            │                       │   │
│  │  Host Browser:8001 ────────┘ (Access test UI)    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

**Key Isolation:**
- Test Claude Code on port 3000 (container-internal)
- Test Relay on port 8001 (exposed to host for browser access)
- Production Claude Code: No network ports, local only
- Shared backend (KGB, Fiedler, Dewey) unchanged

### 2.3 Port Mapping

| Component | Container Port | Host Port | Purpose |
|-----------|---------------|-----------|---------|
| claude-code-test | 3000 | 8001 | Web UI access from host browser |
| stable-relay-test | 8001 | - | Internal (claude→relay) |
| stable-relay (prod) | 8000 | 8000 | Production (unchanged) |
| kgb-proxy | 9000 | 9000 | Shared (unchanged) |
| fiedler-mcp | 8080 | 9010 | Shared (unchanged) |
| dewey-mcp | 9020 | 9020 | Shared (unchanged) |

---

## 3. Implementation Strategy

### 3.1 Claude Code Containerization

**Option A: Official Claude Code Docker Image (Preferred)**
```dockerfile
# If Anthropic provides official image
FROM anthropic/claude-code:latest

ENV CLAUDE_API_KEY=${CLAUDE_API_KEY}
ENV MCP_CONFIG_PATH=/app/mcp-test.json

COPY mcp-test.json /app/mcp-test.json

EXPOSE 3000

CMD ["claude-code", "--web", "--port", "3000"]
```

**Option B: NPM Package (CONFIRMED - USE THIS)**
```dockerfile
FROM node:22-slim

WORKDIR /app

# Install Claude Code via NPM (official method)
RUN npm install -g @anthropic-ai/claude-code

# Authentication: Prefer Max Account, API key is fallback
# CLAUDE_AUTH_METHOD: "max" (default) or "api"
ENV CLAUDE_AUTH_METHOD=${CLAUDE_AUTH_METHOD:-max}
ENV CLAUDE_API_KEY=${CLAUDE_API_KEY}
ENV MCP_CONFIG_PATH=/app/mcp-test.json

COPY mcp-test.json /app/mcp-test.json

EXPOSE 3000

# Run Claude Code in web mode on port 3000
# Will prompt for Max login if CLAUDE_AUTH_METHOD=max
# Will use API key if CLAUDE_AUTH_METHOD=api
CMD ["claude-code", "--web", "--port", "3000"]
```

**Option C: Use Desktop Claude Code via VNC (Most Realistic)**
```dockerfile
FROM ubuntu:24.04

# Install Xvfb, VNC, noVNC for browser access
RUN apt-get update && apt-get install -y \
    xvfb x11vnc novnc websockify \
    wget curl

# Download and install Claude Code
# (Exact method depends on distribution format)

# Setup VNC server
ENV DISPLAY=:99
EXPOSE 6080

CMD ["start-vnc-and-claude.sh"]
```

### 3.2 Test Relay Configuration

**File:** `/mnt/projects/ICCM/claude-code-test/relay-test/config.yaml`
```yaml
backend: "ws://kgb-proxy:9000"
port: 8001
```

**Dockerfile:**
```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY relay.py .
COPY config.yaml .

EXPOSE 8001

CMD ["python", "relay.py"]
```

### 3.3 Test MCP Configuration

**File:** `/mnt/projects/ICCM/claude-code-test/mcp-test.json`
```json
{
  "mcpServers": {
    "fiedler": {
      "transport": {
        "type": "ws",
        "url": "ws://stable-relay-test:8001?upstream=fiedler"
      }
    },
    "dewey": {
      "transport": {
        "type": "ws",
        "url": "ws://stable-relay-test:8001?upstream=dewey"
      }
    }
  }
}
```

**Note:** Points to test relay (8001), not production (8000)

### 3.4 Docker Compose Stack

**File:** `/mnt/projects/ICCM/claude-code-test/docker-compose.yml`
```yaml
version: '3.8'

services:
  stable-relay-test:
    build:
      context: ./relay-test
      dockerfile: Dockerfile
    image: stable-relay-test:latest
    container_name: stable-relay-test
    restart: unless-stopped

    environment:
      - RELAY_BACKEND=ws://kgb-proxy:9000
      - RELAY_PORT=8001

    networks:
      - iccm_network

    healthcheck:
      test: ["CMD", "python", "-c", "import sys; sys.exit(0)"]
      interval: 30s
      timeout: 10s
      retries: 3

  claude-code-test:
    build:
      context: ./claude-test
      dockerfile: Dockerfile
    image: claude-code-test:latest
    container_name: claude-code-test
    restart: unless-stopped

    environment:
      # Auth: "max" for Max Account (default), "api" for API key
      - CLAUDE_AUTH_METHOD=${CLAUDE_AUTH_METHOD:-max}
      - CLAUDE_API_KEY=${CLAUDE_API_KEY}
      - MCP_CONFIG_PATH=/app/mcp-test.json

    ports:
      - "8001:3000"  # Host:8001 → Container:3000

    networks:
      - iccm_network

    depends_on:
      - stable-relay-test

    volumes:
      - claude-test-data:/app/data
      # Mount keys.txt as read-only (for API key fallback)
      - /mnt/projects/keys.txt:/app/keys.txt:ro

networks:
  iccm_network:
    external: true

volumes:
  claude-test-data:
```

---

## 4. Usage Workflow

### 4.1 Setup (One-Time)

```bash
# 1. Create project directory
mkdir -p /mnt/projects/ICCM/claude-code-test/{claude-test,relay-test}

# 2. Create config files
# (See section 3.3 for content)

# 3. Build containers
cd /mnt/projects/ICCM/claude-code-test
docker compose build

# 4. Start test stack
docker compose up -d
```

### 4.2 Access Test Claude Code

**Via Browser:**
```
http://localhost:8001
```

**Check it's using test relay:**
```bash
# Test Claude logs should show connection to stable-relay-test:8001
docker logs claude-code-test --tail 20

# Test relay logs should show connection from Claude
docker logs stable-relay-test --tail 20
```

### 4.3 Debugging Workflow

**Step 1: Use Native Claude Code to modify test stack**
```bash
# Use THIS session (native Claude) to:
# - Edit test relay config
# - Modify test MCP config
# - Change test Claude settings
```

**Step 2: Rebuild test stack**
```bash
cd /mnt/projects/ICCM/claude-code-test
docker compose build
docker compose up -d
```

**Step 3: Test in isolated environment**
```
# Access test Claude at http://localhost:8001
# Try the risky operations
# If it breaks → no problem, native Claude still works
```

**Step 4: Verify with logs**
```bash
docker logs claude-code-test --tail 50
docker logs stable-relay-test --tail 50
docker logs kgb-proxy --tail 50  # Shared, see test traffic
```

**Step 5: If broken → destroy and rebuild**
```bash
docker compose down
# Fix issues using native Claude
docker compose up -d
# Try again
```

### 4.4 Testing Scenarios

**Scenario 1: Test KGB Logging**
1. Access test Claude at localhost:8001
2. Use Fiedler MCP tools
3. Check KGB logs for spy creation
4. Check Winni for conversation
5. If broken → fix in isolation, native Claude unaffected

**Scenario 2: Test Relay Restart Survival**
1. Use test Claude to call Fiedler
2. Restart test relay: `docker restart stable-relay-test`
3. Verify test Claude reconnects
4. Native Claude never affected

**Scenario 3: Test Breaking Changes**
1. Modify test relay to intentionally break
2. Restart test stack
3. Observe failure in test Claude
4. Use native Claude to diagnose and fix
5. Deploy fix when verified

---

## 5. Safety Guarantees

### 5.1 Isolation Checklist

- ✅ **Separate Port Space**: Test (8001) vs Prod (8000)
- ✅ **Separate Containers**: Test stack completely isolated
- ✅ **Separate Configs**: mcp-test.json vs mcp.json
- ✅ **Separate Relay**: stable-relay-test vs stable-relay
- ✅ **Shared Backend Only**: KGB/Fiedler/Dewey unchanged (safe to share)
- ✅ **Native Claude Untouched**: No changes to host Claude Code

### 5.2 Failure Modes & Recovery

| Failure | Impact | Recovery |
|---------|--------|----------|
| Test Claude crashes | SAFE - Test stack only | `docker restart claude-code-test` |
| Test Relay crashes | SAFE - Test stack only | `docker restart stable-relay-test` |
| Bad MCP config in test | SAFE - Test stack only | Edit mcp-test.json, restart |
| KGB crashes | BOTH AFFECTED | Fix KGB (shared component) |
| Native Claude issue | PRODUCTION ONLY | Use test Claude to diagnose (reverse roles!) |
| Complete test stack failure | SAFE - Rebuild | `docker compose down && docker compose up -d` |

### 5.3 Verification Steps

**Verify Isolation Working:**
```bash
# 1. Native Claude should NOT see test relay
netstat -tuln | grep 8001  # Should show Docker, not host process

# 2. Test Claude should ONLY connect to test relay
docker logs claude-code-test | grep -i "connect"
# Should show: ws://stable-relay-test:8001

# 3. Native Claude should ONLY connect to production relay
# Check ~/.config/claude-code/mcp.json
# Should show: ws://localhost:8000
```

---

## 6. Implementation Phases

### Phase 1: Research & Prep (Current)
- ✅ Design architecture
- [ ] Research Claude Code containerization methods
- [ ] Determine if official Docker image exists
- [ ] Test Claude Code installation methods

### Phase 2: Build Test Relay
- [ ] Copy stable-relay code to test directory
- [ ] Modify to use port 8001
- [ ] Build and test relay container standalone
- [ ] Verify connectivity to KGB

### Phase 3: Build Test Claude Code
- [ ] Create Claude Code container (method TBD)
- [ ] Configure with test MCP config
- [ ] Expose on port 8001 (mapped from 3000)
- [ ] Test basic functionality

### Phase 4: Integration Testing
- [ ] Start full test stack
- [ ] Verify Claude→Relay→KGB→Fiedler chain
- [ ] Check KGB spy creation
- [ ] Verify Winni logging
- [ ] Confirm isolation from native Claude

### Phase 5: Debugging Use Cases
- [ ] Use test environment for BUG #1 investigation
- [ ] Test risky configuration changes
- [ ] Verify fixes before deploying to production
- [ ] Document lessons learned

---

## 7. Open Questions & Decisions Needed

### Q1: Claude Code Distribution Method ✅ RESOLVED
**Question:** How is Claude Code distributed? NPM package? Binary? AppImage?

**Answer:** NPM package - `npm install -g @anthropic-ai/claude-code`

**Decision:** Use Option B - NPM package in Node.js container (simple, official, lightweight)

### Q2: Authentication Method ✅ RESOLVED
**Question:** How should containerized Claude Code authenticate?

**Claude Code supports TWO authentication methods:**

**Method 1: Max Account (PRIMARY - Use This)**
- Free tier / subscription-based
- Login via browser/OAuth
- No per-request costs
- **USE FOR:** Regular development, testing, most work

**Method 2: API Key (SECONDARY - Use Sparingly)**
- Pay-per-use (costs extra)
- Direct API access
- Stored in `/mnt/projects/keys.txt` (line 14)
- **USE FOR:** Automated tasks, when Max login not feasible

**Decision for Test Container:**
- **Default:** Max Account login (primary method)
- **Fallback:** API key from environment variable (only if Max login fails)
- **Security:** Mount keys.txt as read-only volume, use env var to select auth method

### Q3: Data Persistence
**Question:** Should test Claude Code persist data?

**Options:**
- A: Ephemeral (destroyed on restart) - cleaner testing
- B: Persistent volume - preserves conversation history

**Decision:** Start ephemeral, add persistence if needed

### Q4: Shared Backend Safety
**Question:** Is it safe to share KGB/Fiedler/Dewey with test traffic?

**Analysis:**
- KGB logs all traffic to Winni (test + prod mixed) - ACCEPTABLE
- Fiedler stateless (just calls LLMs) - SAFE
- Dewey stores in same Winni DB - ACCEPTABLE (can filter by session_id)

**Decision:** SAFE to share, conversations distinguishable by session ID

---

## 8. Success Criteria

**Test Stack is Ready When:**
1. ✅ Test Claude Code accessible at http://localhost:8001
2. ✅ Test Claude connects to stable-relay-test:8001 (not prod :8000)
3. ✅ Test relay connects to KGB:9000
4. ✅ KGB creates spy for test Claude (separate from native)
5. ✅ Test Claude can use Fiedler/Dewey MCP tools
6. ✅ Test traffic logged to Winni (distinguishable from prod)
7. ✅ Native Claude Code completely unaffected
8. ✅ Can destroy/rebuild test stack without impact
9. ✅ Can intentionally break test stack safely

---

## 9. Next Steps

### Immediate Actions
1. **Research:** Investigate Claude Code containerization
   - Check Anthropic docs for official Docker image
   - Test NPM installation method
   - Evaluate desktop app + VNC approach

2. **Prototype:** Build minimal test relay
   - Copy stable-relay to test directory
   - Modify port to 8001
   - Test standalone

3. **Design Decision:** Choose Claude Code container method
   - Based on research findings
   - Document pros/cons
   - Create Dockerfile

### After Container Method Decided
1. Build test stack
2. Verify isolation
3. Use for BUG #1 debugging
4. Deploy fixes to production

---

## 10. Documentation References

**This Design:**
- `/mnt/projects/ICCM/architecture/CONTAINERIZED_CLAUDE_DESIGN.md` (this file)

**Implementation (TBD):**
- `/mnt/projects/ICCM/claude-code-test/README.md` - Setup guide
- `/mnt/projects/ICCM/claude-code-test/docker-compose.yml` - Stack definition
- `/mnt/projects/ICCM/claude-code-test/TESTING_GUIDE.md` - How to use test env

**Related:**
- `/mnt/projects/ICCM/architecture/CURRENT_STATUS.md` - Overall project status
- `/mnt/projects/ICCM/architecture/BUG_TRACKING.md` - What we're debugging
- `/mnt/projects/ICCM/architecture/STABLE_RELAY_DEPLOYMENT.md` - Production relay

---

**STATUS:** Design complete, ready for triplet review via Fiedler (Gemini 2.5 Pro, GPT-4o-mini, DeepSeek-R1).

---

## 11. Pre-Implementation Review

**Before building, this design will be reviewed by:**
1. **Gemini 2.5 Pro** (via Fiedler)
2. **GPT-4o-mini** (via Fiedler)
3. **DeepSeek-R1** (via Fiedler)

**Review Focus:**
- Architecture soundness
- Security/isolation verification
- Implementation risks
- Missing considerations
- Authentication strategy

**After triplet approval:** Proceed to implementation.
