# Anthropic API Gateway Implementation

**Date:** 2025-10-03
**Status:** Ready for testing
**Architecture Decision:** GPT-5's recommended approach (reverse proxy)

---

## Problem Statement

**Gap Identified:** Direct Claude Code ↔ Anthropic API conversations bypass all logging
- MCP tool calls are logged via KGB proxy ✅
- Anthropic API requests go directly to api.anthropic.com ❌
- **Result:** Incomplete conversation history

**Goal:** Capture 100% of Claude Code activity for audit/analysis

---

## Solution: KGB HTTP Gateway (Integrated)

### Triplet Consultation Results

We consulted Gemini 2.5 Pro, GPT-5, and Grok-4 on five approaches:

| Approach | Gemini | GPT-5 | Grok-4 | Decision |
|----------|--------|-------|--------|----------|
| **HTTPS Proxy (mitmproxy)** | ✅ Primary | Fallback | ❌ | Not chosen |
| **Anthropic Gateway (reverse proxy)** | - | ✅ Primary | ❌ | **SELECTED** |
| **API Wrapper (modify Claude Code)** | ❌ | ❌ | ✅ Primary | Not chosen |
| **Claude Code Hook** | N/A | N/A | N/A | Not available |
| **Network Monitor** | ❌ | ❌ | ❌ | Not viable |

**Consensus:** Gateway/Proxy approaches are best
- Gemini + GPT-5 converged on proxy-based solutions
- Grok-4's API wrapper creates maintenance burden

**Key Discovery:** Claude Code supports `ANTHROPIC_BASE_URL` environment variable
- Verified in source: `baseURL:A=i31("ANTHROPIC_BASE_URL")`
- This enables GPT-5's primary recommendation: **Anthropic Gateway**

### Why Anthropic Gateway Won

✅ **No TLS/SSL issues** - No certificate trust required
✅ **Simple HTTP reverse proxy** - Standard web architecture
✅ **Works everywhere** - Identical on bare metal and containers
✅ **Minimal code changes** - Just set environment variable
✅ **Secure by design** - API keys redacted before logging

❌ **Why not mitmproxy?** Requires CA certificate installation (complex, error-prone)
❌ **Why not API wrapper?** Creates Claude Code fork (maintenance burden)

---

## Architecture

### Data Flow

```
Claude Code Container
    ↓ ANTHROPIC_BASE_URL=http://host.docker.internal:8089/v1
KGB HTTP Gateway (Python aiohttp reverse proxy)
    ↓ Forward request to https://api.anthropic.com
    ├→ Capture request (sanitize API keys)
    ├→ Capture response
    ├→ Log to Dewey asynchronously
    └→ Return response to Claude Code

Dewey → Winni PostgreSQL (unified storage)
```

### Components

**1. Claude Code Container** (`/mnt/projects/ICCM/claude-container/`)
- Dockerfile: Node.js 22 + Claude Code CLI
- docker-compose.yml: Volume mounts, environment config
- Environment: `ANTHROPIC_BASE_URL=http://host.docker.internal:8089/v1`

**2. KGB (Enhanced)** (`/mnt/projects/ICCM/kgb/`)
- **WebSocket Spy** (port 9000): MCP tool traffic logging
- **HTTP Gateway** (port 8089): Anthropic API traffic logging
- Both use DeweyClient for unified logging
- Sanitizes headers (redacts x-api-key, authorization, cookie)
- Health checks on `/health`

**3. Integration Points**
- **KGB Unified**: Single proxy for both WebSocket and HTTP traffic
  - Port 9000: WebSocket spy for MCP traffic
  - Port 8089: HTTP gateway for Anthropic API traffic
  - Universal gateway: ANY component can route Anthropic calls through KGB
- **Dewey:** Unified logging endpoint for all traffic types
- **Winni:** Single database for all conversations
- **Future: Fiedler + Anthropic**: When Fiedler adds Anthropic provider support, it will also route through KGB HTTP gateway for unified logging

---

## Implementation Files

### Created/Modified Files

```
/mnt/projects/ICCM/
├── claude-container/
│   ├── Dockerfile
│   └── docker-compose.yml (updated to use KGB)
│
├── kgb/
│   ├── kgb/http_gateway.py (NEW - HTTP reverse proxy)
│   ├── kgb/proxy_server.py (UPDATED - dual protocol)
│   ├── Dockerfile (UPDATED - expose port 8089)
│   ├── docker-compose.yml (UPDATED - expose port 8089)
│   ├── requirements.txt (UPDATED - add aiohttp)
│   └── README.md (UPDATED - document HTTP gateway)
│
└── architecture/triplet_consultations/
    ├── anthropic_api_logging_gemini.md
    ├── anthropic_api_logging_gpt5.md
    └── anthropic_api_logging_grok4.md
```

### Key Code: KGB HTTP Gateway

```python
# kgb/http_gateway.py - aiohttp reverse proxy
async def proxy_request(self, request: web.Request):
    """Proxy request to upstream and log to Dewey."""
    # Create conversation in Dewey
    dewey_client = DeweyClient()
    conv = await dewey_client.begin_conversation(...)

    # Log request (sanitized)
    await dewey_client.store_message(
        conversation_id=conv_id,
        role="user",
        content=sanitize_headers(request.headers)
    )

    # Forward to Anthropic API
    async with aiohttp.ClientSession() as session:
        response = await session.request(...)

    # Log response (sanitized)
    await dewey_client.store_message(
        conversation_id=conv_id,
        role="assistant",
        content=response.body
    )

    return response  # Return to Claude Code
```

---

## Configuration

### Environment Variables

**Claude Code Container:**
```bash
ANTHROPIC_BASE_URL=http://anthropic-gateway:8089/v1
ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}  # From host
```

**Anthropic Gateway:**
```bash
PORT=8089
UPSTREAM_URL=https://api.anthropic.com
DEWEY_URL=http://192.168.1.210:8080/conversations
REDACT_API_KEYS=true
```

### Docker Compose Network

```yaml
networks:
  iccm-network:
    driver: bridge

services:
  anthropic-gateway:
    networks: [iccm-network]
    ports: ["8089:8089"]

  claude-code:
    networks: [iccm-network]
    depends_on: [anthropic-gateway]
    environment:
      - ANTHROPIC_BASE_URL=http://anthropic-gateway:8089/v1
```

---

## Security

### API Key Protection
- ✅ Redacted before logging (x-api-key, authorization, cookie)
- ✅ Never persisted to disk
- ✅ Only forwarded to upstream Anthropic API
- ✅ Sanitization function: `sanitizeHeaders()`

### Network Isolation
- ✅ Containers on isolated `iccm-network` bridge
- ✅ Gateway not exposed to public internet
- ✅ Dewey accessible only from ICCM network

### Failure Handling
- ✅ Async logging (failures don't block API requests)
- ✅ Error logging to console (debugging)
- ✅ Health checks for monitoring

---

## Testing Plan

### Phase 1: Build Infrastructure
```bash
cd /mnt/projects/ICCM/anthropic-gateway
npm install

cd /mnt/projects/ICCM/claude-container
docker-compose build
```

### Phase 2: Start Services
```bash
docker-compose up -d
docker-compose logs -f
```

### Phase 3: Verify Gateway
```bash
# Health check
curl http://localhost:8089/health

# Test proxy (with API key)
curl http://localhost:8089/v1/messages \
  -H "x-api-key: $ANTHROPIC_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"model":"claude-sonnet-4","max_tokens":100,"messages":[{"role":"user","content":"Hello"}]}'
```

### Phase 4: Test Containerized Claude
```bash
docker exec -it claude-code-container claude
# Should connect through gateway
# Verify logs in Dewey/Winni
```

### Phase 5: Verify Logging
```bash
# Check Dewey received logs
curl http://192.168.1.210:8080/conversations

# Query Winni database
psql -h 192.168.1.210 -U dewey -d winni -c "SELECT * FROM conversations ORDER BY created_at DESC LIMIT 5;"
```

---

## Rollback Plan

**If issues occur:**

1. Stop containers:
   ```bash
   docker-compose down
   ```

2. Bare metal Claude Code continues working (unchanged):
   ```bash
   claude  # Works normally
   ```

3. Debug container logs:
   ```bash
   docker logs anthropic-gateway
   docker logs claude-code-container
   ```

4. Fall back to mitmproxy approach if gateway fails (documented in triplet consultations)

---

## Next Steps

1. ✅ Documentation complete
2. ⏳ Commit changes
3. ⏳ Build Docker images
4. ⏳ Test gateway standalone
5. ⏳ Test containerized Claude Code
6. ⏳ Verify logging to Dewey/Winni
7. ⏳ Evaluate making containerized mode default

---

## References

- Triplet consultations: `/mnt/projects/ICCM/architecture/triplet_consultations/`
- Gateway README: `/mnt/projects/ICCM/anthropic-gateway/README.md`
- Architecture overview: `/mnt/projects/ICCM/architecture/CURRENT_ARCHITECTURE_OVERVIEW.md`
- Current status: `/mnt/projects/ICCM/CURRENT_STATUS.md`
