# Claude Code MCP WebSocket Connection Failure - Complete Analysis with Configs

## Problem Statement

Claude Code cannot connect to Fiedler MCP server via WebSocket despite correct configuration and verified network accessibility. All 7 URL variation attempts have failed.

## Configuration Files

### File 1: ~/.config/claude-code/mcp.json (Global Config)
```json
{
  "mcpServers": {
    "fiedler": {
      "transport": {
        "type": "ws",
        "url": "ws://192.168.1.200:9010"
      }
    },
    "desktop-commander": {
      "command": "npx",
      "args": [
        "-y",
        "@executeautomation/desktop-commander-mcp"
      ]
    }
  }
}
```

### File 2: ~/.claude.json (Project Config - Lines 514-519)
```json
"fiedler": {
  "transport": {
    "type": "ws",
    "url": "ws://192.168.1.200:9010"
  }
}
```

**NOTE:** This is within a larger config structure at `projects["/home/aristotle9"].mcpServers.fiedler`

### Other MCP Servers in ~/.claude.json (All using stdio):
- sequential-thinking (npx)
- desktop-commander (npx) 
- gcloud (npx)
- observability (npx)
- google-cloud-mcp (npx)
- herodotus (Python venv)
- grok (npx with env var)

**OBSERVATION:** Fiedler is the ONLY WebSocket transport. All other servers use stdio/command transport and work fine.

## System Architecture

**Target Setup:**
```
Claude Code (bare metal)
    ↓ MCP WebSocket
Fiedler MCP (Docker container, port 9010)
    ↓ 7 LLM providers
```

**Container:** fiedler-mcp
**Port Mapping:** 0.0.0.0:9010 -> container:8080
**Expected Tools:** 5+ MCP tools (fiedler_list_models, fiedler_send, etc.)
**Actual Result:** "No such tool available" - tools never load

## Configuration Attempts (All Failed)

### Attempt #1: Wrong Format
```json
"fiedler": {
  "url": "ws://localhost:9010"
}
```
**Result:** ❌ FAILED - Missing transport wrapper

### Attempt #2: Correct Format, localhost
```json
"fiedler": {
  "transport": {
    "type": "ws",
    "url": "ws://localhost:9010"
  }
}
```
**Result:** ❌ FAILED - Tools not loading

### Attempt #3: Attempted stdio (User corrected as wrong)
**Result:** ❌ ABANDONED - WebSocket is correct per architecture docs

### Attempt #4: localhost retry after triplet consultation
```json
"url": "ws://localhost:9010"
```
**Result:** ❌ FAILED

### Attempt #5: Explicit IPv4 loopback
```json
"url": "ws://127.0.0.1:9010"
```
**Result:** ❌ FAILED

### Attempt #6: Docker host DNS
```json
"url": "ws://host.docker.internal:9010"
```
**Result:** ❌ FAILED

### Attempt #7: Host LAN IP (CURRENT)
```json
"url": "ws://192.168.1.200:9010"
```
**Result:** ❌ FAILED

## Deep Diagnostics Results

### Port Binding ✅
```
docker ps --filter "name=fiedler" --format "{{.Ports}}"
Result: 0.0.0.0:9010->8080/tcp, [::]:9010->8080/tcp
Status: Port correctly bound to all interfaces
```

### Firewall ✅
```
sudo ufw status
Result: Status: inactive
Status: No firewall blocking
```

### Network Accessibility ✅
```
sudo lsof -i :9010
Result: docker-proxy listening on *:9010
Status: Port accessible on network
```

### WebSocket Test ✅
```
wscat -c ws://192.168.1.200:9010
Result: Connection successful (no errors)
Status: WebSocket endpoint accessible from host
```

### Claude Code Logs ❌
```
find ~/.cache -name "*claude*"
Result: No MCP log files found
Status: Cannot determine Claude Code MCP client error
```

## Evidence Summary

**What Works:**
- ✅ Fiedler container is healthy
- ✅ Port 9010 is listening and accessible
- ✅ WebSocket handshake succeeds (wscat test)
- ✅ Firewall is not blocking
- ✅ Config format is correct per MCP specs
- ✅ All other MCP servers (stdio-based) work fine

**What Doesn't Work:**
- ❌ Claude Code MCP client never connects to Fiedler WebSocket
- ❌ No Fiedler MCP tools appear in Claude Code
- ❌ No connection attempts visible in Fiedler logs
- ❌ No MCP error logs from Claude Code found

## Key Observations

1. **External tools can connect:** wscat successfully connects to WebSocket
2. **Config format is correct:** Matches MCP WebSocket specification exactly
3. **All URL variations failed:** localhost, 127.0.0.1, host.docker.internal, LAN IP
4. **No log evidence:** Claude Code produces no visible MCP connection logs
5. **Complete restart verified:** Full process quit and relaunch performed each time
6. **Only WebSocket fails:** All stdio MCP servers (7 others) work perfectly
7. **Two config files:** Both ~/.config/claude-code/mcp.json AND ~/.claude.json exist
8. **Mixed transports:** stdio works, WebSocket doesn't

## Alternative Architecture Available

The system has a **working Stable Relay architecture** already deployed:
```
Claude Code → Stable Relay (port 8000) → KGB (port 9000) → Fiedler (port 8080)
```

This was tested and verified working internally, but Claude Code has never been configured to use it (still trying direct WebSocket to Fiedler).

## Critical Questions for LLM Triplet

1. **Why would Claude Code's MCP WebSocket client fail when wscat succeeds?**
   - Same endpoint, same protocol, different behavior
   - All stdio MCP servers work fine

2. **Is there a known incompatibility between Claude Code and WebSocket MCP transport?**
   - Should we try stdio transport instead despite container architecture?
   - Why do 7 stdio servers work but 1 WebSocket doesn't?

3. **Two config files - could this be the problem?**
   - Both ~/.config/claude-code/mcp.json AND ~/.claude.json contain Fiedler config
   - Could they be conflicting?
   - Which one takes precedence?

4. **Should we abandon direct WebSocket and switch to Stable Relay architecture?**
   - Already deployed, tested, and working
   - Config would be: `ws://localhost:8000?upstream=fiedler`

5. **What diagnostic information would help identify the Claude Code MCP client issue?**
   - No logs available, no error messages visible
   - How to enable verbose MCP logging in Claude Code?

6. **Is there an undocumented Claude Code MCP WebSocket requirement we're missing?**
   - Format matches spec, but never connects
   - Special headers? Authentication? Subprotocols?

7. **Could the presence of multiple MCP servers cause issues?**
   - 7 stdio servers + 1 WebSocket server
   - Does Claude Code have a limit or compatibility issue mixing transports?

Please provide your analysis and recommendations focusing on:
- Why WebSocket fails when stdio works
- The two config file situation
- Whether to switch to Stable Relay
- How to debug Claude Code's MCP WebSocket client
