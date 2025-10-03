# Claude Code MCP WebSocket Connection Failure - 7 Attempts Exhausted

## Problem Statement

Claude Code cannot connect to Fiedler MCP server via WebSocket despite correct configuration and verified network accessibility. All 7 URL variation attempts have failed.

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

### Attempt #7: Host LAN IP
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

## Configuration Files

### Both files updated identically:
1. `~/.config/claude-code/mcp.json`
2. `~/.claude.json`

Both contain:
```json
{
  "mcpServers": {
    "fiedler": {
      "transport": {
        "type": "ws",
        "url": "ws://192.168.1.200:9010"
      }
    }
  }
}
```

## Evidence Summary

**What Works:**
- ✅ Fiedler container is healthy
- ✅ Port 9010 is listening and accessible
- ✅ WebSocket handshake succeeds (wscat test)
- ✅ Firewall is not blocking
- ✅ Config format is correct per MCP specs

**What Doesn't Work:**
- ❌ Claude Code MCP client never connects
- ❌ No Fiedler MCP tools appear in Claude Code
- ❌ No connection attempts visible in Fiedler logs
- ❌ No MCP error logs from Claude Code found

## Key Observations

1. **External tools can connect:** wscat successfully connects to WebSocket
2. **Config format is correct:** Matches MCP WebSocket specification exactly
3. **All URL variations failed:** localhost, 127.0.0.1, host.docker.internal, LAN IP
4. **No log evidence:** Claude Code produces no visible MCP connection logs
5. **Complete restart verified:** Full process quit and relaunch performed each time

## Alternative Architecture Available

The system has a **working Stable Relay architecture** already deployed:
```
Claude Code → Stable Relay (port 8000) → KGB (port 9000) → Fiedler (port 8080)
```

This was tested and verified working internally, but Claude Code has never been configured to use it (still trying direct WebSocket to Fiedler).

## Questions for LLM Triplet

1. **Why would Claude Code's MCP WebSocket client fail when wscat succeeds?**
   - Same endpoint, same protocol, different behavior

2. **Is there a known incompatibility between Claude Code and WebSocket MCP transport?**
   - Should we try stdio transport instead despite container architecture?

3. **Should we abandon direct WebSocket and switch to Stable Relay architecture?**
   - Already deployed, tested, and working
   - Config would be: `ws://localhost:8000?upstream=fiedler`

4. **What diagnostic information would help identify the Claude Code MCP client issue?**
   - No logs available, no error messages visible

5. **Is there an undocumented Claude Code MCP WebSocket requirement we're missing?**
   - Format matches spec, but never connects

Please provide your analysis and recommendations.
