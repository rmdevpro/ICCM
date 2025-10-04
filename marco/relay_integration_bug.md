# Marco Relay Integration Bug

## Context
Marco Internet Gateway deployed successfully, but tools are not exposed through MCP Relay despite successful connection.

## Issue
**Problem:** Marco added to relay with `relay_add_server('marco', 'ws://localhost:9030')` but tools are not accessible

**Evidence:**
1. Marco container running and healthy
   - WebSocket handshake successful: `HTTP/1.1 101 Switching Protocols`
   - Container on port 9030, uptime 1700+ seconds

2. Relay connection established
   - Socket fd 11 created at 14:42 (when Marco added)
   - Relay process PID 2461107 has active connection to Marco

3. Tools not exposed
   - `tools/list` via relay shows 24 tools total
   - Zero Marco/Playwright tools in list
   - Expected: ~7 Playwright tools + marco_reset_browser

4. Health check shows "degraded"
   - `playwright_subprocess: "unresponsive"`
   - But this is expected - subprocess only outputs when processing requests
   - Not the root cause

## What Was Tried

**Attempt 1:** Check relay status via `relay_get_status`
- Result: Tool returned nothing (hook blocked or failed)

**Attempt 2:** Direct WebSocket test
- Result: ✅ Handshake successful, Marco WebSocket working

**Attempt 3:** Query relay for tools via subprocess
- Result: 24 tools total, 0 Marco tools

**Attempt 4:** Check relay process state
- Result: Connection exists (socket fd 11) but tools not propagated

## Question for Triplet

**Why would MCP Relay show an active connection to Marco but not expose its tools?**

Possibilities:
1. Marco not implementing required MCP protocol methods (initialize, tools/list)?
2. Relay not querying Marco for tools after connection?
3. Relay's `notifications/tools/list_changed` not sent to Claude?
4. Marco's tools/list response format incorrect?
5. Something else?

**Marco implementation:**
- WebSocket server on port 8030 (internal) / 9030 (external)
- Spawns Playwright MCP subprocess: `npx @playwright/mcp@0.0.41`
- Bridges WebSocket ↔ stdio for Playwright MCP
- Should expose Playwright's MCP tools

**Expected behavior:**
1. Relay connects to Marco WebSocket ✅
2. Relay sends `initialize` to Marco
3. Marco forwards to Playwright subprocess via stdio
4. Playwright responds with capabilities
5. Relay sends `tools/list` to Marco
6. Marco returns Playwright tools
7. Relay sends `notifications/tools/list_changed` to Claude
8. Claude re-queries and gets Marco tools

**Diagnostic request:**
Please analyze why the tool discovery chain is breaking and recommend fix.

## Files
- `/mnt/projects/ICCM/marco/server.js` - Marco WebSocket MCP server
- `/mnt/projects/ICCM/mcp-relay/mcp_relay.py` - MCP Relay implementation
- `/mnt/projects/ICCM/marco/REQUIREMENTS.md` - Marco specifications
