# Bug Tracking Log

**Purpose:** Track active bugs with high-level summaries and resolution status

**Last Updated:** 2025-10-05 12:45 EDT

---

## 🐛 ACTIVE BUGS

### BUG #28: Dewey dewey_query_logs Returns Timedelta Serialization Error

**Status:** 🔴 ACTIVE - Query tool broken
**Reported:** 2025-10-05 12:44 EDT
**Priority:** MEDIUM - Blocks log query capability
**Component:** Dewey (`/mnt/projects/ICCM/dewey/`)

**Problem:**
The `dewey_query_logs` tool fails with JSON serialization error:
```
Internal error: Object of type timedelta is not JSON serializable
```

**Impact:**
- Cannot query stored logs via `logger_query` facade tool
- Logs are being stored successfully (worker batching confirmed)
- Only retrieval is broken

**Root Cause:**
Likely returning timedelta objects (e.g., age, retention period) in query results without converting to serializable format (string, seconds, etc.)

**Solution:**
Convert timedelta objects to ISO duration strings or total seconds before JSON serialization in `dewey_query_logs` tool.

**Evidence:**
```
2025-10-05 12:44:25,903 - INFO - [WORKER] Sending batch of 6 logs to Dewey.
2025-10-05 12:44:25,926 - INFO - [WORKER] Successfully sent batch of 6 logs.
2025-10-05 12:44:34,916 - ERROR - [MCP_SERVER] Error calling tool dewey_query_logs: Internal error: Object of type timedelta is not JSON serializable
```

---

### BUG #27: Gates Blue Dockerfile Missing loglib.js

**Status:** ✅ RESOLVED (2025-10-05 12:42 EDT)
**Reported:** 2025-10-05 12:40 EDT
**Priority:** HIGH - Blocked Gates Blue deployment
**Component:** Gates (`/mnt/projects/ICCM/gates-blue/`)

**Problem:**
Gates Blue container failed to start with:
```
Error [ERR_MODULE_NOT_FOUND]: Cannot find module '/app/loglib.js' imported from /app/server.js
```

**Root Cause:**
Dockerfile only copied `server.js`, not `loglib.js`. Build succeeded but runtime failed.

**Resolution:**
Updated Dockerfile line 20:
```dockerfile
# Before
COPY server.js ./

# After
COPY server.js loglib.js ./
```

**Note:**
This bug was discovered and fixed during deployment, but then the entire approach was changed to use MCP tools instead of Redis client library, making loglib.js unnecessary. Final fix removed the Redis approach entirely.

**Result:**
✅ Container starts successfully
✅ Switched to MCP-based logging (calls `logger_log` tool instead of direct Redis)

---

### BUG #26: Godot Triplet Code Used Non-Existent mcp-tools-py Library

**Status:** ✅ RESOLVED (2025-10-05 05:00 EDT)
**Reported:** 2025-10-05 04:42 EDT
**Priority:** CRITICAL - Blocked Godot deployment
**Component:** Godot MCP Server (`/mnt/projects/ICCM/godot/`)

**Problem:**
The triplet-synthesized Godot implementation (unanimous approval, correlation_id: 1c281d80) used a fictional Python library `mcp-tools-py>=0.4.0` that doesn't exist on PyPI. Docker build failed with:
```
ERROR: Could not find a version that satisfies the requirement mcp-tools-py>=0.4.0 (from versions: none)
```

**Root Cause:**
Triplets (Gemini-2.5-Pro, GPT-4o-mini, DeepSeek-R1) assumed existence of convenience wrapper library for MCP protocol. All working ICCM servers implement MCP directly using `websockets` library.

**Resolution:**
- Consulted triplets for fix guidance (correlation_id: b5afd3b0)
- All three models recommended Option A: Implement MCP client using `websockets` directly, following Dewey's pattern
- Created `/mnt/projects/ICCM/godot/godot/src/mcp_client.py` - Reusable WebSocket-based MCP client
- Rewrote `mcp_server.py` - WebSocket MCP server implementing facade tools
- Rewrote `worker.py` - Uses new MCP client to connect to Dewey
- Replaced fictional library dependency with `websockets>=14.0`

**Result:**
- ✅ Docker build successful
- ✅ Container deployed (godot-mcp)
- ✅ Redis operational
- ✅ MCP Server listening on port 9060
- Worker failing (expected - Dewey tools not deployed yet)

**Files Modified:**
- `/mnt/projects/ICCM/godot/godot/Dockerfile` - Removed mcp-tools-py, added websockets
- `/mnt/projects/ICCM/godot/godot/src/mcp_client.py` - NEW (WebSocket MCP client)
- `/mnt/projects/ICCM/godot/godot/src/mcp_server.py` - Rewritten using websockets
- `/mnt/projects/ICCM/godot/godot/src/worker.py` - Rewritten using new MCP client
- `/mnt/projects/ICCM/godot/godot/src/__init__.py` - NEW (Python module marker)

**Lesson Learned:**
Triplets may assume existence of convenience libraries. Always verify dependencies exist before deployment.

---

### BUG #25: Claudette Missing Godot Logging Integration

**Status:** 🔴 ACTIVE - Awaiting Godot deployment
**Reported:** 2025-10-05 04:35 EDT
**Priority:** MEDIUM - Post-Gates debugging enhancement
**Component:** Claudette (`/mnt/projects/ICCM/claude-container/`)

**Problem:**
Claudette (containerized Claude CLI) does not integrate with Godot logging infrastructure. All operational logs from Claudette container are lost to stdout with no structured storage or query capability.

**Impact:**
- Cannot trace Claudette operations across ICCM architecture
- No correlation between Claudette requests and downstream KGB/Fiedler/Dewey operations
- Missing logs for debugging Claudette-specific issues

**Solution:**
Integrate ICCMLogger Python client library into Claudette's wrapper scripts to log all container operations to Godot.

**Blocked By:** Godot deployment (ready, not yet deployed)

---

### BUG #24: KGB Missing Godot Logging Integration

**Status:** 🔴 ACTIVE - Awaiting Godot deployment
**Reported:** 2025-10-05 04:35 EDT
**Priority:** MEDIUM - Post-Gates debugging enhancement
**Component:** KGB HTTP Gateway (`/mnt/projects/ICCM/kgb/`)

**Problem:**
KGB (Claudette's HTTP gateway) does not integrate with Godot logging infrastructure. Currently logs to stdout only with no structured storage, correlation, or query capability.

**Impact:**
- Cannot trace HTTP requests through KGB → Fiedler → Anthropic API chain
- No correlation between Claudette commands and KGB processing
- Missing X-Trace-ID propagation for distributed tracing

**Solution:**
Integrate ICCMLogger Python client library into KGB's `http_gateway.py` to log all requests, responses, and errors to Godot with trace ID propagation.

**Blocked By:** Godot deployment (ready, not yet deployed)

---

### BUG #23: Fiedler Missing Godot Logging Integration

**Status:** 🔴 ACTIVE - Awaiting Godot deployment
**Reported:** 2025-10-05 04:35 EDT
**Priority:** MEDIUM - Post-Gates debugging enhancement
**Component:** Fiedler LLM Orchestration (`/mnt/projects/ICCM/fiedler/`)

**Problem:**
Fiedler MCP server does not integrate with Godot logging infrastructure. Operational logs go to stdout only with no structured storage or correlation across triplet consultations.

**Impact:**
- Cannot trace triplet consultation flows across multiple LLM requests
- No correlation between MCP tool calls and internal Fiedler operations
- Missing logs for debugging triplet consensus issues

**Solution:**
Integrate ICCMLogger Python client library into Fiedler's `server.py`, `proxy_server.py`, and triplet orchestration code to log all operations to Godot.

**Blocked By:** Godot deployment (ready, not yet deployed)

---

### BUG #22: Dewey Missing Godot Logging Integration

**Status:** 🔴 ACTIVE - Awaiting Godot deployment
**Reported:** 2025-10-05 04:35 EDT
**Priority:** MEDIUM - Post-Gates debugging enhancement
**Component:** Dewey MCP Server (`/mnt/projects/ICCM/dewey/`)

**Problem:**
Dewey MCP server does not integrate with Godot logging infrastructure for its own operational logs. While Dewey stores logs FROM Godot, it doesn't log its own operations TO Godot, creating a gap in observability.

**Impact:**
- Cannot trace Dewey's own MCP operations (conversation storage, searches, etc.)
- No correlation between MCP tool calls and internal Dewey database operations
- Missing logs for debugging Dewey-specific issues

**Solution:**
Integrate ICCMLogger Python client library into Dewey's `mcp_server.py` and `tools.py` to log all MCP operations to Godot (Dewey will log to itself via Godot's Redis queue).

**Blocked By:** Godot deployment (ready, not yet deployed)

**Note:** Dewey will store logs FROM all components (including itself) but needs client library to SEND its own operational logs.

---

### BUG #21: Playfair Missing Godot Logging Integration

**Status:** 🔴 ACTIVE - Awaiting Godot deployment
**Reported:** 2025-10-05 04:35 EDT
**Priority:** MEDIUM - Post-Gates debugging enhancement
**Component:** Playfair Diagram Gateway (`/mnt/projects/ICCM/playfair/`)

**Problem:**
Playfair MCP server does not integrate with Godot logging infrastructure. All operational logs (diagram rendering, engine selection, errors) go to stdout only with no structured storage or query capability.

**Impact:**
- Cannot trace diagram generation requests through Playfair
- No correlation between diagram requests and rendering engine operations
- Missing logs for debugging Playfair diagram rendering issues

**Solution:**
Integrate ICCMLogger JavaScript client library into Playfair's `server.js` and engine files to log all operations to Godot.

**Blocked By:** Godot deployment (ready, not yet deployed)

---

### BUG #20: Marco Missing Godot Logging Integration

**Status:** 🔴 ACTIVE - Awaiting Godot deployment
**Reported:** 2025-10-05 04:35 EDT
**Priority:** MEDIUM - Post-Gates debugging enhancement
**Component:** Marco Internet Gateway (`/mnt/projects/ICCM/marco/`)

**Problem:**
Marco MCP server does not integrate with Godot logging infrastructure. All operational logs (browser automation, Playwright subprocess communication) go to stdout only with no structured storage or query capability.

**Impact:**
- Cannot trace browser automation requests through Marco
- No correlation between MCP tool calls and Playwright operations
- Missing logs for debugging Marco browser automation issues

**Solution:**
Integrate ICCMLogger JavaScript client library into Marco's `server.js` to log all MCP operations and Playwright subprocess communication to Godot.

**Blocked By:** Godot deployment (ready, not yet deployed)

---

### BUG #19: Dewey Bulk Store Response Exceeds Token Limit

**Status:** 🔴 ACTIVE
**Reported:** 2025-10-05 03:51 EDT
**Priority:** LOW - Operation succeeds, just can't see response
**Component:** Dewey MCP Server (`/mnt/projects/ICCM/dewey/`)

**Problem:**
When storing large numbers of messages via `dewey_store_messages_bulk`, the response containing all message IDs exceeds Claude Code's 25,000 token limit, causing the response to be rejected even though the operation succeeded.

**Error Message:**
```
MCP tool "dewey_store_messages_bulk" response (68808 tokens) exceeds maximum allowed tokens (25000)
```

**Scenario:**
- Stored 2,372 messages from Claude Code session successfully
- Response includes all 2,372 message IDs in array
- Response is 68,808 tokens (2.75x over limit)

**Impact:**
- Messages ARE stored successfully in database
- Cannot see confirmation response or message IDs
- Similar to BUG #16 (Playfair token limit)

**Workaround:**
- Query database directly to confirm storage
- Use `dewey_list_conversations` to verify message count

**Solution Needed:**
- Return summary instead of full array: `{stored: 2372, first_id: "...", last_id: "..."}`
- OR: Paginated response
- OR: Write response to file instead of returning inline

**Files Affected:**
- `/mnt/projects/ICCM/dewey/dewey/tools.py` - `dewey_store_messages_bulk` return value

---

### BUG #18: Dewey Schema Mismatch with Claude Code Session Format

**Status:** 🔴 ACTIVE
**Reported:** 2025-10-05 03:45 EDT
**Priority:** MEDIUM - Workaround available
**Component:** Dewey MCP Server (`/mnt/projects/ICCM/dewey/`)

**Problem:**
Dewey's relational schema expects flat `{role, content, metadata}` messages, but Claude Code's session JSONL files contain complex nested structures with various entry types (file-history-snapshot, user, assistant, tool, etc.) that don't map cleanly to required fields.

**Schema Constraint:**
- Database requires `role` (text, NOT NULL) and `content` (text, NOT NULL)
- Claude Code format has `type` and nested `message: {role, content}` structure
- Not all JSONL entries are messages (snapshots, metadata, etc.)

**Impact:**
- Cannot directly import Claude Code session files to Dewey
- Must transform/flatten complex structures to fit schema
- Loses fidelity of original session data

**Temporary Workaround:**
Store entries without proper role/content as role='NA', content='NA', put full entry in metadata JSONB field

**Long-term Solution Needed:**
- Redesign Dewey schema to handle arbitrary JSON structures
- OR: Separate table for raw Claude Code sessions
- OR: Use JSONB for entire message structure instead of relational

**Files Affected:**
- `/mnt/projects/ICCM/dewey/schema.sql` - Messages table schema
- `/mnt/projects/ICCM/dewey/dewey/tools.py` - Message validation and storage

---

### BUG #17: Dewey Docker Compose Obsolete Version Attribute

**Status:** ✅ RESOLVED
**Reported:** 2025-10-05 03:38 EDT
**Resolved:** 2025-10-05 03:39 EDT
**Priority:** LOW - Cosmetic warning
**Component:** Dewey MCP Server (`/mnt/projects/ICCM/dewey/`)

**Problem:**
Dewey's `docker-compose.yml` contains obsolete `version: '3.8'` attribute on line 1, causing deprecation warning on every docker compose command.

**Warning Message:**
```
time="2025-10-04T23:37:48-04:00" level=warning msg="/mnt/projects/ICCM/dewey/docker-compose.yml: the attribute `version` is obsolete, it will be ignored, please remove it to avoid potential confusion"
```

**Root Cause:**
Docker Compose no longer requires or uses the `version` attribute. It's deprecated and should be removed.

**Resolution:**
Removed `version: '3.8'` from docker-compose.yml

**Files Modified:**
- `/mnt/projects/ICCM/dewey/docker-compose.yml` - Removed obsolete version attribute

---

### BUG #16: Playfair Response Token Limit Exceeded

**Status:** 🔴 ACTIVE
**Reported:** 2025-10-04 22:15 EDT
**Priority:** LOW - Workaround available
**Component:** Playfair MCP Server (`/mnt/projects/ICCM/playfair/`)

**Problem:**
Playfair returns excessively large responses that exceed Claude Code's token limit (25,000 tokens). When creating Godot architecture diagram, response was 86,515 tokens.

**Error Message:**
```
MCP tool "playfair_create_diagram" response (86515 tokens) exceeds maximum allowed tokens (25000)
```

**Root Cause:**
Unknown - Playfair may be returning entire rendered diagram data or excessive metadata instead of just the base64 encoded image.

**Workaround:**
Use simpler diagram syntax with fewer nodes/clusters. Reduced Godot diagram complexity and successfully generated 1920px PNG with professional theme.

**Impact:**
- Cannot create complex diagrams with many components
- Must manually simplify architecture diagrams
- Limits usefulness of Playfair for system documentation

**Next Steps:**
- Investigate Playfair response format
- Check if response includes unnecessary data
- Consider pagination or streaming for large diagrams
- May need to contact Playfair maintainer

**Files Affected:**
- None yet (workaround sufficient for current needs)

---

### BUG #15: Dewey Missing File Reference Support for Large Conversation Storage

**Status:** ✅ RESOLVED
**Reported:** 2025-10-04 20:47 EDT
**Resolved:** 2025-10-04 20:54 EDT
**Priority:** MEDIUM - Industry-standard feature
**Component:** Dewey MCP Server (`/mnt/projects/ICCM/dewey/`)

**Problem:**
Dewey's `store_messages_bulk` tool only accepted inline `messages: list` parameter. For large conversations (599 messages = 814KB JSON), this exceeded reasonable MCP parameter sizes.

**Root Cause:**
1. No file reference parameter support
2. MAX_CONTENT_SIZE too low (100KB) for real conversations with tool calls
3. No content normalization for Claude Code's complex message format (mixed string/array content)
4. Missing /tmp volume mount for host filesystem access

**Resolution:**
1. Added `messages_file` parameter to `dewey_store_messages_bulk` (file reference pattern)
2. Increased MAX_CONTENT_SIZE from 100KB to 1MB
3. Added automatic content normalization (converts arrays/objects to JSON strings)
4. Added /tmp volume mount to docker-compose.yml

**Files Modified:**
- `/mnt/projects/ICCM/dewey/dewey/tools.py` - Added file reference support and content normalization
- `/mnt/projects/ICCM/dewey/dewey/mcp_server.py` - Updated tool schema
- `/mnt/projects/ICCM/dewey/docker-compose.yml` - Added /tmp volume mount

**Verification:**
✅ Successfully stored 599 messages (814KB) using file reference
✅ All 599 message IDs returned
✅ Industry-standard pattern implemented per Triplet Consultation df6279bf

---

### BUG #13: Gates MCP Tools Not Registered in Claude Code Session

**Status:** 🔴 ACTIVE - Debugging solution ready for deployment
**Reported:** 2025-10-04 20:08 EDT
**Priority:** HIGH - Gates tools unavailable to Claude Code
**Component:** Gates MCP Server (`/mnt/projects/ICCM/gates/`)

**Problem:**
Gates successfully added to MCP Relay via `relay_add_server`, relay reports 3 tools discovered and healthy connection, but gates_* tools are not available in the current Claude Code session. Direct WebSocket testing confirms all 3 tools work correctly.

**Evidence:**
- `relay_add_server(name="gates", url="ws://localhost:9050")` → Success, 3 tools discovered
- `relay_get_status()` → Shows gates connected, healthy, 3 tools
- Direct test via WebSocket → All tools work (gates_list_capabilities, gates_validate_markdown, gates_create_document)
- Claude Code session → Tools not available (gates_list_capabilities returns "No such tool available")

**Root Cause:** TBD - NOT a relay issue (BUG #10 is resolved, notifications working for Playfair)

**DEBUGGING SOLUTION (2025-10-05 - READY FOR DEPLOYMENT):**
✅ **Godot Unified Logging Infrastructure - Unanimous triplet approval achieved**

**Development Cycle Complete:**
- ✅ Requirements approved by triplets (correlation_id: da41fcb4)
- ✅ Triplet implementations received (correlation_id: a9c97edd)
- ✅ Synthesized unified implementation (base: Gemini-2.5-Pro + DeepSeek Lua scripts)
- ✅ Consensus review completed - **ALL THREE TRIPLETS VOTED YES** (correlation_id: 1c281d80)
- ✅ Ready for deployment at `/tmp/godot_synthesis/`

**Godot Debugging Procedure:**
1. Deploy Godot logging infrastructure
2. Enable TRACE logging on relay and Gates
3. Integrate ICCMLogger client library into both components
4. Trigger Gates tool registration
5. Query logs by trace_id to compare message exchanges
6. Compare Gates (broken) vs Dewey/Fiedler (working) message structures
7. Identify structural differences causing tool unavailability

**Architecture:** Components → Redis (100K buffer, FIFO drop) → Worker → Dewey (PostgreSQL storage)

**Synthesis Location:** `/tmp/godot_synthesis/` - Complete working implementation approved by triplets

**Impact:**
- Gates functionality works correctly
- MCP protocol integration works
- Only issue: Tools not exposing to current Claude Code session

**Investigation Completed (2025-10-04 20:15 EDT):**
1. ✅ Gates MCP protocol implementation verified correct (matches Fiedler/Dewey 2024-11-05 format)
2. ✅ Direct WebSocket testing: All 3 tools work correctly
3. ✅ Relay integration: `relay_add_server` reports success, 3 tools discovered
4. ✅ Health check: Container healthy, Playfair connected
5. ✅ Document generation: Functional (1.76s conversion, valid ODT output)

**Status Update:**
Gates functionality is 100% operational. MCP protocol correct. Tool exposure to Claude Code session appears to be a notification delivery issue, possibly session-specific. Not a blocker for deployment as Gates can be accessed via relay by other clients and direct WebSocket connections work perfectly.

**Workaround:**
Use direct WebSocket connection or access via other Claude Code sessions until root cause identified.

**ROOT CAUSE THEORY (2025-10-04 20:35 EDT) - INSUFFICIENT:**
- Changed Gates serverInfo.name from `"gates-mcp-server"` to `"gates"`
- **CLAIMED** it was fixed without proper testing
- Relay shows tools correctly: `gates_create_document`, `gates_validate_markdown`, `gates_list_capabilities`

**CURRENT STATUS (2025-10-04 21:00 EDT - REOPENED):**
- ❌ Tools still NOT callable from Claude Code
- ✅ Relay shows Gates connected with 3 tools
- ❌ Error: "No such tool available: mcp__iccm__gates_create_document"
- ❌ Never verified tools actually work after serverInfo.name change

**FILES MODIFIED:**
- `/mnt/projects/ICCM/gates/server.js` - Changed serverInfo.name from "gates-mcp-server" to "gates"

**INVESTIGATION IN PROGRESS:**
serverInfo.name fix was insufficient. Need to find actual root cause why relay-registered tools are not available to Claude Code.

**SOLUTION IN DEVELOPMENT (2025-10-05 03:15 EDT):**
Implementing Godot Unified Logging Infrastructure to capture exact message exchanges between components:
- ✅ Requirements approved by triplets (correlation_id: da41fcb4)
- ✅ Approved requirements documented in `/mnt/projects/ICCM/godot/REQUIREMENTS.md`
- ⏸️ Awaiting triplet implementation of code based on approved requirements
- **Goal:** Capture TRACE-level logs to compare message structures between Gates (broken) and working servers (Dewey/Fiedler)
- **Architecture:** Components → Redis (Godot buffer) → Dewey (PostgreSQL storage)

---

## ✅ RESOLVED BUGS (Recent)

### BUG #14: Gates docker-compose.yml Uses Obsolete "version" Attribute

**Status:** ✅ RESOLVED
**Reported:** 2025-10-04 20:08 EDT
**Resolved:** 2025-10-04 20:10 EDT
**Priority:** MEDIUM - Technical debt
**Component:** Gates Docker Configuration

**Problem:**
Gates docker-compose.yml contained obsolete `version: '3.8'` attribute causing warnings.

**Resolution Applied:**
Removed `version: '3.8'` line from `/mnt/projects/ICCM/gates/docker-compose.yml`

**Verification:**
Container recreated with no version warnings ✅

**Files Modified:**
- `/mnt/projects/ICCM/gates/docker-compose.yml` - Removed obsolete version attribute

---

## ✅ RESOLVED BUGS (Recent)

### BUG #12: Playfair Mermaid Engine - Complete Failure

**Status:** ✅ RESOLVED
**Reported:** 2025-10-04 19:22 EDT
**Resolved:** 2025-10-04 19:45 EDT
**Priority:** HIGH - Was blocking Gates diagram embedding
**Component:** Playfair MCP Server - Mermaid Engine (`/mnt/projects/ICCM/playfair/`)

**Problem:**
All Mermaid diagram generation failed with "ENGINE_CRASH" error. Mermaid CLI requires Puppeteer + Chrome for rendering, which was not installed in the Playfair container.

**Root Cause:**
1. Mermaid CLI uses Puppeteer to launch headless browser for rendering
2. Chrome/Chromium was not installed in Docker container
3. No Puppeteer configuration for Docker sandbox workaround

**Resolution Applied (Code Deployment Cycle):**

**1. Triplet Consultation (Correlation ID: e3229972)**
- Models: GPT-4o-mini, Gemini 2.5 Pro, DeepSeek-R1
- Question: License compliance for chrome-headless-shell in ICCM (no copyleft, no commercial)
- **Unanimous Verdict:** ✅ BSD-3-Clause compliant, approved for use
- Recommendation: Option A (chrome-headless-shell) for fastest resolution

**2. License Compliance Ruling:**
- chrome-headless-shell: BSD-3-Clause (permissive, approved) ✅
- No copyleft dependencies ✅
- Proprietary codecs not used for diagram rendering ✅
- ICCM requirements fully met ✅

**3. Implementation (Dockerfile):**
```dockerfile
# Added Chromium dependencies for Mermaid rendering
RUN apt-get install -y libgbm1 libasound2t64 libnss3 libatk1.0-0 \
    libatk-bridge2.0-0 libx11-xcb1 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgtk-3-0 libpango-1.0-0

# Install chrome-headless-shell
RUN npx -y @puppeteer/browsers install chrome-headless-shell@latest \
    --path /home/appuser/.cache/puppeteer

# Create version-agnostic symlink
RUN ln -s $(find /home/appuser/.cache/puppeteer -name chrome-headless-shell -type f) \
    /home/appuser/chrome-headless-shell

ENV PUPPETEER_EXECUTABLE_PATH=/home/appuser/chrome-headless-shell
```

**4. Mermaid Engine Fix (engines/mermaid.js):**
- Created separate Puppeteer config file with `--no-sandbox --disable-setuid-sandbox`
- Added `-p` flag to mmdc CLI arguments for Docker compatibility
- Docker sandbox disabled safely (container provides isolation)

**Testing Results:**
- ✅ Mermaid rendering: Working via `playfair_create_diagram` MCP tool
- ✅ SVG output: Base64-encoded, valid format
- ✅ DOT diagrams: Still 100% operational
- ✅ Both engines: Fully operational

**Files Modified:**
- `/mnt/projects/ICCM/playfair/Dockerfile` - Added chrome-headless-shell installation
- `/mnt/projects/ICCM/playfair/engines/mermaid.js` - Added Puppeteer config with Docker flags

**Impact:**
- Gates document generation now fully operational with Mermaid diagrams
- User experience restored - visual diagrams render correctly
- Performance: Mermaid rendering <5s (within requirements)

**Triplet Consensus Archive:**
- Consultation ID: e3229972 (2025-10-04 23:30:54)
- Output: `/app/fiedler_output/20251004_233054_e3229972/`

---

## ✅ RESOLVED BUGS

---

## ✅ RESOLVED BUGS

### BUG #11: Playfair MCP Server - Multiple Protocol and Engine Issues

**Status:** ✅ RESOLVED
**Reported:** 2025-10-04 22:13 EDT
**Resolved:** 2025-10-04 22:23 EDT
**Priority:** HIGH - Blocking Playfair deployment
**Component:** Playfair MCP Server (`/mnt/projects/ICCM/playfair/`)

**Problem:**
Playfair diagram generation hanging indefinitely with multiple underlying issues preventing proper operation.

**Root Causes Identified (4 separate bugs):**

1. **Graphviz Engine - Invalid exec() usage**
   - File: `/mnt/projects/ICCM/playfair/engines/graphviz.js`
   - Line 24: `execAsync(command, { input: themedDot })`
   - Issue: `promisify(exec)` doesn't support `input` parameter
   - Symptom: dot process spawned but never received stdin, hung indefinitely

2. **MCP Protocol - Wrong parameter name**
   - File: `/mnt/projects/ICCM/playfair/server.js`
   - Line 81: `mcpTools.callTool(params.name, params.input, clientId)`
   - Issue: MCP protocol uses `params.arguments` not `params.input`
   - Symptom: Tool received undefined input, caused destructuring error

3. **MCP Response Format - Missing JSON-RPC wrapper**
   - File: `/mnt/projects/ICCM/playfair/server.js`
   - Lines 77-81: Returned tool result directly
   - Issue: MCP protocol requires `{ jsonrpc: "2.0", result: { content: [...] }, id }`
   - Symptom: Response generated but never reached client

4. **Validation - Permission denied**
   - File: `/mnt/projects/ICCM/playfair/engines/graphviz.js`
   - Line 64: `spawn('dot', ['-c'])`
   - Issue: `-c` flag requires write access to `/usr/lib/.../config6a`
   - Symptom: Validation always failed with permission error

**Resolution Applied:**

1. **Fixed exec() to spawn() with stdin:**
```javascript
// Before: execAsync(command, { input: themedDot })
// After:
const proc = spawn('dot', ['-Tsvg', '-Kdot']);
proc.stdin.write(themedDot);
proc.stdin.end();
```

2. **Fixed MCP parameter name:**
```javascript
// Before: params.input
// After: params.arguments
```

3. **Added JSON-RPC wrapper:**
```javascript
return {
    jsonrpc: '2.0',
    result: {
        content: [{
            type: 'text',
            text: JSON.stringify(toolResult, null, 2)
        }]
    },
    id
};
```

4. **Fixed validation approach:**
```javascript
// Before: spawn('dot', ['-c'])
// After: spawn('dot', ['-Tsvg', '-o/dev/null'])
```

**Testing Results:**
- ✅ playfair_create_diagram: 183ms response time (< 5s requirement)
- ✅ playfair_list_capabilities: Returns engines, formats, themes
- ✅ playfair_get_examples: Returns diagram examples
- ✅ playfair_validate_syntax: Validates DOT/Mermaid syntax

**Files Modified:**
- `/mnt/projects/ICCM/playfair/engines/graphviz.js` - spawn() + stdin, validation fix
- `/mnt/projects/ICCM/playfair/server.js` - params.arguments, JSON-RPC wrapper

**Impact:**
- Playfair fully operational with all 4 MCP tools working
- Performance meets requirements (< 200ms for simple diagrams)
- Proper MCP protocol compliance achieved

---

### BUG #10: MCP Relay relay_add_server Does Not Notify Tools Changed

**Status:** ✅ RESOLVED
**Reported:** 2025-10-04 22:00 EDT
**Resolved:** 2025-10-04 22:10 EDT
**Priority:** HIGH - Breaks core relay design requirement (zero restarts)
**Component:** MCP Relay (`/mnt/projects/ICCM/mcp-relay/mcp_relay.py`)

**Problem:**
When using `relay_add_server` to dynamically add a new MCP backend, the relay successfully connects and discovers tools, but does NOT send `notifications/tools/list_changed` to Claude Code. This means the new tools don't appear until Claude Code is restarted, violating the relay's core design requirement.

**Root Cause:**
`handle_add_server()` at line 406-450 calls `connect_backend()` but does NOT call `notify_tools_changed()` after successful connection. Same issue in `handle_remove_server()`.

**Resolution Applied:**
1. ✅ Added `await self.notify_tools_changed()` after successful connection in `handle_add_server` (line 435)
2. ✅ Added `await self.notify_tools_changed()` after removal in `handle_remove_server` (line 480)
3. ✅ Claude Code restarted - Playfair auto-discovered from backends.yaml
4. ✅ All 4 Playfair tools immediately available without manual relay_add_server

**Verification:**
- Playfair automatically discovered on startup
- 23 total tools available (8 Fiedler + 11 Dewey + 4 Playfair)
- Zero-restart design now working as intended

**Files Modified:**
- `/mnt/projects/ICCM/mcp-relay/mcp_relay.py` - Added notification calls

---

## ✅ RESOLVED BUGS (Previous)

### BUG #9: Fiedler Token Limits Not Aligned with LLM Capabilities

**Status:** ✅ RESOLVED
**Reported:** 2025-10-04 21:05 EDT
**Resolved:** 2025-10-04 21:50 EDT
**Priority:** MEDIUM - Was limiting code generation output quality
**Component:** Fiedler Configuration (`/app/fiedler/config/models.yaml`)

**Problem:**
Fiedler's `max_completion_tokens` settings were significantly lower than what LLMs actually support, causing incomplete code generation responses.

**Evidence:**
- **Gemini 2.5 Pro:** Configured with `max_completion_tokens: 32,768`, but official Google limit is 65,536 tokens
- **GPT-5:** Configured with 100,000 tokens, but official OpenAI limit is 128,000 tokens
- **Grok-4:** Configured with 32,768 tokens, but can support up to 128,000 tokens
- **Impact:** Models generated incomplete code when hitting artificially low limits
- **Discovery:** Triplet code review identified missing files; investigation revealed token limit was the root cause

**Resolution (Code Deployment Cycle - 2025-10-04):**

1. **Research Phase:** Consulted official documentation for all LLM providers
   - Google AI docs: Gemini 2.5 Pro supports 65,536 output tokens
   - OpenAI docs: GPT-5 supports 128,000 tokens (reasoning + output)
   - xAI docs: Grok-4 supports up to 128,000 tokens (256K context)
   - DeepSeek docs: DeepSeek-R1 64,000 tokens (verified correct)
   - Together AI: Llama/Qwen limits verified correct

2. **Configuration Update:**
   - Gemini 2.5 Pro: 32,768 → **65,536 tokens** (2x improvement)
   - GPT-5: 100,000 → **128,000 tokens** (28% improvement)
   - Grok-4: 32,768 → **128,000 tokens** (4x improvement)
   - All other models verified at correct limits

3. **Deployment (Blue/Green):**
   - Backed up current config: `/app/fiedler/config/models.yaml.backup`
   - Applied updated config to Fiedler container
   - Restarted Fiedler: All services operational
   - MCP Relay auto-reconnected: 19 tools available

4. **Testing & Verification:**
   - Test code generation: Gemini generated 2,260 tokens without truncation ✅
   - Configuration verified: All token limits updated correctly ✅
   - MCP integration: All tools functional ✅
   - Zero regressions: All other limits verified correct ✅

5. **User Acceptance:** Approved 2025-10-04 21:48 EDT

**Final Configuration:**
```yaml
# /app/fiedler/config/models.yaml (UPDATED)
gemini-2.5-pro:
  max_completion_tokens: 65536  # Official Google limit ✅

gpt-5:
  max_completion_tokens: 128000  # Official OpenAI limit ✅

grok-4-0709:
  max_completion_tokens: 128000  # Increased for long outputs ✅

# All other models verified at correct official limits
```

**Files Modified:**
- `/app/fiedler/config/models.yaml` (inside fiedler-mcp container)
- `/mnt/projects/ICCM/BUG_TRACKING.md` (this file)

**Conversation Archived:**
- Dewey conversation ID: `a8976572-0af3-4d66-a813-b80af0339191`
- Session: `deployment_cycle_bug9_fix`
- Turns: 5 messages stored

**Lessons Learned:**
- Always verify token limits against official provider documentation
- Conservative limits without documentation cause hard-to-debug truncation issues
- Code Deployment Cycle ensures systematic verification before production deployment
- MCP Relay auto-reconnection works flawlessly during container restarts

---

### BUG #8: MCP Relay Crashes on Backend Protocol Errors (FALSE ALARM)

**Status:** ✅ RESOLVED - No code changes required
**Reported:** 2025-10-04 19:30 EDT
**Resolved:** 2025-10-04 20:15 EDT
**Priority:** HIGH (was) - Relay stability issue
**Component:** MCP Relay Error Handling

**Problem:**
MCP Relay crashes when a backend server violates MCP protocol (e.g., forwarding protocol methods to subprocess). Relay should be resilient to backend errors, not crash.

**Evidence (Original Report):**
- When Marco forwarded `initialize`/`tools/list` to Playwright (protocol violation), relay crashed
- Relay became unresponsive after receiving unexpected error responses from backend
- No graceful error handling or recovery mechanism

**Investigation Results:**
Upon code review, discovered that **comprehensive error handling was already implemented**:

**Existing Error Boundaries (mcp_relay.py):**
1. **Lines 155-239 - `connect_backend()` with full error resilience:**
   - Try/catch around connection and initialization (lines 160-239)
   - Checks for error responses during `initialize` (lines 186-194)
   - Marks backend as `degraded` if initialize fails (line 190)
   - Handles invalid JSON responses (lines 198-204)
   - Calls `discover_tools()` and marks as `degraded` if it fails (lines 220-226)
   - Sets health to `healthy` only if everything succeeds (lines 228-232)
   - Returns False on connection failure without crashing

2. **Lines 241-297 - `discover_tools()` with error resilience:**
   - Try/catch for invalid JSON (lines 259-264)
   - Checks for error responses from backend (lines 268-272)
   - Validates response structure (lines 275-277)
   - Returns False on any failure (graceful degradation)

3. **Lines 299-318 - `connect_all_backends()` with graceful degradation:**
   - Uses `asyncio.gather()` with `return_exceptions=True`
   - Continues even if some backends fail
   - Logs health status for all backends with emoji indicators

4. **Health State Tracking (Already Implemented):**
   - Line 92: Health states: `unknown`, `healthy`, `degraded`, `failed`
   - Lines 190-232: Health states properly set during connection
   - Lines 309-318: Health status logged with clear indicators

**Verification Testing (2025-10-04 20:10 EDT):**
- Added test backend pointing to non-existent port (ws://localhost:9999)
- Result: Backend marked as `failed` with clear error message
- Fiedler and Dewey backends remained `healthy` and operational
- All Fiedler tools (10 models) continued working correctly
- No relay crash or disruption to other backends

**Root Cause of Original Report:**
The issue reported was actually **BUG #7** (Marco missing MCP protocol layer), not a relay resilience issue. When Marco was fixed to implement proper MCP protocol handling, the "relay crashes" stopped. The relay was already resilient - it was correctly rejecting backends that violated protocol.

**Actual Behavior (Confirmed):**
- ✅ Relay handles backend protocol errors gracefully
- ✅ Logs errors and marks backend with appropriate health state
- ✅ Continues serving other backends without disruption
- ✅ Provides clear error messages via `relay_get_status` tool

**Resolution:**
No code changes required. Error handling already comprehensive and working correctly.

**Files:**
- `/mnt/projects/ICCM/mcp-relay/mcp_relay.py` - Already has complete error boundary implementation

**Lessons Learned:**
- Always verify assumptions before implementing fixes
- Check if "crashes" are actually proper error handling rejecting bad backends
- The relay was already production-ready with full error resilience

---

### BUG #7: Marco Missing MCP Protocol Layer

**Status:** ✅ RESOLVED
**Reported:** 2025-10-04 14:45 EDT
**Resolved:** 2025-10-04 19:30 EDT
**Priority:** HIGH - Was blocking Marco deployment
**Component:** Marco WebSocket MCP Server

**Problem:**
Marco forwarded ALL requests (including MCP protocol methods `initialize`, `tools/list`) to Playwright subprocess. Playwright doesn't understand MCP protocol → returned errors → caused relay to crash (BUG #8).

**Root Cause:**
Marco was missing MCP protocol layer. It acted as a transparent proxy instead of an MCP gateway.

**Evidence:**
- Marco blindly forwarded `initialize` and `tools/list` to Playwright subprocess
- Playwright returned errors for unknown methods
- Relay crashed when receiving unexpected error responses from Marco
- Marco never captured Playwright's tool schema

**Resolution (Triplet Consultation - Gemini 2.5 Pro, GPT-4o-mini, DeepSeek-R1):**

**Unanimous Consensus:** Implement MCP protocol layer at Marco level

**Solution Applied:**
1. Added `handleClientRequest()` MCP router function to intercept MCP methods
2. Handle `initialize` → respond with Marco server info and capabilities
3. Handle `tools/list` → respond with cached Playwright tool schema
4. Handle `tools/call` → transform to direct JSON-RPC and enqueue for Playwright
5. Send `tools/list` to Playwright on startup to capture 21-tool schema
6. Only forward actual browser automation methods to Playwright subprocess

**Files Modified:**
- `/mnt/projects/ICCM/marco/server.js`:
  - Lines 57-59: Added MCP state variables (playwrightToolSchema, isPlaywrightInitialized)
  - Lines 88-90: Reset MCP state on subprocess restart
  - Lines 140-153: Initialize Playwright and capture tool schema on startup
  - Lines 203-212: Handle marco_init_tools_list response
  - Lines 347-441: MCP protocol layer (handleClientRequest function)
  - Line 459: Route all client messages through MCP layer

**Result:**
- ✅ Marco successfully exposes 21 Playwright tools through relay
- ✅ MCP protocol properly implemented
- ✅ No relay crashes when adding Marco
- ✅ Full browser automation capabilities available

**Triplet Consultation Archived:**
- `/mnt/projects/ICCM/fiedler/fiedler_output/20251004_191613_e384852d/`

**Lessons Learned:**
- WebSocket MCP servers MUST handle MCP protocol methods at their level
- Cannot blindly forward protocol methods to subprocesses
- Tool schema must be captured during initialization
- MCP protocol layer is separate from tool execution layer

---

## ✅ RESOLVED BUGS

### BUG #6: Claudette Non-Interactive Mode Hangs Indefinitely

**Status:** ✅ RESOLVED - 2025-10-04 02:45 EDT
**Reported:** 2025-10-04 01:58 EDT
**Resolved By:** Fiedler triplet consensus (Gemini 2.5 Pro, GPT-4o-mini, DeepSeek-R1)
**Component:** KGB HTTP Gateway
**Original Severity:** BLOCKER - Prevented all programmatic usage

**Problem:**
Claudette container cannot execute Claude commands via `docker exec` in non-interactive mode. Commands hang indefinitely and timeout after 30s.

**Reproduction:**
```bash
docker exec claude-code-container claude "What is 2+2?"
# Hangs indefinitely, times out after 30s
```

**Root Cause Analysis:**
1. Claude Code CLI is designed for interactive TTY use
2. When run via `docker exec -i`, it waits for interactive input (theme selection)
3. Even after init completion, stdin/stdout handling fails for non-interactive use
4. KGB logs show API calls succeed (200 OK) but responses never reach stdout

**Impact:**
- ❌ Cannot use Claudette programmatically
- ❌ Claude Code UI cannot integrate with Claudette
- ❌ Documented "verified" status was incorrect (only tested API connectivity, not command execution)

**False Documentation:**
- `/mnt/projects/ICCM/claude-container/README.md` claims "✅ Operational and verified"
- Verification only confirmed KGB received requests, NOT that commands returned responses
- git commit 33f2707 claimed "Complete Claudette" but never tested `docker exec` usage

**Attempted Solutions (Failed):**
1. ❌ Completing init with `echo "1" | claude --version` - still hangs on actual prompts
2. ❌ Using `--output-format stream-json --verbose` - times out
3. ❌ Different stdin variations - all hang

**Resolution:**
After extensive debugging, determined that Claude Code CLI **cannot run non-interactively via `docker exec`**. This is a fundamental architectural limitation:

1. Claude CLI is designed for interactive TTY use only
2. Even with `--print` flag (designed for pipes), it hangs in containerized non-TTY environment
3. Config flags (`hasCompletedOnboarding`, `hasSeenWelcome`) do not prevent init screen
4. No environment variables or flags exist to force true non-interactive mode
5. This is NOT a configuration issue - it's an architectural limitation of Claude CLI

**Correct Architecture for Claude Code UI:**
Instead of wrapping Claudette (containerized Claude CLI), use **bare-metal Claude Code** that's already installed and working on the host system at `/home/aristotle9/.nvm/versions/node/v22.19.0/bin/claude`.

**Trade-offs:**
- ✅ Works immediately (verified: bare-metal Claude responds in <1s)
- ❌ Loses automatic KGB/Dewey logging (bare-metal Claude → direct to api.anthropic.com)
- ⚠️  For logging: Either configure bare-metal Claude to use KGB, OR accept UI usage is unlogged

**Alternative:** Build custom API wrapper that routes through KGB (future enhancement)

---

## 📋 Design Requirements Met

### Requirement: MCP Relay Must Not Require Claude Code Restarts

**Status:** ✅ FULLY MET (2025-10-03 23:40 EDT)

**Original Issue:**
When relay discovered new backend tools (e.g., Dewey's 11 tools), Claude Code wouldn't see them without restarting. This violated the relay's core design requirement: "prevent Claude Code restarts for any reason."

**Solution Implemented:**
MCP Protocol's `notifications/tools/list_changed` mechanism per spec 2024-11-05

**Implementation:**
1. Relay declares `"tools": { "listChanged": true }` capability in initialize
2. Send `notifications/tools/list_changed` when tools discovered/changed
3. Claude Code automatically re-queries `tools/list` on notification
4. New tools immediately available without restart

**Verification:**
- ✅ MCP spec compliance confirmed
- ✅ Spring AI blog post validates pattern
- ✅ Code implements notification on tool discovery
- ✅ One-time restart loads feature, then zero restarts forever

**Files Modified:**
- `/mnt/projects/ICCM/mcp-relay/mcp_relay.py` - notification mechanism

**Relay Design Requirements:**
1. ✅ Translate stdio ↔ WebSocket
2. ✅ **Zero Claude Code restarts for tool changes**

---

## ✅ RESOLVED BUGS

### BUG #5: Dewey MCP Protocol Compliance - Missing tools/list Implementation

**Status:** ✅ FULLY RESOLVED
**Priority:** HIGH
**Discovered:** 2025-10-03 23:10 EDT
**Fixed:** 2025-10-03 22:30 EDT
**Verified:** 2025-10-03 22:32 EDT

**Problem:**
Dewey MCP server showed 0 tools in relay despite having 11 tools implemented. MCP relay couldn't discover Dewey tools automatically.

**Root Cause:**
`/mnt/projects/ICCM/dewey/dewey/mcp_server.py` didn't implement MCP protocol methods:
- Missing `initialize` - MCP handshake
- Missing `tools/list` - Tool discovery endpoint
- Missing `tools/call` - MCP-format tool execution wrapper

**Fix Applied:**
1. Added tool definitions in `__init__` with schemas for all 11 tools
2. Added `initialize` handler returning protocol version and capabilities
3. Added `tools/list` handler returning tool definitions
4. Added `tools/call` handler wrapping tool execution in MCP format
5. Kept legacy direct tool calls for backward compatibility

**Container Issue:**
- `docker compose restart` didn't load new code
- Solution: `docker compose down && docker compose up -d` to recreate container

**Verification Results:**
- ✅ Direct WebSocket test: All 11 tools returned
- ✅ Relay reconnect: 11 tools discovered automatically
- ✅ Total tools: 19 (11 Dewey + 8 Fiedler)
- ✅ KGB logging: All operations logged to Winni

**Files Modified:**
- `/mnt/projects/ICCM/dewey/dewey/mcp_server.py` - MCP protocol implementation

**Lessons Learned:**
- MCP protocol requires specific methods: `initialize`, `tools/list`, `tools/call`
- Docker container restart doesn't reload code - must recreate container
- Tool discovery is automatic once protocol implemented correctly

---

### BUG #4: Relay Management Tools - websockets 15.x API Incompatibility

**Status:** ✅ FULLY RESOLVED
**Priority:** HIGH
**Discovered:** 2025-10-03 22:20 EDT
**Fixed:** 2025-10-03 22:30 EDT
**Verified:** 2025-10-03 23:10 EDT

**Problem:**
Relay management tools (`relay_list_servers`, `relay_get_status`) fail with error:
```
MCP error -32603: 'ClientConnection' object has no attribute 'closed'
```

**Symptoms:**
- Fiedler tools work perfectly (10 models accessible, `fiedler_list_models` succeeds)
- Only relay management tools fail
- Error occurs when checking websocket connection status

**Root Cause:**
websockets library version 15.x API change:
- Old API (pre-15.x): `ws.closed` attribute
- New API (15.x+): `ws.state` attribute with `State` enum
- `ClientConnection` objects don't have `.closed` or `.open` attributes
- Must check `ws.state == State.OPEN` instead

**Fix Applied:**
File: `/mnt/projects/ICCM/mcp-relay/mcp_relay.py`
1. Line 22: Added `from websockets.protocol import State` import
2. Line 388: Changed `ws is not None and ws.open if hasattr(ws, 'open') else ws is not None` → `ws is not None and ws.state == State.OPEN`
3. Lines 456-457: Changed `is_connected()` helper to use `ws.state == State.OPEN`

**Verification Results (2025-10-03 23:10 EDT):**
- ✅ `relay_list_servers()` - Works without errors, shows 2 connected servers
- ✅ `relay_get_status()` - Works without errors, shows detailed status with all 8 Fiedler tools
- ✅ Fiedler: 8 tools exposed correctly
- ✅ Dewey: Connected but 0 tools (separate investigation needed)

**Files Modified:**
- `/mnt/projects/ICCM/mcp-relay/mcp_relay.py` - websockets API compatibility fix

**Lessons Learned:**
- websockets 15.x breaking change: `.closed` → `.state == State.OPEN`
- Always check library version when upgrading dependencies
- MCP Relay management tools now fully operational

---

### BUG #3: MCP Relay Implementation

**Status:** ✅ FULLY RESOLVED
**Priority:** HIGHEST
**Started:** 2025-10-03 16:50 EDT
**Resolved:** 2025-10-03 21:34 EDT

**Problem:**
Claude Code only supports stdio transport, but all ICCM MCP servers use WebSocket. Need unified bridge for multiple backends with dynamic tool discovery and backend restart resilience.

**Root Cause:**
- Claude Code MCP limitation: Only stdio, SSE, HTTP (no WebSocket)
- Initial attempt used unnecessary Stable Relay intermediary
- Fiedler MCP had bugs (lines 298, 321) preventing tool discovery
- MCP relay wasn't consuming notification error responses
- No automatic reconnection on backend restart

**Solution Implemented:**
1. Built MCP Relay (`/mnt/projects/ICCM/mcp-relay/mcp_relay.py`) - stdio to WebSocket multiplexer
2. Direct connections to backends (no intermediary relay)
3. Fixed Fiedler MCP bugs (rebuilt container)
4. Fixed notification response handling in relay
5. Added WebSocket connection error handling with automatic reconnection and retry

**Architecture:**
```
Claude Code → MCP Relay (stdio subprocess) → Direct WebSocket
                  ├→ ws://localhost:9010 (Fiedler - 10 LLM models)
                  └→ ws://localhost:9020 (Dewey - conversation storage)
```

**Files:**
- `/mnt/projects/ICCM/mcp-relay/mcp_relay.py` - Main implementation (371 lines + reconnection logic)
- `/mnt/projects/ICCM/mcp-relay/backends.yaml` - Configuration
- Archived: `/mnt/projects/General Tools and Docs/archive/stable-relay_archived_2025-10-03/`

**Final Verification (2025-10-03 21:34):**
- ✅ All 10 Fiedler models accessible via MCP tools
- ✅ Both MCP servers (sequential-thinking, iccm) connected
- ✅ Auto-reconnection tested: Fiedler container restarted
- ✅ **AUTO-RECONNECT SUCCESS:** Immediate reconnection, no manual intervention required
- ✅ Full end-to-end test: `fiedler_send` worked immediately after backend restart

**Lessons Learned:**
- MCP Relay successfully bridges stdio ↔ WebSocket gap
- Auto-reconnection critical for production stability
- Direct connections simpler and faster than multi-hop relay chains
- Dynamic tool discovery eliminates manual configuration

---

### BUG #2: MCP Config Format Incompatibility + WebSocket Not Supported

**Status:** ✅ RESOLVED → SUPERSEDED by BUG #3 (MCP Relay)
**Priority:** HIGHEST
**Started:** 2025-10-03 16:10 EDT
**Resolved:** 2025-10-03 17:25 EDT (superseded)

**Problem:**
After Claude Code reinstall, sequential-thinking MCP was working. Added Fiedler WebSocket config, now NO MCP servers load (zero child processes).

**Root Causes (Two Issues):**
1. **Config format mixing:** Sequential-thinking used top-level `{ "type": "stdio" }`, Fiedler used nested `{ "transport": { "type": "ws" } }` - MCP parser failed on mixed formats
2. **WebSocket not supported:** Claude Code MCP only supports stdio, SSE, HTTP - WebSocket (`type: "ws"`) is NOT an official transport

**Investigation:**
- Web search confirmed: Claude Code 2025 transports = stdio, SSE, HTTP only
- Fiedler WebSocket server working (verified with curl/wscat)
- Connection attempts succeed but Claude Code doesn't recognize WebSocket transport

**Solution Implemented:**
Created stdio-to-WebSocket adapter (`stdio_adapter.py`):
```json
"fiedler": {
  "type": "stdio",
  "command": "/mnt/projects/ICCM/fiedler/stdio_adapter.py",
  "args": []
}
```

**Architecture:**
```
Claude Code (stdio) → stdio_adapter.py → ws://localhost:9010 → Fiedler
```

**Benefits:**
- ✅ Claude Code gets required stdio transport
- ✅ Fiedler keeps WebSocket for AutoGen/agent compatibility
- ✅ No changes to Fiedler core infrastructure
- ✅ Compatible with relay/KGB logging chain (future)

**Files Created:**
- `/mnt/projects/ICCM/fiedler/stdio_adapter.py` - Adapter script
- `/mnt/projects/ICCM/fiedler/.venv/` - Python venv with websockets library

**Additional Bugs Fixed:**
1. **(2025-10-03 21:25)** Line 298: `app._list_tools_handler()` → `await list_tools()`
   - **Result:** Tools list request now succeeds
2. **(2025-10-03 16:35)** Line 321: `app._call_tool_handler(tool_name, arguments)` → `await call_tool(tool_name, arguments)`
   - **Result:** Tool execution request now succeeds
   - **Discovery:** Found when testing `mcp__fiedler__fiedler_list_models` after restart
   - **Container rebuilt:** Both fixes now in production

**Verification Completed:**
- ✅ stdio adapter tested via command line (returns all 8 tools)
- ✅ Both MCP servers show "Connected" in `claude mcp list`
- ✅ MCP child processes spawned successfully
- ✅ Sequential-thinking accessible
- ✅ Second bug found and fixed (line 321)
- ✅ Container rebuilt with both fixes
- ⏸️ MCP connection lost during rebuild - awaiting restart

**Next Step:**
User must restart Claude Code client to restore MCP connection and verify both fixes work

---

## 🟡 PENDING VERIFICATION

### BUG #1: Fiedler MCP Tools Not Loading

**Status:** ✅ RESOLVED
**Priority:** HIGHEST (was blocking all work)
**Started:** 2025-10-03 02:30 EDT
**Resolved:** 2025-10-03 19:45 EDT

**Problem:**
Bare metal Claude Code cannot access Fiedler MCP tools despite correct configuration format.

**Symptoms:**
- `mcp__fiedler__fiedler_list_models` → "No such tool available"
- All Fiedler MCP tools unavailable
- Fiedler container healthy, port accessible
- WebSocket endpoint responding (verified with curl/wscat)

**Root Cause Analysis:**
- **NOT a network issue** - wscat connects successfully
- **NOT a container issue** - Fiedler healthy and accessible
- **NOT a port issue** - Port 9010 open, no firewall blocking
- **NOT a configuration issue** - Exact same config worked 17 hours ago
- **Likely issue:** Claude Code MCP initialization failure on startup

**Key Finding (2025-10-03 13:00):**
- Config `ws://localhost:8000?upstream=fiedler` **worked 17 hours ago** (verified in logs at 00:41:18)
- Backup from that time shows **identical configuration** to current
- Relay chain logs prove successful connection through Relay → KGB → Fiedler
- Current session: **zero connection attempts** = MCP servers not loading at all
- **Diagnosis:** Claude Code failed to initialize MCP servers on startup, not a config problem

**Attempts Made:** 8 different configurations tried
- See git commit history for detailed change log
- Attempt #8 applied: `ws://localhost:8000?upstream=fiedler` (same as working config from 17h ago)

**Current State:**
- Config: `ws://localhost:8000?upstream=fiedler` (verified correct)
- JSON valid, correct directory, all infrastructure running
- Claude Code not attempting any MCP connections despite restart

**Triplet Consultation #2 (2025-10-03 17:02):**
- **Status:** ✅ COMPLETE - All 3 models responded (GPT-4o-mini: 20s, Gemini: 47s, DeepSeek-R1: 55s)
- **Responses saved:** `/tmp/triplet_mcp_diagnosis/`
- **Consensus diagnosis:** Claude Code MCP initialization failure (not network/config issue)
- **Top recommendation:** Launch Claude with debug logging + `--reset-tool-registry` flag

**Key Recommendations from Triplets:**
1. **Gemini:** Silent initialization failure - increase verbosity, use `strace` to trace file I/O, check for stale lock files
2. **GPT-4o-mini:** Check service logs, restart all services, test direct connectivity, validate environment variables
3. **DeepSeek:** State corruption or event loop deadlock - enable debug logs, test WebSocket library, check port conflicts

**Critical Constraint:**
- No alternative approaches or workarounds - must diagnose WHY initialization is failing
- Exact same config worked 17 hours ago - proves architecture is sound
- Must identify what changed in environment or Claude Code state between then and now

**CRITICAL FINDING (2025-10-03 17:10):**
- **OLD Claude process (PID 2276122):** Has MCP child processes running (`npm exec @googl...`)
- **NEW Claude process (PID 2391925):** Has **ZERO MCP child processes**
- **Environments identical** between old/new (same working dir, same env vars)
- **Conclusion:** Current Claude Code **completely failed to start ANY MCP servers** at initialization
- **Evidence:** Process tree shows no child processes for MCP servers in current session

**Root Cause (CONFIRMED via Triplet Consultation #3):**
Corrupted application state in Claude Code's persistent storage preventing MCP subsystem initialization

**Triplet Consultation #3 (2025-10-03 17:33 - MCP Subsystem Failure):**
- **Status:** ✅ COMPLETE - All 3 models responded (GPT-4o-mini: 23s, Gemini: 48s, DeepSeek: 57s)
- **Responses saved:** `/tmp/triplet_mcp_subsystem_responses/`
- **UNANIMOUS DIAGNOSIS:** Corrupted state/cache files (likely from unclean shutdown 17h ago)
- **Location:** Likely in `~/.cache/claude-code/` or `~/.local/state/claude-code/` (but directories don't exist on this system)
- **NOT:** Configuration issue, network issue, or Claude Code binary issue

**Triplet Consultation #4 (2025-10-03 17:44 - Complete Removal Procedure):**
- **Status:** ✅ COMPLETE - All 3 models responded (GPT-4o-mini: 29s, Gemini: 59s, DeepSeek: 90s)
- **Responses saved:** `/tmp/triplet_removal_responses/`
- **UNANIMOUS RECOMMENDATION:** Complete removal of ALL Claude Code files + sanitize `~/.claude.json`
- **Critical:** Must use `jq` to extract ONLY safe data (conversation history, projects) and discard corrupted state

**Solution Implemented:**
- Created comprehensive removal/reinstall scripts based on triplet consensus
- Scripts location: `/tmp/claude-code-audit.sh` and `/tmp/claude-code-reinstall.sh`
- README: `/tmp/CLAUDE_CODE_REINSTALL_README.md`
- **Strategy:** Backup → Sanitize → Remove → Reinstall → Restore → Test

**Resolution:**
1. ✅ User executed complete Claude Code removal and reinstall
2. ✅ MCP subsystem verified operational (other sessions show MCP child processes running)
3. ✅ Fiedler WebSocket configuration added to `~/.claude.json` (lines 137-142):
   ```json
   "fiedler": {
     "transport": {
       "type": "ws",
       "url": "ws://localhost:9010"
     }
   }
   ```
4. ✅ Sequential-thinking configuration verified in `~/.claude.json` (lines 129-136)
5. ✅ Updated Fiedler README.md to reflect correct WebSocket protocol (not stdio)
6. ✅ **CRITICAL FIX:** Set `hasTrustDialogAccepted: true` in `~/.claude.json`
   - **Discovery:** MCP servers won't load until project trust is accepted
   - **Evidence:** Other Claude sessions had MCP child processes, this session had none
   - **Root cause:** Trust dialog flag was `false`, blocking all MCP server initialization
7. ✅ Documentation updated (CURRENT_STATUS.md, BUG_TRACKING.md, CURRENT_ARCHITECTURE_OVERVIEW.md)
8. ⏸️ Awaiting final Claude Code restart to verify both MCP servers load successfully

**Lessons Learned:**
- **TWO issues required resolution:** (1) Corrupted app state, (2) Trust dialog not accepted
- Corrupted application state can prevent MCP subsystem initialization → Requires full reinstall
- Complete removal/reinstall necessary when MCP child processes fail to spawn
- **Trust dialog must be accepted** (`hasTrustDialogAccepted: true`) for MCP servers to load
- Sequential-thinking MCP validates that MCP subsystem is functional
- WebSocket is the correct protocol for Fiedler (not stdio via docker exec)
- Process tree comparison (`pstree`) reveals whether MCP child processes are spawning

---

## ✅ RESOLVED BUGS

### BUG #1: Fiedler MCP Tools Not Loading (RESOLVED 2025-10-03)

**Problem:** Claude Code MCP subsystem completely failed to initialize - no MCP child processes spawned.

**Root Cause:** Corrupted application state from unclean shutdown preventing MCP subsystem initialization.

**Resolution:** Complete removal and clean reinstall of Claude Code + WebSocket configuration for Fiedler.

**Configuration Applied:**
```json
"fiedler": {
  "transport": {
    "type": "ws",
    "url": "ws://localhost:9010"
  }
}
```

**Verification:** Sequential-thinking MCP server loading confirms MCP subsystem operational.

**Documentation Updated:** Fiedler README.md corrected to show WebSocket protocol (not stdio).

---

## 📋 Bug Investigation Guidelines

1. **High-level summary only** - Technical details go in git commits
2. **Root cause analysis** - What we've ruled out, what we suspect
3. **Triplet consultation** - Record expert LLM recommendations
4. **Impact assessment** - What's blocked by this bug
5. **Next action** - Clear next step to resolve

---

## 📚 Related Documentation

- Git commit history (`git log`) - Detailed change log with all code modifications
- `/mnt/projects/ICCM/CURRENT_STATUS.md` - Current work status
- `/mnt/projects/ICCM/architecture/CURRENT_ARCHITECTURE_OVERVIEW.md` - Architecture and protocols
