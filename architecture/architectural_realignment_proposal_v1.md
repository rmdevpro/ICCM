# ICCM Architectural Realignment Proposal

**Date:** 2025-10-05
**Purpose:** Correct 3 architectural violations discovered during architecture diagram review
**Development Cycle:** Ideation → Draft → Triplet Review → Synthesis → Implementation
**Target Architecture:** Option 4 - Write/Read Separation (Pure Implementation)

---

## Executive Summary

During the October 5th architecture review, three critical violations of the Option 4 architecture were discovered:

1. **Dewey has write tools** (violates READ-only specialist principle)
2. **Fiedler not logging conversations** (violates single gateway principle)
3. **KGB still exists** (unnecessary complexity, creates alternative write path)

This proposal outlines the architectural changes required to bring the ICCM system into full compliance with the Option 4: Write/Read Separation architecture as shown in the three architecture diagrams.

---

## Current State (Violations)

### VIOLATION #1: Dewey Has Write Tools

**Current Reality:**
- Dewey exposes `dewey_store_message` and `dewey_store_messages_bulk` tools
- These tools write directly to PostgreSQL Winni
- KGB uses these tools to log conversations

**Architectural Violation:**
- Option 4 requires Dewey to be READ-only specialist
- All writes should flow through Godot (WRITE specialist)
- Creates multiple write paths to database (not single source of truth)

**Diagram Evidence:**
- Diagram_2_Data_Writes.png: Shows NO arrow from Dewey to Winni
- Diagram_3_Data_Reads.png: Shows Dewey with READ arrows only (green bidirectional)

### VIOLATION #2: Fiedler Not Logging Conversations

**Current Reality:**
- Fiedler routes all LLM traffic (10 models across 4 providers)
- Fiedler logs operational events to Godot
- Fiedler does NOT log LLM conversations to Godot

**Architectural Violation:**
- Fiedler is "single gateway for all LLM access"
- All LLM conversations should be captured and logged
- Diagram_2_Data_Writes.png shows "Logs + Conversations" from Fiedler to Godot

**Diagram Evidence:**
- Diagram_2_Data_Writes.png: Blue arrow from Fiedler labeled "Logs + Conversations"
- Currently only "Logs" flows, missing "Conversations"

### VIOLATION #3: KGB Still Exists

**Current Reality:**
- KGB HTTP proxy (port 8089) routes Claudette → Anthropic API
- KGB WebSocket spy (port 9000) can intercept MCP traffic
- KGB logs conversations via Dewey write tools

**Architectural Violation:**
- Diagram_1_MCP_Traffic.png shows Claudette → MCP Relay (direct MCP connection)
- No KGB component in any diagram
- Creates alternative conversation logging path (Claudette → KGB → Dewey instead of Fiedler → Godot)

**Diagram Evidence:**
- Diagram_1_MCP_Traffic.png: Shows Claudette connected to MCP Relay via MCP protocol
- No HTTP proxy layer shown
- No alternative write paths

---

## Target State (Option 4 Pure Implementation)

### Data Write Flow (Diagram_2_Data_Writes.png)

**Single Source of Truth: Godot**
```
ALL components → Godot (logger_log MCP tool) → PostgreSQL Winni
```

**Write Specialists:**
- **Godot (9060)**: ALL database writes (logs + conversations)
- **Fiedler (9030)**: Logs both operational events AND LLM conversations to Godot
- **All other components**: Log operational events to Godot

**NO Write Capabilities:**
- Dewey has ZERO write tools
- No component writes directly to Winni except Godot

### Data Read Flow (Diagram_3_Data_Reads.png)

**Read Specialist: Dewey**
```
Components → Dewey (query tools) ↔ PostgreSQL Winni (READ request/reply)
```

**Read Capabilities:**
- **Dewey (9022)**: Query logs, conversations, search, analytics
- All components use Dewey tools for data retrieval
- Dewey has SELECT-only database permissions

### MCP Traffic Flow (Diagram_1_MCP_Traffic.png)

**Claude Code (Bare Metal):**
```
Claude Code → MCP Relay → {Fiedler, Dewey, Marco, Gates, Playfair, Godot}
Claude Code → logger_log → Godot (9060)
```

**Claudette (Containerized):**
```
Claudette → MCP Relay → {Fiedler, Dewey, Marco, Gates, Playfair, Godot}
Claudette → fiedler_send → Fiedler → Cloud LLMs
Fiedler → logger_log (conversations + logs) → Godot
```

**Conversation Logging:**
```
User calls fiedler_send
  ↓
Fiedler routes to Cloud LLM
  ↓
Fiedler receives response
  ↓
Fiedler logs {request, response, model, tokens, timing} to Godot
  ↓
Godot writes to Winni conversations table
```

---

## Required Changes

### Change #1: Remove Write Tools from Dewey

**Objective:** Make Dewey truly READ-only

**Implementation:**
1. Remove `dewey_store_message` tool from Dewey
2. Remove `dewey_store_messages_bulk` tool from Dewey
3. Keep all query/read tools:
   - `dewey_get_conversation`
   - `dewey_list_conversations`
   - `dewey_search`
   - `dewey_query_logs`
   - `dewey_get_log_stats`
   - All startup context tools (read/write to contexts table)
4. Database: Revoke INSERT/UPDATE/DELETE on conversations/messages tables from Dewey user
5. Database: Grant SELECT-only on conversations/messages tables to Dewey user

**Affected Files:**
- `/mnt/projects/ICCM/dewey-blue/dewey/tools.py` - Remove write tool implementations
- `/mnt/projects/ICCM/dewey-blue/dewey/mcp_server.py` - Remove write tools from schema
- `/mnt/projects/ICCM/dewey-blue/README.md` - Update documentation
- PostgreSQL Winni - Update dewey user permissions

**Breaking Changes:**
- Any code calling `dewey_store_message` or `dewey_store_messages_bulk` will fail
- KGB conversation logging will break (to be replaced by Fiedler logging)

**Migration Path:**
1. Deploy Fiedler conversation logging FIRST
2. Verify conversations flowing Fiedler → Godot → Winni
3. Then remove Dewey write tools

### Change #2: Add Conversation Logging to Fiedler

**Objective:** Fiedler logs ALL LLM conversations to Godot

**Implementation:**
1. Add Godot MCP client to Fiedler (use existing pattern from Gates/Playfair/Marco)
2. Intercept every `fiedler_send` call
3. Capture request: {prompt, model, max_tokens, temperature, files}
4. Capture response: {content, model, input_tokens, output_tokens, timing}
5. Log conversation via `logger_log` tool to Godot with component='fiedler-conversations'
6. Store in conversations table: {conversation_id, turn_number, role, content, model, tokens, metadata}

**Data Format:**
```python
conversation_log = {
    "conversation_id": str(uuid4()),  # New conversation per fiedler_send call
    "turn_number": 1,  # Request = 1, Response = 2
    "role": "user" | "assistant",
    "content": str,  # Prompt or response
    "metadata": {
        "model": str,  # Which LLM was called
        "input_tokens": int,
        "output_tokens": int,
        "timing_ms": int,
        "files": list,  # If files sent
        "model_specific": dict  # Provider-specific metadata
    }
}
```

**Affected Files:**
- `/mnt/projects/ICCM/fiedler-blue/fiedler/tools.py` - Add conversation logging to fiedler_send
- `/mnt/projects/ICCM/fiedler-blue/fiedler/mcp_client.py` - Add Godot MCP client (if not exists)
- `/mnt/projects/ICCM/fiedler-blue/README.md` - Document conversation logging

**Requirements:**
- Non-blocking logging (don't fail LLM call if Godot unavailable)
- Log BEFORE returning to user (ensure conversation captured)
- Include correlation IDs for tracing

### Change #3: Eliminate KGB

**Objective:** Remove KGB container and route Claudette through MCP Relay

**Implementation:**
1. Update Claudette configuration to connect to MCP Relay directly
2. Claudette uses `fiedler_send` for all LLM calls (not direct Anthropic API)
3. Stop and remove KGB container
4. Remove KGB from docker-compose.yml
5. Archive KGB code to `/mnt/projects/ICCM/archive/kgb/`

**Affected Files:**
- `/mnt/projects/ICCM/claude-container/config/.claude.json` - Update to use MCP Relay
- `/mnt/projects/ICCM/claude-container/docker-compose.yml` - Remove KGB references
- `/mnt/projects/ICCM/kgb/` - Archive entire directory

**Claudette New Configuration:**
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

**Claudette New Flow:**
```
User in Claudette
  ↓
Claudette calls fiedler_send via MCP Relay
  ↓
Fiedler routes to Cloud LLM
  ↓
Fiedler logs conversation to Godot
  ↓
Response returns to Claudette
```

**Benefits:**
- Eliminates unnecessary HTTP proxy layer
- Claudette and Claude Code use identical MCP architecture
- Single conversation logging path (Fiedler → Godot)
- Simpler architecture matches diagrams

---

## Database Schema Changes

### Conversations Table (New Structure)

**Add to Godot's schema_additions.sql:**
```sql
-- Conversations table for Fiedler conversation logging
CREATE TABLE IF NOT EXISTS conversations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID NOT NULL,  -- Groups related messages
    turn_number INTEGER NOT NULL,   -- Sequence within conversation
    role VARCHAR(20) NOT NULL,      -- 'user' or 'assistant'
    content TEXT NOT NULL,
    model VARCHAR(100),              -- Which LLM model used
    input_tokens INTEGER,
    output_tokens INTEGER,
    timing_ms INTEGER,
    metadata JSONB,                  -- Extensible metadata
    created_at TIMESTAMP DEFAULT NOW(),

    UNIQUE(conversation_id, turn_number)
);

CREATE INDEX idx_conversations_conv_id ON conversations(conversation_id);
CREATE INDEX idx_conversations_created ON conversations(created_at);
CREATE INDEX idx_conversations_model ON conversations(model);
CREATE INDEX idx_conversations_metadata ON conversations USING GIN(metadata);

-- Grant permissions
GRANT SELECT ON conversations TO dewey_user;  -- READ-only
GRANT INSERT ON conversations TO godot_log_writer;  -- WRITE-only
```

### Dewey User Permissions

**Revoke write permissions on existing conversations tables (if any):**
```sql
-- Remove write access from Dewey
REVOKE INSERT, UPDATE, DELETE ON conversations FROM dewey_user;
REVOKE INSERT, UPDATE, DELETE ON messages FROM dewey_user;

-- Ensure read access only
GRANT SELECT ON conversations TO dewey_user;
GRANT SELECT ON messages TO dewey_user;
```

---

## Implementation Plan

### Phase 1: Add Fiedler Conversation Logging (Blue/Green)

1. Create Fiedler Blue-v2 with conversation logging
2. Test conversation logging to Godot
3. Verify conversations appear in Winni database via Dewey queries
4. Cutover when verified

**Success Criteria:**
- Every `fiedler_send` call creates 2 log entries (request + response)
- Conversations queryable via Dewey tools
- Zero errors in Fiedler logs

### Phase 2: Remove Dewey Write Tools (Blue/Green)

1. Create Dewey Blue-v2 without write tools
2. Update database permissions (SELECT-only on conversations/messages)
3. Test all READ operations still working
4. Cutover when verified

**Success Criteria:**
- All query tools working
- Write tools return "tool not found" error
- Database permissions enforce READ-only access

### Phase 3: Eliminate KGB

1. Update Claudette configuration to use MCP Relay
2. Test Claudette → Fiedler flow
3. Verify conversations logged via Fiedler → Godot
4. Stop KGB container
5. Archive KGB code

**Success Criteria:**
- Claudette uses MCP Relay for all tools
- Claudette conversations logged via Fiedler
- No KGB container running
- All 12 Claudette tests passing

### Phase 4: Verification & Documentation

1. Run full system test (Claude Code + Claudette)
2. Verify conversation logging working both paths
3. Update all documentation
4. Update architecture diagrams (remove KGB if shown)
5. Close GitHub issues #1, #2, #3

---

## Testing Strategy

### Unit Tests
- Fiedler conversation logging function
- Godot conversation write handler
- Dewey conversation query functions

### Integration Tests
- Full flow: fiedler_send → Godot → Winni → Dewey query
- Verify conversation_id grouping
- Verify turn_number sequencing
- Verify metadata captured correctly

### System Tests
- Claude Code conversation logged via Fiedler
- Claudette conversation logged via Fiedler
- Both retrievable via Dewey queries
- No duplicate logging
- No missing conversations

### Regression Tests
- All existing Dewey query tools still working
- All existing Godot logging still working
- All existing Fiedler LLM calls still working
- Claudette 12-test suite passes

---

## Rollback Plan

### If Fiedler Conversation Logging Fails:
- Rollback to Fiedler Blue-v1 (no conversation logging)
- Keep KGB running temporarily
- Debug and retry

### If Dewey Write Tool Removal Fails:
- Rollback to Dewey Blue-v1 (with write tools)
- Keep database permissions unchanged
- Debug and retry

### If KGB Elimination Fails:
- Revert Claudette configuration
- Restart KGB container
- Debug Claudette → MCP Relay connection
- Retry after fix

---

## Risk Assessment

### LOW RISK: Fiedler Conversation Logging
- **Why:** Additive change, doesn't break existing functionality
- **Mitigation:** Non-blocking logging, fails silently
- **Rollback:** Easy - just use old Fiedler container

### MEDIUM RISK: Dewey Write Tool Removal
- **Why:** Breaking change for any code using write tools
- **Mitigation:** Deploy Fiedler logging first, verify before removing
- **Rollback:** Medium - need to restore tools and permissions

### LOW RISK: KGB Elimination
- **Why:** Only affects Claudette, easily tested
- **Mitigation:** Test thoroughly before removing KGB
- **Rollback:** Easy - restart KGB container

---

## Questions for Triplet Review

1. **Conversation Logging Design:**
   - Should Fiedler create new conversation_id per `fiedler_send` call?
   - Or should it track multi-turn conversations (requires state management)?
   - Proposed: New conversation_id per call (stateless, simpler)

2. **Database Schema:**
   - Should conversations table be partitioned by created_at (like logs)?
   - Proposed: Yes, monthly partitions for scalability

3. **Error Handling:**
   - If Godot unavailable, should Fiedler:
     - A) Fail the LLM call?
     - B) Complete LLM call but log error?
   - Proposed: B (non-blocking logging)

4. **Migration Strategy:**
   - Should we migrate existing conversations from Dewey to new schema?
   - Proposed: No, clean break - old conversations queryable via old tools temporarily

5. **Claudette Emergency Access:**
   - Should Claudette keep emergency direct API access like Claude Code?
   - Proposed: No, Claudette is "inside ecosystem" and must follow all rules

---

## Success Metrics

**Architectural Compliance:**
- ✅ Dewey has ZERO write tools
- ✅ ALL database writes flow through Godot
- ✅ Fiedler logs ALL LLM conversations
- ✅ KGB eliminated
- ✅ Claudette uses MCP Relay

**Functional Correctness:**
- ✅ All LLM calls still working
- ✅ All conversations logged and retrievable
- ✅ All query operations working
- ✅ Zero regressions

**Performance:**
- ✅ No latency increase in LLM calls
- ✅ Logging non-blocking
- ✅ Query performance unchanged

---

## Timeline Estimate

**Phase 1 (Fiedler Logging):** 2-4 hours
- Add Godot MCP client: 30 min
- Implement conversation logging: 1 hour
- Blue/Green deployment: 30 min
- Testing: 1 hour

**Phase 2 (Remove Dewey Writes):** 1-2 hours
- Remove tools: 30 min
- Update permissions: 15 min
- Blue/Green deployment: 30 min
- Testing: 30 min

**Phase 3 (Eliminate KGB):** 1-2 hours
- Update Claudette config: 15 min
- Testing: 30 min
- Archive KGB: 15 min
- Documentation: 30 min

**Phase 4 (Verification):** 1 hour
- System testing: 30 min
- Documentation updates: 30 min

**Total Estimated Time:** 5-9 hours

---

## Deliverables

1. **Code:**
   - Fiedler Blue-v2 with conversation logging
   - Dewey Blue-v2 without write tools
   - Claudette updated configuration
   - Database schema updates

2. **Documentation:**
   - Updated CURRENT_STATUS.md
   - Updated CURRENT_ARCHITECTURE_OVERVIEW.md
   - Updated component READMEs
   - Closed GitHub issues #1, #2, #3

3. **Testing:**
   - Integration test suite for conversation logging
   - Updated Claudette test suite
   - System verification report

4. **Git History:**
   - Commit per phase
   - Tagged release after full implementation
   - Conversation archived to Dewey

---

## Conclusion

This architectural realignment brings the ICCM system into full compliance with Option 4: Write/Read Separation as documented in the three architecture diagrams. The changes are straightforward, low-risk, and can be implemented incrementally using Blue/Green deployment strategy.

The key principle: **Godot is the ONLY component that writes to the database. Dewey is the ONLY component for reading from the database.**

**Recommendation:** Proceed with triplet review of this proposal before implementation.
