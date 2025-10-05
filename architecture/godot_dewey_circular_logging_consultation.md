# Architectural Consultation: Godot-Dewey Circular Logging Problem

## Executive Summary

While implementing Godot logging integration for Dewey Blue, we discovered a **circular dependency issue** that challenges a core architectural constraint (CONST-001: "Dewey is sole gatekeeper for Winni"). We need triplet guidance on the best architectural approach.

---

## Problem Statement

### The Circular Dependency

**Current Architecture (from REQUIREMENTS.md):**
```
Components (Gates, Fiedler, Marco, etc.)
  ↓ (WebSocket MCP: logger_log tool)
Godot MCP Server
  ↓ (pushes to internal Redis queue)
Godot Batch Worker
  ↓ (calls dewey_store_logs_batch via MCP)
Dewey MCP Server
  ↓ (stores in PostgreSQL)
Winni Database (PostgreSQL)
```

**The Problem:**
1. Dewey wants to log its own operations to Godot (like all other components)
2. Godot calls `dewey_store_logs_batch` to store logs in Winni
3. If `dewey_store_logs_batch` execution logs to Godot...
4. **Infinite loop!**

**Specific Example:**
- Dewey's `mcp_server.py` logs: "Tool call started: dewey_store_logs_batch"
- That log goes to Godot
- Godot batches it and calls `dewey_store_logs_batch`
- Dewey logs: "Tool call started: dewey_store_logs_batch"
- Infinite recursion!

### Current Workaround (Implemented but Not Deployed)

In `dewey-blue/dewey/mcp_server.py`, we added:

```python
# CRITICAL: Don't log logging operations (prevents circular logging)
is_logging_tool = tool_name in ('dewey_store_logs_batch', 'dewey_query_logs', 'dewey_clear_logs', 'dewey_get_log_stats')

if LOGGING_ENABLED and not is_logging_tool:
    await log_to_godot('TRACE', 'Tool call started', data={'tool_name': tool_name, 'has_args': bool(arguments)})
```

**Issues with this approach:**
- Fragile - must carefully maintain exclusion list
- Can't debug Dewey's logging operations themselves
- Violates principle of "all MCP servers log consistently"
- Creates special case that future developers might break

---

## Architectural Constraint in Question

From `/mnt/projects/ICCM/godot/REQUIREMENTS.md`:

> **CONST-001:** Dewey is sole gatekeeper for Winni - no direct PostgreSQL access from Godot

**Rationale (assumed):**
- Single point of control for database schema
- Consistent validation and error handling
- Abstraction layer for future database changes
- Clear separation of concerns

**However:** This constraint creates the circular dependency when Godot needs to store logs.

---

## Proposed Solutions

### Option 1: Circular Logging Prevention (Current Implementation)

**Architecture:** Keep current design, exclude logging tools from being logged

**Implementation:**
```python
is_logging_tool = tool_name in ('dewey_store_logs_batch', 'dewey_query_logs', 'dewey_clear_logs', 'dewey_get_log_stats')
if LOGGING_ENABLED and not is_logging_tool:
    await log_to_godot(...)
```

**Pros:**
- ✅ Maintains CONST-001 (Dewey as sole gatekeeper)
- ✅ Minimal code changes
- ✅ No schema changes required
- ✅ Clear separation of concerns preserved

**Cons:**
- ❌ Fragile - easy to break if developer forgets to update exclusion list
- ❌ Can't debug Dewey's logging operations (blind spot)
- ❌ Special case creates inconsistency
- ❌ Must maintain exclusion list as logging tools evolve
- ❌ Violates "all components log the same way" principle

**Risk Level:** Medium - Works but requires ongoing maintenance vigilance

---

### Option 2: Godot Direct PostgreSQL Access for Logs Only

**Architecture:** Allow Godot to write directly to `logs` table in Winni, bypassing Dewey

**Implementation:**
```python
# In Godot batch worker
async def store_logs_direct(logs: list):
    # Direct PostgreSQL connection to Winni
    conn = await asyncpg.connect(
        host='192.168.1.210',
        database='winni',
        user='godot',  # NEW: separate database user
        password='...'
    )
    await conn.executemany(
        "INSERT INTO logs (trace_id, component, level, message, data, created_at) VALUES ($1, $2, $3, $4, $5, $6)",
        log_data
    )
```

**Database Changes Required:**
```sql
-- Create dedicated Godot user with LIMITED permissions
CREATE USER godot WITH PASSWORD '...';
GRANT INSERT ON logs TO godot;
GRANT SELECT ON logs TO godot;  -- For stats only
-- NO other permissions (no DELETE, UPDATE, or access to other tables)
```

**Pros:**
- ✅ Eliminates circular dependency entirely
- ✅ Godot and Dewey fully independent
- ✅ Can debug ALL components including Dewey's logging operations
- ✅ All components log consistently (no special cases)
- ✅ Better performance (one fewer MCP hop)
- ✅ Still maintains data integrity via limited database permissions

**Cons:**
- ❌ Violates CONST-001 architectural constraint
- ❌ Two components writing to same table (though with different permissions)
- ❌ Requires new database user and permission management
- ❌ Godot must duplicate log validation logic (or risk bad data)
- ❌ Future schema changes require updating both Godot AND Dewey

**Risk Level:** Low-Medium - Clean solution but violates stated architecture

---

### Option 3: Dewey Does Not Log to Godot

**Architecture:** Dewey logs only to stdout/stderr, not to Godot centralized logging

**Implementation:**
- Dewey uses Python's `logging` module only
- No Godot integration in Dewey
- Docker captures logs via container logging

**Pros:**
- ✅ No circular dependency
- ✅ Maintains CONST-001 (Dewey as sole gatekeeper)
- ✅ Simple - no special cases needed
- ✅ Dewey remains the single source of truth for Winni

**Cons:**
- ❌ Can't correlate Dewey logs with other components via `trace_id`
- ❌ Can't query Dewey logs through Godot MCP tools
- ❌ Inconsistent - all other components log to Godot
- ❌ Harder to debug distributed issues (logs in different places)
- ❌ Loses centralized logging benefits for Dewey

**Risk Level:** Low - Safe but reduces debugging capability

---

## Comparison Matrix

| Criteria | Option 1: Exclusion | Option 2: Direct Access | Option 3: No Logging |
|----------|---------------------|-------------------------|----------------------|
| **Maintains CONST-001** | ✅ Yes | ❌ No | ✅ Yes |
| **No circular dependency** | ⚠️ Prevented | ✅ Yes | ✅ Yes |
| **Can debug Dewey logging** | ❌ No | ✅ Yes | ⚠️ Partial |
| **Consistent logging** | ❌ Special case | ✅ Yes | ❌ Inconsistent |
| **Trace ID correlation** | ✅ Yes | ✅ Yes | ❌ No |
| **Maintenance burden** | ❌ High | ⚠️ Medium | ✅ Low |
| **Performance** | ⚠️ Extra MCP hop | ✅ Direct | ✅ Local only |
| **Schema coupling** | ✅ Single point | ❌ Two points | ✅ Single point |
| **Future-proof** | ❌ Brittle | ⚠️ Moderate | ✅ Stable |

---

## Questions for Triplet Review

1. **Is CONST-001 a hard architectural requirement, or can it be relaxed for the specific case of logging?**
   - If hard: Option 1 or 3
   - If soft: Option 2 is cleanest

2. **How important is it to debug Dewey's logging operations?**
   - If critical: Option 2
   - If not important: Option 1 or 3

3. **Should all MCP servers log consistently, or is it acceptable for Dewey to be a special case?**
   - If consistency critical: Option 2 or 3
   - If special cases acceptable: Option 1

4. **Is there a fourth option we haven't considered?**
   - Perhaps a hybrid approach?
   - Different architectural pattern altogether?

5. **For Option 2, is limited database access (INSERT/SELECT on logs table only) an acceptable security model?**
   - Database permissions can enforce constraint
   - But requires trust in permission system

---

## Recommendation Request

Please provide:

1. **Preferred architectural option** (1, 2, or 3) with rationale
2. **Risk assessment** of your preferred option
3. **Implementation guidance** for the chosen approach
4. **Any additional constraints or considerations** we should account for

---

## Current Status

- **Dewey Blue** implementation is paused at this decision point
- Code has been written for Option 1 but not deployed
- All other components (Gates, Playfair, Marco, Fiedler) successfully integrated with Godot
- Godot logging infrastructure is operational and working well

---

## Supporting Documentation

- `/mnt/projects/ICCM/godot/REQUIREMENTS.md` - Original requirements with CONST-001
- `/mnt/projects/ICCM/dewey-blue/dewey/mcp_server.py` - Paused implementation with circular prevention
- `/mnt/projects/ICCM/godot/godot/README.md` - MCP-based logging integration guide
- `/mnt/projects/ICCM/CURRENT_STATUS.md` - Current deployment status

---

**Prepared by:** Claude (ICCM Development Session)
**Date:** 2025-10-05
**Session Context:** Godot Logging Integration - Dewey Blue Deployment (Paused)
**Decision Required:** Architectural approach for Dewey logging to resolve circular dependency
