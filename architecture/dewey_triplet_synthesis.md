# Dewey + Winni Requirements: Triplet Review Synthesis

**Date:** 2025-10-02
**Reviewers:** Gemini 2.5 Pro, GPT-5, Grok-4
**Verdict:** All 3 approved WITH CHANGES

---

## Executive Summary

All three reviewers found the MVP design sound but identified critical issues that must be addressed before implementation:

1. **Schema design flaws** requiring structural changes
2. **Missing MCP tools** essential for basic functionality
3. **Security gaps** particularly around network exposure
4. **MCP/Docker transport mismatch** needing architectural clarification
5. **Performance optimizations** requiring specification

---

## Critical Issues (Must Fix)

### 1. Schema Design Flaws

**Problem:** Multiple structural issues in database schema
**Consensus:** All 3 reviewers identified same issues

**Required Changes:**

```sql
-- REMOVE: storage_type column from conversations
-- It's redundant with separate tables for startup_contexts and fiedler_results

-- ADD: updated_at triggers
CREATE TRIGGER conversations_updated_at AFTER UPDATE ON conversations
BEGIN
    UPDATE conversations SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER startup_contexts_updated_at AFTER UPDATE ON startup_contexts
BEGIN
    UPDATE startup_contexts SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

-- ADD: Partial unique index for single active startup context (better than trigger)
CREATE UNIQUE INDEX ux_startup_contexts_single_active
ON startup_contexts(is_active) WHERE is_active = TRUE;

-- ADD: JSON validation
ALTER TABLE conversations ADD CHECK(json_valid(metadata));
ALTER TABLE fiedler_results ADD CHECK(json_valid(metadata));

-- ADD: Missing indices
CREATE INDEX idx_messages_created ON messages(created_at DESC);
CREATE INDEX idx_conversations_updated ON conversations(updated_at DESC);
CREATE INDEX idx_messages_conversation_created ON messages(conversation_id, created_at);
```

**Rationale:**
- `storage_type` causes confusion and complicates queries
- `updated_at` without triggers leads to stale data
- Race condition in trigger can cause multiple active contexts
- Missing indices hurt query performance

---

### 2. Missing MCP Tools

**Problem:** Core functionality gaps in tool interface
**Consensus:** All 3 reviewers identified missing tools

**Required Additions:**

1. **dewey_delete_conversation**
   ```json
   {
     "conversation_id": "string (required)",
     "cascade": "boolean (optional, default: true)"
   }
   ```
   Returns: `{ "deleted": true, "messages_deleted": 15 }`

2. **dewey_delete_startup_context**
   ```json
   {
     "name": "string (required)"
   }
   ```
   Returns: `{ "deleted": true }`

3. **dewey_list_conversations**
   ```json
   {
     "session_id": "string (optional)",
     "limit": "integer (optional, default: 20)",
     "cursor": "string (optional, pagination cursor)",
     "sort_by": "string (optional, default: 'updated_at')"
   }
   ```
   Returns: `{ "conversations": [...], "next_cursor": "..." }`

4. **Pagination in dewey_search**
   - Add `cursor` parameter
   - Add `next_cursor` in return
   - Add `rank` score in results (bm25)

5. **dewey_store_messages_bulk** (GPT-5 suggestion)
   ```json
   {
     "conversation_id": "string (required)",
     "messages": [
       {"role": "user", "content": "..."},
       {"role": "assistant", "content": "..."}
     ]
   }
   ```
   Returns: `{ "stored": 2, "message_ids": [...] }`

**Rationale:**
- Cannot manage data without delete tools
- Pagination essential for scaling beyond 100 conversations
- Bulk insert reduces overhead for batch operations

---

### 3. Security Gaps

**Problem:** Port exposure without authentication, missing encryption
**Consensus:** GPT-5 and Grok-4 flagged; Gemini noted concern

**Critical Decision Required:**

**Option A: MCP Stdio Only (Recommended for MVP)**
```yaml
# docker-compose.yml - NO PORT EXPOSURE
# Claude Code runs Dewey as subprocess via stdio
# NO network exposure, NO authentication needed
ports: []  # Remove 9020:8080 mapping
```

**Option B: Add HTTP API with Authentication**
```yaml
# docker-compose.yml - WITH AUTH
ports:
  - "9020:8080"
environment:
  - DEWEY_AUTH_TOKEN=secret-token-here
  - DEWEY_TLS_CERT=/certs/dewey.crt
  - DEWEY_TLS_KEY=/certs/dewey.key
```

**Additional Security Requirements:**

1. **Prompt Redaction:** Make default (not optional)
2. **Encryption at Rest:** Document SQLCipher option
3. **Audit Logging:** Log all startup_context updates
4. **Input Validation:** Size limits, injection prevention
5. **Data Retention:** Add `dewey_purge_old_data` tool

---

### 4. MCP/Docker Transport Mismatch

**Problem:** MCP uses stdio, but Docker exposes TCP port
**Consensus:** GPT-5 strong concern; others noted confusion

**Clarification Required:**

**Current Confusion:**
- MCP protocol is stdio-based (subprocess communication)
- Docker container with port 9020 suggests HTTP/TCP
- Claude Code expects MCP tools via stdio, not HTTP

**Recommended Resolution (GPT-5):**

```yaml
# Two distinct interfaces:

1. MCP Stdio Interface (for Claude Code):
   - Run Dewey as subprocess: `python -m dewey.mcp_server`
   - No Docker needed for Claude Code integration
   - Use Docker only for development/testing

2. HTTP API Interface (for Fiedler, optional Phase 2):
   - Separate HTTP server: `python -m dewey.http_api`
   - Requires authentication (bearer token or mTLS)
   - Docker deployment with port exposure
   - OpenAPI spec for cross-service calls
```

**Decision for MVP:**
- **Remove port exposure** from docker-compose.yml
- Clarify Dewey runs as **local subprocess** for Claude Code
- Docker used only for **packaging and distribution**, not runtime
- Defer HTTP API to Phase 2 when Fiedler integration needed

---

### 5. Performance Optimizations

**Problem:** Missing PRAGMA settings and operational procedures
**Consensus:** All 3 reviewers requested specifications

**Required Additions:**

```python
# SQLite PRAGMA settings
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;
PRAGMA foreign_keys = ON;
PRAGMA temp_store = MEMORY;
PRAGMA cache_size = -64000;  # 64MB page cache

# FTS5 tokenizer configuration
CREATE VIRTUAL TABLE messages_fts USING fts5(
    content,
    content=messages,
    content_rowid=rowid,
    tokenize='unicode61 remove_diacritics 2'
);
```

**Backup Procedure:**
```python
# Use SQLite online backup API
# Schedule: Daily at 2 AM
# Compress: gzip backup files
# Retention: Keep 7 daily backups
# Document restore procedure in README
```

**Vacuum Strategy:**
```python
# Option 1: VACUUM INTO (preferred)
VACUUM INTO '/app/winni/conversations_compacted.db';

# Option 2: Auto-vacuum incremental
PRAGMA auto_vacuum = INCREMENTAL;
PRAGMA incremental_vacuum(1000);  # 1000 pages

# Schedule: Weekly off-peak (Sunday 3 AM)
```

---

## Important Issues (Should Fix)

### 6. Tool Interface Improvements

**Clarifications Needed:**

1. **dewey_get_startup_context:**
   - Behavior when `name` omitted: "Get active context"
   - Behavior when no active context: Return null or error?

2. **dewey_store_message:**
   - Auto-increment `turn_number`: Use transaction to prevent races
   - Return `message_count` and `last_turn_number` in response

3. **dewey_search:**
   - Add `rank` score (bm25) in results
   - Add optional `highlights` parameter for snippets
   - Standardize error responses across all tools

**Standardized Error Format (GPT-5/Grok-4):**
```json
{
  "success": false,
  "error": {
    "code": "CONVERSATION_NOT_FOUND",
    "message": "No conversation found with id: abc-123"
  }
}
```

---

### 7. Data Model Extensions

**Optional Fields for Future-Proofing:**

**Conversations table:**
```sql
title TEXT,                        -- User-assigned title
tags TEXT,                         -- JSON array of tags
user_id TEXT,                      -- Multi-user support
message_count INTEGER DEFAULT 0,   -- Denormalized for performance
deleted_at TIMESTAMP              -- Soft delete
```

**Messages table:**
```sql
parent_message_id TEXT REFERENCES messages(id),  -- Threading
tokens_prompt INTEGER,                           -- Token counts
tokens_completion INTEGER,
deleted_at TIMESTAMP                            -- Soft delete
```

**Fiedler_results table:**
```sql
conversation_id TEXT REFERENCES conversations(id),  -- Link to conversation
message_id TEXT REFERENCES messages(id),            -- Link to specific message
latency_ms INTEGER                                  -- Performance tracking
```

---

### 8. Testing Expansions

**Additional Test Scenarios:**

**Concurrency Tests:**
- Simultaneous `dewey_store_message` calls (turn_number race)
- FTS trigger synchronization under concurrent writes
- Single active context enforcement with concurrent updates

**Crash/Recovery Tests:**
- Kill process during write, verify WAL recovery
- Database lock contention with multiple readers
- Corruption detection and recovery

**Migration Tests:**
- Schema version upgrade (v1 → v2)
- Data integrity validation after migration
- Rollback procedure

**Security Tests (if HTTP API added):**
- Authentication bypass attempts
- Rate limiting enforcement
- Request size limit validation
- SQL injection via tool parameters

**Load Tests:**
- 100K conversations with realistic message counts
- Search performance at scale
- Concurrent read/write mix

---

## Answers to Open Questions

### Q1: Is the SQLite schema appropriate for MVP?

**Consensus:** Yes, with required fixes:
- Remove `storage_type` column
- Add `updated_at` triggers
- Add missing indices
- Add JSON validation
- Fix single active context enforcement

### Q2: Are the tool signatures clear and complete?

**Consensus:** Mostly clear but incomplete:
- Add delete tools
- Add pagination
- Add bulk insert
- Clarify edge case behaviors
- Standardize error responses

### Q3: Is SQLite FTS5 sufficient for MVP?

**Consensus:** **YES** - All 3 reviewers agree
- Fast enough for 10K-100K conversations
- No external dependencies
- Good relevance ranking with bm25
- Configure with unicode61 tokenizer
- Defer vector search to Phase 3

### Q4: Should Fiedler integration be MVP (Phase 1) or Phase 2?

**Consensus:** **Phase 2** (defer)
- **Gemini:** Defer to stabilize core first
- **GPT-5:** Defer but define network API now
- **Grok-4:** Defer, include schema stubs

**Action:** Keep Fiedler in Phase 2, but clarify how it will connect (HTTP API vs MCP)

### Q5: Does the "write/clear/load" workflow make sense?

**Consensus:** **YES** - Pragmatic and functional
- Maps to user mental model
- Minimal state management
- **Enhancement (GPT-5):** Add `dewey_apply_startup_context()` helper
- **Enhancement (Grok-4):** Add `dewey_clear_chat()` tool for automation

### Q6: Are performance targets realistic?

**Consensus:** **YES** with proper configuration
- 100ms store: Achievable with WAL mode
- 200ms retrieve: Achievable with proper indices
- 500ms search (10K): Achievable with FTS5 + bm25
- **Caveat:** At 100K conversations, search may drift to 700-1200ms
- **Mitigation:** Pagination, caching, bulk inserts

### Q7: Is the migration path reasonable?

**Consensus:** **YES** - Well-considered
- SQLite → sqlite-vss → PostgreSQL is logical
- Phased approach reduces risk
- **Suggestion (GPT-5):** Consider PostgreSQL earlier if multi-writer contention
- **Suggestion (Grok-4):** Make Phase 4 (PostgreSQL) mandatory, not optional

### Q8: Any security concerns?

**Consensus:** **YES** - Critical gap
- Port exposure without auth is dangerous
- **Decision needed:** Stdio-only OR HTTP with auth
- Add encryption-at-rest guidance
- Add data retention/deletion tools
- Enforce prompt redaction by default

### Q9: Should we add extensibility fields now?

**Consensus:** Use `metadata` JSON column for flexibility
- Avoid premature column additions
- `metadata` provides extensibility without schema changes
- **Exception:** Add `message_count`, `deleted_at` if soft-delete desired
- **Exception:** Add `tags`, `title` if common use case

### Q10: Is the testing strategy comprehensive?

**Consensus:** Good but needs expansion
- **Add:** Concurrency tests (all 3 reviewers)
- **Add:** Crash/recovery tests (GPT-5, Grok-4)
- **Add:** Migration tests (GPT-5, Grok-4)
- **Add:** Security tests if HTTP API added
- **Add:** Load tests for 100K conversations

---

## Divergent Opinions

### Schema Structure (Minor Disagreement)

**Grok-4 Position:** Unify all data under conversations table
- Use `storage_type` to differentiate
- Extend `messages` for all types

**Gemini/GPT-5 Position:** Keep separate tables
- Remove `storage_type` as redundant
- Separate tables for separate concerns

**Recommendation:** Follow Gemini/GPT-5 (majority view, cleaner design)

### Fiedler Integration Timing

**GPT-5 Position:** Define network API spec NOW (even if Phase 2)
- Avoid rework later
- Clarify HTTP vs MCP interface early

**Gemini Position:** Defer entirely to Phase 2
- Focus on core functionality
- Design API when needed

**Recommendation:** **Middle ground** - Keep Phase 2, but clarify transport in requirements v2

---

## Requirements v2 Action Items

### Schema Changes (Critical)
- [ ] Remove `storage_type` column from conversations
- [ ] Add `updated_at` triggers (2 tables)
- [ ] Replace trigger with partial unique index for active context
- [ ] Add missing indices (4 new indices)
- [ ] Add JSON validation checks
- [ ] Add FTS5 tokenizer configuration

### MCP Tool Additions (Critical)
- [ ] Add `dewey_delete_conversation`
- [ ] Add `dewey_delete_startup_context`
- [ ] Add `dewey_list_conversations` with pagination
- [ ] Add pagination to `dewey_search` (cursor, rank score)
- [ ] Add `dewey_store_messages_bulk`
- [ ] Clarify edge case behaviors

### Architecture Clarification (Critical)
- [ ] **DECISION:** Stdio-only OR HTTP with auth?
- [ ] Remove port exposure from docker-compose.yml (if stdio-only)
- [ ] Document how Claude Code connects (subprocess)
- [ ] Defer HTTP API to Phase 2 Fiedler integration

### Security Enhancements (Critical)
- [ ] Make prompt redaction default (not optional)
- [ ] Add encryption-at-rest guidance (SQLCipher)
- [ ] Add audit logging for sensitive operations
- [ ] Add data retention/purge tools
- [ ] Remove network exposure OR add authentication

### Performance Specifications (Important)
- [ ] Specify SQLite PRAGMA settings
- [ ] Document backup procedure (online backup API)
- [ ] Document vacuum strategy (VACUUM INTO or incremental)
- [ ] Add bm25 ranking to FTS5 queries
- [ ] Adjust NFR-1.4 to "acceptable degradation"

### Testing Expansions (Important)
- [ ] Add concurrency test scenarios
- [ ] Add crash/recovery test scenarios
- [ ] Add migration test scenarios
- [ ] Add security test scenarios (if HTTP)
- [ ] Add load test scenarios (100K conversations)

### Documentation Updates (Important)
- [ ] Clarify MCP stdio vs Docker deployment
- [ ] Update architecture diagram (remove port, add stdio)
- [ ] Document PRAGMA settings and rationale
- [ ] Document backup/restore procedures
- [ ] Document migration from SQLite to PostgreSQL

---

## Recommended Priority

**Must Fix Before Implementation (P0):**
1. Schema changes (remove storage_type, add triggers, indices)
2. MCP transport clarification (stdio vs HTTP decision)
3. Security decision (remove port OR add auth)
4. Add delete tools (conversation, startup context)

**Should Fix Before Implementation (P1):**
5. Add pagination (search, list conversations)
6. Specify PRAGMA settings
7. Add bulk insert tool
8. Standardize error responses

**Nice to Have (P2):**
9. Data model extensions (soft delete, tags)
10. Testing expansions
11. Documentation improvements

---

## Final Recommendation

**Status:** APPROVED WITH CHANGES (unanimous)

**Next Steps:**
1. Create requirements v2 with P0 changes
2. Make architecture decision: stdio-only (recommended) vs HTTP with auth
3. Re-review requirements v2 internally
4. Proceed to implementation

**Estimated Effort:**
- Requirements v2: 2-3 hours
- Implementation (with changes): 3-5 days
- Testing: 1-2 days
- Total: ~1 week for MVP

---

## Appendix: Full Review Links

- **Gemini 2.5 Pro Review:** `/tmp/dewey_triplet_reviews/gemini-2.5-pro.md`
- **GPT-5 Review:** `/tmp/dewey_triplet_reviews/gpt-5.md`
- **Grok-4 Review:** `/tmp/dewey_triplet_reviews/grok-4-0709.md`
