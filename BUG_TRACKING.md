# Bug Tracking Log

**Purpose:** Track active bugs with high-level summaries and resolution status

**Last Updated:** 2025-10-03

---

## 🐛 ACTIVE BUGS

### BUG #1: Fiedler MCP Tools Not Loading (CRITICAL)

**Status:** 🔴 ACTIVE - Blocking all work
**Priority:** HIGHEST
**Started:** 2025-10-03 02:30 EDT
**Last Updated:** 2025-10-03 13:00 EDT

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

**Root Cause:** Claude Code silently failed to initialize MCP subsystem - no servers launched, no connection attempts made

**Next Action:**
User must restart Claude Code to trigger MCP initialization with current correct config

**Impact:**
- Cannot use Fiedler for LLM orchestration
- Blocks containerized Claude design review
- Blocks all development requiring multi-LLM consultation

---

## ✅ RESOLVED BUGS

*None yet*

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
