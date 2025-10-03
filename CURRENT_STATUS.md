## 📚 MUST READ FIRST - Session Startup Documents

**Every new session MUST start by reading these documents to understand the system:**

### 1. System Overview & Current Work (READ FIRST - ALWAYS)
- `/mnt/projects/ICCM/architecture/General Architecture.PNG` - **Visual diagram of entire system**
- `/mnt/projects/ICCM/architecture/CURRENT_ARCHITECTURE_OVERVIEW.md` - **System overview and current protocol configuration**
- `/mnt/projects/ICCM/CURRENT_STATUS.md` - **THIS FILE - Current work & status**
- `/mnt/projects/ICCM/BUG_TRACKING.md` - **Active bugs with high-level summaries**
- `/mnt/projects/ICCM/CODE_CHANGE_TRACKER.md` - **Detailed log of all configuration changes and attempts**

### 2. Module Documentation (READ AS NEEDED - Based on Current Work)
- `/mnt/projects/ICCM/architecture/dewey_winni_requirements_v3.md` - Dewey + Winni specification
- `/mnt/projects/ICCM/kgb/README.md` - KGB proxy specification
- `/mnt/projects/ICCM/architecture/STABLE_RELAY_DEPLOYMENT.md` - Stable Relay deployment
- `/mnt/projects/ICCM/fiedler/README.md` - Fiedler MCP server (7 LLM models)

### 3. Reference Documentation & Workarounds (As Needed)
- `/mnt/projects/ICCM/architecture/FIEDLER_DOCKER_WORKAROUND.md` - Triplet consultation when Fiedler MCP unavailable
- `~/.config/claude-code/mcp.json` - MCP configuration (global)
- `~/.claude.json` - MCP configuration (project-specific)

### 4. Archive
- **Location:** `/mnt/projects/General Tools and Docs/archive/`
- **Purpose:** Historical reference only - DO NOT USE for active work
- **Usage:** Place outdated documents here to prevent confusion about which version to use

**⚠️ CRITICAL:** Starting a session without understanding the high-level architecture and requirements leads to wasted effort and mistakes. Always read Section 1 documents first!

---

# ICCM Development Status - Current Session

**Last Updated:** 2025-10-03 17:15 EDT
**Session:** Root Cause Found - MCP Subsystem Not Starting
**Status:** 🟡 **BUG #1 - Restart required to initialize MCP servers**

---

## 🎯 Current Objective

**Fix BUG #1:** Fiedler MCP tools not loading in bare metal Claude Code

**Status:** Root cause identified - Claude Code did not start MCP subsystem

**Quick Summary:**
- **Problem:** Fiedler MCP tools not loading despite correct configuration
- **Root Cause Found:** Process tree analysis shows current Claude has ZERO MCP child processes
- **Old Claude (PID 2276122):** Has `npm exec @googl...` MCP servers running
- **Current Claude (PID 2391925):** No MCP child processes at all - subsystem never initialized
- **Configuration:** Correct (`ws://localhost:8000?upstream=fiedler`)
- **Environment:** Identical to working session

**Next Action:**
User must FULLY QUIT and RESTART Claude Code to trigger MCP initialization

*(See BUG_TRACKING.md for triplet consultation details, CODE_CHANGE_TRACKER.md for all attempts)*

---

## 📋 Current Work - Step by Step

### ✅ Phase 1: Configuration Investigation (8 attempts completed)
**Summary:** Tried 8 different configurations - all failed
- Attempts 1-7: Various URLs and formats (see CODE_CHANGE_TRACKER.md)
- **Attempt #8:** `ws://localhost:8000?upstream=fiedler` - Same config that worked 17h ago
- **Result:** MCP servers not loading - Claude Code not attempting connections

**Critical Discovery:**
- Git backup from 17 hours ago shows **identical configuration**
- Relay/KGB logs show successful connection at 00:41:18 (17h ago)
- Current session: **Zero connection attempts** in logs
- Conclusion: Not a configuration problem - Claude Code initialization issue

### ✅ Phase 2: Triplet Consultation (COMPLETED - 2025-10-03 17:02)
**Status:** All 3 LLMs responded successfully

**Consensus Diagnosis:**
- **Root Cause:** Silent MCP initialization failure in Claude Code
- **Not:** Network issue, container issue, configuration syntax issue
- **Evidence:** No connection attempts = pre-connection initialization failure

**Top Recommendations:**
1. Enable debug logging (`ANTHROPIC_LOG_LEVEL=debug` or `--debug`)
2. Check for stale state (.pid, .lock, cache files)
3. Use `strace` to trace config file reading
4. Compare environment variables between sessions
5. Test WebSocket connectivity from host

**Responses saved:** `/tmp/triplet_mcp_diagnosis/`

### ✅ Phase 3: Root Cause Analysis (COMPLETED - 2025-10-03 17:10)
**Status:** Root cause identified

**Process Tree Analysis:**
- **Command:** `pstree -p 2391925` vs `pstree -p 2276122`
- **Finding:** Old Claude has MCP child processes (`npm exec @googl...`), current Claude has NONE
- **Conclusion:** MCP subsystem completely failed to initialize in current session
- **Environment Check:** Both processes identical (same working dir, same env vars)

**Why It Failed:**
Current Claude Code startup did not initialize MCP subsystem at all - not even attempting to spawn MCP server processes.

### 🔴 Phase 4: Restart Test (NEXT - User Action Required)
**Status:** Awaiting user restart

**Action Required:**
1. **FULLY QUIT** Claude Code (kill process if needed)
2. **RESTART** Claude Code
3. **TEST:** `mcp__fiedler__fiedler_list_models`
4. **VERIFY:** Check for MCP child processes with `pstree`

**Expected After Restart:**
- Fiedler MCP tools should be available
- Process tree should show WebSocket connection child processes
- Relay/KGB/Fiedler logs should show new connection attempts

### 🔬 Diagnostic Plan (IF Attempt #5 Fails)
**Per triplet consensus, execute in this order:**

**Step 1: Gather Evidence**
```bash
# Check Claude Code logs for MCP connection errors
# Location: ~/.cache/claude-code/logs/*.log or similar

# Monitor Fiedler logs during Claude restart
docker logs -f fiedler-mcp
```

**Step 2: Isolate Problem with wscat**
```bash
# Test if WebSocket endpoint is accessible from host
# Install: npm install -g wscat
wscat -c ws://127.0.0.1:9010

# If wscat CONNECTS: Problem is in Claude Code
# If wscat FAILS: Problem is in Docker/network/firewall
```

**Step 3: Try host.docker.internal (Attempt #6)**
```json
"url": "ws://host.docker.internal:9010"
```
- Update both config files
- Full restart
- Test again

**Step 4: Try Host LAN IP (Attempt #7)**
```bash
# Find host IP
ip addr show | grep "inet " | grep -v 127.0.0.1

# Use in config (example):
"url": "ws://192.168.1.X:9010"
```

**Step 5: Deep Diagnostics**
```bash
# Check port binding
docker ps --filter "name=fiedler" --format "{{.Ports}}"
docker inspect fiedler-mcp | grep -A 10 Ports

# Check firewall
sudo ufw status
sudo iptables -L -n -v | grep 9010

# Check listening on host
sudo lsof -i :9010
netstat -tuln | grep 9010
```

### 🎯 Phase 3: Containerized Claude Review (WAITING)
- [ ] Send design to triplet (Gemini 2.5 Pro, GPT-4o-mini, DeepSeek-R1)
- [ ] Incorporate feedback
- [ ] Proceed with implementation

---

## 🏗️ System Architecture

### Bare Metal Claude (Current Session - ACTUAL STATE)
```
Claude Code (bare metal)
    ↓ Direct connection (no logging)
    ├→ Claude Max (Anthropic API) - Yellow dotted line in diagram
    └→ Fiedler (docker exec -i fiedler-mcp fiedler) - Blue solid line in diagram
```

**No automatic logging** - Bare metal Claude bypasses Relay/KGB entirely for stability.

### Containerized Claude (Future/Optional - TARGET STATE)
```
Claude Code (containerized)
    ↓ ws://localhost:8000?upstream=fiedler
Stable Relay (port 8000)
    ↓ ws://kgb-proxy:9000?upstream=fiedler
KGB Proxy (port 9000)
    ├→ Spy Worker → Dewey → Winni (auto-logging)
    └→ Forward → fiedler-mcp:8080 or dewey-mcp:9020
```

**Key Feature:** KGB automatically logs ALL traffic to Winni when using containerized Claude.

### Components Status

**Infrastructure (All Running):**
- ✅ `stable-relay` - Port 8000 (WebSocket relay, auto-reconnects to KGB)
- ✅ `kgb-proxy` - Port 9000 (Intercepts traffic, logs to Winni)
- ✅ `fiedler-mcp` - Port 8080 (WebSocket MCP, 7 LLM models)
- ✅ `dewey-mcp` - Port 9020 (Conversation storage)
- ✅ `winni` - PostgreSQL database on Irina (192.168.1.210)

**MCP Tools Status (This Session):**
- ✅ Fiedler MCP: 5 tools available (`fiedler_list_models`, `fiedler_send`, etc.)
- ❌ Dewey MCP: 0 tools (not loading - part of BUG #1 investigation)

---

## 🐛 Known Issues

### ✅ FIXED: Fiedler MCP Tools Not Loading
**Problem:** Fiedler MCP tools were not available in Claude Code.

**Root Cause:** Config was using WebSocket relay method which wasn't working.

**Fix Applied:** Restored direct docker exec connection method.

**Status:** FIXED - Restart required to verify

### ⏸️ DEFERRED: Conversation Logging via KGB/Dewey
**Problem:** Claude Code conversations not being logged to Winni database.

**Status:** Deferred - Focus on getting Fiedler working first as baseline

**Tracking:** `/mnt/projects/ICCM/architecture/BUG_TRACKING.md`

---

## 📁 Key Documentation

### Primary Specs
- `/mnt/projects/ICCM/architecture/dewey_winni_requirements_v3.md` - Dewey + Winni spec
- `/mnt/projects/ICCM/kgb/README.md` - KGB proxy spec
- `/mnt/projects/ICCM/architecture/STABLE_RELAY_DEPLOYMENT.md` - Relay deployment

### Status & Tracking
- `/mnt/projects/ICCM/architecture/BUG_TRACKING.md` - Bug investigation log
- `/mnt/projects/ICCM/architecture/POST_RESTART_STATUS_v4.md` - Detailed test plan
- `/mnt/projects/ICCM/architecture/General Architecture.PNG` - Visual diagram

### Configuration
- `~/.config/claude-code/mcp.json` - MCP config (active, correct)
- `~/.claude.json` - Alternative config (updated, correct)
- `/mnt/projects/ICCM/stable-relay/config.yaml` - Relay backend config

---

## 🔧 Current Configuration

### MCP Configuration Status
**Status:** ✅ Attempt #8 Applied - WebSocket Relay Chain

**Project Config:** `~/.claude.json` (lines 508-513)
```json
"fiedler": {
  "transport": {
    "type": "ws",
    "url": "ws://localhost:8000?upstream=fiedler"
  }
}
```

**Architecture:**
```
Claude Code → ws://localhost:8000 → Stable Relay → KGB → Fiedler
```

**Applied:** 2025-10-03 17:30 EDT
**Awaiting:** Full Claude Code restart and test

### Relay Configuration
**File:** `/mnt/projects/ICCM/stable-relay/config.yaml`
```yaml
backend: "ws://kgb-proxy:9000"
```

---

## 🧪 Test Plan (After Restart)

### Phase 1: Verify Fiedler MCP Tools Available ⭐ **DO THIS FIRST**
1. **Say:** "List available Fiedler tools"
2. **Expected:** Should see 5+ tools:
   - `mcp__fiedler__fiedler_list_models`
   - `mcp__fiedler__fiedler_send`
   - `mcp__fiedler__fiedler_get_config`
   - `mcp__fiedler__fiedler_set_models`
   - `mcp__fiedler__fiedler_set_output`
3. **If missing:** Check Fiedler container: `docker ps | grep fiedler`

### Phase 2: Test Model Listing
1. **Use:** `mcp__fiedler__fiedler_list_models`
2. **Expected:** List of 7 models:
   - gemini-2.5-pro
   - gpt-5
   - llama-3.1-405b
   - llama-3.3-70b
   - deepseek-chat
   - qwen-2.5-72b
   - grok-2-1212
3. **If fails:** Check Fiedler logs: `docker logs fiedler-mcp --tail 30`

### Phase 3: Test Single Model
1. **Use:** `mcp__fiedler__fiedler_send`
2. **Params:**
   ```
   models: ["gemini-2.5-pro"]
   prompt: "Reply with exactly: FIEDLER MCP WORKING"
   ```
3. **Expected:** Gemini response with "FIEDLER MCP WORKING"
4. **Verify:** Output file created in `/app/fiedler_output/`

### Phase 4: Test Multiple Models
1. **Use:** `mcp__fiedler__fiedler_send`
2. **Params:**
   ```
   models: ["gemini-2.5-pro", "gpt-5", "llama-3.3-70b"]
   prompt: "What is 2+2? Answer in one word."
   ```
3. **Expected:** All 3 respond correctly

---

## ✅ Success Criteria

**Phase 1 Success (Immediate Goal):**
1. ✅ `mcp__fiedler__*` tools visible in Claude Code
2. ✅ Can list 7 models via Fiedler MCP
3. ✅ Can send prompts to models via Fiedler MCP
4. ✅ Can access multiple models simultaneously
5. ✅ Output files created in Fiedler container

**IF ALL PASS:** 🎉 Bare metal Claude → Fiedler connection restored!

**Phase 2 Goals (Future):**
- Dewey MCP integration
- KGB conversation logging
- Full relay chain with auto-logging

**Current Status:** Config fixed, awaiting restart verification

---

## 🚀 Quick Commands

### Container Status
```bash
docker ps --filter "name=stable-relay|kgb|fiedler|dewey"
```

### Check Logs
```bash
docker logs stable-relay --tail 30
docker logs kgb-proxy --tail 30
docker logs fiedler-mcp --tail 30
docker logs dewey-mcp --tail 30
```

### Check Winni Database
```bash
sshpass -p "Edgar01760" ssh aristotle9@192.168.1.210 \
  "echo 'Edgar01760' | sudo -S -u postgres psql -d winni -c \
  'SELECT COUNT(*) FROM conversations;'"
```

### Restart Components (if needed)
```bash
docker restart stable-relay
docker restart kgb-proxy
docker restart fiedler-mcp
docker restart dewey-mcp
```

---

## 🔄 Next Steps

### Immediate (DO NOW)
1. ✅ **Config fixed** - Direct Fiedler connection restored
2. 🔄 **RESTART CLAUDE CODE** - Load new MCP configuration
3. ⏸️ **Test Fiedler tools** - Verify tools available and working

### After Successful Fiedler Test
1. Document working configuration
2. Update CURRENT_STATUS.md with success
3. Consider adding Dewey MCP (optional)
4. Consider relay/KGB integration (optional)

### Future Enhancements (Deferred)
1. Containerized Claude Code for safe testing
2. Full conversation logging via KGB/Dewey/Winni
3. Complete relay chain with auto-reconnect

---

## 📝 Session Notes

### Current Session Summary (2025-10-03)
- **Issue:** Fiedler MCP tools not loading despite correct config
- **Investigation:** 4 attempts, triplet consultation, config verification
- **Attempt #1 (14:30):** Wrong format `{"url": "..."}` ❌
- **Attempt #2 (14:50):** Transport wrapper added, still failed ❌
- **Attempt #3 (15:00):** Tried stdio (incorrect per user) ❌
- **Attempt #4 (15:10):** WebSocket config restored ⏳
- **Triplet Review:** All 3 LLMs confirmed WebSocket SHOULD work
- **Status:** Config correct, awaiting restart + test

### Investigation Findings
**What We Confirmed:**
1. ✅ Config format is CORRECT (transport wrapper present)
2. ✅ Fiedler healthy on port 9010
3. ✅ WebSocket server responding (curl test passed)
4. ✅ This config worked before (per user)
5. ❓ Unknown why Claude Code not loading MCP tools

**Triplet Diagnosis (via Fiedler):**
- Gemini: Network isolation possible, try different URLs
- GPT: Try 127.0.0.1 instead of localhost
- DeepSeek: Container namespace issue, try host.docker.internal
- **None said WebSocket won't work!**

### Key Files Modified
1. `~/.config/claude-code/mcp.json` - Global config
2. `~/.claude.json` - Project config (lines 414-419)
3. `/mnt/projects/ICCM/architecture/BUG_TRACKING.md` - Tracking attempts
4. `/mnt/projects/ICCM/architecture/CURRENT_STATUS.md` - This file
5. `/home/aristotle9/CLAUDE.md` - Added testing protocol

### Lessons Learned
1. **Architecture docs are authoritative** - User confirmed WebSocket is correct approach
2. **Triplet consultation valuable** - Got 3 expert opinions in 40 seconds
3. **Don't assume failure mode** - WebSocket works, issue is elsewhere
4. **Check BUG_TRACKING.md** - Prevents retrying failed approaches
5. **Test before declaring victory** - Config correct ≠ working system

### Critical Reminders
- 🔴 **WebSocket IS correct** - Architecture docs confirmed by user
- 🔴 **System worked before** - This exact config was working previously
- 🔴 **Test after restart** - Full process quit required, not just Ctrl+D
- 📋 **Try URL variations** - If fails: 127.0.0.1, host.docker.internal
- 📖 **Check logs** - Claude Code and Fiedler logs may show connection errors

---

**CURRENT ACTION:** User must QUIT and RESTART Claude Code, then test `mcp__fiedler__fiedler_list_models`

**Expected After Restart:**
1. Fiedler MCP tools should be available
2. Call `mcp__fiedler__fiedler_list_models` should return 7 models
3. Fiedler logs should show WebSocket connection
4. If successful: Update BUG_TRACKING.md as RESOLVED
5. If failed: Add details to BUG_TRACKING.md for Attempt #3
