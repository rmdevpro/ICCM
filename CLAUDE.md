# Claude Code Session Instructions

## System Information
- OS: Linux Ubuntu 24.04
- Node.js: v22.19.0 (via NVM)
- Working Directory: /home/aristotle9
- Key Projects: /mnt/projects/ICCM/
- API Keys: /mnt/projects/keys.txt
- sudo Password: Edgar01760

---

## 📚 DOCUMENTATION STRUCTURE

**ALWAYS read these on session startup:**
1. `/mnt/projects/ICCM/CURRENT_STATUS.md` - **Current work and next steps**
2. **GitHub Issues** (`gh issue list`) - **Active bugs and tasks** (https://github.com/rmdevpro/ICCM/issues)
3. `/mnt/projects/ICCM/architecture/General Architecture.PNG` - **System architecture diagram**
4. `/mnt/projects/ICCM/architecture/CURRENT_ARCHITECTURE_OVERVIEW.md` - **Protocol configuration**

**Supporting documentation:**
- Module-specific docs (README.md, requirements) - Read as needed for current work
- Git history (`git log`) - Detailed change log with all code modifications

**Process documentation:**
- `/mnt/projects/ICCM/architecture/TRIPLET_CONSULTATION_PROCESS.md` - Standard process for consulting the LLM triplet
- `/mnt/projects/ICCM/architecture/FIEDLER_DOCKER_WORKAROUND.md` - Triplet consultation when Fiedler MCP unavailable

**Documentation hierarchy:**
- **CURRENT_STATUS.md** = Where we are, what's next
- **GitHub Issues** = What's broken, what needs doing (bugs, features, tech debt)
- **CURRENT_ARCHITECTURE_OVERVIEW.md** = How it's configured
- **Git commits** = What changed (detailed technical history)

**Bug tracking workflow:**
- Start session: `gh issue list --assignee @me --state open`
- Work on issue: Reference in commits with `fixes #N` or `relates to #N`
- Issue auto-closes when commit pushed with `fixes #N`
- Query bugs: `gh issue list --label bug --state open`

---

## 🛠️ COMPONENT MODIFICATION PROTOCOL

**CRITICAL: Never break working systems**

**Before modifying ANY production system:**
1. Read README.md and specifications
2. Verify current system works as documented
3. Get user approval for changes

**During modifications:**
1. **Blue/Green Deployment** - Build changes in separate copy
2. Test thoroughly before switching
3. Keep original running until replacement verified

**After modifications:**
1. Update README.md and specifications
2. Verify documentation matches reality
3. Final system test

**VIOLATION RISKS DESTROYING WORKING SYSTEMS**

---

## ⚠️ TOOL-FIRST POLICY

**Use provided MCP tools - NEVER bypass with direct file edits**

**When tools fail:**
1. Diagnose why (logs, status, connections)
2. Fix root cause
3. Retry tool
4. Never bypass without user permission

---

## 📝 FILE EDITING

**Use native `Edit` tool** - Shows visual diff (red/green)
**NOT** `mcp__desktop-commander__edit_block` - No visual feedback

---

## 🧪 TESTING PROTOCOL

**CRITICAL: Never declare victory without testing**

**Bugs are NOT resolved until:**
1. Fix has been applied
2. System has been restarted (if required)
3. Test has been executed successfully
4. Results verified and documented

**Do NOT say "fixed" or "resolved" - say "applied fix, awaiting test"**

---

*This file ensures consistent behavior across all Claude Code sessions*
