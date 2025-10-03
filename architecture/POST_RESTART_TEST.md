# Post-Restart Test Plan
**Created:** 2025-10-02 19:55 EDT
**Updated:** 2025-10-02 21:05 EDT

## TL;DR - What Was Fixed
1. **Restart #1:** Fixed MCP config format (wrong JSON structure)
2. **Restart #2:** Fixed trust dialog (was blocking ALL MCP servers)
3. **Restart #3:** Partial success - Fiedler working ✅, Dewey not loading ❌

## Current Status (Session 5)
- ✅ Fiedler: 5 tools loaded, connected via relay→KGB→Winni
- ❌ Dewey: 0 tools loaded, connection failing
- 🔍 Investigating: Why dewey route not working while fiedler works

---

## Quick Verification (30 seconds)

### 1. Check if tools loaded
```
Say: "list available fiedler tools"
Expected: See mcp__fiedler__* tools
```

### 2. Check connection logs
```bash
docker logs stable-relay --tail 20
# Should see: "Client X connected" with recent timestamps

docker logs kgb-proxy --tail 30
# Should see: "Spy X recruited for upstream: fiedler"
```

### 3. Test Fiedler
```
Use: mcp__fiedler__fiedler_list_models
Expected: List of 7 models (Gemini, GPT-5, Llama, DeepSeek, Qwen, Grok)
```

---

## Full Test: Send Message Through Chain

### Test Command
```
mcp__fiedler__fiedler_send
- models: ["gemini-2.5-pro"]
- prompt: "Say 'Hello from Gemini via Fiedler!' in exactly 5 words."
- output_file: /tmp/gemini_test.txt
```

### What Should Happen
1. **Claude Code** → sends via MCP to stable-relay:8000
2. **stable-relay** → routes to kgb-proxy:9000 (transparent relay)
3. **kgb-proxy** → intercepts, logs to Winni, forwards to fiedler-mcp:8080
4. **fiedler-mcp** → sends to Gemini 2.5 Pro API
5. **Response** → flows back through chain
6. **KGB** → logs conversation with UUID to Winni database

### Verify Logging
```bash
# Check KGB captured the conversation
docker logs kgb-proxy --tail 50 | grep -E "(conversation:|Spy.*logging)"

# Expected output:
# INFO - Spy X logging to conversation: <UUID>
# INFO - Logged user message (123 bytes)
# INFO - Logged assistant response (456 bytes)
```

---

## Success Criteria

✅ **MCP Tools Available**
- fiedler_list_models works
- fiedler_send works
- fiedler_get_config works

✅ **Connection Chain Works**
- stable-relay logs show Claude Code connections
- kgb-proxy logs show spy recruitment
- fiedler-mcp logs show requests received

✅ **Message Flow Works**
- Can send prompts to any of 7 models
- Responses return successfully
- Output written to specified files

✅ **Logging Works**
- KGB logs show conversation UUIDs
- Messages logged to Winni database
- Both user and assistant messages captured

---

## If Something Fails

**Tools still not available:**
- Check: `python3 -c "import json; print(json.load(open('/home/aristotle9/.claude.json'))['projects']['/home/aristotle9']['hasTrustDialogAccepted'])"`
- Should output: `True`

**Connections not happening:**
- Check: `docker ps | grep -E "(stable-relay|kgb-proxy|fiedler-mcp)"`
- All should show "Up" and "healthy"

**Messages failing:**
- Check API keys: `/mnt/projects/keys.txt`
- Check fiedler logs: `docker logs fiedler-mcp --tail 50`

---

## Next Steps After Success

1. **Test all 7 models** - Verify each LLM responds correctly
2. **Verify Winni integration** - Check database has conversation records
3. **Test Dewey tools** - Conversation retrieval and management
4. **Stress test** - Send multiple concurrent requests

See `/mnt/projects/ICCM/architecture/RESUME_HERE.md` for full context.
