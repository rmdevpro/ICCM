# Triplet Consultation: Playfair Mermaid Puppeteer/Chromium Sandbox Issue

## Problem Statement

Playfair MCP server's Mermaid diagram rendering fails with "ENGINE_CRASH" error. The root cause is that `mmdc` (Mermaid CLI) cannot launch Chromium/Puppeteer due to sandbox restrictions in the Docker container.

## Context

### What Works
- **Manual `mmdc` execution with puppeteer config file**: Successfully renders Mermaid diagrams when run directly in container with explicit puppeteer config file
- **DOT/Graphviz rendering**: Works perfectly through Playfair
- **Chrome-headless-shell installation**: Properly installed and accessible at `/home/appuser/chrome-headless-shell`
- **Environment variables**: `PUPPETEER_EXECUTABLE_PATH` and `PUPPETEER_ARGS` are set correctly in Dockerfile

### What Fails
- **Mermaid rendering via MCP tool**: All attempts fail with generic "ENGINE_CRASH" error
- **Error visibility**: No error details reach logs - errors are swallowed before being logged

### Manual Test Success
```bash
# This works:
docker exec playfair-mcp mmdc -i /tmp/test.mmd -o /tmp/test.svg -p /tmp/pconfig.json
# Where pconfig.json contains: {"args": ["--no-sandbox", "--disable-setuid-sandbox"]}
```

### Current Code Approach
```javascript
// engines/mermaid.js
const puppeteerConfig = {
    "args": ["--no-sandbox", "--disable-setuid-sandbox"]
};
await fs.writeFile(puppeteerConfigPath, JSON.stringify(puppeteerConfig));

const args = [
    '-i', tempInputDir,
    '-o', tempOutputDir,
    '-c', tempConfigPath,
    '-p', puppeteerConfigPath,  // Puppeteer config file
    '-w', '1920'
];

await execFileAsync('mmdc', args, { timeout: RENDER_TIMEOUT, env });
```

## What Has Been Tried

1. ✅ **Verified Puppeteer config file creation** - Code creates correct JSON file
2. ✅ **Added environment variables to execFileAsync** - `PUPPETEER_ARGS` passed in env
3. ✅ **Confirmed Chrome installation** - chrome-headless-shell v143 installed and executable
4. ✅ **Manual testing** - Direct `mmdc` calls work with same config
5. ❌ **Debug logging** - Attempted to capture error details but errors occur before catch blocks
6. ❌ **Multiple rebuilds** - No improvement after code changes

## Dockerfile Relevant Sections

```dockerfile
# Install Chromium dependencies
RUN apt-get install -y libgbm1 libasound2t64 libnss3 libatk1.0-0 \
    libatk-bridge2.0-0 libx11-xcb1 libdrm2 libxkbcommon0 \
    libxcomposite1 libxdamage1 libxrandr2 libgtk-3-0 libpango-1.0-0

# Install chrome-headless-shell for appuser
USER appuser
RUN npx -y @puppeteer/browsers install chrome-headless-shell@latest \
    --path /home/appuser/.cache/puppeteer

# Create symlink
RUN ln -s $(find /home/appuser/.cache/puppeteer -name chrome-headless-shell -type f) \
    /home/appuser/chrome-headless-shell

# Set environment variables
ENV PUPPETEER_EXECUTABLE_PATH=/home/appuser/chrome-headless-shell
ENV PUPPETEER_ARGS="--no-sandbox --disable-setuid-sandbox"
```

## Questions for Triplet Consultation

1. **Why does `mmdc` work manually but fail when called via Node.js `execFileAsync`?**
   - Is there a difference in how environment variables are inherited?
   - Does the puppeteer config file get properly passed through execFileAsync?

2. **Is the puppeteer config file approach correct for `mmdc` CLI?**
   - Should we use environment variables instead?
   - Is there a different format or location required?

3. **Are there additional Chrome/Puppeteer sandbox flags needed?**
   - The Dockerfile sets `PUPPETEER_ARGS` but `mmdc` might not respect it
   - Should we modify how mmdc launches Puppeteer?

4. **Is there a permissions/ownership issue?**
   - Container runs as `appuser` (non-root)
   - Chrome-headless-shell owned by `appuser`
   - Could there be a permission mismatch during execFileAsync?

5. **Should we investigate alternative approaches?**
   - Run `mmdc` via shell instead of execFile?
   - Use a different Mermaid rendering library?
   - Modify mmdc's Puppeteer configuration directly?

## Desired Outcome

Mermaid diagrams should render successfully through the Playfair MCP tool, just like DOT diagrams do, using the properly sandboxed Chrome/Puppeteer setup.

## Related Issues

- GitHub Issue #19: Playfair Mermaid engine crashes
- GitHub Issue #20: Playfair not logging to Godot (blocks error visibility)
