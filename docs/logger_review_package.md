# Unified Logging Infrastructure Review Request

## Context
We are implementing a unified logging infrastructure for the ICCM project to solve a critical debugging problem: we cannot see what data is being exchanged between components, specifically between the MCP relay and backend servers.

**The Problem We're Solving:**
- BUG #13: Gates MCP tools are registered in relay but not callable from Claude Code
- We need to see EXACTLY what Gates sends vs what working servers (Dewey, Fiedler) send
- Current logging is insufficient - no visibility into tool registration data structures
- Need to compare full message exchanges to find subtle differences

## Architecture Overview

### Components
1. **Logger MCP Server** (port 9060) - Receives and stores logs to Dewey's PostgreSQL
2. **Python Client Library** (`loglib.py`) - For Python services (Dewey, Fiedler, Relay)
3. **JavaScript Client Library** (`loglib.js`) - For Node.js services (Gates)

### Storage
- Dewey's PostgreSQL database (new `logs` table)
- Full-text search on messages
- JSONB storage for structured data (entire request/response objects)

### Log Levels
- `ERROR` - Errors only
- `WARN` - Warnings + errors
- `INFO` - Key events (default)
- `DEBUG` - Detailed operation traces
- `TRACE` - Full message dumps (everything sent/received)

### MCP Tools Exposed
- `logger_log(component, level, message, data)` - Write log entry
- `logger_query(component, level, limit, start_time, end_time, search)` - Query logs
- `logger_list_components()` - List all components logging
- `logger_clear(component, before_time)` - Clear old logs
- `logger_set_level(component, level)` - Set logging level dynamically
- `logger_get_level(component)` - Get current level

## Implementation Files

### Database Schema Addition (dewey/schema.sql)
```sql
-- Logs table for unified logging infrastructure
CREATE TABLE IF NOT EXISTS logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    component TEXT NOT NULL,
    level TEXT NOT NULL CHECK (level IN ('ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE')),
    message TEXT NOT NULL,
    data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_logs_component ON logs(component);
CREATE INDEX IF NOT EXISTS idx_logs_level ON logs(level);
CREATE INDEX IF NOT EXISTS idx_logs_created_at ON logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_component_created_at ON logs(component, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_logs_message_fts ON logs USING gin(to_tsvector('english', message));
CREATE INDEX IF NOT EXISTS idx_logs_data ON logs USING gin(data);
```

### Python Client Library (logger/loglib.py)
```python
class ICCMLogger:
    """Async logging client that sends logs to Logger MCP server."""

    def __init__(self, component: str, logger_url: str = "ws://localhost:9060", default_level: str = "INFO"):
        self.component = component
        self.logger_url = logger_url
        self.current_level = default_level.upper()
        self.level_order = ['ERROR', 'WARN', 'INFO', 'DEBUG', 'TRACE']
        self._local_logger = logging.getLogger(f"iccm.{component}")
        self._ws = None

    async def _send_log(self, level: str, message: str, data: Optional[Dict[str, Any]] = None):
        """Send log entry to Logger MCP server."""
        if not self._should_log(level):
            return

        try:
            await self._ensure_connected()
            
            request = {
                "jsonrpc": "2.0",
                "id": self._message_id,
                "method": "tools/call",
                "params": {
                    "name": "logger_log",
                    "arguments": {
                        "component": self.component,
                        "level": level,
                        "message": message,
                        "data": data
                    }
                }
            }
            
            await self._ws.send(json.dumps(request))
            response = await asyncio.wait_for(self._ws.recv(), timeout=5.0)
            # ... error handling ...
        except Exception as e:
            # Fallback to local logging if remote logging fails
            self._local_logger.error(f"Failed to send log: {e}")

    def info(self, message: str, data: Optional[Dict[str, Any]] = None):
        asyncio.create_task(self._send_log("INFO", message, data))
        self._local_logger.info(f"{message} {data or ''}")
```

## Planned Integration Points

### 1. MCP Relay Integration
```python
from logger.loglib import ICCMLogger

logger = ICCMLogger('relay', default_level='TRACE')

# Log ALL WebSocket messages to/from backends
async def forward_to_backend(backend_name, request):
    logger.trace(f"→ {backend_name}", {"request": request})
    response = await backend_ws.send(request)
    logger.trace(f"← {backend_name}", {"response": response})
    return response

# Log ALL stdio messages to/from Claude Code
def handle_stdio_request(request):
    logger.trace("→ Claude Code", {"request": request})
    # ... process ...
    logger.trace("← Claude Code", {"response": response})
```

### 2. Gates Integration
```javascript
const logger = new ICCMLogger('gates', 'ws://localhost:9060', 'TRACE');

// Log tools/list response
case 'tools/list':
  const response = { jsonrpc: '2.0', id, result: { tools: TOOLS } };
  logger.trace('tools/list response', { response, tools: TOOLS });
  return response;

// Log tools/call request
case 'tools/call':
  logger.trace('tools/call request', { params });
  const result = await handleToolCall(params.name, params['arguments']);
  logger.trace('tools/call result', { result });
  return { jsonrpc: '2.0', id, result: { content: [...] } };
```

## Review Questions

### 1. Architecture & Design
- Is the overall architecture sound for debugging message exchanges?
- Should logging be synchronous or async? (Currently async with fire-and-forget)
- Is WebSocket the right protocol for logging, or should we use HTTP POST?
- Should we have local fallback if Logger MCP is unavailable?

### 2. Database Schema
- Is the `logs` table schema appropriate?
- Are we missing any critical indexes?
- Should we have retention policies / auto-cleanup?
- JSONB for full message dumps - will this scale?

### 3. Python Client Library
- Is the async implementation correct? (fire-and-forget via `asyncio.create_task`)
- Should we wait for log confirmation or is fire-and-forget acceptable?
- Error handling: fallback to local logging appropriate?
- Thread safety: is `_ws_lock` sufficient?

### 4. Integration Strategy
- Plan is to integrate into relay and Gates at TRACE level
- This will capture EVERY message exchanged
- Potential performance impact?
- Should we make logging optional/configurable?

### 5. Missing Components
- JavaScript client library (`loglib.js`) not yet implemented - review design?
- Should we add log rotation/archival?
- Should we add log aggregation/correlation (e.g., request IDs)?
- Need metrics/alerting on top of logs?

### 6. Security & Privacy
- Logs will contain full message payloads - any sensitive data concerns?
- Should we have log access controls?
- Should we sanitize/redact certain fields?

### 7. Operational Concerns
- How to handle Logger MCP downtime? (Currently falls back to local logging)
- How to prevent infinite loops if Logger itself has issues?
- Log volume estimates - how much data per hour at TRACE level?
- PostgreSQL performance with high-volume log inserts?

## Success Criteria

After implementation, we should be able to:

1. Set relay and Gates to TRACE level
2. Call `mcp__iccm__gates_list_capabilities`
3. Query logs to see:
   - Exact JSON Claude Code sends to relay
   - Exact JSON relay sends to Gates
   - Exact JSON Gates returns to relay
   - Exact JSON relay returns to Claude Code
4. Compare Gates logs to Dewey/Fiedler logs (working servers)
5. Identify the structural difference causing BUG #13

## Request

Please provide:
1. **Critical flaws** in architecture or implementation
2. **Improvements** to design or code
3. **Missing considerations** we haven't thought of
4. **Alternative approaches** if this design is fundamentally wrong
5. **Go/No-Go recommendation** - should we proceed with this implementation?

Please be thorough and critical - this is infrastructure that will be used across all ICCM components.
