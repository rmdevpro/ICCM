const WebSocket = require('ws');

const GODOT_URL = process.env.GODOT_URL || 'ws://godot-mcp:9060';
const LOGGING_ENABLED = (process.env.LOGGING_ENABLED || 'true').toLowerCase() === 'true';
const COMPONENT_NAME = 'playfair';

/**
 * Send log to Godot via MCP logger_log tool
 * Non-blocking - fails silently on error
 */
async function logToGodot(level, message, data = null, traceId = null) {
  if (!LOGGING_ENABLED) return;

  try {
    const ws = new WebSocket(GODOT_URL, { handshakeTimeout: 1000 });

    await new Promise((resolve, reject) => {
      ws.on('open', () => resolve());
      ws.on('error', reject);
      setTimeout(() => reject(new Error('Connection timeout')), 1000);
    });

    const request = {
      jsonrpc: '2.0',
      method: 'tools/call',
      params: {
        name: 'logger_log',
        arguments: {
          level: level.toUpperCase(),
          message,
          component: COMPONENT_NAME,
          data,
          trace_id: traceId
        }
      },
      id: 1
    };

    ws.send(JSON.stringify(request));

    // Wait for response or timeout
    await Promise.race([
      new Promise(resolve => ws.on('message', resolve)),
      new Promise((_, reject) => setTimeout(() => reject(new Error('Response timeout')), 1000))
    ]);

    ws.close();
  } catch (err) {
    // Silently fail - logging should never break the application
    // Fallback to console for local debugging
    if (process.env.DEBUG_LOGGING === 'true') {
      console.error(`[Godot fallback] ${level.toUpperCase()}: ${message}`, data || '');
    }
  }
}

module.exports = {
  trace: (data, message, traceId) => logToGodot('trace', message || '', data, traceId),
  debug: (data, message, traceId) => logToGodot('debug', message || '', data, traceId),
  info: (data, message, traceId) => logToGodot('info', message || '', data, traceId),
  warn: (data, message, traceId) => logToGodot('warn', message || '', data, traceId),
  error: (data, message, traceId) => logToGodot('error', message || '', data, traceId),
};
