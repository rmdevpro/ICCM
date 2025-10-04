#!/usr/bin/env node
/**
 * Marco - Internet Gateway Service for ICCM
 *
 * WebSocket MCP server that bridges Playwright browser automation
 * to multiple clients via a FIFO queue to a single browser instance.
 *
 * Architecture: WebSocket Server → stdio Bridge → Playwright MCP → Chromium
 */

const http = require('http');
const WebSocket = require('ws');
const { spawn } = require('child_process');
const crypto = require('crypto');

// ============================================================================
// Configuration from Environment Variables
// ============================================================================

const CONFIG = {
  port: parseInt(process.env.MARCO_PORT || '8030', 10),
  browserType: process.env.BROWSER_TYPE || 'chromium',
  headless: process.env.HEADLESS !== 'false',
  logLevel: process.env.LOG_LEVEL || 'info',
  maxRestarts: 3,
  restartBackoffMs: 1000,
  healthCheckTimeoutMs: 5000,
  stableRuntimeMs: 10000,
};

const LOG_LEVELS = { debug: 0, info: 1, warn: 2, error: 3 };
const CURRENT_LOG_LEVEL = LOG_LEVELS[CONFIG.logLevel] || 1;

// ============================================================================
// Global State
// ============================================================================

let server;
let wss;
let playwrightProcess = null;
let restartAttempts = 0;
let isShuttingDown = false;

const serverStartTime = Date.now();
let lastSubprocessActivity = Date.now();

// Request queue (FIFO)
const requestQueue = [];
let isProcessingQueue = false;

// Pending requests waiting for responses: requestId -> { clientId, resolve, reject, startTime }
const pendingRequests = new Map();

// Connected clients: clientId -> { ws, contexts: Set<contextId> }
const clients = new Map();

// ============================================================================
// Structured JSON Logger
// ============================================================================

function log(level, message, context = {}) {
  const levelNum = LOG_LEVELS[level] || 0;
  if (levelNum >= CURRENT_LOG_LEVEL) {
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      service: 'marco',
      message,
      ...context,
    };
    console.log(JSON.stringify(entry));
  }
}

// ============================================================================
// Playwright Subprocess Management
// ============================================================================

function startPlaywrightSubprocess() {
  if (isShuttingDown) {
    return;
  }

  if (restartAttempts >= CONFIG.maxRestarts) {
    log('error', 'Max restart attempts reached, Playwright subprocess will not restart', {
      restartAttempts,
      maxRestarts: CONFIG.maxRestarts,
    });
    return;
  }

  log('info', 'Starting Playwright MCP subprocess', {
    command: 'npx',
    args: ['@playwright/mcp@1.43.0'],
    attempt: restartAttempts + 1,
    maxAttempts: CONFIG.maxRestarts,
  });

  playwrightProcess = spawn('npx', ['@playwright/mcp@1.43.0'], {
    stdio: ['pipe', 'pipe', 'pipe'],
    env: {
      ...process.env,
      PW_BROWSER_TYPE: CONFIG.browserType,
      PW_HEADLESS: CONFIG.headless ? '1' : '0',
    },
  });

  let stdoutBuffer = '';

  // Handle stdout: parse newline-delimited JSON-RPC messages
  playwrightProcess.stdout.on('data', (data) => {
    lastSubprocessActivity = Date.now();
    stdoutBuffer += data.toString();

    // Process complete messages (newline-delimited)
    let newlineIndex;
    while ((newlineIndex = stdoutBuffer.indexOf('\n')) !== -1) {
      const messageLine = stdoutBuffer.slice(0, newlineIndex).trim();
      stdoutBuffer = stdoutBuffer.slice(newlineIndex + 1);

      if (messageLine) {
        handlePlaywrightMessage(messageLine);
      }
    }
  });

  // Handle stderr
  playwrightProcess.stderr.on('data', (data) => {
    log('warn', 'Playwright subprocess stderr', { data: data.toString().trim() });
  });

  // Handle process exit
  playwrightProcess.on('exit', (code, signal) => {
    log('warn', 'Playwright subprocess exited', { code, signal, restartAttempts });
    playwrightProcess = null;

    // Reject all pending requests
    pendingRequests.forEach(({ reject, clientId }) => {
      reject(new Error('Playwright subprocess terminated'));
    });
    pendingRequests.clear();

    // Attempt restart with exponential backoff
    if (!isShuttingDown && restartAttempts < CONFIG.maxRestarts) {
      restartAttempts++;
      const delay = CONFIG.restartBackoffMs * Math.pow(2, restartAttempts - 1);
      log('info', 'Scheduling Playwright subprocess restart', { delay, attempt: restartAttempts });
      setTimeout(startPlaywrightSubprocess, delay);
    }
  });

  playwrightProcess.on('error', (err) => {
    log('error', 'Failed to spawn Playwright subprocess', { error: err.message });
  });

  // Reset restart attempts after stable operation
  setTimeout(() => {
    if (playwrightProcess && !playwrightProcess.killed) {
      log('info', 'Playwright subprocess stable, resetting restart counter');
      restartAttempts = 0;
    }
  }, CONFIG.stableRuntimeMs);
}

// ============================================================================
// Playwright Message Handling
// ============================================================================

function handlePlaywrightMessage(messageLine) {
  let message;
  try {
    message = JSON.parse(messageLine);
  } catch (err) {
    log('error', 'Failed to parse Playwright message', { messageLine, error: err.message });
    return;
  }

  log('debug', 'Received message from Playwright subprocess', { message });

  // Handle responses (have an id)
  if (message.id !== undefined && message.id !== null) {
    const pending = pendingRequests.get(message.id);
    if (pending) {
      const { clientId, resolve, startTime, method } = pending;
      pendingRequests.delete(message.id);

      // Get client object
      const client = clients.get(clientId);

      // Track context creation (CRITICAL FIX: Gemini/DeepSeek/GPT-4o-mini consensus)
      if (client && method === 'browser.newContext' && message.result && message.result.guid) {
        client.contexts.add(message.result.guid);
        log('debug', 'Tracking new context for client', { clientId, contextId: message.result.guid });
      }

      // Send response to client
      if (client && client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(JSON.stringify(message));

        // Log tool invocation completion (with tool name)
        const duration = Date.now() - startTime;
        log('info', 'Tool invocation completed', {
          event: 'tool_invocation',
          tool_name: method,
          client_id: clientId,
          duration_ms: duration,
          status: message.error ? 'error' : 'success',
        });
      }

      resolve();
      processNextRequest(); // Process next queued request
    } else {
      log('debug', 'Received response for unknown request ID (possibly cleanup request)', { id: message.id });
    }
  }
  // Handle requests from subprocess (not expected in Phase 1)
  else if (message.method) {
    log('warn', 'Received unexpected request from subprocess', { message });
    const errorResponse = {
      jsonrpc: '2.0',
      id: message.id,
      error: {
        code: -32601,
        message: 'Method not supported in multiplexed mode',
      },
    };
    if (playwrightProcess && !playwrightProcess.killed) {
      playwrightProcess.stdin.write(JSON.stringify(errorResponse) + '\n');
    }
  }
  // Handle notifications (no id) - broadcast to all clients
  else {
    log('debug', 'Broadcasting notification to all clients', { notification: message });
    wss.clients.forEach((ws) => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(message));
      }
    });
  }
}

// ============================================================================
// FIFO Request Queue Processing
// ============================================================================

function processNextRequest() {
  if (isProcessingQueue || requestQueue.length === 0 || !playwrightProcess || playwrightProcess.killed) {
    isProcessingQueue = false;
    return;
  }

  isProcessingQueue = true;
  const { clientId, request } = requestQueue.shift();

  // Check if client is still connected
  const client = clients.get(clientId);
  if (!client || client.ws.readyState !== WebSocket.OPEN) {
    log('debug', 'Skipping request from disconnected client', { clientId, requestId: request.id });
    isProcessingQueue = false;
    processNextRequest();
    return;
  }

  try {
    const requestId = request.id;
    const startTime = Date.now();

    // Track context creation: wrap resolve to capture context ID
    let resolve = () => {};
    let reject = (err) => {
      log('error', 'Request failed', { clientId, requestId, error: err.message });
      if (client.ws.readyState === WebSocket.OPEN) {
        client.ws.send(JSON.stringify({
          jsonrpc: '2.0',
          id: requestId,
          error: {
            code: -32603,
            message: 'Internal error',
            data: { originalError: err.message },
          },
        }));
      }
    };

    // Intercept browser.newContext to track created contexts
    if (request.method === 'browser.newContext') {
      const originalResolve = resolve;
      resolve = () => {
        // Context ID will be in the response, handled when we receive it
        originalResolve();
      };
    }

    // Store pending request with method for context tracking
    if (requestId !== undefined && requestId !== null) {
      pendingRequests.set(requestId, { clientId, resolve, reject, startTime, method: request.method });
    }

    // Send to Playwright subprocess
    const requestStr = JSON.stringify(request);
    log('debug', 'Sending request to Playwright subprocess', {
      clientId,
      requestId,
      method: request.method,
    });

    playwrightProcess.stdin.write(requestStr + '\n');
  } catch (err) {
    log('error', 'Failed to process request', { error: err.message, request });
    isProcessingQueue = false;
    processNextRequest();
  }
}

function enqueueRequest(clientId, request) {
  requestQueue.push({ clientId, request });
  log('debug', 'Request enqueued', {
    clientId,
    requestId: request.id,
    method: request.method,
    queueLength: requestQueue.length,
  });

  if (!isProcessingQueue) {
    processNextRequest();
  }
}

// ============================================================================
// WebSocket Connection Handling
// ============================================================================

function handleClientConnection(ws) {
  const clientId = crypto.randomUUID();
  clients.set(clientId, { ws, contexts: new Set() });

  log('info', 'Client connected', { clientId, totalClients: clients.size });

  ws.on('message', (data) => {
    try {
      const message = JSON.parse(data.toString());
      log('debug', 'Received message from client', { clientId, message });

      // Enqueue request
      enqueueRequest(clientId, message);
    } catch (err) {
      log('error', 'Failed to parse client message', { clientId, error: err.message });
      ws.send(JSON.stringify({
        jsonrpc: '2.0',
        id: null,
        error: {
          code: -32700,
          message: 'Parse error',
          data: { originalError: err.message },
        },
      }));
    }
  });

  ws.on('close', () => {
    cleanupClient(clientId);
  });

  ws.on('error', (err) => {
    log('error', 'WebSocket error', { clientId, error: err.message });
  });
}

function cleanupClient(clientId) {
  const client = clients.get(clientId);
  if (!client) return;

  log('info', 'Client disconnected, cleaning up contexts', {
    clientId,
    contextsToCleanup: client.contexts.size,
    totalClients: clients.size - 1,
  });

  // Queue context dispose requests for cleanup (best effort)
  // NOTE: Playwright MCP uses 'context.dispose' not 'context.close'
  client.contexts.forEach((contextId) => {
    const disposeRequest = {
      jsonrpc: '2.0',
      id: crypto.randomUUID(),
      method: 'context.dispose',
      params: { guid: contextId },
    };

    // Enqueue cleanup request (no response tracking)
    requestQueue.push({ clientId, request: disposeRequest });
  });

  clients.delete(clientId);

  // Trigger queue processing if cleanup requests were added
  if (client.contexts.size > 0 && !isProcessingQueue) {
    processNextRequest();
  }
}

// ============================================================================
// Health Check Endpoint
// ============================================================================

function handleHealthCheck(req, res) {
  const uptime = (Date.now() - serverStartTime) / 1000;
  const isSubprocessAlive = playwrightProcess !== null && !playwrightProcess.killed;
  const timeSinceLastActivity = Date.now() - lastSubprocessActivity;
  const isSubprocessResponsive = isSubprocessAlive && timeSinceLastActivity < CONFIG.healthCheckTimeoutMs;

  const status = isSubprocessAlive && isSubprocessResponsive ? 'healthy' : 'degraded';
  const httpStatus = status === 'healthy' ? 200 : 503;

  const health = {
    status,
    browser: isSubprocessAlive ? 'alive' : 'dead',
    uptime_seconds: parseFloat(uptime.toFixed(2)),
    playwright_subprocess: isSubprocessResponsive ? 'responsive' : 'unresponsive',
  };

  res.writeHead(httpStatus, { 'Content-Type': 'application/json' });
  res.end(JSON.stringify(health));

  log('debug', 'Health check', { health, httpStatus });
}

// ============================================================================
// HTTP and WebSocket Server Setup
// ============================================================================

// Create HTTP server for health check
server = http.createServer((req, res) => {
  if (req.url === '/health' && req.method === 'GET') {
    handleHealthCheck(req, res);
  } else {
    res.writeHead(404, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ error: 'Not Found' }));
  }
});

// Create WebSocket server (no separate HTTP server)
wss = new WebSocket.Server({ noServer: true });

// Handle WebSocket upgrade on same port as HTTP
server.on('upgrade', (request, socket, head) => {
  wss.handleUpgrade(request, socket, head, (ws) => {
    wss.emit('connection', ws, request);
  });
});

wss.on('connection', handleClientConnection);

// ============================================================================
// Graceful Shutdown
// ============================================================================

function gracefulShutdown(signal) {
  if (isShuttingDown) return;
  isShuttingDown = true;

  log('info', 'Graceful shutdown initiated', { signal });

  // Close all WebSocket connections
  wss.clients.forEach((ws) => {
    if (ws.readyState === WebSocket.OPEN) {
      ws.close();
    }
  });

  // Close WebSocket server
  wss.close(() => {
    log('info', 'WebSocket server closed');
  });

  // Close HTTP server
  server.close(() => {
    log('info', 'HTTP server closed');

    // Terminate Playwright subprocess
    if (playwrightProcess) {
      log('info', 'Terminating Playwright subprocess');
      playwrightProcess.kill('SIGTERM');

      // Force kill after timeout
      setTimeout(() => {
        if (playwrightProcess && !playwrightProcess.killed) {
          log('warn', 'Force killing Playwright subprocess');
          playwrightProcess.kill('SIGKILL');
        }
        process.exit(0);
      }, 2000);
    } else {
      process.exit(0);
    }
  });

  // Force exit after timeout
  setTimeout(() => {
    log('error', 'Graceful shutdown timeout, forcing exit');
    process.exit(1);
  }, 5000);
}

process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
process.on('SIGINT', () => gracefulShutdown('SIGINT'));

// ============================================================================
// Application Startup
// ============================================================================

server.listen(CONFIG.port, () => {
  log('info', 'Marco server started', {
    port: CONFIG.port,
    browserType: CONFIG.browserType,
    headless: CONFIG.headless,
    logLevel: CONFIG.logLevel,
  });

  startPlaywrightSubprocess();
});
