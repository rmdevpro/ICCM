const http = require('http');
const { WebSocketServer } = require('ws');
const { v4: uuidv4 } = require('uuid');
const logger = require('./utils/logger');
const McpTools = require('./mcp-tools');

const PORT = process.env.PORT || 8040;

// --- Main Application Setup ---

const server = http.createServer();
const wss = new WebSocketServer({ noServer: true });

const mcpTools = new McpTools();

// --- WebSocket Connection Handling ---

wss.on('connection', (ws, request) => {
    const clientId = uuidv4();
    ws.clientId = clientId;
    logger.info({ clientId, ip: request.socket.remoteAddress }, 'Client connected');

    ws.on('message', async (message) => {
        let request;
        try {
            request = JSON.parse(message);
            logger.debug({ clientId, request }, 'Received request');
        } catch (error) {
            logger.error({ clientId, error: error.message }, 'Invalid JSON message received');
            ws.send(JSON.stringify({ error: true, code: 'INVALID_JSON', message: 'Message must be valid JSON.' }));
            return;
        }

        const response = await handleClientRequest(request, clientId);
        
        try {
            ws.send(JSON.stringify(response));
            logger.debug({ clientId, response }, 'Sent response');
        } catch (error) {
            logger.error({ clientId, error: error.message }, 'Failed to send response');
        }
    });

    ws.on('close', () => {
        logger.info({ clientId }, 'Client disconnected');
    });

    ws.on('error', (error) => {
        logger.error({ clientId, error: error.message }, 'WebSocket error');
    });
});

// --- MCP Protocol Handler ---

async function handleClientRequest(request, clientId) {
    const { method, params, id } = request;

    switch (method) {
        case 'initialize':
            return {
                jsonrpc: '2.0',
                result: {
                    name: 'Playfair',
                    version: '1.0.0',
                    description: 'ICCM Diagram Generation Gateway',
                    capabilities: ['tools'],
                    protocol_version: '1.0'
                },
                id
            };

        case 'tools/list':
            return {
                jsonrpc: '2.0',
                result: {
                    tools: mcpTools.listTools()
                },
                id
            };

        case 'tools/call':
            if (!params || !params.name) {
                return {
                    jsonrpc: '2.0',
                    error: {
                        code: -32602,
                        message: 'Missing tool name in params.'
                    },
                    id
                };
            }

            const toolResult = await mcpTools.callTool(params.name, params.arguments, clientId);

            // Wrap tool result in MCP protocol format
            return {
                jsonrpc: '2.0',
                result: {
                    content: [{
                        type: 'text',
                        text: JSON.stringify(toolResult, null, 2)
                    }]
                },
                id
            };

        default:
            return {
                jsonrpc: '2.0',
                error: {
                    code: -32601,
                    message: `Method '${method}' is not supported.`
                },
                id
            };
    }
}

// --- HTTP Server for Health Checks ---

server.on('request', (req, res) => {
    if (req.url === '/health') {
        const healthStatus = mcpTools.getHealthStatus();
        if (healthStatus.status === 'healthy') {
            res.writeHead(200, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(healthStatus));
        } else {
            res.writeHead(503, { 'Content-Type': 'application/json' });
            res.end(JSON.stringify(healthStatus));
        }
    } else {
        // For non-WebSocket requests, respond with an error
        res.writeHead(426); // Upgrade Required
        res.end('This is a WebSocket server. Please connect using a WebSocket client.');
    }
});


// --- Server Startup ---

server.on('upgrade', (request, socket, head) => {
    // Handle WebSocket upgrade requests
    wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request);
    });
});

server.listen(PORT, () => {
    logger.info(`Playfair MCP Server listening on ws://localhost:${PORT}`);
    logger.info(`Health check available at http://localhost:${PORT}/health`);
});

process.on('SIGTERM', () => {
    logger.info('SIGTERM signal received. Shutting down gracefully.');
    server.close(() => {
        logger.info('HTTP server closed.');
        process.exit(0);
    });
});