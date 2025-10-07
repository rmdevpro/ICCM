#!/usr/bin/env node

/**
 * Gates Document Generation Gateway
 * WebSocket MCP Server - Markdown to ODT Conversion
 * Version: 1.0.0
 */

import { WebSocketServer } from 'ws';
import { createServer } from 'http';
import markdownIt from 'markdown-it';
import markdownItMultimdTable from 'markdown-it-multimd-table';
import markdownItAttrs from 'markdown-it-attrs';
import markdownItTaskLists from 'markdown-it-task-lists';
import { execa } from 'execa';
import PQueue from 'p-queue';
import pino from 'pino';
import { writeFile, readFile, unlink, mkdir } from 'fs/promises';
import { tmpdir } from 'os';
import { join, isAbsolute } from 'path';
import { randomUUID } from 'crypto';
import { existsSync } from 'fs';
import { WebSocket as WebSocketClient } from 'ws';

const logger = pino({
  level: process.env.LOG_LEVEL || 'info'
});

// Configuration
const PORT = process.env.GATES_PORT || 8050;
const HOST = process.env.GATES_HOST || '0.0.0.0';
const PLAYFAIR_URL = process.env.PLAYFAIR_URL || 'ws://playfair-mcp:8040';
const MAX_QUEUE_DEPTH = 10;
const CONVERSION_TIMEOUT = 120000; // 120 seconds
const DIAGRAM_TIMEOUT = 10000; // 10 seconds
const MAX_MARKDOWN_SIZE = 10 * 1024 * 1024; // 10MB
const MAX_ODT_SIZE = 50 * 1024 * 1024; // 50MB
const MAX_IMAGE_SIZE = 10 * 1024 * 1024; // 10MB

// Conversion queue (FIFO, single worker)
const conversionQueue = new PQueue({
  concurrency: 1,
  timeout: CONVERSION_TIMEOUT,
  throwOnTimeout: true
});

// Playfair connection state
let playfairClient = null;
let playfairConnected = false;
let playfairReconnectTimer = null;

// Tool definitions
const TOOLS = [
  {
    name: 'gates_create_document',
    description: 'Convert Markdown to ODT document with embedded diagrams',
    inputSchema: {
      type: 'object',
      properties: {
        markdown: {
          type: 'string',
          description: 'Markdown content to convert (use this OR input_file, not both)'
        },
        input_file: {
          type: 'string',
          description: 'Path to markdown file to convert (use this OR markdown, not both)'
        },
        metadata: {
          type: 'object',
          properties: {
            title: { type: 'string' },
            author: { type: 'string' },
            date: { type: 'string' },
            keywords: { type: 'array', items: { type: 'string' } }
          }
        },
        output_path: {
          type: 'string',
          description: 'Optional: file path for output (default: temp file in /mnt/irina_storage/files/temp/gates/)'
        }
      },
      required: []
    }
  },
  {
    name: 'gates_validate_markdown',
    description: 'Validate Markdown syntax and check for potential ODT conversion issues',
    inputSchema: {
      type: 'object',
      properties: {
        markdown: { type: 'string' }
      },
      required: ['markdown']
    }
  },
  {
    name: 'gates_list_capabilities',
    description: 'List supported Markdown features and current configuration',
    inputSchema: {
      type: 'object',
      properties: {}
    }
  }
];

/**
 * Connect to Playfair MCP server
 */
async function connectPlayfair() {
  return new Promise((resolve) => {
    try {
      playfairClient = new WebSocketClient(PLAYFAIR_URL);

      playfairClient.on('open', () => {
        logger.info('Connected to Playfair MCP server');
        playfairConnected = true;
        resolve(true);
      });

      playfairClient.on('error', (error) => {
        logger.warn({ error: error.message }, 'Playfair connection error');
        playfairConnected = false;
        schedulePlayfairReconnect();
        resolve(false);
      });

      playfairClient.on('close', () => {
        logger.info('Playfair connection closed');
        playfairConnected = false;
        schedulePlayfairReconnect();
      });
    } catch (error) {
      logger.warn({ error: error.message }, 'Failed to connect to Playfair');
      playfairConnected = false;
      resolve(false);
    }
  });
}

/**
 * Schedule Playfair reconnection
 */
function schedulePlayfairReconnect() {
  if (playfairReconnectTimer) return;

  playfairReconnectTimer = setTimeout(async () => {
    playfairReconnectTimer = null;
    logger.info('Attempting to reconnect to Playfair...');
    await connectPlayfair();
  }, 5000);
}

/**
 * Call Playfair MCP tool
 */
async function callPlayfair(toolName, args) {
  if (!playfairConnected || !playfairClient) {
    throw new Error('Playfair not connected');
  }

  return new Promise((resolve, reject) => {
    const requestId = randomUUID();
    const timeout = setTimeout(() => {
      reject(new Error('Playfair request timeout'));
    }, DIAGRAM_TIMEOUT);

    const messageHandler = (data) => {
      try {
        const response = JSON.parse(data.toString());
        if (response.id === requestId) {
          clearTimeout(timeout);
          playfairClient.off('message', messageHandler);

          if (response.error) {
            reject(new Error(response.error.message || 'Playfair error'));
          } else {
            resolve(response.result);
          }
        }
      } catch (error) {
        // Ignore parse errors for other messages
      }
    };

    playfairClient.on('message', messageHandler);

    const request = {
      jsonrpc: '2.0',
      id: requestId,
      method: 'tools/call',
      params: {
        name: toolName,
        arguments: args
      }
    };

    playfairClient.send(JSON.stringify(request));
  });
}

/**
 * Create Playfair plugin for markdown-it
 */
function createPlayfairPlugin() {
  return (md) => {
    const defaultFenceRenderer = md.renderer.rules.fence || function(tokens, idx, options, env, slf) {
      return slf.renderToken(tokens, idx, options);
    };

    md.renderer.rules.fence = function(tokens, idx, options, env, slf) {
      const token = tokens[idx];
      const info = token.info.trim();

      // Check for playfair-* fence types
      if (info.startsWith('playfair-')) {
        const format = info.replace('playfair-', '');
        const content = token.content;

        // Store for async processing
        const placeholderId = `PLAYFAIR_${randomUUID()}`;
        env.playfairDiagrams = env.playfairDiagrams || [];
        env.playfairDiagrams.push({
          id: placeholderId,
          format,
          content
        });

        // Return placeholder
        return `<img id="${placeholderId}" alt="Playfair diagram" />`;
      }

      return defaultFenceRenderer(tokens, idx, options, env, slf);
    };
  };
}

/**
 * Create image processing plugin for markdown-it
 * Handles file paths, URLs, and base64 data in markdown images
 */
function createImagePlugin() {
  return (md) => {
    const defaultImageRenderer = md.renderer.rules.image || function(tokens, idx, options, env, slf) {
      return slf.renderToken(tokens, idx, options);
    };

    md.renderer.rules.image = function(tokens, idx, options, env, slf) {
      const token = tokens[idx];
      const src = token.attrGet('src');

      if (src) {
        // Store image for async processing
        const placeholderId = `IMAGE_${randomUUID()}`;
        env.markdownImages = env.markdownImages || [];
        env.markdownImages.push({
          id: placeholderId,
          src,
          alt: token.content || 'Image'
        });

        // Return placeholder
        return `<img id="${placeholderId}" alt="${token.content || 'Image'}" />`;
      }

      return defaultImageRenderer(tokens, idx, options, env, slf);
    };
  };
}

/**
 * Process Playfair diagrams in HTML
 */
async function processPlayfairDiagrams(html, diagrams) {
  if (!diagrams || diagrams.length === 0) {
    return { html, warnings: [] };
  }

  const warnings = [];
  let processedHtml = html;

  for (const diagram of diagrams) {
    try {
      const result = await callPlayfair('playfair_create_diagram', {
        content: diagram.content,
        format: diagram.format,
        output_format: 'png',
        output_mode: 'base64',
        theme: 'professional'
      });

      // Extract image from Playfair result
      const imageData = result?.content?.[0]?.text;
      let base64Data = null;

      if (imageData) {
        try {
          const jsonData = JSON.parse(imageData);

          // Playfair can return base64 data, file path, or both depending on output_mode
          if (jsonData?.result?.data) {
            // Direct base64 data from Playfair
            base64Data = jsonData.result.data;
            logger.info({ diagramId: diagram.id }, 'Using base64 data from Playfair');
          } else if (jsonData?.result?.path) {
            // Fallback: Read from file path
            const imagePath = jsonData.result.path;
            const containerPath = imagePath.replace('/mnt/projects/ICCM/irina_storage_test/files', '/mnt/irina_storage/files');

            try {
              const imageBuffer = await readFile(containerPath);
              base64Data = imageBuffer.toString('base64');
              logger.info({ imagePath: containerPath, diagramId: diagram.id }, 'Loaded diagram image from file path');
            } catch (readError) {
              logger.error({ error: readError.message, imagePath: containerPath, diagramId: diagram.id }, 'Failed to read Playfair image file');
            }
          } else {
            logger.warn({ playfairResponse: jsonData, diagramId: diagram.id }, 'Playfair response missing both "result.data" and "result.path"');
          }
        } catch (e) {
          logger.error({ error: e.message, rawResponse: imageData, diagramId: diagram.id }, 'Failed to parse JSON from Playfair response');
        }
      } else {
        logger.warn({ playfairResponse: result, diagramId: diagram.id }, 'Invalid Playfair response structure: content[0].text is missing');
      }

      if (base64Data) {
        // Replace placeholder with actual image
        const imgTag = `<img src="data:image/png;base64,${base64Data}" alt="Diagram" />`;
        processedHtml = processedHtml.replace(
          `<img id="${diagram.id}" alt="Playfair diagram" />`,
          imgTag
        );
      } else {
        // Fallback: Render original diagram source as code block
        logger.warn({ diagramId: diagram.id }, 'Rendering fallback diagram due to missing base64 data');

        const fallback = `<pre><code><!-- WARNING: Playfair diagram generation failed -->
<!-- Diagram specification below: -->

${diagram.content}</code></pre>`;

        processedHtml = processedHtml.replace(
          `<img id="${diagram.id}" alt="Playfair diagram" />`,
          fallback
        );

        warnings.push(`Diagram ${diagram.id} failed to generate: No base64 data`);
      }
    } catch (error) {
      logger.error({ error: error.message, diagram: diagram.id }, 'Playfair diagram generation failed');

      // Replace with fallback code block
      const fallback = `<pre><code><!-- WARNING: Playfair diagram generation failed -->
<!-- Error: ${error.message} -->
<!-- Original diagram specification below: -->

${diagram.content}</code></pre>`;

      processedHtml = processedHtml.replace(
        `<img id="${diagram.id}" alt="Playfair diagram" />`,
        fallback
      );

      warnings.push(`Diagram failed to generate: ${error.message}`);
    }
  }

  return { html: processedHtml, warnings };
}

/**
 * Process markdown images (file paths, URLs, base64)
 */
async function processMarkdownImages(html, images) {
  if (!images || images.length === 0) {
    return { html, warnings: [] };
  }

  const warnings = [];
  let processedHtml = html;

  for (const image of images) {
    try {
      let base64Data = null;
      let mimeType = 'image/png';

      // Check if src is already base64
      if (image.src.startsWith('data:')) {
        // Already base64, use as-is
        const imgTag = `<img src="${image.src}" alt="${image.alt}" />`;
        processedHtml = processedHtml.replace(
          `<img id="${image.id}" alt="${image.alt}" />`,
          imgTag
        );
        continue;
      }

      // Check if src is a URL
      if (image.src.startsWith('http://') || image.src.startsWith('https://')) {
        // For now, leave URLs as-is (could fetch and embed in future)
        const imgTag = `<img src="${image.src}" alt="${image.alt}" />`;
        processedHtml = processedHtml.replace(
          `<img id="${image.id}" alt="${image.alt}" />`,
          imgTag
        );
        continue;
      }

      // Treat as file path
      let filePath = image.src;

      // Convert host path to container path if needed
      if (filePath.includes('/mnt/projects/ICCM/irina_storage_test/files')) {
        filePath = filePath.replace('/mnt/projects/ICCM/irina_storage_test/files', '/mnt/irina_storage/files');
      } else if (!isAbsolute(filePath)) {
        // Relative path - resolve against Horace temp storage
        filePath = join('/mnt/irina_storage/files/temp', filePath);
      }

      // Detect MIME type from extension
      const ext = filePath.split('.').pop().toLowerCase();
      const mimeTypes = {
        'png': 'image/png',
        'jpg': 'image/jpeg',
        'jpeg': 'image/jpeg',
        'gif': 'image/gif',
        'svg': 'image/svg+xml',
        'webp': 'image/webp'
      };
      mimeType = mimeTypes[ext] || 'image/png';

      // Read and convert to base64
      try {
        const imageBuffer = await readFile(filePath);
        base64Data = imageBuffer.toString('base64');
        logger.info({ filePath, imageId: image.id }, 'Loaded markdown image from file');
      } catch (readError) {
        logger.error({ error: readError.message, filePath, imageId: image.id }, 'Failed to read markdown image file');
        warnings.push(`Failed to load image: ${image.src}`);
        continue;
      }

      if (base64Data) {
        const imgTag = `<img src="data:${mimeType};base64,${base64Data}" alt="${image.alt}" />`;
        processedHtml = processedHtml.replace(
          `<img id="${image.id}" alt="${image.alt}" />`,
          imgTag
        );
      }
    } catch (error) {
      logger.error({ error: error.message, imageId: image.id }, 'Markdown image processing failed');
      warnings.push(`Image processing failed: ${image.src}`);
    }
  }

  return { html: processedHtml, warnings };
}

/**
 * Convert Markdown to HTML
 */
async function markdownToHTML(markdown) {
  const md = markdownIt({
    html: true,
    linkify: true,
    typographer: true
  })
    .use(markdownItMultimdTable, {
      multiline: true,
      rowspan: true,
      headerless: true
    })
    .use(markdownItAttrs)
    .use(markdownItTaskLists)
    .use(createPlayfairPlugin())
    .use(createImagePlugin());

  const env = {};
  const html = md.render(markdown, env);

  // Process Playfair diagrams
  const { html: processedHtml1, warnings: warnings1 } = await processPlayfairDiagrams(html, env.playfairDiagrams);

  // Process markdown images
  const { html: processedHtml2, warnings: warnings2 } = await processMarkdownImages(processedHtml1, env.markdownImages);

  return { html: processedHtml2, warnings: [...(warnings1 || []), ...(warnings2 || [])] };
}

/**
 * Convert HTML to ODT using LibreOffice
 */
async function htmlToODT(html, outputPath, metadata = {}) {
  const workDir = join(tmpdir(), `gates-${randomUUID()}`);
  await mkdir(workDir, { recursive: true });

  try {
    // Create HTML file with metadata
    const htmlContent = `<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <title>${metadata.title || 'Document'}</title>
  <meta name="author" content="${metadata.author || ''}" />
  <meta name="date" content="${metadata.date || new Date().toISOString()}" />
  <style>
    @page {
      size: A4;
      margin: 2.54cm;
      @bottom-center {
        content: counter(page);
        font-size: 10pt;
      }
    }
    body {
      font-family: 'Liberation Serif', 'Times New Roman', serif;
      font-size: 12pt;
      line-height: 1.5;
      margin: 0;
      padding: 0;
      counter-reset: figure;
    }
    /* Improved heading hierarchy with spacing */
    h1 {
      font-family: 'Liberation Sans', 'Arial', sans-serif;
      font-size: 20pt;
      font-weight: bold;
      margin-top: 24pt;
      margin-bottom: 12pt;
      page-break-after: avoid;
    }
    h2 {
      font-family: 'Liberation Sans', 'Arial', sans-serif;
      font-size: 16pt;
      font-weight: bold;
      margin-top: 18pt;
      margin-bottom: 6pt;
      page-break-after: avoid;
    }
    h3 {
      font-family: 'Liberation Sans', 'Arial', sans-serif;
      font-size: 14pt;
      font-weight: bold;
      margin-top: 14pt;
      margin-bottom: 4pt;
      page-break-after: avoid;
    }
    h4, h5, h6 {
      font-family: 'Liberation Sans', 'Arial', sans-serif;
      font-size: 12pt;
      font-weight: bold;
      margin-top: 12pt;
      margin-bottom: 3pt;
      page-break-after: avoid;
    }
    p {
      margin-top: 0;
      margin-bottom: 6pt;
      text-align: justify;
    }
    /* Image/Figure styling with captions */
    img {
      display: block;
      margin: 12pt auto;
      max-width: 90%;
      height: auto;
      border: 0.5pt solid #000;
      page-break-inside: avoid;
    }
    /* Lists with proper indentation */
    ul, ol {
      margin: 6pt 0;
      padding-left: 24pt;
    }
    li {
      margin-bottom: 3pt;
    }
    /* Code blocks */
    code {
      font-family: 'Liberation Mono', 'Courier New', monospace;
      font-size: 10pt;
      background-color: #f5f5f5;
      padding: 0.2em 0.4em;
      border-radius: 3px;
    }
    pre {
      font-family: 'Liberation Mono', 'Courier New', monospace;
      font-size: 10pt;
      background-color: #f5f5f5;
      padding: 12pt;
      line-height: 1.2;
      margin: 12pt 0;
      border: 1pt solid #ddd;
      page-break-inside: avoid;
      overflow-x: auto;
    }
    /* Table styling */
    table {
      border-collapse: collapse;
      width: 100%;
      margin: 12pt 0;
      page-break-inside: avoid;
    }
    th, td {
      border: 1pt solid #333;
      padding: 6pt 8pt;
      text-align: left;
      vertical-align: top;
    }
    th {
      background-color: #e8e8e8;
      font-weight: bold;
      font-size: 11pt;
    }
    td {
      font-size: 11pt;
    }
  </style>
</head>
<body>
${html}
</body>
</html>`;

    const htmlPath = join(workDir, 'input.html');
    await writeFile(htmlPath, htmlContent, 'utf-8');

    // Convert to ODT using LibreOffice
    logger.info({ workDir }, 'Converting HTML to ODT with LibreOffice');

    const result = await execa('libreoffice', [
      '--headless',
      '--convert-to', 'odt',
      '--outdir', workDir,
      htmlPath
    ], {
      timeout: CONVERSION_TIMEOUT,
      cleanup: true,
      killSignal: 'SIGTERM',
      env: {
        HOME: workDir // Prevent LibreOffice config conflicts
      }
    });

    logger.info({ stdout: result.stdout }, 'LibreOffice conversion completed');

    // Move ODT to output path
    const generatedODT = join(workDir, 'input.odt');
    if (!existsSync(generatedODT)) {
      throw new Error('LibreOffice did not generate ODT file');
    }

    // Read and write to final location
    const odtContent = await readFile(generatedODT);
    await writeFile(outputPath, odtContent);

    logger.info({ outputPath, size: odtContent.length }, 'ODT file created successfully');

    return {
      success: true,
      size: odtContent.length
    };
  } finally {
    // Cleanup temp directory
    try {
      await unlink(join(workDir, 'input.html')).catch(() => {});
      await unlink(join(workDir, 'input.odt')).catch(() => {});
    } catch (error) {
      logger.warn({ error: error.message }, 'Failed to cleanup temp files');
    }
  }
}

/**
 * Main document creation function
 */
async function createDocument(args) {
  const { markdown, input_file, metadata = {}, output_path } = args;

  // Load markdown from file or use provided content
  let markdownContent;
  if (input_file) {
    if (markdown) {
      throw new Error('Cannot specify both "markdown" and "input_file" parameters');
    }
    try {
      markdownContent = await readFile(input_file, 'utf-8');
      logger.info({ input_file, size: markdownContent.length }, 'Loaded markdown from file');
    } catch (error) {
      throw new Error(`Failed to read input file: ${error.message}`);
    }
  } else if (markdown) {
    markdownContent = markdown;
  } else {
    throw new Error('Must specify either "markdown" or "input_file" parameter');
  }

  // Validate markdown size
  if (markdownContent.length > MAX_MARKDOWN_SIZE) {
    throw new Error(`Markdown exceeds maximum size of ${MAX_MARKDOWN_SIZE / 1024 / 1024}MB`);
  }

  const startTime = Date.now();

  // Generate output path with default to Horace temp storage
  const defaultOutputDir = process.env.DEFAULT_OUTPUT_DIR || '/tmp';
  let outputPath;
  if (output_path) {
    // If absolute path, use as-is; if relative, join with default dir
    outputPath = isAbsolute(output_path) ? output_path : join(defaultOutputDir, output_path);
  } else {
    // No path specified, generate default filename
    outputPath = join(defaultOutputDir, `document-${randomUUID()}.odt`);
  }

  // Convert Markdown to HTML
  const { html, warnings } = await markdownToHTML(markdownContent);

  // Convert HTML to ODT
  const { size } = await htmlToODT(html, outputPath, metadata);

  if (size > MAX_ODT_SIZE) {
    await unlink(outputPath);
    throw new Error(`Generated ODT exceeds maximum size of ${MAX_ODT_SIZE / 1024 / 1024}MB`);
  }

  const conversionTime = Date.now() - startTime;

  return {
    success: true,
    odt_file: outputPath,  // Already resolved to absolute path
    size_bytes: size,
    metadata: {
      title: metadata.title || 'Document',
      author: metadata.author || '',
      conversion_time_ms: conversionTime
    },
    warnings: warnings || []
  };
}

/**
 * Validate Markdown
 */
async function validateMarkdown(args) {
  const { markdown } = args;

  const warnings = [];
  const statistics = {
    heading_count: 0,
    paragraph_count: 0,
    code_block_count: 0,
    table_count: 0,
    diagram_count: 0,
    estimated_page_count: 0
  };

  // Parse with markdown-it
  const md = markdownIt();
  const env = {};
  const tokens = md.parse(markdown, env);

  // Analyze tokens
  for (const token of tokens) {
    if (token.type === 'heading_open') {
      statistics.heading_count++;
    } else if (token.type === 'paragraph_open') {
      statistics.paragraph_count++;
    } else if (token.type === 'fence') {
      if (token.info.startsWith('playfair-')) {
        statistics.diagram_count++;
      } else {
        statistics.code_block_count++;
      }
    } else if (token.type === 'table_open') {
      statistics.table_count++;
    }
  }

  // Estimate page count (rough: 500 words per page)
  const wordCount = markdown.split(/\s+/).length;
  statistics.estimated_page_count = Math.ceil(wordCount / 500);

  return {
    valid: true,
    warnings,
    statistics
  };
}

/**
 * List capabilities
 */
async function listCapabilities() {
  return {
    version: '1.0',
    markdown_features: [
      'CommonMark',
      'GFM tables',
      'Task lists',
      'Nested lists (4 levels)',
      'Fenced code blocks',
      'Playfair diagrams (dot, mermaid)'
    ],
    diagram_formats: ['playfair-dot', 'playfair-mermaid'],
    output_formats: ['odt'],
    size_limits: {
      max_markdown_size_mb: MAX_MARKDOWN_SIZE / 1024 / 1024,
      max_odt_size_mb: MAX_ODT_SIZE / 1024 / 1024,
      max_image_size_mb: MAX_IMAGE_SIZE / 1024 / 1024
    },
    playfair_status: playfairConnected ? 'operational' : 'unavailable',
    queue_status: {
      current_depth: conversionQueue.size,
      max_depth: MAX_QUEUE_DEPTH,
      processing: conversionQueue.pending > 0
    }
  };
}

/**
 * Handle tool calls
 */
async function handleToolCall(toolName, args) {
  logger.info({ toolName, args }, 'Handling tool call');

  switch (toolName) {
    case 'gates_create_document':
      if (conversionQueue.size >= MAX_QUEUE_DEPTH) {
        throw new Error('SERVER_BUSY: Queue full (10 requests)');
      }
      return conversionQueue.add(() => createDocument(args));

    case 'gates_validate_markdown':
      return validateMarkdown(args);

    case 'gates_list_capabilities':
      return listCapabilities();

    default:
      throw new Error(`Unknown tool: ${toolName}`);
  }
}

/**
 * Handle MCP request
 */
async function handleMCPRequest(request) {
  const { id, method, params } = request;

  try {
    switch (method) {
      case 'initialize':
        return {
          jsonrpc: '2.0',
          id,
          result: {
            protocolVersion: '2024-11-05',
            capabilities: {
              tools: {}
            },
            serverInfo: {
              name: 'gates-mcp-server',
              version: '1.0.0'
            }
          }
        };

      case 'tools/list':
        logger.info({ tools: TOOLS }, 'Returning tools list');
        return {
          jsonrpc: '2.0',
          id,
          result: {
            tools: TOOLS
          }
        };

      case 'tools/call':
        const result = await handleToolCall(params.name, params['arguments'] || {});
        return {
          jsonrpc: '2.0',
          id,
          result: {
            content: [{
              type: 'text',
              text: JSON.stringify(result, null, 2)
            }]
          }
        };

      default:
        throw new Error(`Unknown method: ${method}`);
    }
  } catch (error) {
    logger.error({ error: error.message, request }, 'MCP request failed');
    return {
      jsonrpc: '2.0',
      id,
      error: {
        code: -32603,
        message: error.message
      }
    };
  }
}

/**
 * Start WebSocket MCP server
 */
async function startServer() {
  // Connect to Playfair
  await connectPlayfair();

  // Create HTTP server for health checks
  const httpServer = createServer((req, res) => {
    if (req.url === '/health') {
      res.writeHead(200, { 'Content-Type': 'application/json' });
      res.end(JSON.stringify({
        status: 'healthy',
        playfair: playfairConnected ? 'connected' : 'disconnected',
        queue_depth: conversionQueue.size,
        queue_processing: conversionQueue.pending > 0
      }));
    } else {
      res.writeHead(404);
      res.end();
    }
  });

  // Create WebSocket server
  const wss = new WebSocketServer({ server: httpServer });

  wss.on('connection', (ws) => {
    logger.info('Client connected');

    ws.on('message', async (data) => {
      try {
        const request = JSON.parse(data.toString());
        const response = await handleMCPRequest(request);
        ws.send(JSON.stringify(response));
      } catch (error) {
        logger.error({ error: error.message }, 'Failed to handle message');
        ws.send(JSON.stringify({
          jsonrpc: '2.0',
          id: null,
          error: {
            code: -32700,
            message: 'Parse error'
          }
        }));
      }
    });

    ws.on('close', () => {
      logger.info('Client disconnected');
    });

    ws.on('error', (error) => {
      logger.error({ error: error.message }, 'WebSocket error');
    });
  });

  httpServer.listen(PORT, HOST, () => {
    logger.info({ port: PORT, host: HOST }, 'Gates MCP server started');
  });

  // Graceful shutdown
  process.on('SIGTERM', () => {
    logger.info('SIGTERM received, shutting down gracefully');
    wss.close(() => {
      httpServer.close(() => {
        process.exit(0);
      });
    });
  });
}

// Start server
startServer().catch((error) => {
  logger.error({ error: error.message }, 'Failed to start server');
  process.exit(1);
});
