const { exec } = require('child_process');
const { promisify } = require('util');
const BaseEngine = require('./base');
const logger = require('../utils/logger');
const svgProcessor = require('../themes/svg-processor');
const themes = require('../themes/graphviz-themes.json');

const execAsync = promisify(exec);
const RENDER_TIMEOUT = parseInt(process.env.RENDER_TIMEOUT_MS, 10) || 60000;

class GraphvizEngine extends BaseEngine {
    constructor() {
        super('graphviz');
    }

    async render(content, options) {
        const theme = options.theme || 'modern';
        const themedDot = this._applyTheme(content, theme);

        // Command to render SVG using the high-quality Cairo renderer
        const command = 'dot -Tsvg -Kdot';

        try {
            const { stdout: rawSvg } = await execAsync(command, {
                input: themedDot,
                timeout: RENDER_TIMEOUT,
                maxBuffer: 50 * 1024 * 1024, // 50MB
            });

            // Post-process the SVG for modern aesthetics (gradients, fonts, etc.)
            const processedSvg = await svgProcessor.process(rawSvg, 'graphviz', theme);
            return Buffer.from(processedSvg);

        } catch (error) {
            logger.error({ engine: this.name, error: error.stderr || error.message }, 'Graphviz rendering failed');
            const parsedError = this._parseError(error.stderr || error.message);
            throw parsedError;
        }
    }

    async validate(content) {
        // Use the '-c' flag for syntax checking without generating output
        const command = 'dot -c';
        try {
            await execAsync(command, { input: content, timeout: 10000 });
            return { valid: true, errors: [] };
        } catch (error) {
            const parsedError = this._parseError(error.stderr || error.message);
            return { valid: false, errors: [{ line: parsedError.line || null, message: parsedError.message }] };
        }
    }

    _applyTheme(dotContent, themeName) {
        const theme = themes[themeName];
        if (!theme) {
            logger.warn({ themeName }, 'Unknown theme requested, using default.');
            return dotContent;
        }

        // Create attribute strings from the theme JSON
        const graphAttrs = Object.entries(theme.graph).map(([k, v]) => `${k}="${v}"`).join(' ');
        const nodeAttrs = Object.entries(theme.node).map(([k, v]) => `${k}="${v}"`).join(' ');
        const edgeAttrs = Object.entries(theme.edge).map(([k, v]) => `${k}="${v}"`).join(' ');

        const themeInjection = `
  // Theme: ${themeName}
  graph [${graphAttrs}];
  node [${nodeAttrs}];
  edge [${edgeAttrs}];
`;
        // Inject theme attributes after the opening brace of the digraph/graph
        return dotContent.replace(/digraph\s*.*?\{/i, `$&${themeInjection}`);
    }

    _parseError(stderr) {
        const error = new Error();
        error.engine = this.name;

        const lineMatch = stderr.match(/error in line (\d+)/i);
        if (lineMatch) {
            error.code = 'SYNTAX_ERROR';
            error.line = parseInt(lineMatch[1], 10);
            error.message = stderr.split('\n')[0] || 'Graphviz syntax error.';
            error.suggestion = `Check the DOT syntax near line ${error.line}.`;
        } else if (stderr.includes('timeout')) {
            error.code = 'TIMEOUT';
            error.message = 'Graphviz rendering timed out.';
            error.suggestion = 'The diagram is too complex. Try simplifying it.';
        } else {
            error.code = 'ENGINE_CRASH';
            error.message = 'Graphviz engine failed unexpectedly.';
            error.suggestion = 'Please check the diagram content for errors.';
        }
        return error;
    }
}

module.exports = GraphvizEngine;