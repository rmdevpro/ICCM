# Gates - Document Generation Gateway

**Version:** 1.0
**Status:** ✅ **DEPLOYED**
**Component Type:** WebSocket MCP Server
**Port:** 9050 (host) / 8050 (container)

**Recent Updates (2025-10-07)**:
- **File Input/Output Support**:
  - Added `input_file` parameter for large markdown documents
  - File output to Horace temp storage at `/mnt/irina_storage/files/temp/gates/`
  - Relative paths automatically joined with DEFAULT_OUTPUT_DIR
  - Absolute paths supported for custom locations
- **Horace Storage Integration**:
  - Mounted irina_storage_files volume
  - All documents automatically cataloged by Horace
  - Consistent with Playfair and Fiedler architecture

---

## Overview

Gates is the ICCM Document Generation Gateway that converts Markdown to professional ODT (OpenDocument Text) documents with embedded diagrams. While LLMs excel at generating markdown, business and academic environments require formatted documents (ODT, DOCX, PDF). Gates bridges this gap.

**Problem Solved:**
- LLMs output markdown, not formatted documents
- Manual conversion is time-consuming and error-prone
- Diagrams need to be embedded, not linked
- Professional formatting required for academia/business

**Solution:**
- Markdown → HTML → ODT pipeline via LibreOffice
- Automatic Playfair diagram embedding
- Professional themes and styling
- WebSocket MCP protocol for LLM integration

---

## Quick Start

### Prerequisites
- Docker & docker-compose
- ICCM network (`iccm_network`)
- MCP Relay configured
- Playfair MCP server running (for diagram embedding)

### Launch Gates

```bash
cd /mnt/projects/ICCM/gates
docker compose up -d
```

### Check Health

```bash
curl http://localhost:9050/health
# Expected: {"status":"healthy","queue_depth":0,"playfair_connected":true}
```

### Use from LLM (via MCP Relay)

```javascript
// Option 1: Direct markdown content (small documents)
gates_create_document({
  markdown: "# My Document\n\nContent here...",
  metadata: {
    title: "My Document",
    author: "Author Name",
    date: "2025-10-07"
  },
  output_path: "my-document.odt"  // Relative or absolute
})

// Option 2: File input (large documents, recommended)
gates_create_document({
  input_file: "/mnt/irina_storage/files/temp/gates/source.md",
  metadata: {
    title: "My Document",
    author: "Author Name"
  },
  output_path: "output.odt"  // Optional, auto-generates if omitted
})
// Returns: { success: true, odt_file: "/mnt/irina_storage/files/temp/gates/output.odt", size_bytes: 15234, ... }
```

---

## Supported Features

### Markdown Features (CommonMark + GFM)

- **Headings:** H1-H6 with automatic styling
- **Paragraphs:** Standard text formatting
- **Lists:** Ordered, unordered, nested (4 levels)
- **Tables:** GFM tables with multiline, rowspan, headerless
- **Code blocks:** Fenced with syntax highlighting
- **Task lists:** `- [ ] Task` and `- [x] Done`
- **Bold/Italic:** Standard markdown emphasis
- **Links:** Hyperlinks preserved
- **Images:** Multiple formats supported (see Image Handling below)

### Image Handling

Gates supports flexible image handling for multiple source types:

#### 1. Playfair Diagrams (Embedded)
```markdown
\`\`\`playfair-dot
digraph { A -> B; }
\`\`\`
```
- Rendered by Playfair in real-time
- Embedded as base64 PNG in ODT
- Fallback to source code if rendering fails

#### 2. Inline Base64 Images (Embedded)
```markdown
![Logo](data:image/png;base64,iVBORw0KGgo...)
```
- Already in base64 format
- Embedded directly in ODT
- No external file access needed

#### 3. File Path Images (Embedded)
```markdown
![Diagram](./path/to/image.png)
![Diagram](/absolute/path/to/image.png)
```
- Reads image from filesystem
- Converts to base64 and embeds in ODT
- Supports: PNG, JPG, JPEG, GIF, SVG, WEBP
- Relative paths resolved against Horace temp storage

#### 4. URL Images (Linked)
```markdown
![External](https://example.com/image.png)
```
- Currently preserved as URL links
- Future: Option to fetch and embed

### Diagram Embedding

Gates automatically detects and embeds diagrams via Playfair:

````markdown
```playfair-dot
digraph { A -> B -> C; }
```

```playfair-mermaid
graph TD
    A[Start] --> B[End]
```
````

Diagrams are:
- Rendered by Playfair in real-time
- Embedded as PNG images in the ODT
- Fallback to source code if rendering fails

---

## File I/O Architecture

### Input Options

1. **Direct Markdown (`markdown` parameter)**:
   - For small documents (<1MB recommended)
   - Passed directly in MCP call
   - Good for quick conversions

2. **File Input (`input_file` parameter)**:
   - For large documents (>1MB, up to 10MB)
   - Reads from filesystem path
   - Preferred for production use
   - Path must be accessible from container

**Note:** Cannot specify both `markdown` and `input_file` parameters.

### Output Behavior

- **No `output_path` specified**: Auto-generates filename in DEFAULT_OUTPUT_DIR
  - Format: `document-{uuid}.odt`
  - Location: `/mnt/irina_storage/files/temp/gates/`

- **Relative `output_path`**: Joined with DEFAULT_OUTPUT_DIR
  - Example: `output_path: "my-doc.odt"` → `/mnt/irina_storage/files/temp/gates/my-doc.odt`

- **Absolute `output_path`**: Used as-is
  - Example: `output_path: "/tmp/custom.odt"` → `/tmp/custom.odt`

---

## Configuration

### Environment Variables

```yaml
GATES_PORT: 8050                    # Internal port
GATES_HOST: 0.0.0.0                # Bind address
PLAYFAIR_URL: ws://playfair-mcp:8040  # Playfair WebSocket URL
DEFAULT_OUTPUT_DIR: /mnt/irina_storage/files/temp/gates  # Default output location
LOG_LEVEL: info                     # Logging level
NODE_ENV: production               # Environment
```

### Resource Limits

```yaml
MAX_MARKDOWN_SIZE: 10MB          # Maximum input size
MAX_ODT_SIZE: 50MB               # Maximum output size
MAX_IMAGE_SIZE: 10MB             # Maximum embedded image size
MAX_QUEUE_DEPTH: 10              # Concurrent conversion limit
CONVERSION_TIMEOUT: 120000ms     # 2 minutes per document
```

### MCP Relay Integration

Add to `/mnt/projects/ICCM/mcp-relay/backends.yaml`:

```yaml
backends:
  - name: gates
    url: ws://localhost:9050
```

---

## API Reference

### `gates_create_document`

Convert Markdown to ODT document with embedded diagrams.

**Parameters:**
- `markdown` (string, optional): Markdown content to convert
- `input_file` (string, optional): Path to markdown file to convert
- `metadata` (object, optional):
  - `title` (string): Document title
  - `author` (string): Author name
  - `date` (string): Document date (ISO format or display string)
  - `keywords` (array): List of keyword strings
- `output_path` (string, optional): Output file path (relative or absolute)

**Returns:**
```json
{
  "success": true,
  "odt_file": "/mnt/irina_storage/files/temp/gates/document.odt",
  "size_bytes": 17374,
  "metadata": {
    "title": "Document Title",
    "author": "Author Name",
    "conversion_time_ms": 1876
  },
  "warnings": []
}
```

**Errors:**
- `MISSING_CONTENT`: Neither `markdown` nor `input_file` specified
- `INVALID_INPUT`: Both `markdown` and `input_file` specified
- `FILE_NOT_FOUND`: Input file doesn't exist
- `MARKDOWN_TOO_LARGE`: Input exceeds 10MB
- `ODT_TOO_LARGE`: Output exceeds 50MB
- `SERVER_BUSY`: Queue full (10 requests)

### `gates_validate_markdown`

Validate Markdown syntax and check for ODT conversion issues.

**Parameters:**
- `markdown` (string): Markdown content to validate

**Returns:**
```json
{
  "valid": true,
  "warnings": ["Detected very long paragraph (>500 words)"],
  "statistics": {
    "heading_count": 15,
    "paragraph_count": 42,
    "code_block_count": 8,
    "table_count": 3,
    "diagram_count": 5,
    "estimated_page_count": 12
  }
}
```

### `gates_list_capabilities`

List supported Markdown features and current configuration.

**Returns:**
```json
{
  "version": "1.0",
  "markdown_features": ["CommonMark", "GFM tables", "Task lists", ...],
  "diagram_formats": ["playfair-dot", "playfair-mermaid"],
  "output_formats": ["odt"],
  "size_limits": {
    "max_markdown_size_mb": 10,
    "max_odt_size_mb": 50,
    "max_image_size_mb": 10
  },
  "playfair_status": "operational",
  "queue_status": {
    "current_depth": 0,
    "max_depth": 10,
    "processing": false
  }
}
```

---

## Architecture Integration

```
┌────────────────────────────────────────┐
│  LLM (via Fiedler)                     │
│  - Generates markdown specifications   │
│  - Calls gates_create_document         │
└────────────┬───────────────────────────┘
             │
             │ MCP WebSocket
             ↓
┌────────────────────────────────────────┐
│  MCP Relay                              │
│  - Routes to Gates backend              │
│  - ws://localhost:9050                  │
└────────────┬───────────────────────────┘
             │
             │ WebSocket
             ↓
┌────────────────────────────────────────┐
│  Gates Container                        │
│  ┌──────────────────────────────────┐  │
│  │ Conversion Queue (FIFO, 1 worker)│  │
│  │ - Markdown-it parser             │  │
│  │ - Playfair diagram embedding     │  │
│  │ - LibreOffice HTML→ODT           │  │
│  └──────────────────────────────────┘  │
└────────────┬───────────────────────────┘
             │
             ↓
    ODT Document in Horace Storage
    (cataloged automatically)
```

**Optional Integration:**
- Godot → Logging (future)
- Horace → File catalog registration (automatic via watcher)
- Playfair → Diagram rendering (required)

---

## Performance

**Conversion Speed:**
- Simple documents (<10 pages): <2 seconds
- Medium documents (10-50 pages): <5 seconds
- Complex documents (50-200 pages): <15 seconds
- Timeout: 120 seconds maximum

**Concurrency:**
- Queue: FIFO, single worker
- Max depth: 10 concurrent requests
- Resource limits: 1GB memory, 2 CPU cores

---

## Troubleshooting

### Container won't start
```bash
docker logs gates-mcp
# Check for LibreOffice or dependency errors
```

### Playfair not connected
- Verify Playfair is running: `docker ps | grep playfair`
- Check network: Both should be on `iccm_network`
- Diagrams will fall back to source code if Playfair unavailable

### File not found errors
- Verify input file path is absolute
- Check file exists in container: `docker exec gates-mcp ls -la /path/to/file.md`
- Ensure volume mounts are correct

### ODT file not created
- Check DEFAULT_OUTPUT_DIR environment variable
- Verify volume mount: `docker exec gates-mcp ls -la /mnt/irina_storage/files/temp/gates/`
- Check return value for actual file path

---

## Example Workflows

### Academic Paper Generation

```javascript
// 1. Generate diagrams using Playfair
const diagram1 = await playfair_create_diagram({
  content: "digraph architecture { ... }",
  format: "dot",
  output_path: "diagrams/architecture.svg"
});

// 2. Write markdown with embedded diagrams
const markdown = `
# Paper Title

## Abstract
...

## Architecture

\`\`\`playfair-dot
digraph architecture { ... }
\`\`\`

## Conclusion
...
`;

// 3. Save to file
writeFile("/mnt/irina_storage/files/temp/gates/paper-source.md", markdown);

// 4. Generate ODT
const result = await gates_create_document({
  input_file: "/mnt/irina_storage/files/temp/gates/paper-source.md",
  metadata: {
    title: "My Research Paper",
    author: "Dr. Researcher",
    date: "2025-10-07",
    keywords: ["AI", "Agents", "Architecture"]
  },
  output_path: "research-paper-v1.odt"
});
// Output: /mnt/irina_storage/files/temp/gates/research-paper-v1.odt
```

### Technical Documentation

```javascript
const result = await gates_create_document({
  input_file: "/mnt/irina_storage/files/temp/gates/api-docs.md",
  metadata: {
    title: "API Documentation v2.0",
    author: "Engineering Team"
  }
  // output_path omitted - auto-generates filename
});
console.log(result.odt_file);
// /mnt/irina_storage/files/temp/gates/document-{uuid}.odt
```

---

## Development Roadmap

### ✅ Phase 1: Core Functionality (Complete)
- WebSocket MCP server
- Markdown → ODT conversion via LibreOffice
- Playfair diagram embedding
- File input/output support
- Horace storage integration
- Queue management

### Phase 2: Enhanced Features (Future)
- PDF output format
- DOCX output format
- Custom ODT templates
- Header/footer customization
- Table of contents generation
- Bibliography support

### Phase 3: Advanced Features (Future)
- Math equation rendering (LaTeX)
- Citation management
- Multi-document merging
- Version comparison
- Collaborative editing metadata

---

## License Compliance

**All Components - 100% Permissive:**

| Component      | License    | Type       |
|----------------|------------|------------|
| LibreOffice    | MPL-2.0    | Permissive |
| markdown-it    | MIT        | Permissive |
| ws             | MIT        | Permissive |
| pino           | MIT        | Permissive |
| p-queue        | MIT        | Permissive |
| execa          | MIT        | Permissive |

✅ **No copyleft dependencies**
✅ **Commercial use allowed**
✅ **Redistribution allowed**

---

## Support & Documentation

- **Requirements:** `/mnt/projects/ICCM/gates/REQUIREMENTS.md`
- **Triplet Review:** `/mnt/projects/ICCM/gates/TRIPLET_REVIEW_SYNTHESIS.md`
- **MCP Protocol:** MCP Specification 2024-11-05
- **Markdown Spec:** CommonMark + GFM extensions

---

## Credits

**Development:** Triplet-driven development
**Rendering Engine:** LibreOffice (HTML → ODT conversion)
**Markdown Parser:** markdown-it
**Diagram Integration:** Playfair MCP
**Protocol:** Model Context Protocol (MCP)
**Part of:** ICCM (Integrated Cognitive Capabilities Module)

---

**Status:** ✅ Deployed and operational
**Next Step:** Phase 2 (PDF/DOCX output formats)
