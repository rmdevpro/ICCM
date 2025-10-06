# Horace: NAS Gateway Requirements

**Version:** 2.0
**Date:** 2025-10-06
**Status:** DRAFT - Architecture Revision

---

## Executive Summary

Horace is the **centralized NAS gateway** for the ICCM ecosystem, providing a file system abstraction backed by ZFS. All services mount `/mnt/irina_storage` and treat it as a standard POSIX filesystem, while Horace automatically catalogs, versions, and manages all content.

---

## 1. Core Principles

### 1.1 NAS Gateway Pattern
- **Services write directly** to `/mnt/irina_storage` using standard file operations
- **Horace watches automatically** and catalogs all changes
- **No manual registration** required - Horace discovers everything
- **Transparent to applications** - services don't know they're using a managed storage layer

### 1.2 ZFS Backing Store
- **Copy-on-write** provides immutability by default
- **Snapshots** for time-travel and versioning
- **Checksums** for content verification (SHA-256)
- **Tiered storage** (cache → SSD → HDD) for performance
- **Compression** (lz4) to save space
- **Deduplication** for identical content

---

## 2. Architecture

### 2.1 Storage Layout

```
/mnt/irina_storage/              # ZFS mount point
├── files/                       # Active file tree
│   ├── fiedler/                 # Per-service namespaces
│   ├── playfair/
│   ├── gates/
│   ├── shared/                  # Cross-service collaboration
│   └── user/                    # User-owned content
└── .horace/                     # Horace metadata (hidden)
    ├── catalog.db               # SQLite catalog
    ├── indexes/                 # Search indexes
    └── snapshots/               # ZFS snapshot metadata
```

### 2.2 Service Access Pattern

**All services mount the same volume:**
```yaml
volumes:
  - irina_storage:/mnt/irina_storage
```

**Services write normally:**
```python
# Playfair creates diagram
Path("/mnt/irina_storage/files/playfair/diagrams/paper_00_arch.svg").write_text(svg_content)

# Horace automatically detects, catalogs, and checksums it
# No registration needed!
```

### 2.3 Horace's Role

**File System Watcher:**
- Uses `inotify` (Linux) to watch `/mnt/irina_storage/files/`
- Detects `CREATE`, `MODIFY`, `DELETE` events
- Auto-catalogs on write completion

**Catalog Management:**
- Extracts metadata (size, mime-type, owner from path)
- Computes checksum (SHA-256)
- Creates ZFS snapshot: `horace/files@<timestamp>`
- Stores catalog entry in SQLite

**Query Interface (MCP):**
- `horace_search_files` - query the catalog
- `horace_get_file_info` - get metadata + version history
- `horace_restore_version` - restore from ZFS snapshot
- `horace_list_collections` - logical groupings

---

## 3. ZFS Configuration

### 3.1 Dataset Properties
```bash
zfs create -o compression=lz4 \
           -o checksum=sha256 \
           -o atime=off \
           -o dedup=on \
           tank/horace
```

### 3.2 Snapshot Schedule
- **Every write:** Immediate snapshot tagged with `file_id`
- **Hourly:** Consolidated snapshot for time-travel
- **Daily:** Kept for 30 days
- **Monthly:** Kept for 1 year

### 3.3 Tiered Storage (Future)
- **L2ARC:** NVMe cache for hot files
- **Primary:** SSD pool for active files
- **Archive:** HDD pool for cold data (auto-tiered by access patterns)

---

## 4. Implementation Requirements

### 4.1 Docker Compose Integration

**Shared ZFS volume:**
```yaml
volumes:
  irina_storage:
    driver: local
    driver_opts:
      type: zfs
      device: tank/horace/files
      o: bind
```

**All services mount it:**
```yaml
services:
  playfair-mcp:
    volumes:
      - irina_storage:/mnt/irina_storage

  gates-mcp:
    volumes:
      - irina_storage:/mnt/irina_storage

  horace-mcp:
    volumes:
      - irina_storage:/mnt/irina_storage
```

### 4.2 Horace Service Requirements

**File watcher:**
- Use `watchdog` (Python) or `inotify-tools`
- Queue events for async processing
- Batch catalog updates for performance

**Catalog schema:**
```sql
CREATE TABLE files (
    file_id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    owner TEXT,
    size INTEGER,
    mime_type TEXT,
    checksum TEXT,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    zfs_snapshot TEXT  -- e.g., 'horace/files@2025-10-06T19:30:00'
);
```

**MCP API endpoints:** (existing, no changes)
- Search, get info, restore version, collections

---

## 5. Migration Path

### Phase 1: Add ZFS backing
- Create ZFS dataset
- Mount in Horace container
- Migrate existing test data

### Phase 2: Add file watcher
- Implement inotify listener
- Auto-catalog on file writes
- Create snapshots per write

### Phase 3: Service integration
- Add volume mounts to all service docker-compose files
- Update service code to write to `/mnt/irina_storage/files/<service>/`
- Remove manual `register_file` calls

### Phase 4: Advanced features
- Implement tiered storage
- Add deduplication
- Build search indexes

---

## 6. Benefits

1. **Zero-touch cataloging** - services just write files
2. **Automatic versioning** - every write creates a snapshot
3. **Content verification** - ZFS checksums detect corruption
4. **Time-travel** - restore any file to any point in time
5. **Efficient storage** - compression + dedup save space
6. **Future-proof** - tiered storage handles scale

---

## 7. Example Workflows

### Playfair creates diagram:
```python
# Playfair just writes to its namespace
svg_path = Path("/mnt/irina_storage/files/playfair/paper_00_quaternary.svg")
svg_path.parent.mkdir(parents=True, exist_ok=True)
svg_path.write_text(svg_content)

# Done! Horace automatically:
# - Detects the new file
# - Catalogs it with metadata
# - Creates ZFS snapshot
# - Makes it searchable
```

### Gates embeds diagram:
```python
# Gates references by path (Horace resolves)
diagram_path = "/mnt/irina_storage/files/playfair/paper_00_quaternary.svg"
embed_diagram(diagram_path)

# Or Gates searches Horace catalog:
results = horace_search_files(tags=["paper_00"], owner="playfair")
diagram_id = results[0]["file_id"]
```

### User retrieves version history:
```bash
# Via Horace MCP API
horace_get_file_info(file_id="abc-123", include_versions=True)

# Returns all ZFS snapshots:
# v1: horace/files@2025-10-06T10:00:00
# v2: horace/files@2025-10-06T11:30:00
# v3: horace/files@2025-10-06T15:45:00
```

---

## 8. Summary

Horace evolves from a **file registry** to a **NAS gateway** backed by ZFS:

**Before:** Services manually register files → Horace tracks paths
**After:** Services write to Horace's NAS → Horace auto-catalogs with ZFS superpowers

This matches the ICCM principle: **infrastructure should be transparent and just work**.
