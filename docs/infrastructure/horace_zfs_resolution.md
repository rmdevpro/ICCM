# CRITICAL: Horace Not Managing ZFS on Physical Server

**Severity:** CRITICAL - Complete Build Failure
**Status:** BLOCKING ALL WORK
**Date Discovered:** 2025-10-07
**Discovered By:** User inspection during /mnt/irina_storage setup

---

## Problem Statement

**Horace is currently NOT managing a ZFS filesystem on the physical server (Irina at 192.168.1.210) as designed.**

Instead, the current implementation uses a bind mount to `/mnt/projects/ICCM/irina_storage_test/files/` which is:
1. NOT on ZFS
2. NOT managed by Horace's versioning system
3. Located in the git repository directory (WRONG)
4. A temporary test configuration that became production

**This is a foundational architecture failure.**

---

## Current (Incorrect) Configuration

From `/mnt/projects/Joshua/mads/horace/docker-compose.yml`:

```yaml
volumes:
  # TEST DEPLOYMENT: Using regular directories (ZFS not available on this host)
  # For production, use ZFS datasets as documented in README.md
  irina_storage_files:
    driver: local
    driver_opts:
      type: none
      device: /mnt/projects/ICCM/irina_storage_test/files  # ❌ WRONG
      o: bind
```

**This comment says "TEST DEPLOYMENT" but it's running in production!**

---

## Expected (Correct) Configuration

Per Horace README.md, it should be:

```yaml
volumes:
  irina_storage_files:
    driver: local
    driver_opts:
      type: none
      device: /mnt/tank/horace/files  # ✅ ZFS dataset on Irina
      o: bind
```

Where `/mnt/tank/horace/files` is a ZFS dataset on the physical server (192.168.1.210) with:
- Compression enabled
- SHA-256 checksums
- Automatic snapshots
- Version tracking
- Proper backup strategy

---

## Impact

**ALL ecosystem components are affected:**
- Fiedler writes to wrong location
- Dewey/Godot logs may be in wrong location
- Playfair diagrams in wrong location
- Gates documents in wrong location
- No versioning, no snapshots, no ZFS benefits
- Files are in the git repository (!) - terrible practice

**This affects:**
- File durability (no ZFS protection)
- Version history (Horace catalog may be wrong)
- Backups (no ZFS snapshots)
- Performance (no ZFS caching/compression)
- Architecture integrity

---

## Root Cause

The docker-compose.yml has a comment "ZFS not available on this host" - meaning:
1. Horace was deployed without ZFS being set up first
2. A test configuration was used
3. That test configuration became the production configuration
4. Nobody verified ZFS was actually working

---

## Required Fix

### On Physical Server (192.168.1.210 - Irina)

1. **Install ZFS** (if not already installed):
   ```bash
   sudo apt install zfsutils-linux
   ```

2. **Create ZFS Pool** (if doesn't exist):
   ```bash
   # Identify disk (e.g., /dev/sdb)
   lsblk

   # Create pool
   sudo zpool create tank /dev/sdb
   ```

3. **Create Horace Datasets**:
   ```bash
   sudo zfs create tank/horace
   sudo zfs create -o compression=lz4 \
                   -o checksum=sha256 \
                   -o atime=off \
                   -o dedup=off \
                   -o mountpoint=/mnt/tank/horace/files \
                   tank/horace/files

   sudo zfs create -o compression=lz4 \
                   -o atime=off \
                   -o mountpoint=/mnt/tank/horace/metadata \
                   tank/horace/metadata

   sudo chown -R 1000:1000 /mnt/tank/horace/files
   sudo chown -R 1000:1000 /mnt/tank/horace/metadata
   ```

4. **Migrate Data**:
   ```bash
   # Copy existing data from wrong location to ZFS
   sudo rsync -av /mnt/projects/ICCM/irina_storage_test/files/ \
               /mnt/tank/horace/files/
   ```

5. **Update docker-compose.yml**:
   ```yaml
   volumes:
     irina_storage_files:
       driver: local
       driver_opts:
         type: none
         device: /mnt/tank/horace/files
         o: bind

     irina_storage_metadata:
       driver: local
       driver_opts:
         type: none
         device: /mnt/tank/horace/metadata
         o: bind
   ```

6. **Enable ZFS Snapshots**:
   ```yaml
   environment:
     - ENABLE_ZFS_SNAPSHOTS=true
     - ZFS_DATASET=tank/horace/files
   ```

7. **Restart Horace**:
   ```bash
   cd /mnt/projects/Joshua/mads/horace
   docker-compose down
   docker-compose up -d
   ```

8. **Verify**:
   ```bash
   sudo zfs list
   docker exec horace-mcp ls -la /mnt/irina_storage/files/
   ```

---

## Verification Checklist

- [ ] ZFS installed on 192.168.1.210
- [ ] ZFS pool created
- [ ] Horace datasets created with correct properties
- [ ] Existing data migrated to ZFS
- [ ] docker-compose.yml updated to use ZFS paths
- [ ] ZFS snapshots enabled
- [ ] Horace container restarted
- [ ] All services can write to new location
- [ ] Horace catalog tracking working
- [ ] ZFS snapshots being created
- [ ] Old test location removed

---

## Priority

**CRITICAL - BLOCKING ALL WORK**

No development or deployment should proceed until this is fixed. This is a foundational infrastructure issue that affects:
- Data durability
- System architecture
- Backup strategy
- All ecosystem components

---

## Next Steps

1. User to verify ZFS status on 192.168.1.210
2. Create ZFS pool and datasets if needed
3. Migrate data
4. Update configuration
5. Test and verify
6. Document in CLAUDE.md

---

**Assigned To:** User + Claude
**Blocks:** All development and deployment work
**Estimated Time:** 1-2 hours (if ZFS pool exists), 2-4 hours (if creating from scratch)

---

## RESOLUTION

**Status:** ✅ RESOLVED
**Date Resolved:** 2025-10-07
**Resolution Time:** ~2 hours

### What Was Done

1. **Installed ZFS** on physical server (192.168.1.210):
   ```bash
   sudo apt-get install -y zfsutils-linux
   ```

2. **Created ZFS pool** using file-based approach on existing 44TB RAID:
   ```bash
   sudo mkdir -p /mnt/storage/zfs
   sudo truncate -s 100G /mnt/storage/zfs/irina-pool.img
   sudo zpool create irina /mnt/storage/zfs/irina-pool.img
   ```

3. **Created Horace datasets** with proper settings:
   ```bash
   sudo zfs create irina/horace
   sudo zfs create -o compression=lz4 -o checksum=sha256 -o atime=off \
                   -o mountpoint=/mnt/irina_storage/files irina/horace/files
   sudo zfs create -o compression=lz4 -o checksum=sha256 -o atime=off \
                   -o mountpoint=/mnt/irina_storage/metadata irina/horace/metadata
   sudo chown -R 1000:1000 /mnt/irina_storage/files
   sudo chown -R 1000:1000 /mnt/irina_storage/metadata
   ```

4. **Migrated data** from test location:
   ```bash
   rsync -av /mnt/projects/ICCM/irina_storage_test/files/ /mnt/irina_storage/files/
   ```

5. **Updated docker-compose.yml** on both local and remote machines to use ZFS paths

6. **Mounted ZFS remotely** via SSHFS on local machine:
   ```bash
   sudo mkdir -p /mnt/irina_storage
   sudo sshfs -o password_stdin,allow_other,default_permissions \
              aristotle9@192.168.1.210:/mnt/irina_storage /mnt/irina_storage
   ```

7. **Restarted Horace** - now running with ZFS snapshots enabled

### Verification Results

- ✅ ZFS pool "irina" operational (100GB)
- ✅ Compression working (1.43x ratio on files dataset)
- ✅ SHA-256 checksums enabled
- ✅ 96 files automatically cataloged on startup
- ✅ File watcher detecting new files
- ✅ Horace logging: "ZFS snapshotting enabled for dataset 'irina/horace/files'"
- ✅ All ecosystem components can now write to ZFS-backed storage

### Architecture Notes

**Physical Server (192.168.1.210):**
- ZFS pool: `irina` (100GB file-based on ext4 RAID)
- Datasets automatically mounted:
  - `irina/horace/files` → `/mnt/irina_storage/files`
  - `irina/horace/metadata` → `/mnt/irina_storage/metadata`

**Local Machine (Development):**
- SSHFS mount: `/mnt/irina_storage` → `aristotle9@192.168.1.210:/mnt/irina_storage`
- Horace container uses bind mounts to this SSHFS location

**Benefits Achieved:**
- Automatic compression (43% space savings)
- End-to-end data integrity via SHA-256
- ZFS snapshot capability for point-in-time recovery
- Automatic file cataloging and versioning
- All ecosystem components protected by ZFS
