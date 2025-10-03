# ICCM System Requirements - Extracted from Research Papers

## Executive Summary
- Total requirements extracted: 45
- Requirements by component: A: 12, B: 8, C: 7, D: 6, E: 5, F: 3, G: 4
- Critical requirements: 18
- Key dependencies identified: Foundation layer (F) prerequisites for all components; CET-D Training Pipeline (A) depends on LLM Orchestra (B) and Conversation Storage (E); Test Lab (D) integrates with Requirements Domain (C)
- Ambiguities requiring user decision: 2

## Requirements by Component

### A. CET-D Training Pipeline
REQ-CET-001: Four-Phase Progressive Training Framework

Source: Paper 01 - Intelligent Context and Conversation Management (ICCM): Learning Context Engineering Through Progressive Training with Interactive Feedback (Section 4)

Description:  
Implement a training pipeline that supports four progressive phases: Phase 1 for subject expertise acquisition using RAG-grounded training, Phase 2 for context engineering skills using Phase 1 data, Phase 3 for interactive optimization with LLM feedback, and Phase 4 for minimal self-improvement through self-critique.

Acceptance Criteria:  
- Pipeline executes all four phases sequentially with data flow from Phase N to N+1.  
- Phase transitions preserve at least 95% of conversation data.  
- System logs phase completion with metrics (e.g., loss values).

Priority: Critical

Dependencies: [REQ-ORC-001, REQ-CONV-001]

Notes: Phase 4 is minimal for PoC; focus on offline self-critique.

REQ-CET-002: Phase 1 RAG-Grounded Training

Source: Paper 01 - Intelligent Context and Conversation Management (ICCM): Learning Context Engineering Through Progressive Training with Interactive Feedback (Section 4.1); Paper 02 - Four-Phase Progressive Training for Context Engineering Transformers: Requirements Engineering Specialization (Section 2)

Description:  
Build RAG-grounded training for Phase 1 using multi-LLM supervision on requirements engineering standards (IEEE 29148-2018, ISO/IEC/IEEE 12207, SWEBOK) and 100 Python applications.

Acceptance Criteria:  
- System retrieves and synthesizes from knowledge bases with >90% relevance.  
- Generates 10,000+ conversation histories validated by multiple LLMs.  
- Stores conversations for Phase 2 use.

Priority: High

Dependencies: [REQ-FOUND-001]

Notes: Use pgvector for RAG.

REQ-CET-003: Phase 2 Context Transformation Training

Source: Paper 01 (Section 4.2); Paper 02 (Section 3)

Description:  
Implement training on Phase 1 conversation pairs to learn context transformation from poor to excellent, handling varying documentation quality.

Acceptance Criteria:  
- Transforms incomplete inputs into structured requirements with >80% accuracy.  
- Handles well-documented, partially documented, and undocumented code.  
- Produces transformation patterns for Phase 3.

Priority: High

Dependencies: [REQ-CET-002]

Notes: Use quality gradients for training pairs.

REQ-CET-004: Phase 3 Interactive Optimization

Source: Paper 01 (Section 4.3); Paper 02 (Section 4); Paper 04A (Section 4)

Description:  
Create interactive training loop where CET generates context, observes LLM responses, evaluates quality, and refines strategies using reconstruction testing feedback.

Acceptance Criteria:  
- Achieves >75% test pass rate on reconstructions.  
- Learns from multi-LLM response variance.  
- Updates context engineering strategies based on outcomes.

Priority: Critical

Dependencies: [REQ-ORC-002, REQ-LAB-001]

Notes: Critical phase for learning effectiveness.

REQ-CET-005: Phase 4 Minimal Self-Improvement

Source: Paper 01 (Section 4.4); Paper 02 (Section 5); Paper 04B (Section 5)

Description:  
Implement offline self-critique and refinement during deployment, using production feedback for continuous improvement.

Acceptance Criteria:  
- Performs self-assessment with >85% accuracy.  
- Improves reconstruction pass rate by +3-5% monthly.  
- Maintains database of error patterns.

Priority: High

Dependencies: [REQ-CET-004, REQ-CONV-003]

Notes: Minimal for PoC; offline only.

REQ-CET-006: CET-D Specialization for Requirements Engineering

Source: Paper 01 (Section 5.3.2); Paper 05 (Section 1)

Description:  
Specialize CET-D for requirements engineering domain with ~3-7B parameters, focusing on software development.

Acceptance Criteria:  
- Achieves >75% test pass rate on reconstructed applications.  
- Outperforms RAG baseline by >15% (p<0.05).  
- Uses 50-application dataset (40 train/10 holdout).

Priority: Critical

Dependencies: [REQ-REQ-001]

Notes: Initial PoC focus.

REQ-CET-007: Multi-LLM Supervision in Phase 1

Source: Paper 02 (Section 2.4)

Description:  
Implement multi-LLM supervision for Phase 1 using premium APIs, Together.AI models, and local models for diverse requirement quality evaluation.

Acceptance Criteria:  
- Aggregates evaluations from at least 3 LLMs per conversation.  
- Computes consensus scoring with >85% agreement.  
- Updates CET based on feedback loss.

Priority: High

Dependencies: [REQ-ORC-001]

Notes: None

REQ-CET-008: Reconstruction Testing in Phase 3

Source: Paper 02 (Section 4.4); Paper 06 (Section 4)

Description:  
Integrate reconstruction testing where LLMs implement from requirements and execute original tests for feedback.

Acceptance Criteria:  
- Achieves >75% average test pass rate.  
- Analyzes failures to identify missing/ambiguous requirements.  
- Updates CET weights based on reconstruction loss.

Priority: Critical

Dependencies: [REQ-LAB-002]

Notes: Uses 5 LLMs for diversity.

REQ-CET-009: Self-Critique Mechanism in Phase 4

Source: Paper 02 (Section 5.2)

Description:  
Build self-critique for Phase 4 where CET assesses its context and refines if confidence <0.8.

Acceptance Criteria:  
- Self-critique accuracy >85%.  
- Refines context leading to >80% production pass rate.  
- Logs improvement trends.

Priority: Medium

Dependencies: [REQ-CET-005]

Notes: Offline for PoC.

REQ-CET-010: Training Infrastructure Requirements

Source: Paper 01 (Section 5.2); Paper 08 (Section 2.1)

Description:  
Set up training infrastructure with M5 server (4x P40, 1x V100) for CET training and LLM orchestra.

Acceptance Criteria:  
- Supports Phase 3 with multiple LLM inference.  
- Achieves training on 3-7B models.  
- Integrates with Irina for storage.

Priority: Critical

Dependencies: [REQ-FOUND-002]

Notes: Constrained to existing hardware.

REQ-CET-011: Progressive Training Timeline

Source: Paper 02 (Section 6.3)

Description:  
Implement training timeline: Phase 1 (2 months), Phase 2 (2 months), Phase 3 (3 months), Phase 4 ongoing.

Acceptance Criteria:  
- Completes to production in 7 months.  
- Tracks milestones with metrics.  
- Adjusts based on phase outputs.

Priority: High

Dependencies: [REQ-CROSS-001]

Notes: Total 7 months to production.

REQ-CET-012: Catastrophic Forgetting Prevention

Source: Paper 02 (Section 7.3)

Description:  
Implement canary set of 10 applications for regression detection during Phase 4.

Acceptance Criteria:  
- Tests canary set every 1,000 steps.  
- Triggers rollback if >5% degradation.  
- Uses experience replay for affected categories.

Priority: Medium

Dependencies: [REQ-REQ-002]

Notes: Separate from train/holdout sets.

### B. LLM Orchestra
REQ-ORC-001: Multi-LLM Team Composition

Source: Paper 01 (Section 4.3); Paper 10 (Section 2.1)

Description:  
Assemble LLM team with premium APIs, Together.AI models, and local models for diverse feedback.

Acceptance Criteria:  
- Includes at least 5 models from different families.  
- Provides diverse response patterns.  
- Integrates with training loop.

Priority: Critical

Dependencies: None

Notes: Simulates varied downstream behaviors.

REQ-ORC-002: Model Rotation Strategy

Source: Paper 10 (Section 3.4)

Description:  
Implement phase-specific model rotation every 4-12 hours for diversity.

Acceptance Criteria:  
- Rotates without >1% overhead.  
- Maintains 15-20 unique models per phase.  
- Logs rotation events.

Priority: High

Dependencies: [REQ-FOUND-003]

Notes: Frequency varies by phase.

REQ-ORC-003: Intelligent Routing and Load Balancing

Source: Paper 10 (Section 5.1)

Description:  
Build router that selects models based on speed, quality, or specialization.

Acceptance Criteria:  
- Routes with <10ms overhead.  
- Balances load across tiers.  
- Prefers local over cloud.

Priority: High

Dependencies: [REQ-ORC-001]

Notes: Includes fallback chain.

REQ-ORC-004: Response Caching

Source: Paper 10 (Section 6.1)

Description:  
Implement semantic caching with Redis for repeated prompts.

Acceptance Criteria:  
- Achieves 35-42% hit rate.  
- TTL 1 hour.  
- Max size 100GB.

Priority: Medium

Dependencies: [REQ-FOUND-001]

Notes: Uses SentenceTransformer for similarity.

REQ-ORC-005: Parallel Processing

Source: Paper 10 (Section 7.2)

Description:  
Enable parallel inference across loaded models.

Acceptance Criteria:  
- 3-6x throughput improvement.  
- Handles batches of 20-30 requests.  
- <1% error rate.

Priority: High

Dependencies: [REQ-ORC-002]

Notes: Uses ThreadPoolExecutor.

REQ-ORC-006: Cost Management

Source: Paper 10 (Section 8.2)

Description:  
Implement cost-aware routing preferring local models.

Acceptance Criteria:  
- Keeps monthly cost < $200.  
- Alerts on spikes >$50/day.  
- Routes 95% to local.

Priority: High

Dependencies: [REQ-CROSS-002]

Notes: Throttles expensive APIs.

REQ-ORC-007: Monitoring and Alerting

Source: Paper 10 (Section 10.2)

Description:  
Set up metrics dashboard and alerts for critical issues.

Acceptance Criteria:  
- Tracks latency, availability, cost.  
- Alerts on all models offline.  
- Daily summaries.

Priority: Medium

Dependencies: [REQ-FOUND-003]

Notes: Simple Slack/email alerts.

REQ-ORC-008: Model Library Management

Source: Paper 10 (Section 3.3)

Description:  
Manage 50+ model variants on Irina storage with dynamic loading.

Acceptance Criteria:  
- Loads models in <15s from RAM cache.  
- Supports quantization.  
- Handles multi-GPU distribution.

Priority: Critical

Dependencies: [REQ-LAB-001]

Notes: Total ~2TB for variants.

### C. Requirements Engineering Domain
REQ-REQ-001: 50-Application Dataset Preparation

Source: Paper 01 (Section 6.4); Paper 02 (Section 2.3)

Description:  
Prepare dataset of 50 Python applications (40 train/10 holdout) with varying complexity.

Acceptance Criteria:  
- 100% manual validation.  
- Mix of simple/medium/complex apps.  
- Includes code, tests, documentation.

Priority: Critical

Dependencies: None

Notes: Quality over quantity.

REQ-REQ-002: Requirements Extraction Capabilities

Source: Paper 05 (Section 2.1)

Description:  
Implement extraction of functional, non-functional, and technical requirements from code.

Acceptance Criteria:  
- Extracts with >85% completeness.  
- Structures according to standards.  
- Handles implicit requirements.

Priority: High

Dependencies: [REQ-CET-006]

Notes: Uses taxonomy.

REQ-REQ-003: Reconstruction Testing Validation

Source: Paper 06 (Section 4.1)

Description:  
Validate requirements via multi-LLM reconstruction and test execution.

Acceptance Criteria:  
- >75% test pass rate.  
- >90% API compatibility.  
- Analyzes failures.

Priority: Critical

Dependencies: [REQ-LAB-003]

Notes: Objective metric.

REQ-REQ-004: Gold Standard Baseline

Source: Paper 02 (Section 9.4)

Description:  
Create manual gold standard requirements with two reviewers + adjudication.

Acceptance Criteria:  
- >85% test pass rate.  
- Documents decisions.  
- Used for comparisons.

Priority: High

Dependencies: None

Notes: Establishes upper bound.

REQ-REQ-005: RAG Baseline Comparison

Source: Paper 02 (Section 7.4)

Description:  
Implement RAG baseline using pgvector for comparison.

Acceptance Criteria:  
- ~60% test pass rate.  
- Used in paired t-test.  
- Detects 15% improvement.

Priority: Medium

Dependencies: [REQ-FOUND-001]

Notes: Competitive automated baseline.

REQ-REQ-006: Statistical Validation

Source: Paper 01 (Section 6.4)

Description:  
Perform paired t-test (α=0.05, 80% power) on 40 training apps.

Acceptance Criteria:  
- CET-D > RAG by 15% (p<0.05).  
- Uses hold-out set for generalization.  
- Reports confidence intervals.

Priority: High

Dependencies: [REQ-REQ-003]

Notes: Null hypothesis: CET-D ≤ RAG.

REQ-REQ-007: Application Analysis Capabilities

Source: Paper 05 (Section 3.1)

Description:  
Build analysis of application structure, behavior, API, data flow.

Acceptance Criteria:  
- Identifies entry points, interactions, logic.  
- Analyzes for requirements extraction.  
- Handles Python apps.

Priority: High

Dependencies: None

Notes: For requirements domain.

### D. Test Lab Infrastructure
REQ-LAB-001: Containerized Isolation

Source: Paper 09 (Section 2.1)

Description:  
Use Docker for isolated code execution with network_mode: none.

Acceptance Criteria:  
- Supports 15+ languages.  
- 600-1000 executions/day.  
- Zero security incidents.

Priority: Critical

Dependencies: [REQ-FOUND-002]

Notes: Simple for small labs.

REQ-LAB-002: Multi-Language Support

Source: Paper 09 (Section 2.1)

Description:  
Create Docker images for Python, JavaScript, Java, etc.

Acceptance Criteria:  
- Executes code in isolated containers.  
- Tiered pre-warming.  
- Handles 82% usage in top 2 languages.

Priority: High

Dependencies: [REQ-LAB-001]

Notes: Minimal base images.

REQ-LAB-003: Execution Workflow

Source: Paper 09 (Section 3.1)

Description:  
Implement code execution API with timeout and result parsing.

Acceptance Criteria:  
- <2s latency for Python.  
- Handles tests with parsing.  
- Batch execution support.

Priority: High

Dependencies: [REQ-LAB-001]

Notes: Uses container pooling.

REQ-LAB-004: Hardware Setup

Source: Paper 08 (Section 2.1)

Description:  
Configure M5 (4x P40, 1x V100) and Irina (2x P4, 60TB).

Acceptance Criteria:  
- Supports LLM orchestra and training.  
- 156GB total VRAM.  
- Tiered storage.

Priority: Critical

Dependencies: None

Notes: $7,840 total cost.

REQ-LAB-005: Monitoring

Source: Paper 09 (Section 5.1)

Description:  
Implement basic logging and daily summaries.

Acceptance Criteria:  
- Logs executions with metrics.  
- Generates reports.  
- No enterprise stack.

Priority: Medium

Dependencies: [REQ-CROSS-003]

Notes: Simple for small teams.

REQ-LAB-006: Security Configuration

Source: Paper 09 (Section 4.2)

Description:  
Apply network isolation, resource limits, read-only filesystem.

Acceptance Criteria:  
- Prevents LLM accidents.  
- Handles infinite loops, deletions.  
- Zero incidents in 6 months.

Priority: High

Dependencies: [REQ-LAB-001]

Notes: For trusted labs.

### E. Conversation Storage & Retrieval
REQ-CONV-001: Core Conversation Schema

Source: Paper 12 (Section 3.1)

Description:  
Implement PostgreSQL schema for conversations and messages with embeddings.

Acceptance Criteria:  
- Stores phase, tags, metadata.  
- Supports GIN indices.  
- pgvector for embeddings.

Priority: Critical

Dependencies: [REQ-FOUND-001]

Notes: Uses vector(1536).

REQ-CONV-002: Phase-Specific Tables

Source: Paper 12 (Section 3.2-3.5)

Description:  
Create tables for Phase 1-4 specific data.

Acceptance Criteria:  
- References conversations table.  
- Indices for common queries.  
- Handles JSONB metadata.

Priority: High

Dependencies: [REQ-CONV-001]

Notes: Phase 3 has llm_responses array.

REQ-CONV-003: Retrieval Patterns

Source: Paper 12 (Section 4.1-4.3)

Description:  
Implement queries for phase transitions and semantic search.

Acceptance Criteria:  
- Sub-100ms latency.  
- Semantic search with distance.  
- Caches hot queries.

Priority: High

Dependencies: [REQ-CONV-001]

Notes: Uses <=> for vector search.

REQ-CONV-004: Data Lifecycle Management

Source: Paper 12 (Section 6.1)

Description:  
Automate archival to slow tier based on age.

Acceptance Criteria:  
- Moves data per retention policies.  
- Compresses archived data.  
- Maintains 40% headroom.

Priority: Medium

Dependencies: [REQ-CONV-001]

Notes: Fast tier for active data.

REQ-CONV-005: Integration with Training

Source: Paper 12 (Section 8.1)

Description:  
Store Phase 1 conversations with RAG sources.

Acceptance Criteria:  
- Inserts with embeddings.  
- Aggregates messages.  
- Handles 100K conversations.

Priority: High

Dependencies: [REQ-CET-001]

Notes: For Phase 2 use.

### F. Foundation Layer
REQ-FOUND-001: Data Management and Backup

Source: Paper 01 (Section 6.4); Paper 08 (Section 5.3)

Description:  
Implement 3-2-1 backup with nightly snapshots to Irina NAS.

Acceptance Criteria:  
- RTO 24 hours, RPO 24 hours.  
- SHA-256 checksum verification.  
- Quarterly recovery drills.

Priority: Critical

Dependencies: None

Notes: For all data.

REQ-FOUND-002: Networking and Connectivity

Source: Paper 08 (Section 4.1)

Description:  
Configure TP-Link ER7206 router and TL-SG1428PE switch with bonding.

Acceptance Criteria:  
- 2Gb/s aggregate bandwidth.  
- VLAN support if needed.  
- <1% packet loss.

Priority: High

Dependencies: None

Notes: Resolves bottlenecks.

REQ-FOUND-003: Monitoring and Observability

Source: Paper 08 (Section 7.1)

Description:  
Set up Prometheus, Grafana for metrics; Elasticsearch for logs.

Acceptance Criteria:  
- Tracks GPU utilization, latency.  
- Alerts on critical issues.  
- Custom dashboards.

Priority: Medium

Dependencies: None

Notes: For all components.

### G. Cross-Cutting Concerns
REQ-CROSS-001: Operational Requirements

Source: Paper 02 (Section 6.3)

Description:  
Ensure system supports 10-16 week timeline for PoC.

Acceptance Criteria:  
- Weekly progress checkpoints.  
- Adjusts for iterations.  
- Meets milestones.

Priority: High

Dependencies: None

Notes: 2.5-4 months total.

REQ-CROSS-002: Performance Requirements

Source: Paper 01 (Section 7.1)

Description:  
Achieve >70% reduction in irrelevant information, >30% task accuracy improvement.

Acceptance Criteria:  
- Meets token efficiency targets.  
- >25% faster inference.  
- Monitored in all phases.

Priority: Critical

Dependencies: None

Notes: Theoretical projections.

REQ-CROSS-003: Testing and Validation Requirements

Source: Paper 01 (Section 6.1)

Description:  
Implement context quality metrics (relevance, coherence, efficiency).

Acceptance Criteria:  
- Measures downstream task accuracy.  
- Compares vs baselines.  
- Statistical rigor.

Priority: High

Dependencies: [REQ-REQ-006]

Notes: Paired t-test.

REQ-CROSS-004: Security Requirements

Source: Paper 09 (Section 4.1)

Description:  
Enforce container isolation, resource limits for code execution.

Acceptance Criteria:  
- Zero incidents.  
- Prevents exhaustion, deletions.  
- Audits data flows.

Priority: High

Dependencies: [REQ-LAB-001]

Notes: For trusted labs.

## Traceability Matrix

| Requirement ID | Source Papers | Priority | Status |
|----------------|---------------|----------|--------|
| REQ-CET-001    | 01, 02        | Critical | New    |
| REQ-CET-002    | 01, 02        | High     | New    |
| REQ-CET-003    | 01, 02        | High     | New    |
| REQ-CET-004    | 01, 02, 04A   | Critical | New    |
| REQ-CET-005    | 01, 02, 04B   | High     | New    |
| REQ-CET-006    | 01, 05        | Critical | New    |
| REQ-CET-007    | 02            | High     | New    |
| REQ-CET-008    | 02, 06        | Critical | New    |
| REQ-CET-009    | 02            | Medium   | New    |
| REQ-CET-010    | 01, 08        | Critical | New    |
| REQ-CET-011    | 02            | High     | New    |
| REQ-CET-012    | 02            | Medium   | New    |
| REQ-ORC-001    | 01, 10        | Critical | New    |
| REQ-ORC-002    | 10            | High     | New    |
| REQ-ORC-003    | 10            | High     | New    |
| REQ-ORC-004    | 10            | Medium   | New    |
| REQ-ORC-005    | 10            | High     | New    |
| REQ-ORC-006    | 10            | High     | New    |
| REQ-ORC-007    | 10            | Medium   | New    |
| REQ-ORC-008    | 10            | Critical | New    |
| REQ-REQ-001    | 01, 02        | Critical | New    |
| REQ-REQ-002    | 05            | High     | New    |
| REQ-REQ-003    | 02, 06        | Critical | New    |
| REQ-REQ-004    | 02            | High     | New    |
| REQ-REQ-005    | 02            | Medium   | New    |
| REQ-REQ-006    | 01            | High     | New    |
| REQ-REQ-007    | 05            | High     | New    |
| REQ-LAB-001    | 09            | Critical | New    |
| REQ-LAB-002    | 09            | High     | New    |
| REQ-LAB-003    | 09            | High     | New    |
| REQ-LAB-004    | 08            | Critical | New    |
| REQ-LAB-005    | 09            | Medium   | New    |
| REQ-LAB-006    | 09            | High     | New    |
| REQ-CONV-001   | 12            | Critical | New    |
| REQ-CONV-002   | 12            | High     | New    |
| REQ-CONV-003   | 12            | High     | New    |
| REQ-CONV-004   | 12            | Medium   | New    |
| REQ-CONV-005   | 12            | High     | New    |
| REQ-FOUND-001  | 01, 08        | Critical | New    |
| REQ-FOUND-002  | 08            | High     | New    |
| REQ-FOUND-003  | 08            | Medium   | New    |
| REQ-CROSS-001  | 02            | High     | New    |
| REQ-CROSS-002  | 01            | Critical | New    |
| REQ-CROSS-003  | 01            | High     | New    |
| REQ-CROSS-004  | 09            | High     | New    |

## Ambiguities and Questions

1. **Phase 4 Self-Improvement Scope**: Papers 01 and 02 have conflicting statements about Phase 4.
   - Paper 01 says: "Continuous self-improvement during deployment" (Section 4.4)
   - Paper 02 says: "Minimal continuous self-improvement (offline self-critique)" (Section 5)
   - Recommendation: Follow scope.md - minimal offline for PoC, defer production scaling.

2. **Dataset Size**: Paper 01 mentions 50 applications, but Paper 05 mentions 100 Python applications.
   - Paper 01 says: "50 high-quality real-world applications" (Section 6.4)
   - Paper 05 says: "100 Python Applications" (Section 2.3)
   - Recommendation: Use 50 as per scope.md for PoC; note as potential expansion.

## Dependencies and Ordering

1. **Foundation requirements** must be satisfied before:
   - All components (A-G)

2. **CET-D Phase 1** requires:
   - REQ-FOUND-001, REQ-ORC-001, REQ-CONV-001

3. **Critical path:** REQ-FOUND-001 → REQ-LAB-001 → REQ-ORC-001 → REQ-CET-001 → REQ-REQ-003

## Implementation Recommendations

Based on requirements extraction:
1. Build foundation layer first (storage, networking, monitoring)
2. Prototype LLM Orchestra and Test Lab early for integration testing
3. Validate Phase 1 training before proceeding to Phase 2
