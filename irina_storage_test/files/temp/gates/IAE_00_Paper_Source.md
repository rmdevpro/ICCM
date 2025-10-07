# Intelligent Agentic Engineering (IAE): A Unified Discipline for Agent Assembly

**Authors:** Council of Elders (Gemini 2.5 Pro, GPT-5, Claude Opus 4)
**Version:** 1.1
**Date:** October 6, 2025
**Status:** Academic/Business Publication Draft

---

## Abstract

We present **Intelligent Agentic Engineering (IAE)**, a comprehensive discipline for designing, assembling, and operating intelligent agents built on the **Multipurpose Agentic Duo (MAD)** architecture pattern. IAE unifies four specialized disciplines—context management (ICCM), decision engineering (IDE), execution engineering (IEE), and agent assembly (IAE)—into a cohesive framework that separates cognition from action. Through canonical contracts, versioned conversation protocols, and a centralized State Manager, IAE enables independent evolution of components while maintaining system coherence. We demonstrate how the operational feedback loop—from decision through execution to state update—creates a continuous improvement cycle that enhances agent performance over time.

**Keywords:** Intelligent Agents, MAD Architecture, Agent Assembly, Context Engineering, Decision Engineering, Execution Engineering, State Management

---

## 1. Introduction

### 1.1 Motivation

Modern intelligent agents face a fundamental challenge: how to combine deliberative reasoning with rapid execution while maintaining auditability, safety, and adaptability. Traditional monolithic architectures tightly couple decision-making with action, creating brittle systems that are difficult to test, validate, and evolve.

**Intelligent Agentic Engineering (IAE)** addresses this challenge through systematic separation of concerns. IAE is not merely an architecture—it is a complete discipline for agent assembly that integrates four specialized engineering domains into a unified whole.

### 1.2 Core Insight: Separation of Cognition from Action

IAE produces agents based on the **Multipurpose Agentic Duo (MAD)** pattern, which separates:

- **Thinking Engine:** Deliberative, auditable, model-agnostic cognition
- **Doing Engine:** Fast, sandboxed, domain-specific execution

This separation enables independent optimization, testing, and evolution of each subsystem while maintaining strong contracts at their boundary.

![Figure 1: MAD Architecture showing separation of Thinking and Doing Engines](file:///mnt/irina_storage/files/temp/playfair/iae_diagrams/01_mad_architecture.svg)

*Figure 1: The MAD (Multipurpose Agentic Duo) architecture separates cognition (Thinking Engine) from action (Doing Engine), with the State Manager providing authoritative memory.*

---

## 2. Theoretical Foundation

### 2.1 The Quaternary Structure

IAE integrates four specialized disciplines, each addressing a distinct aspect of intelligent agent construction:

1. **ICCM (Intelligent Context & Conversation Management):** Transforms raw data into structured context through the Context Engineering Transformation (CET). Manages conversation archives and active sessions.

2. **IDE (Intelligent Decision Engineering):** Combines deterministic policy (Rules Engine) with synthesis under uncertainty (Decision Engine for Reasoning, DER) to produce executable decisions.

3. **IEE (Intelligent Execution Engineering):** Provides domain-specific tools and APIs, validates safety, and synthesizes execution outcomes.

4. **IAE (Intelligent Agentic Engineering):** Assembles Full MADs by integrating components from the other three disciplines. Owns the State Manager, which provides authoritative memory for all agents.

![Figure 2: Quaternary Structure showing integration of four disciplines](file:///mnt/irina_storage/files/temp/playfair/iae_diagrams/02_quaternary_structure.svg)

*Figure 2: The IAE Quaternary Structure integrates ICCM (context), IDE (decisions), IEE (execution), and IAE (assembly) into cohesive Full MADs.*

### 2.2 Thinking Engine Composition

The Thinking Engine comprises four components:

- **CET (ICCM):** Transformation-only context engineering. Converts raw inputs into structured representations.
- **Rules Engine (IDE):** Deterministic policy enforcement, safety guardrails, and pattern-matched responses for known regimes.
- **DER (IDE):** Synthesis under uncertainty, multi-objective arbitration, and consultative reasoning.
- **State Manager (IAE):** Authoritative memory system with tripartite data model (World Model, Task Context, Execution State).

**Note:** LLM Orchestra capability is provided externally by the **LLM Conductor** Half-MAD and accessed via conversations, not embedded in the Thinking Engine.

### 2.3 Doing Engine Philosophy

The Doing Engine (IEE) executes decisions through domain-specific tools and APIs. Key principles:

- **Sandboxed execution:** All actions occur in controlled environments with rollback capability.
- **Safety-first validation:** Preconditions verified before execution; outcomes validated afterward.
- **Outcome synthesis:** Raw results transformed into structured Execution Outcome Packages.
- **State reporting:** All execution results persisted via the State Manager for feedback loop integration.

### 2.4 Operational Feedback Loop

IAE's continuous improvement mechanism flows through six stages:

1. **Decision (IDE):** Rules Engine and DER produce a Decision Package.
2. **Execution (IEE):** Doing Engine performs the selected action.
3. **Outcome (IEE):** Safety validation and result synthesis.
4. **State Update (IAE):** Execution Outcome persisted; World Model updated.
5. **Context Refresh (ICCM):** New world state enriches context quality.
6. **Better Decision (IDE):** Improved context enables refined decisions.

![Figure 3: Operational Feedback Loop](file:///mnt/irina_storage/files/temp/playfair/iae_diagrams/03_feedback_loop.svg)

*Figure 3: The Operational Feedback Loop creates continuous improvement: decisions inform execution, outcomes update state, and refreshed context enables better future decisions.*

---

## 3. Architecture Components and Specifications

### 3.1 State Manager: The Authoritative Memory System

The State Manager (IAE-owned) is the backbone of all MADs. It provides:

#### 3.1.1 Tripartite Data Model

- **World Model:** Facts, entities, and relationships representing the agent's environment. Versioned and content-addressable.
- **Task Context:** Active goals, constraints, and problem frames for current work.
- **Execution State:** Historical record of decisions, outcomes, and state transitions.

#### 3.1.2 Global Properties

- **Versioning:** Immutable-by-default records with version IDs (ULIDs).
- **Content addressing:** Artifacts referenced by cryptographic hash.
- **Optimistic concurrency:** Readers never block writers; conflicts resolved through versioning.
- **Access control:** Tenancy separation and role-based permissions.
- **Time-travel reads:** Query any historical state version.

#### 3.1.3 Core APIs

**World Model APIs:**
- `get_world_snapshot(version_id)` → current world state
- `put_world_fact(fact, metadata)` → new version ID
- `query_world(predicate)` → matching facts

**Task Context APIs:**
- `create_task_context(objectives, constraints)` → task_id
- `read_task_context(task_id)` → current context
- `update_task_context(task_id, changes)` → new version

**Execution State APIs:**
- `start_execution(decision_id)` → execution_id
- `update_execution(execution_id, progress)` → status
- `complete_execution(execution_id, outcome)` → persisted

**Cross-Cutting APIs:**
- `persist_decision_package(decision)` → decision_id
- `persist_execution_outcome(outcome)` → outcome_id
- `get_reasoning_trace(decision_id)` → trace DAG

### 3.2 Canonical Contracts (Version 1)

IAE defines five canonical data schemas that enable independent component evolution:

#### 3.2.1 Structured Context (CET → IDE)

```json
{
  "context_id": "string (ULID)",
  "schema_version": "string",
  "task_id": "string",
  "problem_frame": {
    "objectives": [],
    "constraints": []
  },
  "features": [
    {"name": "...", "value": "..."}
  ],
  "world_refs": {
    "world_version_id": "..."
  }
}
```

#### 3.2.2 Rule Engine Output (Rules → DER)

```json
{
  "rule_output_id": "string",
  "schema_version": "string",
  "status": "enum {HIGH_CONFIDENCE_MATCH, PARTIAL_MATCH, NO_MATCH}",
  "matches": [
    {"rule_id": "...", "action_proposal": "..."}
  ],
  "guardrails_triggered": []
}
```

#### 3.2.3 Decision Package (DER → Doing Engine)

```json
{
  "decision_id": "string",
  "schema_version": "string",
  "task_id": "string",
  "selected_action": {
    "name": "string",
    "parameters": {},
    "preconditions": [],
    "expected_effects": []
  },
  "safety_assertions": [],
  "confidence_score": "0-1",
  "human_review_required": "bool",
  "reasoning_trace_ref": "string",
  "references": {
    "context_id": "...",
    "rule_output_id": "...",
    "world_version_id": "..."
  },
  "consultations": [
    {"provider": "LLM Conductor", "consultation_id": "..."}
  ]
}
```

#### 3.2.4 Execution Outcome Package (Doing Engine → State Manager)

```json
{
  "outcome_id": "string",
  "schema_version": "string",
  "decision_id": "string",
  "status": "enum {success, failure, partial, aborted}",
  "observed_effects": [],
  "deviations": [
    {"expected": "...", "observed": "..."}
  ],
  "safety_validation_results": [],
  "telemetry": {},
  "artifacts": [],
  "world_version_id_before": "string",
  "world_version_id_after": "string",
  "reengagement_advice": {}
}
```

#### 3.2.5 Reasoning Trace (Audit and Replay)

```json
{
  "trace_id": "string",
  "decision_id": "string",
  "schema_version": "string",
  "structure": "directed acyclic graph of nodes"
}
```

![Figure 4: Canonical Contracts Data Flow](file:///mnt/irina_storage/files/temp/playfair/iae_diagrams/04_canonical_contracts.svg)

*Figure 4: The five canonical contracts standardize data flow between components, enabling independent evolution while maintaining system coherence.*

---

## 4. Half-MADs: Reusable Capabilities

**Half-MADs** are minimal MADs that provide specific capabilities to all Full MADs via versioned conversation protocols. The seven canonical Half-MADs are:

1. **LLM Conductor:** Provides LLM Orchestra capability—multi-model consultative reasoning for synthesis under uncertainty.

2. **Dewey:** Conversation retrieval from immutable archives. Enables historical query and audit.

3. **Godot:** Active conversation management. Maintains real-time session state.

4. **Marco:** Session orchestration and budgeting. Manages resource allocation and conversation lifecycle.

5. **Horace:** File and artifact catalog with provenance tracking. Versioned storage for all generated artifacts.

6. **Gates:** Document generation with style and compliance enforcement. Produces formatted outputs (PDF, DOCX, ODT).

7. **Playfair:** Diagram and visualization generation. Creates professional graphics from structured specifications.

![Figure 5: Seven Canonical Half-MADs](file:///mnt/irina_storage/files/temp/playfair/iae_diagrams/05_halfmads_ecosystem.svg)

*Figure 5: The Half-MADs ecosystem provides reusable capabilities to all Full MADs through standardized conversation protocols.*

### 4.1 Conversation-First Integration

Half-MADs integrate via **conversations, not API calls**. Benefits:

- **Versioned protocols:** Schema evolution without breaking changes.
- **Dialogic interaction:** Multi-turn exchanges enable clarification and refinement.
- **Auditable:** All conversations archived in Dewey for compliance and debugging.
- **Composable:** MADs can chain Half-MAD conversations to build complex capabilities.

---

## 5. Integration Boundaries

IAE enforces strict boundaries between components:

- **ICCM → IDE:** `Structured Context` contract
- **IDE → IEE:** `Decision Package` contract
- **IEE → State Manager (IAE):** `Execution Outcome Package` contract
- **All components ↔ State Manager (IAE):** Versioned APIs
- **MAD ↔ Half-MADs:** Conversation protocols

These boundaries enable:

- **Independent testing:** Each component testable in isolation with contract mocks.
- **Incremental deployment:** Update one component without redeploying the entire system.
- **Heterogeneous implementation:** Components can use different languages, frameworks, or hosting models.

---

## 6. Design Principles and Tenets

IAE is grounded in five core tenets:

### 6.1 Separation of Concerns

Model-agnostic thinking (Thinking Engine) decouples from domain-specific doing (Doing Engine). This enables independent optimization of deliberation and execution.

### 6.2 Contracts First

Canonical schemas are defined before implementation. All components must honor contract semantics, but implementation details remain private.

### 6.3 Conversations Over Calls

MAD-to-MAD and MAD-to-Half-MAD interactions use versioned conversation protocols, not direct API calls. This supports schema evolution and dialogic refinement.

### 6.4 State as the Spine

The IAE-owned State Manager is the single source of truth for world model, task context, and execution state. All components read and write state through versioned APIs.

### 6.5 Feedback Loop Centrality

The operational feedback loop—decision → execution → outcome → state update → context refresh → better decision—is the primary mechanism for continuous improvement.

---

## 7. Implications for Agent Engineering

### 7.1 Auditability and Compliance

All decisions produce Reasoning Traces stored in the State Manager. Execution outcomes link to their originating decisions. This creates a complete audit trail from context to outcome.

### 7.2 Safety and Validation

Safety assertions in Decision Packages are validated before execution. Execution outcomes include safety validation results and deviation reports. The State Manager preserves pre- and post-execution world versions for rollback.

### 7.3 Independent Evolution

Canonical contracts enable component teams to evolve independently. A new CET implementation, for example, can replace the existing one without changes to IDE or IEE, as long as it produces valid `Structured Context` schemas.

### 7.4 Scalability and Reuse

Half-MADs provide reusable capabilities across all Full MADs. A single LLM Conductor instance, for instance, serves consultative reasoning for hundreds of agents.

---

## 8. Related Work

*(To be expanded with academic context and comparisons to existing agent frameworks, cognitive architectures, and decision systems.)*

---

## 9. Conclusion and Future Directions

Intelligent Agentic Engineering (IAE) provides a comprehensive discipline for assembling intelligent agents through separation of cognition from action, canonical contracts, and a centralized State Manager. The MAD architecture pattern, produced within IAE, enables independent evolution of components while maintaining system coherence through versioned protocols.

The operational feedback loop creates continuous improvement: each execution cycle refines the world model, enriches context quality, and enables better future decisions. Half-MADs provide reusable capabilities across all agents, promoting ecosystem-wide efficiency.

Future work includes formal verification of canonical contracts, empirical evaluation of feedback loop convergence, and extension of the Half-MAD ecosystem with domain-specific capabilities.

---

## Appendix A: Master Glossary

- **IAE (Intelligent Agentic Engineering):** The overarching discipline for agent assembly, integrating ICCM, IDE, IEE, and IAE-specific components.
- **MAD (Multipurpose Agentic Duo):** The architecture pattern produced by IAE, separating Thinking Engine from Doing Engine.
- **Full MAD:** A complete agent assembled from ICCM, IDE, IEE, and IAE components.
- **Half-MAD:** A minimal MAD providing a reusable capability to other MADs via conversations.
- **Thinking Engine:** The four cognitive components: CET (ICCM), Rules Engine (IDE), DER (IDE), State Manager (IAE).
- **Doing Engine:** Domain-specific execution component produced by IEE.
- **State Manager:** IAE-owned authoritative memory system (World Model, Task Context, Execution State).
- **Canonical Contracts:** The five core data schemas: Structured Context, Rule Engine Output, Decision Package, Execution Outcome Package, Reasoning Trace.
- **CET (Context Engineering Transformation):** ICCM component transforming raw inputs into structured context.
- **DER (Decision Engine for Reasoning):** IDE component for synthesis under uncertainty and multi-objective arbitration.
- **LLM Conductor:** Half-MAD providing LLM Orchestra (multi-model consultative reasoning).
- **Dewey:** Half-MAD for conversation retrieval (immutable archives).
- **Godot:** Half-MAD for active conversation management.
- **Marco:** Half-MAD for session orchestration and budgeting.
- **Horace:** Half-MAD for file and artifact catalog with provenance.
- **Gates:** Half-MAD for document generation with style/compliance.
- **Playfair:** Half-MAD for diagram and visualization generation.

---

## Appendix B: Diagram References

All diagrams generated using Playfair visualization system and stored in Horace catalog:

1. **Figure 1:** MAD Architecture (`01_mad_architecture.svg`)
2. **Figure 2:** Quaternary Structure (`02_quaternary_structure.svg`)
3. **Figure 3:** Operational Feedback Loop (`03_feedback_loop.svg`)
4. **Figure 4:** Canonical Contracts (`04_canonical_contracts.svg`)
5. **Figure 5:** Half-MADs Ecosystem (`05_halfmads_ecosystem.svg`)

---

**Document Generated:** October 6, 2025
**Generation Tools:** Playfair (diagrams), Gates (formatting)
**Revision History:** v1.0 (initial), v1.1 (quaternary structure, State Manager specification, canonical contracts v1)
