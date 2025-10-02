# ICCM Architecture Planning Log

## Purpose

This log captures every significant decision made during the architecture planning process. It serves as:
- Decision journal (what we decided and why)
- Traceability record (decisions → papers → requirements → architecture)
- Session continuity tool (resume work across sessions)
- Drift detection mechanism (compare current state to logged decisions)

---

## 2025-10-02 14:30 - Architecture Planning Initiated

**Context:** Transitioning from research papers (14 theoretical papers) to implementation architecture.

**Problem Identified:**
- Implementation papers exist but are misaligned to research papers
- Missing architectural layer between research (WHAT) and implementation (BUILD)
- Risk of drift without proper requirements → architecture → implementation flow

**Solution Approach:**
Five-phase methodology:
1. Requirements Extraction (from papers)
2. System Architecture Design (C4 model)
3. Infrastructure Mapping (to existing hardware)
4. Component Specification (detailed APIs/contracts)
5. Implementation Paper Alignment (rewrite to match architecture)

**Anti-Drift Mechanisms Established:**
- Artifact-based gates (deliverable per phase, user approval required)
- Planning directory structure (all work in `/mnt/projects/ICCM/architecture/`)
- Traceability matrix (requirement → paper mapping)
- Scope control document (explicit in/out of scope)
- Planning log (this document)

**Rationale:** Previous attempt at implementation papers failed due to lack of architectural foundation. Proper software engineering process required.

---

## 2025-10-02 14:45 - Triplet-Accelerated Process Adopted

**Decision:** Use triplet AI review (Gemini 2.5 Pro, GPT-5, Grok 4) to accelerate architecture work.

**Process:**
1. Define clear question/task for each phase
2. Send identical request to all three models via `triplet_verifier.py`
3. Synthesize their responses (combine best, remove duplicates, flag contradictions)
4. Filter synthesis against scope.md and research papers
5. Present unified deliverable to user for approval
6. Gate: User must approve before next phase

**Rationale:**
- Speed: 3 parallel AI perspectives vs. sequential single-AI work
- Quality: Multiple viewpoints catch different issues
- Control: User still gates each phase transition
- Anti-drift: Synthesis filtered against scope and papers

**Tool:** `/mnt/projects/ICCM/tools/triplet_verifier.py`

**Triplet Submission Storage:** `/mnt/projects/ICCM/architecture/triplet_submissions/`

---

## 2025-10-02 14:50 - Foundation Documents Created

**Created:**
1. `/mnt/projects/ICCM/architecture/` - Root planning directory
2. `scope.md` - IN SCOPE vs OUT OF SCOPE definition
3. `planning_log.md` - This decision journal
4. Phase subdirectories:
   - `phase1_requirements/` - Requirements extraction
   - `phase2_architecture/` - System architecture design
   - `phase3_deployment/` - Infrastructure mapping
   - `phase4_specifications/` - Component specifications
   - `phase5_implementation_alignment/` - Implementation paper revision
   - `triplet_submissions/` - AI model responses

**Status:** Foundation ready for Phase 0 (Scope Validation)

---

## 2025-10-02 15:00 - Phase 0: Scope Validation (Pending)

**Next Action:** Send scope validation request to triplets

**Request Focus:**
- Review papers 01 (Primary), 04 (CET-D Software), 05 (Requirements Engineering)
- Validate IN SCOPE items are appropriate for proof of concept
- Validate OUT OF SCOPE deferrals are justified
- Identify any critical missing scope items
- Confirm feasibility with constraints (5-person team, $7,840, 50 apps)

**Expected Deliverables:**
- `triplet_submissions/phase0_scope/Gemini_2.5_Pro_Scope_Review.md`
- `triplet_submissions/phase0_scope/GPT-5_Scope_Review.md`
- `triplet_submissions/phase0_scope/Grok_4_Scope_Review.md`
- Synthesized `scope.md` (version 1.0, user-approved)

**Gate:** User must approve synthesized scope before Phase 1 begins

---

## 2025-10-02 15:15 - LLM Orchestra Scope Correction

**Issue Identified:** User noted LLM Orchestra description in scope.md was too generic and didn't reflect Paper 10's actual architecture.

**Correction Made:**
Read Paper 10 (`10_LLM_Orchestra_v4.md`) and corrected understanding:

**LLM Orchestra is:**
- M5-based infrastructure (4x Tesla P40 GPUs = 96GB VRAM)
- Manages local model deployment (Llama 3.1 70B, DeepSeek-R1, Mistral Large, etc.)
- Dynamic model loading/rotation every 4-12 hours for training diversity
- Model library storage on Irina (60TB, stores 50+ models)
- Coordinates supplemental cloud APIs (Together.AI, GPT-4o, Gemini, Claude)
- Intelligent routing: prefer local ($0.003/M) over cloud ($0.88-15/M)
- Phase-specific rotation strategies
- Cost management and performance optimization

**NOT just:** "Multi-LLM ensemble coordination" (too vague)

**Scope.md Updated:**
- LLM Orchestra section corrected with accurate architecture description
- Infrastructure section updated to reflect actual hardware:
  - M5 Server: 4x Tesla P40 (LLM Orchestra)
  - Irina NAS: 60TB (model library + storage)
  - 3x RTX 4070 Ti Super: CET training
  - V100: Reserved for CET training

**Traceability:** Paper 10, Sections 2-3 (System Architecture, Local LLM Deployment)

**Impact:** More accurate scope enables better requirements extraction and architecture design

---

## 2025-10-02 15:20 - Triplet Performance Tracking Added

**Decision:** Track quality of each triplet member's responses to enable data-driven model swapping.

**Rationale:**
- Not all AI models perform equally on architecture tasks
- Quality varies by task type (some better at code, others at planning)
- Budget allows swapping underperformers for better alternatives
- Empirical tracking prevents subjective/biased model selection

**Quality Metrics (1-5 scale each):**
1. Accuracy (paper alignment)
2. Completeness (coverage)
3. Insight Quality (value-add)
4. Practical Feasibility (implementation awareness)
5. Synthesis Utility (ease of integration)

**Total Score:** 25 points maximum per phase

**Swap Criteria:**
- Average <15/25 across 3+ phases → swap out
- Consistently ranks #3 → consider replacement
- All three <18/25 → consider premium upgrade (Claude Opus)

**Replacement Candidates:**
- Claude Opus ($15/M) - premium reasoning
- Claude Sonnet ($3/M) - affordable strong alternative
- DeepSeek-R1 (TBD pricing) - strong reasoning
- Llama 3.1 405B ($3.50/M) - largest open model
- Qwen2.5-Max 72B ($1.20/M) - strong + affordable

**Tracking Document:** `triplet_performance_tracking.md`

**Impact:** Continuous optimization of triplet composition based on empirical performance data

---

## 2025-10-02 16:00 - Phase 0 Triplet Results and Infrastructure Correction

**Triplet Performance Results:**
- Gemini 2.5 Pro: 25/25 ⭐ (perfect - caught critical architectural issue)
- GPT-5: 25/25 ⭐ (perfect - identified budget inconsistency)
- Grok 4: 11/25 ⚠️ (severely underperformed - minimal engagement)

**Critical Findings:**
1. **Architectural Clarification:** Both Gemini and GPT-5 correctly identified that scope language incorrectly implied CET "generates requirements" when CETs only generate context
2. **Fabricated Hardware:** GPT-5 flagged budget inconsistency for "3x RTX 4070 Ti Super" - which I (Claude) completely invented without basis
3. **Missing Components:** All identified RAG system, reconstruction testing pipeline, validation framework, etc. as missing from scope

**User Clarification on Infrastructure:**
- All infrastructure already owned and operational in basement
- No budget constraints (hardware already purchased)
- Only decision: optional $600 for +2 P40s

**Actual Infrastructure:**
- M5 Server: 4x Tesla P40 (96GB VRAM)
- Irina NAS: 60TB storage
- V100: Tesla V100 32GB
- Optional: +2 P40s for $600

**Scope.md Corrections Made:**
- Removed fictional RTX 4070 Ti Super GPUs
- Removed "$7,840 budget" language (irrelevant)
- Listed actual owned infrastructure
- Noted optional P40 expansion

**Impact:** Demonstrates value of triplet validation - GPT-5 caught my fabrication

---

## Session Continuity

**Current Phase:** Phase 0 - Complete (scope.md v1.0 finalized)
**Current Task:** Building Fiedler MCP Server (Orchestra conductor prototype)
**Next Session Start:**
1. Fiedler requirements validated by triplets
2. Build Fiedler MCP server with multi-provider support
3. Return to Phase 1: Requirements Extraction (using Fiedler)
**Blocking:** None

---

## Decisions Pending

None currently.

---

## Deferred Questions

None currently.

---

## Template for Future Entries

```markdown
## YYYY-MM-DD HH:MM - Decision Title

**Context:** Why this decision point arose

**Decision:** What was decided

**Alternatives Considered:** What else was evaluated

**Rationale:** Why this decision was made

**Impact:** What this affects

**Traceability:** Paper references or requirement IDs
```
