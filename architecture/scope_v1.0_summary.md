# Scope v1.0 - Summary of Changes

## Version Control
- **Version:** 1.0 (Triplet Validated)
- **Previous:** 0.1 (Initial Draft)
- **Date:** 2025-10-02
- **Validation:** Gemini 2.5 Pro (25/25), GPT-5 (25/25), Grok 4 (11/25)

---

## Critical Corrections Applied

### 1. **Architectural Clarity (CRITICAL)**
**Issue:** Scope implied CET "generates requirements"
**Fix:** Clarified throughout that:
- **CET-D generates:** Optimized context ONLY
- **LLM Orchestra generates:** Actual outputs (requirements, code, tests)

**Updated sections:**
- CET-D Training Pipeline: Added explicit note about context-only output
- LLM Orchestra: Added role clarification
- Requirements Engineering Domain: Changed "extraction" to "engineering context that enables extraction"
- Success Criteria: Added #8 requiring verifiable distinction between CET context and LLM outputs

### 2. **Infrastructure Correction (CRITICAL)**
**Issue:** Listed fictional "3x RTX 4070 Ti Super" GPUs
**Fix:** Removed fabricated hardware, listed actual owned infrastructure

**Actual Infrastructure:**
- M5 Server: 4x Tesla P40 (96GB VRAM)
- Irina NAS: 60TB storage
- V100: Tesla V100 32GB
- Optional: +2 Tesla P40 for $600

### 3. **Phase 4 Clarification**
**Issue:** Ambiguity about Phase 4 scope (in or out?)
**Fix:**
- **IN SCOPE:** Minimal Phase 4 (offline self-critique, simulated loops) for PoC
- **OUT OF SCOPE:** Production-scale continuous learning, deployment monitoring

---

## Components Added (7 New Top-Level Items)

### 1. RAG System
**Why:** Paper 01 Phase 1 requires "RAG-grounded training"
**Contents:**
- Vector database (pgvector)
- Embedding models
- Retrieval logic (chunking, top-k, reranking)
- Knowledge base indexing

### 2. Reconstruction Testing Pipeline
**Why:** Core validation methodology (Papers 05, 06)
**Contents:**
- LLM Orchestra orchestration for multi-LLM implementations
- Test Lab execution coordination
- Pass rate calculation
- Comparison framework (CET-D vs RAG vs Gold Standard)

### 3. Validation & Metrics Framework
**Why:** Statistical validation essential for proof of concept
**Contents:**
- Experiment runner
- Metrics collection (pass rates, compatibility, reconstruction success)
- Statistical testing (paired t-test)
- Baseline comparisons and reporting

### 4. Gold-Standard Protocol
**Why:** Paper 05 requires manual baseline
**Contents:**
- Two-reviewer independent requirements generation
- Third reviewer adjudication
- Time budget (6-10 hours/app)
- Decision documentation

### 5. Dataset Preparation & Management
**Why:** 50-app dataset requires significant workflow
**Contents:**
- Selection criteria (language, size, test coverage)
- License compliance
- Ingestion and code analysis
- Partitioning (40 training / 10 hold-out)

### 6. Experiment Tracking
**Why:** Research reproducibility
**Contents:**
- Config versioning (Git)
- Dataset snapshot hashes
- Seed capture
- Results storage (PostgreSQL/CSV)
- Re-run scripts

### 7. API Cost Guardrails
**Why:** Budget protection
**Contents:**
- Daily/monthly spend caps
- Fallback routing policy
- Alert thresholds
**Added to:** LLM Orchestra component

---

## Timeline Added

**Duration:** 10-16 weeks (2.5-4 months)

**Breakdown:**
- Weeks 1-3: Infrastructure setup + RAG baseline
- Weeks 2-6: Gold standard + Phase 1/2 training
- Weeks 5-10: Phase 3 training + reconstruction pipeline
- Weeks 9-12: Minimal Phase 4 + statistical analysis
- Weeks 13-16: Buffer for iterations and final validation

---

## Success Criteria Enhanced

**Added 3 new criteria (#8-10):**

8. **CET-D outputs only optimized context, verifiably distinct from LLM Orchestra's actual requirement specifications**
9. CET-D + LLM Orchestra ensemble outperforms RAG baseline and approaches gold standard on requirements extraction
10. Demonstrates that CET-engineered context enables smaller/faster systems to match or exceed generalist 70B+ baseline performance within requirements engineering domain

---

## Validation Dataset Clarified

**Original:** "50 high-quality applications"
**Enhanced:** "50 high-quality applications (40 training / 10 hold-out)"

---

## Document Control Updates

**Version:** 1.0 - Triplet Validated
**Status:** Awaiting User Final Approval
**Triplet Scores:**
- Gemini 2.5 Pro: 25/25 ⭐
- GPT-5: 25/25 ⭐
- Grok 4: 11/25 ⚠️

---

## Files Updated

1. `/mnt/projects/ICCM/architecture/scope.md` - v1.0 (this document)
2. `/mnt/projects/ICCM/architecture/triplet_performance_tracking.md` - Phase 0 scores
3. `/mnt/projects/ICCM/architecture/phase0_synthesis.md` - Detailed analysis
4. `/mnt/projects/ICCM/architecture/planning_log.md` - Decision log updated

---

## Next Steps

**Upon User Approval:**
1. Mark scope.md v1.0 as approved
2. Proceed to Phase 1: Requirements Extraction
3. Send requirements extraction request to triplets
4. Continue triplet performance tracking (monitor underperforming models)

**If Grok 4 continues underperforming:**
- After Phase 1 and 2, if Grok 4 avg remains <15/25
- Swap for Claude Sonnet ($3/M) or other alternative
