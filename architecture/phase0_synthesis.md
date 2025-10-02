# Phase 0: Scope Validation - Synthesis

## Triplet Performance Summary

**Scores:**
- Gemini 2.5 Pro: 25/25 ⭐
- GPT-5: 25/25 ⭐
- Grok 4: 11/25 ⚠️ (underperformed significantly)

**Quality:** Gemini and GPT-5 provided excellent, comprehensive feedback. Grok 4 provided minimal value.

---

## Critical Issues Identified

### 1. **CRITICAL ARCHITECTURAL CLARIFICATION** (Gemini, GPT-5)

**Issue:** Scope.md contains language implying CET-D "generates requirements" or "extracts requirements."

**Reality:** CETs generate ONLY optimized context. LLM Orchestra generates actual outputs (requirements, code, tests).

**Source:** CET Architecture Clarification Summary, Papers 01 and 05

**Impact:** Fundamental architectural misunderstanding that affects entire system design

**Required Action:** Revise ALL scope language to clarify:
- CET-D generates: Optimized context
- LLM Orchestra generates: Requirements, code, tests (using CET context)

---

### 2. **INFRASTRUCTURE BUDGET INCONSISTENCY** (GPT-5)

**Issue:** Scope lists hardware exceeding $7,840 budget:
- M5 Server: 4x Tesla P40
- 3x RTX 4070 Ti Super (NEW - not in Paper 07 BOM)
- V100: Tesla V100 32GB
- Irina NAS: 60TB

**Reality:** Paper 07's BOM (~$7,490) does NOT include 3x RTX 4070 Ti Super

**Required Action:** Remove 3x RTX 4070 Ti Super from scope OR increase budget accordingly. Align with Paper 07's actual infrastructure.

---

### 3. **PHASE 4 SCOPE AMBIGUITY** (Gemini, GPT-5, Grok)

**Issue:** Scope defers "continuous self-improvement" to OUT OF SCOPE, but Paper 01 describes Phase 4 as essential to progressive training thesis.

**Resolution:** Phase 4 should be IN SCOPE but simplified:
- IN SCOPE: Minimal Phase 4 (offline self-critique, simulated production loops)
- OUT OF SCOPE: Production continuous learning deployment

**Required Action:** Clarify Phase 4 inclusion with explicit constraints

---

## Missing Critical Components

All three models (primarily Gemini and GPT-5) identified missing top-level components:

### 1. **RAG System** (All three)
**Why Critical:** Paper 01 Phase 1 requires "RAG-grounded training"
**What's Missing:**
- Vector database (pgvector)
- Embedding models
- Retrieval logic
- Chunking strategy
- Top-k selection
- Reranking (cross-encoder)

**Required Action:** Add as top-level IN SCOPE component

### 2. **Reconstruction Testing Pipeline** (Gemini, GPT-5)
**Why Critical:** Core validation methodology per Papers 05 and 06
**What's Missing:**
- LLM Orchestra orchestration for multi-LLM implementations
- Test Lab execution coordination
- Pass rate calculation
- Comparison framework

**Required Action:** Elevate from sub-point to top-level IN SCOPE component

### 3. **Validation & Metrics Framework** (Gemini, GPT-5)
**Why Critical:** Statistical validation is core to proof of concept
**What's Missing:**
- Experiment runner
- Metrics collection (pass rates, compatibility scores)
- Baseline comparisons (RAG, Gold Standard)
- Statistical test execution (paired t-test)
- Results reporting

**Required Action:** Add as top-level IN SCOPE component

### 4. **Gold-Standard Protocol** (GPT-5)
**Why Critical:** Paper 05 requires manual gold standard as baseline
**What's Missing:**
- Two-reviewer + adjudicator workflow
- Time budget (6-10 hours/app)
- Documentation of decisions
- Quality criteria

**Required Action:** Add to IN SCOPE

### 5. **Dataset Preparation & Management** (Gemini, GPT-5)
**Why Critical:** 50-app dataset requires significant workflow
**What's Missing:**
- Ingestion pipeline
- Code analysis
- Cleaning
- Partitioning (40 training / 10 hold-out)
- License compliance
- Selection criteria (language, size, test coverage)

**Required Action:** Add as IN SCOPE component or toolset

### 6. **Experiment Tracking** (GPT-5)
**Why Critical:** Research reproducibility and statistical claims
**What's Missing:**
- Config versioning (Git)
- Dataset snapshot hashes
- Seed capture
- Results storage (Postgres/CSV)
- Re-run scripts

**Required Action:** Add minimal experiment tracking to IN SCOPE

### 7. **API Cost Guardrails** (GPT-5)
**Why Critical:** Budget protection for 5-person lab
**What's Missing:**
- Daily/monthly spend caps
- Fallback routing policy
- Alert thresholds

**Required Action:** Add to LLM Orchestra component

---

## Feasibility Assessment

### Infrastructure
**Gemini:** Feasible with current hardware IF aligned to Paper 07 BOM
**GPT-5:** NOT feasible as written (budget conflict), feasible if corrected
**Consensus:** **REVISE** - Remove 3x RTX 4070 Ti Super OR increase budget

### Team Size (5 people)
**Gemini:** Feasible but ambitious
**GPT-5:** Feasible with gold standard taking ~300-500 hours over 6-10 weeks
**Consensus:** **FEASIBLE** with efficient tooling

### Dataset Size (50 apps)
**Gemini:** Statistically justified per Papers 01 and 05
**GPT-5:** Adequate for paired t-test (α=0.05, power 0.8, 15% effect)
**Consensus:** **APPROPRIATE** for proof of concept

### Timeline
**Gemini:** 6-9 months realistic
**GPT-5:** 10-12 weeks achievable with detailed breakdown
**Consensus:** **FEASIBLE** - recommend 10-16 weeks (2.5-4 months)

---

## Recommended Scope Revisions

### 1. Architecture Clarity (CRITICAL)
**Change ALL instances of:**
- "Requirements extraction" → "Engineering context for requirements extraction"
- "CET-D generates requirements" → "CET-D generates optimized context; LLM Orchestra generates requirements"

**Add to Success Criteria:**
- "CET-D's sole output is optimized context, verifiably distinct from LLM Orchestra's requirement specifications"

### 2. Infrastructure Correction (CRITICAL)
**Option A (Recommended):** Align to Paper 07 BOM
- Keep: M5 (4x P40), V100, Irina NAS
- Remove: 3x RTX 4070 Ti Super
- Budget: $7,490-7,840 ✓

**Option B:** Increase budget
- Keep all hardware
- Increase budget to ~$10,000-12,000

### 3. Add Missing Components

**IN SCOPE - Core System Components:**
```markdown
**CET-D Training Pipeline:** [existing]

**LLM Orchestra:** [existing + add API cost guardrails]

**RAG System:** (NEW)
- Vector database (pgvector)
- Embedding models
- Retrieval logic (chunking, top-k, reranking)
- Knowledge base indexing

**Reconstruction Testing Pipeline:** (NEW - elevated from sub-point)
- Multi-LLM implementation generation
- Test Lab execution coordination
- Pass rate calculation
- Baseline comparison framework

**Requirements Engineering Domain:** [existing]

**Test Lab Infrastructure:** [existing]

**Validation & Metrics Framework:** (NEW)
- Experiment runner
- Metrics collection (pass rates, compatibility)
- Statistical testing (paired t-test)
- Baseline comparisons (RAG, Gold Standard)
- Results reporting

**Gold-Standard Protocol:** (NEW)
- Two-reviewer + adjudicator workflow
- Time budget (6-10 hours/app)
- Decision documentation

**Dataset Preparation & Management:** (NEW)
- 50-app selection (language, size, test coverage criteria)
- License compliance
- Ingestion and partitioning (40/10 split)

**Experiment Tracking:** (NEW - minimal)
- Config versioning (Git)
- Dataset snapshot hashes
- Results storage

**Conversation Storage & Retrieval:** [existing]

**Model Management:** [existing]
```

### 4. Clarify Phase 4 Scope

**IN SCOPE:**
- Minimal Phase 4 implementation (offline self-critique, simulated loops)
- Validates progressive training thesis completion

**OUT OF SCOPE:**
- Production continuous learning
- Deployment monitoring
- Real-time adaptation

### 5. Add Timeline

**Recommended:** 10-16 weeks (2.5-4 months) for proof of concept
- Weeks 1-3: Infrastructure + RAG baseline
- Weeks 2-6: Gold standard + Phase 1/2 training
- Weeks 5-10: Phase 3 training + reconstruction pipeline
- Weeks 9-12: Minimal Phase 4 + statistical analysis
- Weeks 13-16: Buffer for iterations and final validation

---

## Paper Alignment Confirmation

**Papers 01, 05, 10:** Scope aligned with corrections above
**CET Architecture Clarification:** NOW aligned (after language corrections)
**Paper 07 (Infrastructure):** Will be aligned after budget correction

---

## Next Steps

1. User reviews this synthesis
2. Update scope.md with approved changes
3. Mark scope.md v1.0 as approved
4. Proceed to Phase 1: Requirements Extraction

---

## Grok 4 Performance Note

**Grok 4 scored 11/25** - significantly underperformed vs Gemini (25/25) and GPT-5 (25/25)

**Issues:**
- Minimal detail (1,445 bytes vs 9,460 and 8,065)
- Missed critical architectural issue
- Missed budget inconsistency
- No meaningful insights
- Failed to follow requested format

**Recommendation:** Monitor Grok 4 in Phase 1 and 2. If still underperforming (<15/25), swap for Claude Sonnet ($3/M) or another alternative.
