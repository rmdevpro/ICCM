# Requirements-First CET Implementation Action Plan

## Executive Summary

This action plan restructures the ICCM proof of concept to use requirements engineering as the primary subject domain for CET-D, maintaining the 4-phase progressive training methodology while providing clearer validation metrics. Papers will be versioned semantically: 

- All existing papers to be renamed v1.x where x is the version number.

- Papers to be edited as v2.1.0 (major version, minor version, minor updates)

- At the end of rewrites, restructure archive directory for major version

- All papers that did not recieve the new major version number will be copied to the archive. The copy that remains in the primary directory will be renamed with the major versionin so that all papers in the primary director are v2.1.0 indicating a new staring point.

- All other papers outside of the core papers and the master document will be archived a v1.x such that only the core papers and the master document remain in the primary directory.

## Core Paradigm Shift

### From Code Generation to Requirements Engineering

**Original Approach (v1.x papers):**

- CET-D learns to generate code directly
- Validation is subjective (is the code "good"?)
- Massive subject area (all of programming)
- Unclear success metrics

**Requirements-First Approach (v2.x direction):**

- CET-D learns requirements extraction/generation first
- Validation is objective (does reconstructed app work?)
- Focused subject area (requirements engineering)
- Clear success metrics (compilation, tests, equivalence)

## How This Maps to the 4-Phase Training

The 4-phase methodology from Paper 00 remains unchanged - only the subject domain shifts:

| Phase   | Original Focus                 | Requirements-First Focus                  |
| ------- | ------------------------------ | ----------------------------------------- |
| Phase 1 | Learn coding                   | Learn requirements engineering            |
| Phase 2 | Transform descriptions to code | Transform descriptions to requirements    |
| Phase 3 | LLM feedback on generated code | LLM feedback on implementing requirements |
| Phase 4 | Improve code generation        | Improve requirements extraction           |

## Paper Versioning Strategy

### Papers Needing v2.x Updates (Major Changes)

These papers need significant rewriting to reflect requirements-first approach:

1. **01_Progressive_Training_Methodology_v2.1** ✅ CONFIRMED NEEDS REWRITE
   
   - Currently focuses on coding data (freeCodeCamp, Exercism, GitHub)
   - Update to requirements engineering data (IEEE standards, specifications)
   - Maintain 4-phase structure but change all content examples
   - Reduce from 1703 lines to ~1000 (right-size)

2. **03A_Code_Execution_Feedback_v2.1** ✅ CONFIRMED NEEDS REWRITE
   
   - Currently focuses on code errors (imports, API misuse, type mismatches)
   - Shift to "requirements validation through reconstruction"
   - Keep Docker execution infrastructure, change validation focus
   - Reduce from 1836 lines to ~900 (right-size)

3. **03B_Production_Learning_Pipeline_v2.1** ✅ ADDED - NEEDS REWRITE
   
   - Currently focuses on debugging code errors
   - Shift to production requirements learning
   - Keep pipeline infrastructure, change learning focus
   - Reduce from 1938 lines to ~900 (right-size)

4. **04_CET_D_Software_Implementation_v2.1** ✅ CONFIRMED NEEDS REWRITE
   
   - Currently all about code generation context
   - Reframe as requirements engineering specialist
   - Keep model architecture, change domain focus
   - Reduce from 1380 lines to ~900

5. **05_Automated_Validation_Framework_v2.1** ✅ CONFIRMED NEEDS REWRITE
   
   - Currently about test generation for code
   - Change to requirements validation through reconstruction
   - Keep automation infrastructure
   - Keep at ~1000 lines (already reasonable)

6. **06A_Self_Bootstrapping_Development_v2.1** & **06B_Continuous_Self_Improvement_v2.1** ✅ CONFIRMED
   
   - Currently about code self-bootstrapping
   - Reframe as requirements complexity bootstrapping
   - Keep learning loop architecture

### Papers Needing v1.x Updates (Minor Changes)

These papers need only minor adjustments or remain unchanged:

1. **00_ICCM_Primary_Paper_v1.1** ✅ CONFIRMED MINOR UPDATE
   
   - Framework explicitly supports different subjects (line 116)
   - Add footnote about requirements engineering as the chosen subject
   - No structural changes needed

2. **02_CET_Architecture_Specialization_v1.1** ✅ CONFIRMED MINOR UPDATE
   
   - Architecture (3-7B models) remains identical
   - Add note that CET-D specializes in requirements instead of code
   - No structural changes needed

3. **07_Test_Lab_Infrastructure_v1.1** ✅ CONFIRMED MINOR UPDATE
   
   - Infrastructure is implementation-agnostic
   - Add note about supporting requirements validation
   - No changes to hardware/network specs

4. **08_Containerized_Code_Execution_v1.1** ✅ CONFIRMED MINOR UPDATE
   
   - Execution environment unchanged
   - Now used for reconstruction testing
   - No infrastructure changes

5. **09_LLM_Orchestra_v1.1** ✅ CONFIRMED MINOR UPDATE
   
   - Orchestra configuration unchanged
   - Add note about test evaluation models being valuable for requirements
   - No changes to model rotation strategy

6. **10_Testing_Infrastructure_v1.1** ✅ CONFIRMED MINOR UPDATE
   
   - Testing infrastructure unchanged
   - Now tests reconstructed apps from requirements
   - No infrastructure changes

7. **11_Conversation_Storage_Retrieval_v1.1** ✅ CONFIRMED MINOR UPDATE
   
   - Storage infrastructure unchanged
   - Now stores requirements-focused conversations
   - No database changes

### Papers Remaining at v1.0

These future-looking papers don't need updates:

- **F01_Bidirectional_Processing_v1.0** ✅ CONFIRMED NO CHANGES (future work)
- **F02_Edge_CET_P_v1.0** ✅ CONFIRMED NO CHANGES (future work)
- **F03_Requirements_Reverse_Engineering_v1.0** ✅ BECOMES PRIMARY REFERENCE!
  - This paper ALREADY DESCRIBES the requirements-first approach
  - Use as implementation guide for the POC

## Detailed Action Items

### Phase 1: Documentation Updates (Papers)

**High Priority (v2.x rewrites):**

1. **Paper 01 (Progressive Training)** - NEEDS MAJOR REWRITE
   
   - [ ] Update Phase 1 to focus on requirements expertise
   - [ ] Change Phase 2 examples to requirements transformation
   - [ ] Revise Phase 3 to show requirements→code→validation loop
   - [ ] Update Phase 4 for requirements improvement
   - [ ] Reduce from 1703 lines to ~800-1000 (right-size)

2. **Paper 03A (Code Execution Feedback)** - NEEDS MAJOR REWRITE
   
   - [ ] Reframe as "Requirements Validation Through Execution"
   - [ ] Focus on reconstruction testing methodology
   - [ ] Keep Docker infrastructure, change validation focus
   - [ ] Reduce from 1836 lines to ~900 (right-size)

3. **Paper 04 (CET-D Implementation)** - NEEDS MODERATE REWRITE
   
   - [ ] Redefine CET-D as requirements engineering specialist
   - [ ] Update training objectives for requirements focus
   - [ ] Maintain infrastructure and model architecture
   - [ ] Reduce from 1380 lines to ~900

4. **Paper 05 (Validation Framework)** - NEEDS MODERATE REWRITE
   
   - [ ] Change validation metrics to requirements-focused
   - [ ] Add reconstruction testing as primary measure
   - [ ] Update success criteria
   - [ ] Keep at ~1000 lines (already reasonable)

**Low Priority (v1.x updates):**

5. **Paper 00** - MINOR UPDATE
   
   - [ ] Add footnote about requirements-first implementation
   - [ ] No structural changes

6. **Papers 02, 07-11** - MINOR UPDATES
   
   - [ ] Add notes about requirements focus where relevant
   - [ ] No major changes needed

### Phase 2: Implementation Planning

**Training Data Preparation:**

- [ ] Identify 100 Python applications for requirements extraction
- [ ] Create requirements annotations for 20 apps (training baseline)
- [ ] Set up automated reconstruction testing

**Infrastructure Setup:**

- [ ] Configure LLM orchestra for Phase 3 (10-15 models)
- [ ] Set up containerized execution for reconstruction testing
- [ ] Implement model rotation strategy (<1% overhead)

**Metrics and Monitoring:**

- [ ] Define requirements extraction accuracy metrics
- [ ] Implement reconstruction success tracking
- [ ] Set up dashboards for training progress

## Implementation Timeline

### Month 1: Documentation Sprint

- Week 1-2: Rewrite Papers 01, 03A (v2.1)
- Week 3: Rewrite Papers 04, 05 (v2.1)
- Week 4: Minor updates to remaining papers

### Month 2-3: Phase 1 Training

- Implement requirements extraction with RAG grounding
- Generate conversation histories
- Validate with LLM orchestra

### Month 4-5: Phase 2 Training

- Transform Phase 1 conversations to context pairs
- Train requirements transformation skills

### Month 6-7: Phase 3 Training

- Implement interactive feedback loop
- Requirements → Code → Execution → Learning
- 10-15 model diversity

### Month 8: Phase 4 Deployment

- Production validation
- Self-improvement loops
- Performance measurement

### Month 9: Evaluation

- Final metrics collection
- Comparison with baselines
- Documentation of results

## Success Metrics

### Documentation Success

- All v2.x papers clearly explain requirements-first approach
- Papers right-sized (target 800-1000 lines for most)
- Consistent messaging across all documents

### Implementation Success

- 85% requirements extraction accuracy
- 75% successful app reconstruction
- 10-15 model diversity in Phase 3
- <1% model rotation overhead
- $300-500/month operational cost

## Risk Mitigation

### Documentation Risks

- **Over-editing**: Keep infrastructure papers (07, 08) mostly unchanged
- **Inconsistency**: Review all papers together after updates
- **Scope creep**: Focus only on requirements-first changes

### Implementation Risks

- **Data quality**: Start with simple, well-documented apps
- **Validation complexity**: Use existing test suites where possible
- **Model diversity**: Leverage existing LLM orchestra from Paper 09

## Summary for Clean Thread Transition

### Review Results ✅ ALL PAPERS REVIEWED

**Papers Confirmed for v2.x Rewrite (6 papers):**

- 01 Progressive Training - Currently all coding data, needs requirements focus
- 03A Code Execution - Currently code errors, needs reconstruction validation
- 03B Production Pipeline - Currently debugging, needs requirements learning
- 04 CET-D Implementation - Currently code context, needs requirements context
- 05 Validation Framework - Currently code testing, needs requirements validation
- 06A/B Self-Improvement - Currently code bootstrapping, needs requirements bootstrapping

**Papers Confirmed for v1.x Minor Updates (8 papers):**

- 00 Primary Paper - Framework supports any subject
- 02 Architecture - Model architecture unchanged
- 07-11 Infrastructure papers - All implementation-agnostic

**Papers Staying v1.0 (3 papers):**

- F01, F02 - Future work
- F03 - **Already describes requirements-first approach!**

### Key Decisions Validated:

1. **Requirements-first approach** compatible with existing framework
2. **4-phase training** methodology needs no changes
3. **Paper F03** already provides the implementation guide
4. **Semantic versioning**: v2.x for content rewrites, v1.x for notes

### Core Insight:

Requirements engineering is simply a different **subject domain** for CET-D to master - the framework, infrastructure, and training methodology remain unchanged. Only the content of 6 papers needs updating.

### Next Steps:

1. Sonnet rewrites 6 v2.x papers with requirements focus + right-sizing
2. Add minor notes to 8 v1.x papers
3. Use F03 as implementation guide

### Infrastructure Confirmed:

- Hardware: $7,840 investment (M5, Irina, network)
- LLM Orchestra: 10-15 models via three-tier strategy
- Monthly cost: $300-500 (85-92% savings vs cloud)

This approach transforms the POC from "generate good code" (subjective) to "extract requirements that enable reconstruction" (objective), providing clearer success metrics while maintaining the theoretical framework.