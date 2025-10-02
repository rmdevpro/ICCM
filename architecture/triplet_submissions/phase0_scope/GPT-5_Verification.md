## Scope Validation Summary

### Overall Assessment
REVISIONS NEEDED

### IN SCOPE Review
- CET-D Training Pipeline (4 phases): Correct, with clarifications needed
  - Require explicit statement that CET generates context only; downstream LLMs generate requirements, code, and tests (per Papers 01 and 05).
  - Scope Phase 4 to a minimal, proof-of-concept variant (offline/self-critique and simulated production loops), not production continuous learning.
- LLM Orchestra (local-first, hybrid cloud, rotation, cost controls): Correct
  - Aligned with Paper 09; include explicit spend guardrails and model rotation cadence in scope to protect budget.
- Requirements Engineering Domain (reconstruction testing, 50-app dataset): Correct
  - Aligned with Paper 05; keep 40/10 split and gold-standard + RAG baselines.
- Test Lab Infrastructure (Docker sandboxing, 600–1000 exec/day): Correct
  - Aligned with Papers 07/08/10; ensure “network_mode: none”, resource limits, and simple monitoring are specified.
- Conversation Storage & Retrieval (persist phase data): Correct
  - Aligned with Paper 11; Postgres + pgvector design is appropriate for the mixed relational + vector queries.
- Model Management (checkpoints, versioning, perf monitoring): Correct
  - Add light experiment tracking for reproducibility (see Missing Components).
- Quality Standards (paired t-test, α=0.05, 80% power @ 15% effect): Correct
  - Keep paired design across the same apps; pre-register analysis plan.
- Infrastructure Constraints: Needs revision
  - Conflict with budget and with prior system description in Paper 07. The scope lists 3x RTX 4070 Ti Super + V100 + 4x P40 + 60TB NAS within $7,840, which is not realistic. Paper 07’s BOM (~$7,490) did not include 3x 4070 Ti Super and is internally consistent. Recommend removing the 3x 4070 Ti Super and aligning with Paper 07’s M5 + Irina + V100 configuration to meet budget.

### OUT OF SCOPE Review
- CET-P (edge deployment, federated learning): Justified
  - Future work per Paper F02/14; keep out of initial CET-D POC.
- CET-T (team context): Justified
  - Future work per Paper 01/02; exclude from POC.
- Bidirectional Processing (reverse-pass adaptation): Justified
  - Future work per Paper F01; unidirectional only in POC.
- Production Deployment (enterprise hardening, multi-tenant, observability stacks): Justified
  - Out of research-lab scope; aligns with Papers 07/08 design rationale.
- Large-Scale Training (500–3000+ apps): Justified
  - Future scale per Paper 05 roadmap; not needed for POC.
- Self-bootstrapping development and continuous improvement at production scale (Papers 06A/06B): Justified
  - Keep as future work; not required for the proof-of-concept.

### Missing Critical Components
1. RAG Baseline Implementation Details (Paper 05/01)
   - Specify pgvector-backed store, embedding model, chunking, k, and reranking settings to ensure a competitive baseline.
2. Gold-Standard Protocol (Paper 05)
   - Include two-reviewer independent requirements + third reviewer tie-break protocol, timebox (6–10 hours/app), and documentation of adjudication decisions.
3. Experiment Tracking and Reproducibility
   - Minimal stack (Git + immutable dataset snapshot hashes + config/seed capture + results in PostgreSQL/CSV). Needed to support statistical claims and reruns.
4. Data Selection and Licensing Plan
   - Criteria for the 50 apps (language, size bands, test coverage thresholds), license compliance, and documentation of inclusion/exclusion.
5. Ablation/Canary Set and Forgetting Guard (Paper 05 Appx)
   - Add the 10-app canary set and a simple schedule for regression checks to detect forgetting during Phase 3/4 loops.
6. API Cost Guardrails
   - Daily/monthly caps + fallback policy for premium and Together.AI usage to protect budget.
7. Scope wording to reflect “CET generates context only”
   - Avoid any phrasing that implies the CET generates requirements/code; downstream LLMs generate, CET engineers context (Papers 01/05).

### Feasibility Concerns
- Infrastructure:
  - As written, the hardware list exceeds the $7,840 budget if 3x 4070 Ti Super are included. Feasible if aligned to Paper 07’s BOM (M5: 4x P40 + V100, Irina NAS, workstation, Pharaoh), which delivered ~156 GB total VRAM and fits ~$7.5k.
- Team size:
  - 5-person team can complete gold standard + validation (≈300–500 hours total) over 6–10 weeks, while one member focuses on pipeline/infrastructure; feasible.
- Dataset size:
  - 50 apps with 40/10 split is adequate for the paired t-test at α=0.05, power 0.8 for detecting a 15% improvement (per Paper 05). Ensure baseline SD assumptions hold; otherwise adjust effect or sample.
- Timeline:
  - POC achievable in ~10–12 weeks: Weeks 1–3 infra + RAG baseline; Weeks 2–6 gold standard + Phase 1/2; Weeks 5–10 Phase 3 + reconstruction pipeline + analysis; Weeks 9–12 minimal Phase 4 (offline self-critique).

### Recommended Changes
1. Align Infrastructure with Paper 07 (BOM)
   - Remove the 3x RTX 4070 Ti Super from scope. Use M5 (4x P40) + V100 + Irina NAS as in Paper 07 to hit the $7.5k–$7.84k target. Cite Paper 07 cost breakdown.
2. Clarify CET Output Boundary (Papers 01/05)
   - Update scope language everywhere to “CET generates optimized context; downstream LLMs generate requirements/code/tests.” Prevents architectural drift.
3. Constrain Phase 4 for POC (Papers 01/10)
   - Limit Phase 4 to offline self-critique/simulated loops, not production continuous learning or deployment. Keep full production learning as future work.
4. Add RAG Baseline Spec (Paper 05)
   - Specify pgvector, embedding model, chunk size/overlap, top-k, and cross-encoder reranking. This ensures a strong, reproducible comparator.
5. Add Gold-Standard Protocol (Paper 05)
   - Include two-reviewer + adjudicator workflow and logging of decisions. Budget the time (6–10 hours/app).
6. Add Minimal Experiment Tracking
   - Git-committed config + dataset hash; results table (Postgres/CSV) with seeds, metrics, model/commit IDs; scripts to re-run experiments.
7. Add API Budget Guardrails (Paper 09)
   - State monthly caps and fallback routing (local-first, Together.AI selective, premium only for validation). Include alerting thresholds.
8. Reword Success Criterion
   - Change “CET-D (5B) outperforms 70B+ LLMs” to “CET-D + LLM ensemble outperforms a strong RAG baseline and approaches the manual gold standard on requirements extraction; CET-D enables smaller/faster systems to exceed generalist 70B baselines within this domain” (per Papers 01/05).

### Paper Alignment Issues
- Paper 01 (Progressive Training, CET generates context only)
  - Scope needs explicit language to avoid implying CET generates requirements or code. Add this boundary throughout.
  - Phase 4 should be scoped to a minimal demonstration; full continuous deployment is future work.
- Paper 05 (Requirements Engineering + Reconstruction Testing)
  - Scope is aligned on domain and evaluation. Missing explicit RAG baseline configuration and gold-standard protocol; add both.
  - Ensure reconstruction testing pipeline (multi-LLM implementations + test execution) is explicitly listed as an in-scope deliverable (it is implied but should be itemized).
- Paper 10 (Testing Infrastructure)
  - Scope aligns with a Docker-based, multi-language execution lab and CI-style validation. Add explicit mention of unit/integration test orchestration and simple daily reporting to reflect Paper 10 practices.
- Infrastructure Budget vs. Paper 07
  - Inconsistency: current scope lists extra 3x 4070 Ti Super within the same ~$7.8k budget, which conflicts with Paper 07’s detailed costed infrastructure. Align scope to Paper 07 hardware or increase budget accordingly (recommend align to Paper 07).
- Success Criteria Language
  - “CET-D (5B) outperforms 70B+ LLMs” is too strong and not consistent with Paper 05’s framing. Rephrase to focus on beating a strong RAG baseline and approaching gold standard for requirements extraction via reconstruction testing.

