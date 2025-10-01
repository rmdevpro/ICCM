# Requirements Reverse Engineering: Learning Software Understanding Through Reconstruction

## Abstract

We propose a novel methodology for training Context Engineering Transformers to understand software requirements through reverse engineering: learning from real-world deployed applications by extracting requirements, regenerating code, and validating through reconstruction fidelity. Our approach harvests 3,000+ open-source applications from GitHub and other repositories, deploys them in isolated containers, trains CET-D to extract comprehensive requirements (functional, non-functional, and technical), then validates understanding by measuring how well the CET can regenerate functionally equivalent applications. Success is measured through test pass rate on original test suites (target: >75%), API compatibility, feature parity, and behavioral equivalence. This bidirectional learning pipeline—Real App → Requirements → Regenerate → Compare—represents a fundamentally new approach to requirements engineering, with applications in legacy system modernization, automatic documentation, cross-platform migration, and compliance verification.

## 1. Introduction

### 1.1 The Requirements Understanding Problem

Traditional approaches to training code generation models follow a unidirectional path: requirements → code. This assumes we have well-specified requirements as training data. In reality:

- Most real-world software has **incomplete or outdated requirements**
- Requirements documents rarely match deployed systems
- Developers learn by **reading existing code**, not just specifications
- The question "what does this system actually do?" is often unanswered

### 1.2 A Novel Approach: Learning by Reconstruction

We propose learning requirements understanding through **reconstruction**:

1. **Observe**: Deploy real-world applications and observe their behavior
2. **Extract**: Generate comprehensive requirements from code + docs + runtime
3. **Regenerate**: Implement those requirements as new code
4. **Validate**: Does the regenerated app pass the original's test suite?

If the CET can successfully reconstruct an application from its own understanding, it has **truly captured the requirements**.

### 1.3 Why This Hasn't Been Done

This approach combines elements from multiple research areas (program synthesis, reverse engineering, code generation) but the complete pipeline doesn't exist because:

1. **Infrastructure complexity**: Requires safely running thousands of containerized applications
2. **Evaluation difficulty**: "Requirements correctness" is hard to measure—our solution: reconstruction success
3. **Computational cost**: Training loop is expensive (deploy → analyze → extract → regenerate → compare)
4. **Novel problem framing**: Most research focuses on better code generation, not requirements understanding

### 1.4 Contributions

- **Novel methodology**: Requirements learning via reconstruction validation
- **Rigorous metrics**: Test pass rate, feature parity, API compatibility as ground truth
- **Scalable pipeline**: 3,000+ real-world applications across diverse categories
- **Practical applications**: Legacy modernization, auto-documentation, cross-platform migration

## 2. Related Work

### 2.1 Program Synthesis

**Existing work**:
- Synthesizing programs from input-output examples (FlashFill, programming by demonstration)
- Focus on **small functions**, not full applications
- Examples: "Synthesizing Programs from Input-Output Examples" (MIT)

**Our contribution**: Synthesizing from **deployed applications + documentation + runtime behavior**, targeting full application reconstruction.

### 2.2 Reverse Engineering

**Existing work**:
- Decompilers (Ghidra, IDA Pro): binary → source code
- API documentation generators: code → API specs
- UML generators: code → diagrams
- All are **static analysis**, not learning-based

**Our contribution**: **Learning-based requirements extraction** from running systems, not just static code analysis.

### 2.3 Code Understanding Models

**Existing work**:
- CodeBERT, GraphCodeBERT: understand code semantics
- Used for search, completion, bug detection
- Not focused on requirements extraction

**Our contribution**: Understanding measured by **reconstruction ability**, not embedding similarity.

### 2.4 Specification Mining

**Existing work**:
- "Mining Specifications" (Ammons et al., POPL 2002)
- Extracts API usage patterns from code
- Static analysis of calling patterns

**Our contribution**: **Full functional requirements** from running systems, validated through code generation and execution.

### 2.5 Neural Code Generation

**Existing work**:
- GitHub Copilot, Amazon CodeWhisperer, Replit Ghostwriter
- Generate code from natural language descriptions
- **Forward only**: requirements → code

**Our contribution**: **Bidirectional pipeline** with validation through reconstruction.

## 3. Methodology

### 3.1 Overview

**Four-Phase Pipeline**:

1. **Application Harvesting**: Collect 3,000+ real-world apps from GitHub, GitLab, Docker Hub
2. **Containerized Deployment**: Deploy apps in isolated containers for safe analysis
3. **Requirements Extraction Training**: Train CET-D to generate requirements from deployed apps
4. **Reverse Engineering Validation**: Regenerate apps from requirements, measure reconstruction fidelity

### 3.2 Application Harvesting

**Sources**:
- **GitHub**: Public repositories with stars >100, active maintenance, comprehensive tests
- **GitLab**: Similar quality filters
- **Docker Hub**: Official images with source code available

**Quality Criteria**:
```python
application_filters = {
    'stars': '>100',
    'languages': ['python', 'javascript', 'typescript', 'go', 'rust'],
    'has_tests': True,
    'test_coverage': '>60%',
    'has_documentation': True,
    'active_maintenance': 'last_6_months',
    'license': 'permissive'  # Apache, MIT, BSD
}
```

**Categories** (3,000+ total):
- **Web Applications** (1,000): Flask, Django, FastAPI, Express, React, Vue
- **CLI Tools** (500): Python, Go, Rust, Node.js command-line utilities
- **APIs** (800): REST, GraphQL, gRPC, WebSocket services
- **Data Processing** (400): ETL pipelines, data analysis, ML pipelines
- **Microservices** (300): Kubernetes apps, Docker Compose stacks

### 3.3 Containerized Deployment

**Security Isolation**:
```yaml
container_security:
  network: 'isolated'  # No internet access
  filesystem: 'writable'  # App needs to run normally
  user: 'appuser'  # Non-root
  capabilities: 'drop ALL'
  resource_limits:
    cpu: '1.0'
    memory: '2GB'
    disk: '5GB'
    timeout: '1h'
```

**Deployment Pipeline**:
1. Clone repository
2. Analyze structure (language, framework, dependencies)
3. Build container with appropriate base image
4. Deploy with strict isolation
5. Run test suite to verify deployment
6. Expose metrics and logs for analysis

### 3.4 Requirements Extraction

**Multi-Source Analysis**:

1. **Documentation Analysis**:
   - README, API docs, user guides
   - Extract: features, usage patterns, constraints

2. **Code Structure Analysis**:
   - Static analysis of source code
   - Extract: data models, APIs, business logic, architecture patterns

3. **Runtime Behavioral Analysis**:
   - Execute test suite, observe behavior
   - Extract: performance characteristics, error handling, resource usage

4. **Synthesis into Structured Requirements**:
```python
requirements_structure = {
    'functional_requirements': {
        'features': [],        # What the app does
        'apis': [],           # API contracts
        'data_models': [],    # Data structures
        'business_logic': []  # Core algorithms
    },
    'non_functional_requirements': {
        'performance': {},     # Latency, throughput
        'scalability': {},     # Resource usage patterns
        'security': {},        # Auth, validation
        'reliability': {}      # Error handling
    },
    'technical_requirements': {
        'dependencies': [],    # Libraries, frameworks
        'architecture': {},    # Design patterns
        'deployment': {}       # How to deploy
    }
}
```

### 3.5 Training CET on Requirements Engineering

**Training Loop**:
```python
for epoch in range(10):
    for app in deployed_applications:
        # 1. CET extracts requirements from deployed app
        cet_requirements = cet.generate_requirements(
            deployed_app=app.container,
            documentation=app.docs,
            source_code=app.source,
            runtime_logs=app.behavior
        )

        # 2. Extract ground truth requirements (multi-source)
        true_requirements = extract_ground_truth(app)

        # 3. LLM orchestra evaluates CET's requirements
        evaluations = [
            llm.evaluate_requirements(cet_requirements, true_requirements)
            for llm in llm_orchestra
        ]

        # 4. Update CET based on consensus
        consensus = aggregate_evaluations(evaluations)
        loss = compute_requirements_loss(cet_requirements, consensus)
        cet.update(loss)
```

### 3.6 Reverse Engineering Validation

**The Ultimate Test: Reconstruction**

```python
def validate_requirements_understanding(original_app):
    """Can CET reconstruct the application?"""

    # Step 1: CET generates requirements from original app
    cet_requirements = cet.generate_requirements(
        deployed_app=original_app.container,
        documentation=original_app.docs,
        source_analysis=original_app.source_metadata
    )

    # Step 2: Generate code from CET's requirements
    regenerated_code = code_generator.generate_from_requirements(
        requirements=cet_requirements,
        language=original_app.language,
        framework=original_app.framework
    )

    # Step 3: Deploy regenerated application
    regenerated_app = deploy_in_container(
        code=regenerated_code,
        config=cet_requirements['technical_requirements']
    )

    # Step 4: Run original tests on regenerated app
    test_results = run_original_tests(
        tests=original_app.test_suite,
        app=regenerated_app
    )

    # Step 5: Multi-dimensional comparison
    return {
        'test_pass_rate': test_results.pass_rate,
        'api_compatibility': compare_apis(original_app, regenerated_app),
        'feature_parity': compare_features(original_app, regenerated_app),
        'behavioral_equivalence': compare_behavior(original_app, regenerated_app),
        'performance_delta': compare_performance(original_app, regenerated_app)
    }
```

### 3.7 Reconstruction Metrics

**Primary Metric: Test Pass Rate**
- Can the regenerated app pass the original app's test suite?
- **Most rigorous validation**: Tests encode expected behavior
- Target: >75% average across all apps

**Secondary Metrics**:
- **API Compatibility**: Can original clients work with regenerated API? (Target: >80%)
- **Feature Parity**: Are all documented features present? (Target: >85%)
- **Behavioral Equivalence**: Same outputs for same inputs? (Target: >80%)
- **Performance Preservation**: Similar latency/throughput? (Target: ±20%)

**Overall Reconstruction Score**:
```python
weights = {
    'test_pass_rate': 0.30,              # Critical
    'feature_reconstruction_rate': 0.25,  # Important
    'api_reconstruction_rate': 0.20,      # Important
    'behavioral_similarity': 0.15,        # Secondary
    'architecture_preservation': 0.10     # Tertiary
}

reconstruction_score = sum(metrics[m] * weights[m] for m in weights)
```

**Grading**:
- A (≥90%): Excellent requirements understanding
- B (≥80%): Good requirements understanding
- C (≥70%): Adequate requirements understanding
- D (≥60%): Poor requirements understanding
- F (<60%): Failed to capture requirements

## 4. Training Data Generation

### 4.1 Learning from Success and Failure

After each reconstruction attempt, generate training data:

**Positive Examples** (what CET got right):
- Requirements that led to passing tests
- Features successfully reconstructed
- APIs correctly identified

**Negative Examples** (what CET missed):
- Test failures → missing requirements
- Feature gaps → incomplete analysis
- API mismatches → incorrect specifications

### 4.2 Continuous Improvement Loop

```python
for iteration in range(100):
    # 1. Harvest new batch of applications
    apps = harvest_applications(batch_size=50)

    # 2. Deploy and analyze
    deployed_apps = [deploy_app(app) for app in apps]

    # 3. CET generates requirements
    cet_requirements = [
        cet.generate_requirements(app)
        for app in deployed_apps
    ]

    # 4. Regenerate from requirements
    regenerated_apps = [
        generate_from_requirements(req)
        for req in cet_requirements
    ]

    # 5. Validate reconstruction
    validation_results = [
        validate_reconstruction(original, regenerated)
        for original, regenerated in zip(deployed_apps, regenerated_apps)
    ]

    # 6. Generate training pairs from results
    training_pairs = extract_training_data(validation_results)

    # 7. Update CET
    cet.train_on_pairs(training_pairs)

    # 8. Track improvement
    log_metrics({
        'iteration': iteration,
        'avg_reconstruction_score': mean(validation_results),
        'improvement_delta': calculate_improvement()
    })
```

### 4.3 Progressive Difficulty

**Phase 1: Simple Applications** (Iterations 1-30)
- Single-file scripts
- CLI tools with <500 LOC
- Simple REST APIs
- Target: >80% reconstruction

**Phase 2: Intermediate Applications** (Iterations 31-60)
- Multi-file projects
- Web frameworks with databases
- Microservices with 2-3 services
- Target: >70% reconstruction

**Phase 3: Complex Applications** (Iterations 61-100)
- Large codebases (>5,000 LOC)
- Distributed systems
- Full-stack applications
- Target: >60% reconstruction

## 5. Infrastructure Requirements

### 5.1 Storage (Irina)

```yaml
application_library:
  location: '/mnt/irina/fast/applications/'
  capacity: '5TB'
  contents:
    source_code: '2TB'        # 3,000 apps × ~700MB avg
    containers: '2TB'          # Built container images
    test_suites: '500GB'       # Test data and fixtures
    analysis_results: '500GB'  # Extracted requirements, metrics
```

### 5.2 Container Orchestration

```yaml
kubernetes_cluster:
  nodes: ['M5', 'Irina']
  max_concurrent_containers: 100
  isolation: 'strict network isolation'
  orchestration: 'Kubernetes'
  monitoring: 'Prometheus + Grafana'
```

### 5.3 Compute Resources

**For Requirements Extraction** (CPU-bound):
- M5 + Irina CPUs
- Parallel static analysis
- Behavioral observation (test execution)

**For Code Generation** (GPU-bound):
- 4x P40 GPUs (M5)
- Models: Llama 3.1 70B, DeepSeek-R1 70B
- Requirements → Code implementation

**For Validation** (CPU-bound):
- Parallel test execution
- API compatibility testing
- Behavioral comparison

### 5.4 Network Requirements

```yaml
network_architecture:
  container_network: 'isolated'  # No external access
  storage_network: 'dual 1Gb bonded'  # M5 ↔ Irina
  monitoring_network: 'management VLAN'
```

## 6. Expected Results

### 6.1 Reconstruction Success Rates

**Target Metrics** (after 100 iterations):

```yaml
simple_applications:
  test_pass_rate: '>80%'
  feature_reconstruction: '>85%'
  api_compatibility: '>90%'
  overall_score: '>85%'

intermediate_applications:
  test_pass_rate: '>70%'
  feature_reconstruction: '>75%'
  api_compatibility: '>80%'
  overall_score: '>75%'

complex_applications:
  test_pass_rate: '>60%'
  feature_reconstruction: '>65%'
  api_compatibility: '>70%'
  overall_score: '>65%'

overall_average:
  reconstruction_score: '>75%'
  improvement_per_iteration: '>2%'
  convergence: '~50 iterations'
```

### 6.2 Requirements Quality Metrics

```yaml
requirements_completeness:
  functional_coverage: '>85%'    # % of features captured
  api_coverage: '>90%'           # % of APIs documented
  data_model_coverage: '>80%'    # % of models identified

requirements_accuracy:
  false_positive_rate: '<10%'    # Invented requirements
  false_negative_rate: '<15%'    # Missed requirements
  specification_correctness: '>85%'  # Correctly specified
```

### 6.3 Learning Curve

**Expected progression**:
- **Iterations 1-20**: Rapid initial improvement (50% → 65% reconstruction)
- **Iterations 21-50**: Steady improvement (65% → 75% reconstruction)
- **Iterations 51-80**: Plateau with gradual gains (75% → 78% reconstruction)
- **Iterations 81-100**: Fine-tuning (78% → 80% reconstruction)

## 7. Applications

### 7.1 Legacy System Modernization

**Problem**: Outdated systems with no current requirements documentation

**Solution**:
1. Deploy legacy system in container
2. CET extracts requirements through observation
3. Generate modern implementation (new language/framework)
4. Validate via original test suite

**Industry Value**: Mainframe → cloud migration, COBOL → Python/Java

### 7.2 Automatic Documentation Generation

**Problem**: Documentation drifts from actual implementation

**Solution**:
1. Deploy current system
2. CET generates up-to-date requirements
3. Auto-generate accurate documentation
4. Keep synchronized through continuous observation

**Industry Value**: Always-current system documentation

### 7.3 Cross-Platform Migration

**Problem**: Port application to different platform (iOS → Android, desktop → web)

**Solution**:
1. Deploy source platform app
2. CET extracts platform-agnostic requirements
3. Generate implementation for target platform
4. Validate functional equivalence

**Industry Value**: Multi-platform development from single specification

### 7.4 Compliance Verification

**Problem**: Verify system does what it claims (regulatory compliance)

**Solution**:
1. Deploy system in container
2. CET extracts actual behavior (requirements)
3. Compare to claimed behavior (specification)
4. Report gaps and discrepancies

**Industry Value**: SOC2, HIPAA, PCI-DSS compliance validation

### 7.5 Security Auditing

**Problem**: Identify undocumented features (backdoors, data leaks)

**Solution**:
1. Deploy system with behavioral monitoring
2. CET extracts all observed behaviors
3. Compare to official requirements
4. Flag undocumented/suspicious behavior

**Industry Value**: Supply chain security, third-party software auditing

## 8. Challenges and Limitations

### 8.1 Technical Challenges

**1. Incomplete Test Coverage**:
- If original app has poor tests, validation is weak
- Mitigation: Auto-generate additional tests using CodeT5, GraphCodeBERT

**2. Non-Deterministic Behavior**:
- Apps with randomness, timestamps, external APIs
- Mitigation: Mock external dependencies, normalize non-deterministic outputs

**3. Deployment Complexity**:
- Some apps require complex infrastructure (databases, message queues)
- Mitigation: Docker Compose for multi-service apps, mocked dependencies

**4. Proprietary Dependencies**:
- Apps requiring licensed software/APIs
- Mitigation: Focus on open-source stack apps initially

### 8.2 Evaluation Challenges

**1. Semantic Equivalence**:
- Different code can implement same requirements
- Mitigation: Focus on behavioral equivalence, not code similarity

**2. Performance Requirements**:
- Original app may be poorly optimized
- Mitigation: Separate functional from non-functional requirements validation

**3. Subjective Quality**:
- "Good requirements" is partially subjective
- Mitigation: Reconstruction success is objective ground truth

### 8.3 Scalability Challenges

**1. Computational Cost**:
- Training loop is expensive (deploy → analyze → generate → compare)
- Mitigation: Batch processing, parallel execution, incremental learning

**2. Storage Requirements**:
- 3,000 apps + containers + results = 5TB
- Mitigation: Irina has 60TB capacity, only 5TB needed

**3. Time to Convergence**:
- 100 iterations × 50 apps/iteration = 5,000 reconstruction attempts
- Mitigation: Parallel processing, efficient container orchestration

## 9. Future Directions

### 9.1 Beyond Software: General Requirements Engineering

**Extend to other domains**:
- **Hardware design**: Extract requirements from circuit designs
- **Business processes**: Extract workflows from process logs
- **Scientific computing**: Extract algorithms from research code
- **Game design**: Extract game mechanics from deployed games

### 9.2 Real-Time Requirements Tracking

**Continuous observation**:
- Monitor production systems in real-time
- Automatically update requirements as system evolves
- Detect requirement drift and unauthorized changes
- Maintain living documentation

### 9.3 Collaborative Requirements Engineering

**Human-AI collaboration**:
- CET proposes requirements from observation
- Human experts review and refine
- Iterative improvement through feedback
- Best of both worlds: AI scale + human insight

### 9.4 Requirements-Driven Testing

**Generate tests from requirements**:
- CET extracts requirements → generates comprehensive test suite
- Ensure all requirements have test coverage
- Mutation testing to validate test quality
- Close the loop: requirements → code → tests → validation

### 9.5 Multi-Language Reconstruction

**Translate across languages**:
- Extract language-agnostic requirements
- Regenerate in different programming language
- Validate functional equivalence across languages
- Enable true polyglot development

## 10. Conclusion

Requirements reverse engineering through reconstruction validation represents a fundamentally new approach to training AI systems for software understanding. By learning from 3,000+ real-world applications and validating through the rigorous metric of "can you rebuild it?", we create a CET that truly understands what software does, not just how to generate code.

This methodology addresses a critical gap in current AI-assisted development: understanding existing systems. While current tools excel at generating new code, they struggle with comprehending deployed applications. Our approach directly tackles this through:

1. **Learning from reality**: Real deployed apps, not toy examples
2. **Rigorous validation**: Test pass rate as objective ground truth
3. **Practical value**: Legacy modernization, documentation, compliance
4. **Novel metrics**: Reconstruction fidelity measures true understanding

The applications extend far beyond academic interest—legacy system modernization alone represents a multi-billion dollar industry need. Automatic documentation generation, cross-platform migration, and compliance verification all become achievable with a CET trained through reconstruction.

While this represents a future research direction requiring significant infrastructure and computational resources, the potential impact on software engineering practice is substantial. The ability to automatically understand, document, and modernize existing systems could fundamentally change how we approach software maintenance and evolution.

## References

### Program Synthesis
- Gulwani, S. "Synthesizing Programs from Input-Output Examples" (MIT)
- Solar-Lezama, A. "Program Synthesis by Sketching" (Berkeley)

### Reverse Engineering
- Ammons, G. et al. "Mining Specifications" (POPL 2002)
- Ernst, M. "Dynamic Discovery of Program Invariants" (PhD Thesis)

### Code Understanding
- Feng, Z. et al. "CodeBERT: A Pre-Trained Model for Programming and Natural Languages" (EMNLP 2020)
- Guo, D. et al. "GraphCodeBERT: Pre-training Code Representations with Data Flow" (ICLR 2021)

### Neural Code Generation
- Chen, M. et al. "Evaluating Large Language Models Trained on Code" (GitHub Copilot)
- Austin, J. "Program Synthesis with Large Language Models" (DeepMind)

### Specification Mining
- Gabel, M. "Mining Temporal Specifications from Object Usage" (ASE 2011)
- Pradel, M. "TypeDevil: Dynamic Type Inconsistency Detection" (ICSE 2015)

### Legacy System Modernization
- Sneed, H. "Estimating the Costs of Software Maintenance Tasks" (ICSM 1995)
- Canfora, G. "New Frontiers of Reverse Engineering" (FOSE 2007)

---

**Paper Status**: Future Directions (F03)
**Dependencies**: Papers 1, 3, 4, 5, 8, 9
**Target Venue**: FSE (Foundations of Software Engineering) or ASE (Automated Software Engineering)
**Target Timeline**: Q4 2025 - Q1 2026
