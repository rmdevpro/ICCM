# Code Execution Feedback Mechanisms for Context Engineering

## Abstract

We present a comprehensive framework for using code execution feedback as training signals for Context Engineering Transformers (CETs). Unlike traditional static metrics, our approach leverages multiple execution-based feedback mechanisms: error messages, multi-LLM solution variance, test results, compilation errors, performance benchmarks, and security scans. Each feedback type provides structured, actionable signals that guide context optimization. We demonstrate how error messages serve as explicit supervision, how solution variance reveals context ambiguity, and how test coverage guides comprehensive context development. This paper establishes the foundational feedback mechanisms that enable CETs to learn practical context engineering from real code execution.

## 1. Introduction

The quality of context in code generation can only be truly measured by execution: does the code compile, do tests pass, is it secure and efficient? This paper presents the feedback mechanisms that transform code execution outcomes into training signals for context optimization. We focus on six primary feedback sources: (1) error messages and their patterns, (2) multi-LLM solution variance, (3) test suite results and coverage, (4) compilation error analysis, (5) performance benchmarks, and (6) security vulnerability detection. Together, these mechanisms provide comprehensive feedback for learning effective context engineering.

## 2. Code Correctness as Primary Signal

### 2.1 Binary and Gradient Signals
```python
def evaluate_code_quality(code, tests):
    signals = {
        'binary': {
            'compiles': does_compile(code),
            'tests_pass': all_tests_pass(code, tests)
        },
        'gradient': {
            'test_pass_rate': count_passing_tests(code, tests) / len(tests),
            'performance_score': benchmark_performance(code),
            'complexity_score': measure_cyclomatic_complexity(code)
        }
    }
    return signals
```

### 2.2 Error Messages as Learning Features

Compilation and runtime errors provide structured, actionable feedback that directly guides context optimization. Unlike vague quality metrics, errors explicitly identify what information was missing or incorrect in the context.

**Error Classification and Context Mapping:**

```python
class ErrorContextAnalyzer:
    def __init__(self):
        self.error_patterns = {
            'missing_import': {
                'indicators': ['undefined name', 'cannot find symbol', 'not defined'],
                'context_fix': 'add_import_statements',
                'severity': 'high'
            },
            'type_mismatch': {
                'indicators': ['type error', 'cannot convert', 'incompatible types'],
                'context_fix': 'add_type_information',
                'severity': 'high'
            },
            'api_misuse': {
                'indicators': ['wrong number of arguments', 'no such method', 'attribute error'],
                'context_fix': 'add_api_documentation',
                'severity': 'medium'
            },
            'logic_error': {
                'indicators': ['assertion failed', 'unexpected result', 'test failed'],
                'context_fix': 'add_example_usage',
                'severity': 'medium'
            }
        }

    def analyze_error_for_context_learning(self, error_msg, code, original_context):
        """Extract learning signal from error messages"""

        # Classify error type
        error_type = self.classify_error(error_msg)

        # Determine what context was missing
        missing_context = self.identify_missing_context(
            error_type=error_type,
            error_msg=error_msg,
            code=code,
            original_context=original_context
        )

        # Generate training signal
        training_signal = {
            'error_type': error_type,
            'error_location': self.extract_location(error_msg),
            'missing_context': missing_context,
            'context_update': self.generate_context_fix(error_type, code),
            'learning_weight': self.calculate_importance(error_type)
        }

        return training_signal

    def identify_missing_context(self, error_type, error_msg, code, original_context):
        """Determine exactly what context should have been included"""

        if error_type == 'missing_import':
            # Extract the undefined symbol
            symbol = self.extract_undefined_symbol(error_msg)

            # Find where this symbol should have been imported from
            correct_import = self.lookup_import_for_symbol(symbol, code)

            return {
                'type': 'import_statement',
                'symbol': symbol,
                'import': correct_import,
                'should_have_included': f"import {correct_import}"
            }

        elif error_type == 'api_misuse':
            # Extract method/function being called incorrectly
            api_call = self.extract_api_call(error_msg, code)

            # Find correct API documentation
            correct_usage = self.lookup_api_signature(api_call)

            return {
                'type': 'api_documentation',
                'api': api_call,
                'correct_signature': correct_usage,
                'should_have_included': f"API docs: {correct_usage}"
            }

        elif error_type == 'type_mismatch':
            # Extract type conflict
            expected_type, actual_type = self.extract_type_conflict(error_msg)

            return {
                'type': 'type_information',
                'expected': expected_type,
                'actual': actual_type,
                'should_have_included': f"Type requirement: {expected_type}"
            }

        return None
```

**Learning from Error Patterns:**

The CET maintains a growing database of error patterns and their context solutions:

```python
class ErrorPatternLearner:
    def __init__(self):
        self.error_context_patterns = defaultdict(list)
        self.fix_success_rates = defaultdict(float)

    def learn_from_compilation_failure(self, context, code, error, fixed_context):
        """Learn which context additions resolve which errors"""

        # Record the pattern
        pattern = {
            'original_context_features': self.extract_features(context),
            'error_signature': self.create_error_signature(error),
            'context_delta': self.compute_delta(context, fixed_context),
            'resolution_success': True
        }

        self.error_context_patterns[pattern['error_signature']].append(pattern)

        # Update success rates
        self.update_fix_effectiveness(pattern)

    def predict_context_fix(self, context, error):
        """Given an error, predict what context should be added"""

        error_sig = self.create_error_signature(error)

        # Find similar past errors
        similar_patterns = self.error_context_patterns[error_sig]

        if not similar_patterns:
            return None

        # Find highest success rate fix
        best_fix = max(similar_patterns,
                      key=lambda p: self.fix_success_rates[p['context_delta']])

        return best_fix['context_delta']

    def create_error_signature(self, error):
        """Create normalized error signature for pattern matching"""

        # Extract key components
        error_type = self.classify_error_type(error)
        error_location = self.extract_location_type(error)
        error_symbols = self.extract_key_symbols(error)

        # Create signature that generalizes across similar errors
        signature = {
            'type': error_type,
            'location_pattern': error_location,
            'symbols': sorted(error_symbols),
            'hash': self.compute_hash(error_type, error_location, error_symbols)
        }

        return signature['hash']
```

**Progressive Error Learning:**

CETs learn to anticipate errors before they occur:

```python
class ProactiveErrorPrevention:
    def __init__(self, error_learner):
        self.error_learner = error_learner
        self.prevention_patterns = {}

    def analyze_context_for_potential_errors(self, context, task):
        """Predict likely errors and preemptively add context"""

        predictions = []

        # Check for common missing context patterns
        if self.detects_api_usage(context) and not self.has_api_docs(context):
            predictions.append({
                'likely_error': 'api_misuse',
                'confidence': 0.85,
                'suggested_addition': 'api_documentation'
            })

        if self.detects_type_usage(context) and not self.has_type_info(context):
            predictions.append({
                'likely_error': 'type_mismatch',
                'confidence': 0.75,
                'suggested_addition': 'type_signatures'
            })

        if self.detects_imports_needed(context) and not self.has_import_context(context):
            predictions.append({
                'likely_error': 'missing_import',
                'confidence': 0.90,
                'suggested_addition': 'import_statements'
            })

        # Add preventive context
        enhanced_context = context
        for prediction in predictions:
            if prediction['confidence'] > 0.7:
                enhanced_context = self.add_preventive_context(
                    enhanced_context,
                    prediction['suggested_addition'],
                    task
                )

        return enhanced_context, predictions
```

**Error Message as Direct Supervision:**

Unlike traditional unsupervised or weakly supervised learning, error messages provide explicit supervision:

1. **Binary Signal**: Code compiles or doesn't (clear success/failure)
2. **Localized Signal**: Error points to exact problem location
3. **Actionable Signal**: Error message suggests what's needed
4. **Verifiable Signal**: Fix can be tested immediately

This creates a tight feedback loop where each error directly improves context quality for similar future situations.

## 3. Multi-LLM Code Generation

### 3.1 Diverse Solution Generation
```python
def generate_diverse_solutions(context, llm_team):
    solutions = {}
    for llm in llm_team:
        solutions[llm.name] = llm.generate_code(context)
    return solutions
```

### 3.2 Learning from Solution Variance

When multiple LLMs generate different solutions from the same context, the variance itself becomes a valuable learning signal. Different approaches reveal what the context emphasized, what it omitted, and how different reasoning patterns interpret the same information.

**Variance Analysis Framework:**

```python
class SolutionVarianceAnalyzer:
    def __init__(self, llm_team):
        self.llm_team = llm_team
        self.variance_patterns = {}

    def analyze_multi_llm_solutions(self, context, task, solutions):
        """Analyze what variance in solutions reveals about context quality"""

        analysis = {
            'solution_diversity': self.measure_diversity(solutions),
            'approach_variance': self.classify_approaches(solutions),
            'correctness_correlation': self.correlate_with_correctness(solutions),
            'context_ambiguity': self.detect_ambiguity(context, solutions),
            'missing_constraints': self.identify_missing_constraints(solutions, task)
        }

        return analysis

    def measure_diversity(self, solutions):
        """Quantify how different the solutions are"""

        # Structural diversity (different algorithms/patterns)
        structural = self.compute_structural_similarity(solutions)

        # Implementation diversity (different syntax/style)
        implementation = self.compute_implementation_similarity(solutions)

        # Semantic diversity (different interpretations)
        semantic = self.compute_semantic_similarity(solutions)

        return {
            'structural_diversity': 1.0 - structural,
            'implementation_diversity': 1.0 - implementation,
            'semantic_diversity': 1.0 - semantic,
            'overall_diversity': (1.0 - structural + 1.0 - semantic) / 2
        }

    def correlate_with_correctness(self, solutions):
        """Find which solution patterns correlate with correctness"""

        # Execute all solutions
        execution_results = {}
        for llm_name, code in solutions.items():
            result = execute_code(code)
            execution_results[llm_name] = {
                'compiles': result.compiles,
                'tests_pass': result.tests_pass,
                'approach': self.classify_approach(code)
            }

        # Find patterns in successful solutions
        successful_approaches = [
            r['approach'] for r in execution_results.values()
            if r['compiles'] and r['tests_pass']
        ]

        failed_approaches = [
            r['approach'] for r in execution_results.values()
            if not (r['compiles'] and r['tests_pass'])
        ]

        # What context led successful LLMs to choose good approaches?
        context_correlation = {
            'successful_patterns': Counter(successful_approaches),
            'failed_patterns': Counter(failed_approaches),
            'context_features_in_successful': self.extract_context_features_used(
                successful_approaches
            )
        }

        return context_correlation
```

**Learning from Divergent Solutions:**

High variance often indicates ambiguous or incomplete context:

```python
class ContextAmbiguityDetector:
    def detect_context_issues_from_variance(self, context, solutions, task):
        """Identify context problems based on solution variance"""

        issues = []

        # Different algorithms chosen -> missing algorithmic constraints
        if self.detects_algorithmic_variance(solutions):
            issues.append({
                'type': 'missing_algorithmic_constraint',
                'evidence': self.extract_algorithms_used(solutions),
                'fix': 'add_performance_requirements' if task.has_performance_req
                       else 'add_algorithm_hint'
            })

        # Different data structures -> missing structural guidance
        if self.detects_data_structure_variance(solutions):
            issues.append({
                'type': 'missing_data_structure_guidance',
                'evidence': self.extract_data_structures_used(solutions),
                'fix': 'add_data_structure_context'
            })

        # Different error handling -> missing error requirements
        if self.detects_error_handling_variance(solutions):
            issues.append({
                'type': 'missing_error_requirements',
                'evidence': self.extract_error_handling_styles(solutions),
                'fix': 'add_error_handling_requirements'
            })

        # Different edge case handling -> missing edge case context
        if self.detects_edge_case_variance(solutions):
            issues.append({
                'type': 'missing_edge_case_context',
                'evidence': self.extract_edge_case_handling(solutions),
                'fix': 'add_edge_case_examples'
            })

        return issues

    def generate_context_improvement(self, context, variance_issues, task):
        """Use variance analysis to improve context"""

        improved_context = context.copy()

        for issue in variance_issues:
            if issue['type'] == 'missing_algorithmic_constraint':
                # Add performance or algorithmic guidance
                improved_context = self.add_algorithmic_context(
                    improved_context,
                    task.performance_requirements
                )

            elif issue['type'] == 'missing_data_structure_guidance':
                # Add data structure constraints or examples
                improved_context = self.add_data_structure_examples(
                    improved_context,
                    task.data_characteristics
                )

            elif issue['type'] == 'missing_error_requirements':
                # Add error handling requirements
                improved_context = self.add_error_handling_spec(
                    improved_context,
                    task.error_requirements
                )

        return improved_context
```

**Consensus Learning:**

When multiple LLMs converge on similar solutions, it validates context quality:

```python
class ConsensusLearning:
    def learn_from_convergence(self, context, solutions, execution_results):
        """Learn from cases where LLMs agree and are correct"""

        # Measure agreement
        agreement_level = self.measure_agreement(solutions)

        # Check correctness
        all_correct = all(r.tests_pass for r in execution_results.values())

        if agreement_level > 0.8 and all_correct:
            # High agreement + all correct = excellent context
            self.record_excellent_context_pattern(context)
            return {
                'quality': 'excellent',
                'confidence': agreement_level,
                'reason': 'consensus_with_correctness'
            }

        elif agreement_level > 0.8 and not all_correct:
            # High agreement + all wrong = misleading context
            self.record_misleading_context_pattern(context)
            return {
                'quality': 'misleading',
                'confidence': agreement_level,
                'reason': 'consensus_but_incorrect',
                'issue': 'context_biases_toward_wrong_solution'
            }

        elif agreement_level < 0.3 and any(r.tests_pass for r in execution_results.values()):
            # Low agreement + some correct = ambiguous context
            self.record_ambiguous_context_pattern(context)
            return {
                'quality': 'ambiguous',
                'confidence': 1.0 - agreement_level,
                'reason': 'divergence_with_some_success',
                'issue': 'context_allows_multiple_interpretations'
            }

        return None
```

**Cross-Model Pattern Mining:**

Different LLMs have different strengths; variance reveals what context elements activate which capabilities:

```python
class CrossModelPatternMiner:
    def mine_model_specific_patterns(self, context_history):
        """Discover which context patterns work best for which LLMs"""

        patterns = defaultdict(lambda: {'strengths': [], 'weaknesses': []})

        for context, solutions, results in context_history:
            for llm_name, result in results.items():
                if result.tests_pass:
                    # What context features led to this LLM's success?
                    features = self.extract_context_features(context)
                    patterns[llm_name]['strengths'].extend(features)
                else:
                    # What context features confused this LLM?
                    features = self.extract_context_features(context)
                    patterns[llm_name]['weaknesses'].extend(features)

        # Identify model-specific patterns
        model_insights = {}
        for llm_name, pattern_data in patterns.items():
            model_insights[llm_name] = {
                'responds_well_to': Counter(pattern_data['strengths']).most_common(5),
                'struggles_with': Counter(pattern_data['weaknesses']).most_common(5),
                'unique_strengths': self.find_unique_strengths(llm_name, patterns)
            }

        return model_insights

    def optimize_context_for_ensemble(self, context, task, model_insights):
        """Optimize context to leverage ensemble diversity"""

        # Include elements that activate different models' strengths
        optimized = context.copy()

        for llm_name, insights in model_insights.items():
            for feature, count in insights['responds_well_to']:
                if feature not in optimized:
                    optimized = self.add_feature(optimized, feature)

        # Avoid elements that confuse multiple models
        common_weaknesses = self.find_common_weaknesses(model_insights)
        for weakness in common_weaknesses:
            optimized = self.remove_or_clarify(optimized, weakness)

        return optimized
```

**Variance as Training Signal:**

```python
class VarianceGradient:
    def compute_variance_gradient(self, context, solutions, results):
        """Use solution variance to compute context update direction"""

        # High variance + low success rate -> context too vague
        if self.high_variance(solutions) and self.low_success(results):
            gradient = {
                'direction': 'add_specificity',
                'magnitude': self.variance_magnitude(solutions),
                'target_features': self.identify_ambiguous_features(context)
            }

        # Low variance + low success rate -> context misleading
        elif self.low_variance(solutions) and self.low_success(results):
            gradient = {
                'direction': 'correct_misleading_elements',
                'magnitude': self.consensus_strength(solutions),
                'target_features': self.identify_misleading_features(context, solutions)
            }

        # Low variance + high success rate -> context excellent (no update needed)
        elif self.low_variance(solutions) and self.high_success(results):
            gradient = {
                'direction': 'maintain',
                'magnitude': 0.0,
                'target_features': self.identify_excellent_features(context)
            }

        # High variance + high success rate -> context allows valid alternatives (good)
        else:
            gradient = {
                'direction': 'slight_constraint',
                'magnitude': 0.3,
                'target_features': self.identify_optional_constraints(context)
            }

        return gradient
```

By analyzing not just whether code works, but how different models interpret the same context, CETs develop sophisticated understanding of what makes context clear, complete, and actionable.

## 4. Test Suite Integration

### 4.1 Test-Driven Context Engineering

Test suites provide explicit specifications of expected behavior. By analyzing which tests pass or fail, CETs learn exactly what context is needed to generate code that meets requirements.

**Test-Driven Context Learning:**

```python
class TestDrivenContextLearner:
    def __init__(self):
        self.test_context_patterns = {}
        self.requirement_extractors = {}

    def learn_from_test_suite(self, context, generated_code, test_suite, results):
        """Extract context lessons from test execution results"""

        lessons = {
            'passing_tests': [],
            'failing_tests': [],
            'context_gaps': [],
            'context_strengths': []
        }

        for test in test_suite:
            result = results[test.name]

            if result.passed:
                # What context enabled this test to pass?
                lessons['passing_tests'].append({
                    'test': test,
                    'context_features': self.extract_relevant_context(context, test),
                    'code_pattern': self.extract_code_pattern(generated_code, test)
                })
                lessons['context_strengths'].extend(
                    self.identify_effective_context_elements(context, test)
                )
            else:
                # What context was missing for this test?
                lessons['failing_tests'].append({
                    'test': test,
                    'failure_reason': result.failure_message,
                    'missing_context': self.infer_missing_context(test, result)
                })
                lessons['context_gaps'].append(
                    self.identify_context_gap(context, test, result)
                )

        return lessons

    def infer_missing_context(self, test, result):
        """Determine what context would have enabled test success"""

        # Analyze test structure to understand requirements
        test_requirements = self.extract_requirements_from_test(test)

        # Analyze failure to understand what was missing
        if "assertion failed" in result.failure_message:
            # Logic error - context didn't convey algorithm/logic clearly
            return {
                'type': 'algorithmic_clarity',
                'needed': 'clearer algorithm description or example',
                'test_expectation': test_requirements['expected_behavior']
            }

        elif "attribute error" in result.failure_message:
            # Missing method/attribute - context didn't specify interface
            return {
                'type': 'interface_specification',
                'needed': 'complete interface/API specification',
                'missing_element': self.extract_missing_attribute(result.failure_message)
            }

        elif "type error" in result.failure_message:
            # Type mismatch - context didn't specify types clearly
            return {
                'type': 'type_specification',
                'needed': 'explicit type information',
                'type_conflict': self.extract_type_conflict(result.failure_message)
            }

        return None

    def extract_requirements_from_test(self, test):
        """Parse test code to understand what it's testing"""

        requirements = {
            'expected_behavior': None,
            'input_output_examples': [],
            'edge_cases': [],
            'error_conditions': []
        }

        # Parse test assertions
        for assertion in test.assertions:
            if assertion.type == 'assertEqual':
                requirements['input_output_examples'].append({
                    'input': assertion.actual_args,
                    'expected_output': assertion.expected
                })

            elif assertion.type == 'assertRaises':
                requirements['error_conditions'].append({
                    'input': assertion.args,
                    'expected_error': assertion.exception_type
                })

            elif assertion.type == 'assertTrue':
                requirements['expected_behavior'] = assertion.condition

        # Identify edge cases from test names and data
        if 'edge' in test.name.lower() or 'boundary' in test.name.lower():
            requirements['edge_cases'].append(self.extract_edge_case(test))

        return requirements
```

**Context Optimization Through Test Feedback:**

```python
class TestFeedbackOptimizer:
    def optimize_context_from_test_results(self, context, test_results):
        """Iteratively improve context based on test failures"""

        optimized_context = context.copy()

        # Categorize test failures
        failure_categories = self.categorize_failures(test_results)

        # Address each category of failures
        for category, failures in failure_categories.items():
            if category == 'missing_functionality':
                # Tests failed because code didn't implement required features
                for failure in failures:
                    required_feature = self.extract_required_feature(failure.test)
                    optimized_context = self.add_feature_requirement(
                        optimized_context,
                        required_feature
                    )

            elif category == 'incorrect_logic':
                # Tests failed due to wrong algorithm/logic
                for failure in failures:
                    correct_logic = self.extract_correct_logic_from_test(failure.test)
                    optimized_context = self.add_logic_example(
                        optimized_context,
                        correct_logic
                    )

            elif category == 'edge_case_failures':
                # Tests failed on edge cases
                for failure in failures:
                    edge_case = self.extract_edge_case(failure.test)
                    optimized_context = self.add_edge_case_context(
                        optimized_context,
                        edge_case
                    )

            elif category == 'performance_failures':
                # Tests failed due to timeout/performance
                for failure in failures:
                    perf_requirement = self.extract_performance_requirement(failure.test)
                    optimized_context = self.add_performance_constraint(
                        optimized_context,
                        perf_requirement
                    )

        return optimized_context

    def add_feature_requirement(self, context, feature):
        """Add explicit feature requirement to context"""

        if 'requirements' not in context:
            context['requirements'] = []

        context['requirements'].append({
            'feature': feature.name,
            'description': feature.description,
            'test_coverage': feature.test_names
        })

        return context

    def add_logic_example(self, context, logic):
        """Add example demonstrating correct logic"""

        if 'examples' not in context:
            context['examples'] = []

        context['examples'].append({
            'scenario': logic.scenario,
            'correct_approach': logic.algorithm,
            'input_output': logic.example_io
        })

        return context
```

### 4.2 Coverage-Guided Optimization

Code coverage metrics reveal which parts of the specification are being addressed and which are ignored. CETs use coverage analysis to ensure comprehensive context.

**Coverage-Based Context Enhancement:**

```python
class CoverageGuidedOptimizer:
    def __init__(self):
        self.coverage_thresholds = {
            'line_coverage': 0.90,
            'branch_coverage': 0.85,
            'path_coverage': 0.75
        }

    def optimize_for_coverage(self, context, test_suite, coverage_report):
        """Enhance context to improve code coverage"""

        optimized_context = context.copy()

        # Identify uncovered code paths
        uncovered_lines = coverage_report['uncovered_lines']
        uncovered_branches = coverage_report['uncovered_branches']
        uncovered_paths = coverage_report['uncovered_paths']

        # For each uncovered element, determine what context would cover it
        for line_num in uncovered_lines:
            # Find tests that should have covered this line
            relevant_tests = self.find_tests_for_line(test_suite, line_num)

            if relevant_tests:
                # Tests exist but line not covered - context didn't guide implementation
                missing_context = self.infer_context_for_coverage(
                    line_num,
                    relevant_tests,
                    coverage_report
                )
                optimized_context = self.add_coverage_context(
                    optimized_context,
                    missing_context
                )

        # Handle branch coverage gaps
        for branch in uncovered_branches:
            branch_condition = self.extract_branch_condition(branch)

            # Find what triggers this branch
            trigger_context = self.find_branch_trigger(branch, test_suite)

            if trigger_context:
                optimized_context = self.add_branch_coverage_context(
                    optimized_context,
                    branch_condition,
                    trigger_context
                )

        return optimized_context

    def identify_uncovered_paths(self, coverage_report):
        """Identify execution paths not covered by tests"""

        uncovered = []

        for path in coverage_report['all_paths']:
            if not path.covered:
                uncovered.append({
                    'path': path,
                    'conditions': path.branch_conditions,
                    'trigger': self.determine_trigger_conditions(path)
                })

        return uncovered

    def augment_context_for_paths(self, context, missing_paths):
        """Add context to ensure uncovered paths are implemented"""

        augmented = context.copy()

        for path_info in missing_paths:
            # Determine what context would guide implementation of this path
            path_context = {
                'condition': path_info['conditions'],
                'trigger_scenario': path_info['trigger'],
                'expected_behavior': self.infer_path_behavior(path_info['path'])
            }

            # Add to context
            if 'execution_paths' not in augmented:
                augmented['execution_paths'] = []

            augmented['execution_paths'].append(path_context)

        return augmented

    def learn_coverage_patterns(self, context_history, coverage_history):
        """Learn which context patterns lead to better coverage"""

        patterns = []

        for context, coverage in zip(context_history, coverage_history):
            if coverage['line_coverage'] > 0.9:
                # High coverage - what context patterns enabled this?
                patterns.append({
                    'context_features': self.extract_features(context),
                    'coverage_achieved': coverage,
                    'quality': 'high_coverage'
                })

        # Find common features in high-coverage contexts
        high_coverage_features = self.find_common_features(
            [p for p in patterns if p['quality'] == 'high_coverage']
        )

        return {
            'effective_patterns': high_coverage_features,
            'coverage_correlation': self.compute_feature_coverage_correlation(patterns)
        }
```

**Test Suite as Context Specification:**

```python
class TestSuiteContextExtractor:
    def extract_context_from_tests(self, test_suite):
        """Use test suite as source of context about requirements"""

        extracted_context = {
            'functional_requirements': [],
            'edge_cases': [],
            'error_handling': [],
            'performance_requirements': [],
            'interface_specification': {}
        }

        for test in test_suite:
            # Extract functional requirements from test assertions
            for assertion in test.assertions:
                req = self.assertion_to_requirement(assertion)
                extracted_context['functional_requirements'].append(req)

            # Extract edge cases from boundary tests
            if self.is_edge_case_test(test):
                edge_case = self.extract_edge_case_spec(test)
                extracted_context['edge_cases'].append(edge_case)

            # Extract error handling requirements
            if test.expects_exception:
                error_req = self.extract_error_requirement(test)
                extracted_context['error_handling'].append(error_req)

            # Extract performance requirements from timeout tests
            if test.has_timeout:
                perf_req = self.extract_performance_requirement(test)
                extracted_context['performance_requirements'].append(perf_req)

            # Build interface specification from test usage
            interface_usage = self.extract_interface_usage(test)
            self.merge_interface_spec(
                extracted_context['interface_specification'],
                interface_usage
            )

        return extracted_context
```

By treating tests as both validation and specification, CETs learn to generate context that naturally leads to code satisfying all test requirements while achieving comprehensive coverage.

## 5. Compilation Error Analysis

### 5.1 Error Pattern Recognition

Compilation errors follow predictable patterns. By recognizing these patterns, CETs learn systematic approaches to context improvement rather than ad-hoc fixes.

**Error Pattern Taxonomy:**

```python
class ErrorPatternRecognizer:
    def __init__(self):
        self.error_taxonomy = {
            'syntax_errors': {
                'patterns': [
                    r'SyntaxError: invalid syntax',
                    r'unexpected token',
                    r'expected .* before .*'
                ],
                'common_causes': ['missing_punctuation', 'incorrect_indentation', 'language_confusion'],
                'context_fixes': ['add_syntax_examples', 'clarify_language_version']
            },
            'name_errors': {
                'patterns': [
                    r'NameError: name .* is not defined',
                    r'undefined name',
                    r'cannot find symbol'
                ],
                'common_causes': ['missing_import', 'typo', 'out_of_scope'],
                'context_fixes': ['add_import_context', 'add_scope_context']
            },
            'type_errors': {
                'patterns': [
                    r'TypeError: .* expected .* got .*',
                    r'incompatible types',
                    r'cannot convert'
                ],
                'common_causes': ['wrong_type_usage', 'missing_type_cast', 'api_misunderstanding'],
                'context_fixes': ['add_type_annotations', 'add_type_examples']
            },
            'import_errors': {
                'patterns': [
                    r'ImportError: cannot import',
                    r'ModuleNotFoundError',
                    r'package .* does not exist'
                ],
                'common_causes': ['missing_dependency', 'wrong_package_name', 'version_incompatibility'],
                'context_fixes': ['add_dependency_list', 'add_import_examples']
            },
            'attribute_errors': {
                'patterns': [
                    r'AttributeError: .* has no attribute',
                    r'no such method',
                    r'undefined method'
                ],
                'common_causes': ['api_version_mismatch', 'wrong_api_usage', 'missing_method'],
                'context_fixes': ['add_api_documentation', 'add_version_context']
            }
        }

    def recognize_pattern(self, error_message):
        """Classify error into known patterns"""

        for category, info in self.error_taxonomy.items():
            for pattern in info['patterns']:
                if re.search(pattern, error_message):
                    return {
                        'category': category,
                        'likely_causes': info['common_causes'],
                        'recommended_fixes': info['context_fixes'],
                        'confidence': self.compute_pattern_confidence(pattern, error_message)
                    }

        return {'category': 'unknown', 'confidence': 0.0}

    def extract_error_context(self, error_message, code, line_number):
        """Extract relevant context around error"""

        return {
            'error_line': code.lines[line_number],
            'surrounding_code': code.lines[max(0, line_number-3):line_number+3],
            'symbols_in_scope': self.get_symbols_at_line(code, line_number),
            'imports_available': self.get_imports(code),
            'error_specifics': self.parse_error_details(error_message)
        }
```

**Pattern-Based Learning:**

```python
class PatternBasedLearner:
    def __init__(self, recognizer):
        self.recognizer = recognizer
        self.pattern_solutions = defaultdict(list)

    def learn_from_error_resolution(self, error, original_context, fixed_context, success):
        """Learn which context changes resolve which error patterns"""

        # Recognize the error pattern
        pattern = self.recognizer.recognize_pattern(error.message)

        if success:
            # This context change successfully fixed this pattern
            context_delta = self.compute_context_difference(original_context, fixed_context)

            self.pattern_solutions[pattern['category']].append({
                'error_specifics': error,
                'context_change': context_delta,
                'success_rate': 1.0,
                'timestamp': datetime.now()
            })

            # Update success rates for this solution
            self.update_solution_effectiveness(pattern['category'], context_delta)

    def suggest_fix_for_error(self, error, current_context):
        """Suggest context improvement based on learned patterns"""

        pattern = self.recognizer.recognize_pattern(error.message)

        if pattern['category'] in self.pattern_solutions:
            # Find most successful solutions for this pattern
            solutions = self.pattern_solutions[pattern['category']]
            best_solution = max(solutions, key=lambda s: s['success_rate'])

            return {
                'suggested_context_change': best_solution['context_change'],
                'confidence': best_solution['success_rate'],
                'rationale': f"This fix resolved {pattern['category']} errors {int(best_solution['success_rate']*100)}% of the time"
            }

        # Fall back to general recommendations
        return {
            'suggested_context_change': self.generic_fix_for_category(pattern['category']),
            'confidence': 0.5,
            'rationale': 'No specific learned solution, using general approach'
        }
```

**Cross-Language Pattern Recognition:**

```python
class CrossLanguagePatternRecognizer:
    def __init__(self):
        self.language_specific_patterns = {
            'python': {
                'indentation_error': r'IndentationError',
                'name_error': r'NameError',
                'import_error': r'ImportError|ModuleNotFoundError'
            },
            'java': {
                'symbol_not_found': r'cannot find symbol',
                'type_mismatch': r'incompatible types',
                'import_error': r'package .* does not exist'
            },
            'javascript': {
                'undefined_var': r'ReferenceError: .* is not defined',
                'type_error': r'TypeError: Cannot read property',
                'syntax_error': r'SyntaxError: Unexpected token'
            },
            'rust': {
                'borrow_error': r'cannot borrow .* as mutable',
                'type_error': r'mismatched types',
                'trait_error': r'the trait .* is not implemented'
            }
        }

    def recognize_cross_language_patterns(self, error, language):
        """Identify patterns that appear across languages"""

        # Language-specific recognition
        specific = self.recognize_language_specific(error, language)

        # Map to universal pattern
        universal = self.map_to_universal_pattern(specific, language)

        return {
            'language_specific': specific,
            'universal_pattern': universal,
            'cross_language_solutions': self.get_cross_language_solutions(universal)
        }

    def map_to_universal_pattern(self, specific_pattern, language):
        """Map language-specific errors to universal concepts"""

        mapping = {
            ('python', 'name_error'): 'undefined_symbol',
            ('java', 'symbol_not_found'): 'undefined_symbol',
            ('javascript', 'undefined_var'): 'undefined_symbol',
            ('python', 'import_error'): 'missing_dependency',
            ('java', 'import_error'): 'missing_dependency',
            # ... more mappings
        }

        return mapping.get((language, specific_pattern), 'unknown')
```

### 5.2 Context Adjustments

Once error patterns are recognized, CETs learn specific context adjustments that systematically prevent these errors.

**Systematic Context Adjustment:**

```python
class ContextAdjuster:
    def __init__(self, pattern_recognizer):
        self.pattern_recognizer = pattern_recognizer
        self.adjustment_strategies = {}

    def adjust_context_for_error(self, context, error, code):
        """Generate improved context based on error analysis"""

        # Recognize error pattern
        pattern = self.pattern_recognizer.recognize_pattern(error.message)

        # Get error-specific context
        error_context = self.pattern_recognizer.extract_error_context(
            error.message, code, error.line_number
        )

        # Apply appropriate adjustment strategy
        if pattern['category'] == 'name_errors':
            return self.adjust_for_undefined_name(context, error_context)
        elif pattern['category'] == 'type_errors':
            return self.adjust_for_type_mismatch(context, error_context)
        elif pattern['category'] == 'import_errors':
            return self.adjust_for_import_issue(context, error_context)
        elif pattern['category'] == 'attribute_errors':
            return self.adjust_for_api_misuse(context, error_context)
        else:
            return self.generic_adjustment(context, error_context)

    def adjust_for_undefined_name(self, context, error_context):
        """Add context to prevent undefined name errors"""

        adjusted = context.copy()

        undefined_symbol = error_context['error_specifics']['symbol']

        # Find likely source of symbol
        likely_import = self.find_import_for_symbol(undefined_symbol)

        if likely_import:
            # Add import context
            if 'imports' not in adjusted:
                adjusted['imports'] = []
            adjusted['imports'].append({
                'symbol': undefined_symbol,
                'source': likely_import,
                'usage': f"from {likely_import} import {undefined_symbol}"
            })

        # Add scope context if symbol should be defined locally
        elif self.should_be_local(undefined_symbol, error_context):
            if 'local_definitions' not in adjusted:
                adjusted['local_definitions'] = []
            adjusted['local_definitions'].append({
                'symbol': undefined_symbol,
                'suggested_type': self.infer_type(undefined_symbol, error_context),
                'usage_context': error_context['surrounding_code']
            })

        return adjusted

    def adjust_for_type_mismatch(self, context, error_context):
        """Add type information to prevent type errors"""

        adjusted = context.copy()

        type_conflict = error_context['error_specifics']
        expected_type = type_conflict.get('expected')
        actual_type = type_conflict.get('actual')

        # Add explicit type annotations to context
        if 'type_annotations' not in adjusted:
            adjusted['type_annotations'] = []

        adjusted['type_annotations'].append({
            'location': error_context['error_line'],
            'expected_type': expected_type,
            'reason': f"Must be {expected_type}, not {actual_type}",
            'example': self.generate_type_example(expected_type)
        })

        return adjusted

    def adjust_for_import_issue(self, context, error_context):
        """Add dependency and import context"""

        adjusted = context.copy()

        missing_module = error_context['error_specifics']['module']

        # Add to dependencies
        if 'dependencies' not in adjusted:
            adjusted['dependencies'] = []

        adjusted['dependencies'].append({
            'module': missing_module,
            'installation': self.get_install_command(missing_module),
            'import_statement': self.get_import_statement(missing_module),
            'version': self.get_recommended_version(missing_module)
        })

        return adjusted

    def adjust_for_api_misuse(self, context, error_context):
        """Add API documentation to context"""

        adjusted = context.copy()

        api_element = error_context['error_specifics']['attribute']
        object_type = error_context['error_specifics']['object']

        # Find correct API usage
        correct_api = self.lookup_api(object_type, api_element)

        if 'api_documentation' not in adjusted:
            adjusted['api_documentation'] = []

        adjusted['api_documentation'].append({
            'object': object_type,
            'available_methods': correct_api['methods'],
            'available_attributes': correct_api['attributes'],
            'usage_examples': correct_api['examples'],
            'common_mistakes': correct_api['pitfalls']
        })

        return adjusted
```

**Learning Adjustment Effectiveness:**

```python
class AdjustmentLearner:
    def __init__(self):
        self.adjustment_history = []
        self.effectiveness_scores = defaultdict(float)

    def track_adjustment(self, original_context, adjusted_context, error, result):
        """Track whether adjustment successfully resolved error"""

        adjustment_type = self.classify_adjustment(original_context, adjusted_context)
        error_type = self.classify_error(error)

        record = {
            'timestamp': datetime.now(),
            'error_type': error_type,
            'adjustment_type': adjustment_type,
            'context_delta': self.compute_delta(original_context, adjusted_context),
            'success': result.compiles and result.tests_pass
        }

        self.adjustment_history.append(record)

        # Update effectiveness scores
        key = (error_type, adjustment_type)
        old_score = self.effectiveness_scores[key]
        new_score = old_score * 0.9 + (1.0 if record['success'] else 0.0) * 0.1
        self.effectiveness_scores[key] = new_score

    def recommend_adjustment(self, error):
        """Recommend most effective adjustment for error type"""

        error_type = self.classify_error(error)

        # Find adjustments for this error type
        relevant_adjustments = [
            (adj_type, score)
            for (err_type, adj_type), score in self.effectiveness_scores.items()
            if err_type == error_type
        ]

        if relevant_adjustments:
            best_adjustment = max(relevant_adjustments, key=lambda x: x[1])
            return {
                'adjustment_type': best_adjustment[0],
                'confidence': best_adjustment[1],
                'historical_success_rate': best_adjustment[1]
            }

        return None
```

Through systematic pattern recognition and learned adjustments, CETs develop expert-level understanding of how to prevent compilation errors through better context engineering.

## 6. Performance Benchmarking

### 6.1 Execution Time Analysis

Performance benchmarks provide quantitative feedback about code quality beyond correctness. CETs learn which context patterns lead to efficient implementations.

**Performance-Aware Context Learning:**

```python
class PerformanceContextLearner:
    def __init__(self):
        self.performance_patterns = {}
        self.efficiency_thresholds = {
            'fast': 0.8,  # Top 20% performance
            'acceptable': 0.5,  # Median performance
            'slow': 0.2  # Bottom 20% performance
        }

    def learn_from_performance(self, context, code, benchmark_results):
        """Learn which context features correlate with performance"""

        # Classify performance
        perf_category = self.classify_performance(benchmark_results)

        # Extract context features
        context_features = self.extract_context_features(context)

        # Extract code patterns
        code_patterns = self.analyze_code_patterns(code)

        # Record correlation
        learning_record = {
            'context_features': context_features,
            'code_patterns': code_patterns,
            'performance': benchmark_results,
            'category': perf_category,
            'context_to_performance_map': self.map_features_to_performance(
                context_features, code_patterns, benchmark_results
            )
        }

        # Update performance patterns
        self.update_performance_patterns(learning_record)

        return learning_record

    def map_features_to_performance(self, context_features, code_patterns, perf):
        """Identify which context features led to which performance characteristics"""

        mappings = []

        # Did context mention performance requirements?
        if 'performance_constraint' in context_features:
            mappings.append({
                'feature': 'explicit_performance_requirement',
                'present': True,
                'resulting_performance': perf['execution_time'],
                'correlation': 'positive' if perf['fast'] else 'negative'
            })

        # Did context include algorithmic hints?
        if 'algorithm_hint' in context_features:
            algo_type = context_features['algorithm_hint']
            mappings.append({
                'feature': f'algorithm_hint_{algo_type}',
                'resulting_complexity': code_patterns['time_complexity'],
                'execution_time': perf['execution_time']
            })

        # Did context include optimization examples?
        if 'optimization_example' in context_features:
            mappings.append({
                'feature': 'optimization_example_present',
                'code_used_optimization': code_patterns['uses_optimization'],
                'performance_gain': perf['speedup_vs_baseline']
            })

        return mappings

    def optimize_context_for_performance(self, context, task, target_performance):
        """Enhance context to encourage high-performance implementations"""

        optimized = context.copy()

        # Add performance requirements explicitly
        if 'performance_requirements' not in optimized:
            optimized['performance_requirements'] = {}

        optimized['performance_requirements'].update({
            'target_time_complexity': self.infer_optimal_complexity(task),
            'target_space_complexity': self.infer_space_requirements(task),
            'max_execution_time': target_performance.get('max_time', '1s'),
            'optimization_priority': 'speed'  # vs 'memory' or 'balanced'
        })

        # Find high-performance patterns for similar tasks
        similar_tasks = self.find_similar_tasks(task)
        high_perf_contexts = [
            p['context'] for p in self.performance_patterns.values()
            if p['task'] in similar_tasks and p['performance']['category'] == 'fast'
        ]

        # Extract common features from high-performing contexts
        if high_perf_contexts:
            common_features = self.find_common_features(high_perf_contexts)

            for feature in common_features:
                if feature not in optimized:
                    optimized[feature] = common_features[feature]

        # Add performance-oriented examples
        optimized['examples'] = self.get_high_performance_examples(task)

        return optimized
```

**Benchmarking Integration:**

```python
class BenchmarkIntegration:
    def __init__(self, benchmark_suite):
        self.benchmark_suite = benchmark_suite
        self.baseline_performance = {}

    def run_performance_analysis(self, context, generated_code, task):
        """Execute comprehensive performance analysis"""

        results = {
            'execution_time': {},
            'memory_usage': {},
            'cpu_usage': {},
            'scalability': {}
        }

        # Run multiple benchmark scenarios
        for scenario in self.benchmark_suite.get_scenarios(task):
            # Time execution
            exec_time = self.measure_execution_time(generated_code, scenario)
            results['execution_time'][scenario.name] = exec_time

            # Measure memory
            memory = self.measure_memory_usage(generated_code, scenario)
            results['memory_usage'][scenario.name] = memory

            # Measure CPU usage
            cpu = self.measure_cpu_usage(generated_code, scenario)
            results['cpu_usage'][scenario.name] = cpu

        # Test scalability
        results['scalability'] = self.test_scalability(generated_code, task)

        # Compare to baseline
        results['vs_baseline'] = self.compare_to_baseline(results, task)

        # Identify performance bottlenecks
        results['bottlenecks'] = self.identify_bottlenecks(generated_code, results)

        return results

    def identify_bottlenecks(self, code, benchmark_results):
        """Identify what's causing performance issues"""

        bottlenecks = []

        # Analyze algorithmic complexity
        complexity = self.analyze_complexity(code)
        if complexity['time'] > 'O(n)':
            bottlenecks.append({
                'type': 'algorithmic_complexity',
                'current': complexity['time'],
                'impact': 'high',
                'suggestion': 'Use more efficient algorithm'
            })

        # Check for inefficient patterns
        inefficient_patterns = self.detect_inefficient_patterns(code)
        for pattern in inefficient_patterns:
            bottlenecks.append({
                'type': 'inefficient_pattern',
                'pattern': pattern.name,
                'location': pattern.line_numbers,
                'impact': pattern.performance_impact,
                'suggestion': pattern.optimization
            })

        # Analyze I/O usage
        io_analysis = self.analyze_io_patterns(code)
        if io_analysis['blocking_calls'] > 0:
            bottlenecks.append({
                'type': 'blocking_io',
                'count': io_analysis['blocking_calls'],
                'impact': 'medium',
                'suggestion': 'Use async I/O or batching'
            })

        return bottlenecks

    def generate_performance_feedback(self, context, code, results):
        """Generate actionable feedback for context improvement"""

        feedback = {
            'performance_summary': self.summarize_performance(results),
            'context_issues': [],
            'context_improvements': []
        }

        # If performance is poor, identify what context was missing
        if results['vs_baseline']['speedup'] < 0.5:  # 2x slower than baseline
            # What context would have led to better performance?
            feedback['context_issues'].append({
                'issue': 'no_performance_guidance',
                'evidence': 'context lacks performance requirements',
                'impact': f"{results['vs_baseline']['speedup']:.1%} of baseline performance"
            })

            feedback['context_improvements'].append({
                'improvement': 'add_performance_requirements',
                'specific': {
                    'add_complexity_constraint': True,
                    'add_optimization_examples': True,
                    'add_benchmark_targets': True
                }
            })

        # Check if context led to wrong algorithm choice
        if 'bottlenecks' in results:
            for bottleneck in results['bottlenecks']:
                if bottleneck['type'] == 'algorithmic_complexity':
                    feedback['context_issues'].append({
                        'issue': 'suboptimal_algorithm_choice',
                        'current_complexity': bottleneck['current'],
                        'evidence': 'context did not guide to efficient algorithm'
                    })

                    feedback['context_improvements'].append({
                        'improvement': 'add_algorithmic_guidance',
                        'specific': {
                            'mention_complexity_requirement': True,
                            'provide_algorithm_hint': bottleneck['suggestion']
                        }
                    })

        return feedback
```

### 6.2 Memory Usage Patterns

Memory efficiency is as important as execution speed. CETs learn to generate context that leads to memory-efficient implementations.

**Memory-Aware Context Learning:**

```python
class MemoryContextLearner:
    def __init__(self):
        self.memory_patterns = {}
        self.memory_thresholds = {
            'small': 1_000_000,      # 1 MB
            'medium': 100_000_000,   # 100 MB
            'large': 1_000_000_000   # 1 GB
        }

    def learn_from_memory_usage(self, context, code, memory_profile):
        """Learn which context patterns lead to memory-efficient code"""

        analysis = {
            'peak_memory': memory_profile['peak'],
            'allocations': memory_profile['total_allocations'],
            'deallocations': memory_profile['total_deallocations'],
            'leaks': memory_profile['leaks'],
            'efficiency_category': self.categorize_memory_usage(memory_profile)
        }

        # Extract memory-relevant context features
        context_features = {
            'mentions_memory': self.context_mentions_memory(context),
            'data_structure_hints': self.extract_data_structure_hints(context),
            'size_constraints': self.extract_size_constraints(context),
            'streaming_hints': self.detect_streaming_hints(context)
        }

        # Extract memory-relevant code patterns
        code_patterns = {
            'data_structures_used': self.extract_data_structures(code),
            'uses_generators': self.uses_generators(code),
            'uses_streaming': self.uses_streaming(code),
            'preallocates': self.preallocates_memory(code),
            'has_memory_leaks': len(analysis['leaks']) > 0
        }

        # Learn correlations
        self.record_memory_correlation(context_features, code_patterns, analysis)

        return analysis

    def optimize_context_for_memory(self, context, task, memory_constraints):
        """Add context to encourage memory-efficient implementations"""

        optimized = context.copy()

        # Add explicit memory constraints
        if 'memory_constraints' not in optimized:
            optimized['memory_constraints'] = {}

        optimized['memory_constraints'].update({
            'max_memory': memory_constraints.get('max_memory', '100MB'),
            'streaming_preferred': task.data_size > self.memory_thresholds['large'],
            'avoid_copies': True if task.data_size > self.memory_thresholds['medium'] else False
        })

        # Add memory-efficient patterns based on task
        if task.data_size > self.memory_thresholds['large']:
            # Large data - need streaming/generators
            optimized['implementation_hints'] = {
                'use_generators': 'Process data in chunks to avoid loading all at once',
                'avoid_list_comprehensions': 'Use generator expressions instead',
                'example': self.get_streaming_example(task)
            }

        elif task.involves_large_objects:
            # Large objects - need careful allocation
            optimized['implementation_hints'] = {
                'preallocate': 'Preallocate arrays/buffers to avoid repeated allocation',
                'reuse_buffers': 'Reuse buffers instead of creating new ones',
                'example': self.get_preallocation_example(task)
            }

        # Add examples of memory-efficient implementations
        similar_efficient = self.find_memory_efficient_solutions(task)
        if similar_efficient:
            optimized['memory_efficient_examples'] = similar_efficient

        return optimized

    def analyze_memory_patterns(self, context_history, memory_history):
        """Identify patterns in context that lead to memory efficiency"""

        patterns = []

        for context, memory_profile in zip(context_history, memory_history):
            if memory_profile['peak'] < self.memory_thresholds['small']:
                # Very memory efficient
                patterns.append({
                    'context_features': self.extract_features(context),
                    'efficiency': 'high',
                    'peak_memory': memory_profile['peak'],
                    'key_techniques': self.identify_memory_techniques(context)
                })

        # Find common features in memory-efficient contexts
        efficient_features = self.find_common_features(
            [p for p in patterns if p['efficiency'] == 'high']
        )

        return {
            'efficient_patterns': efficient_features,
            'memory_correlation': self.compute_memory_correlation(patterns)
        }
```

**Memory Profiling Integration:**

```python
class MemoryProfiler:
    def __init__(self):
        self.profiler = memory_profiler.MemoryProfiler()

    def profile_memory_usage(self, code, inputs):
        """Detailed memory profiling of generated code"""

        profile = {
            'peak_memory': 0,
            'allocations': [],
            'hotspots': [],
            'leaks': [],
            'timeline': []
        }

        # Run with memory profiling
        with self.profiler.start():
            result = execute_code(code, inputs)

            # Collect memory timeline
            profile['timeline'] = self.profiler.get_timeline()

            # Find peak memory
            profile['peak_memory'] = max(t['memory'] for t in profile['timeline'])

            # Identify allocation hotspots
            profile['hotspots'] = self.profiler.get_allocation_hotspots()

            # Check for leaks
            profile['leaks'] = self.profiler.detect_leaks()

        # Analyze allocation patterns
        profile['allocation_analysis'] = self.analyze_allocations(profile)

        return profile

    def generate_memory_feedback(self, context, code, profile):
        """Generate feedback about memory usage"""

        feedback = {
            'memory_summary': f"Peak: {profile['peak_memory'] / 1e6:.1f} MB",
            'context_issues': [],
            'context_improvements': []
        }

        # Check for excessive memory usage
        if profile['peak_memory'] > 100_000_000:  # > 100 MB
            feedback['context_issues'].append({
                'issue': 'excessive_memory_usage',
                'peak': f"{profile['peak_memory'] / 1e6:.1f} MB",
                'cause': 'context did not emphasize memory efficiency'
            })

            feedback['context_improvements'].append({
                'improvement': 'add_memory_constraints',
                'specific': {
                    'add_max_memory_limit': True,
                    'suggest_streaming': True,
                    'example_memory_efficient_approach': self.get_efficient_example()
                }
            })

        # Check for memory leaks
        if profile['leaks']:
            feedback['context_issues'].append({
                'issue': 'memory_leaks',
                'count': len(profile['leaks']),
                'cause': 'context did not emphasize proper resource cleanup'
            })

            feedback['context_improvements'].append({
                'improvement': 'add_resource_management_context',
                'specific': {
                    'emphasize_cleanup': True,
                    'show_context_manager_pattern': True,
                    'example': 'with open(...) or try/finally patterns'
                }
            })

        # Check for inefficient data structures
        hotspots = profile['hotspots']
        for hotspot in hotspots:
            if hotspot['type'] == 'repeated_allocation':
                feedback['context_issues'].append({
                    'issue': 'repeated_allocations',
                    'location': hotspot['location'],
                    'cause': 'context did not suggest preallocating or reusing buffers'
                })

        return feedback
```

By learning from performance benchmarks and memory profiles, CETs develop understanding of how context influences not just correctness, but efficiency and resource usage.

## 7. Security Vulnerability Detection

### 7.1 Security Scanning Integration

Security vulnerabilities provide critical feedback about code safety. CETs learn to generate contexts that promote secure coding practices by default.

**Security-Aware Context Learning:**

```python
class SecurityContextLearner:
    def __init__(self):
        self.vulnerability_patterns = {}
        self.secure_coding_patterns = {}

    def learn_from_security_scan(self, context, code, scan_results):
        """Learn which context patterns prevent vulnerabilities"""

        if scan_results['vulnerabilities']:
            # Learn what context was missing
            for vuln in scan_results['vulnerabilities']:
                missing_context = self.infer_missing_security_context(vuln, context)
                self.record_vulnerability_pattern(vuln, missing_context)
        else:
            # Learn what context promoted security
            self.record_secure_pattern(context, code)

        return self.generate_security_improvements(context, scan_results)
```

### 7.2 Vulnerability Prevention Through Context

CETs learn to prevent common vulnerability classes through proactive context engineering:

**OWASP Top 10 Context Patterns:**

```python
security_context_patterns = {
    'sql_injection': {
        'context_addition': 'Always use parameterized queries, never concatenate SQL',
        'example': 'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))'
    },
    'xss': {
        'context_addition': 'Sanitize all user input before rendering',
        'example': 'html.escape(user_input) or use template auto-escaping'
    },
    'authentication': {
        'context_addition': 'Use established authentication libraries, never roll your own',
        'example': 'Use bcrypt for password hashing, JWT for tokens'
    }
}
```

The integration of security scanning ensures that CETs learn to prioritize security alongside functionality, embedding secure coding practices into the context generation process.

## 8. Conclusion

This paper presented six comprehensive feedback mechanisms for training Context Engineering Transformers through code execution. Error messages provide explicit, localized signals for context improvement. Multi-LLM solution variance reveals context ambiguity and guides specificity. Test suites serve dual roles as validation and specification extraction. Compilation error patterns enable systematic context adjustments. Performance benchmarks guide efficiency-aware context generation. Security scanning ensures contexts promote secure coding practices.

Together, these mechanisms transform code execution from a binary success/failure metric into rich, multi-dimensional training signals. The key insight is that each type of execution feedback identifies specific, actionable context improvements rather than vague quality assessments. This specificity enables systematic learning of context engineering patterns.

The feedback mechanisms presented here form the foundation for production-scale context learning systems. The companion paper (Paper 03B: Production Learning Pipeline Integration) builds on these mechanisms to address system integration, learning algorithms, and deployment at scale.

## References

[To be added]

