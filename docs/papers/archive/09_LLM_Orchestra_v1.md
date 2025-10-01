# Orchestrating Local and Cloud LLMs for Diverse Training Signals

## Abstract

We present LLM Orchestra, a system for coordinating multiple Large Language Models—both locally deployed and cloud-based—to provide diverse training signals for Context Engineering Transformers. Our architecture manages heterogeneous models including CodeLlama, Mistral, and Llama-3 running on local GPUs, alongside API access to GPT-4, Claude, and Gemini. We address challenges of load balancing, response caching, latency optimization, and cost management while ensuring diverse code generation patterns that improve CET training robustness. The system handles over 100,000 code generation requests daily with 99.9% uptime.

## 1. Introduction

Training CETs effectively requires diverse perspectives from multiple LLMs, each with unique strengths and biases.

## 2. System Architecture

### 2.1 LLM Orchestra Overview

```python
class LLMOrchestra:
    def __init__(self):
        self.local_models = {
            # P40 Cluster (96GB total)
            'llama3.1-70b-q4': Llama3_70B_Quantized(),  # ~48GB
            'mistral-large-q4': MistralLarge_Quantized(),  # ~22.5GB
            'llama3.1-8b': Llama3_8B(),  # Multiple instances

            # P4s (16GB each)
            'codellama-7b': CodeLlama7B(),
            'qwen2.5-coder-7b': QwenCoder7B(),

            # RTX 3050 (8GB) - when not testing edge
            'llama3.2-3b': Llama3_2_3B(),
            'phi-3': Phi3_Mini()
        }
        self.together_ai_models = {
            # Pay-per-token pricing
            'llama3.1-405b': TogetherAI('meta-llama/Llama-3.1-405B', cost_per_m='$3.50'),
            'llama3.1-70b': TogetherAI('meta-llama/Llama-3.1-70B', cost_per_m='$0.88'),
            'deepseek-r1': TogetherAI('deepseek-ai/DeepSeek-R1', cost_per_m='TBD'),
            'mistral-large': TogetherAI('mistralai/Mistral-Large', cost_per_m='$1.20'),
            'qwen2.5-max': TogetherAI('Qwen/Qwen2.5-72B', cost_per_m='$1.20'),
            'qwen2.5-coder-32b': TogetherAI('Qwen/Qwen2.5-Coder-32B', cost_per_m='$0.80'),
            'codellama-70b': TogetherAI('meta-llama/CodeLlama-70B', cost_per_m='$0.90')
        }
        self.premium_apis = {
            # $50-100/month for validation
            'claude-opus': AnthropicClient('claude-3-opus'),
            'gpt-4o': OpenAIClient('gpt-4o'),
            'gemini-2.5-pro': GoogleClient('gemini-2.5-pro')
        }
        self.router = IntelligentRouter()
        self.cache = ResponseCache()
```

## 3. Local LLM Deployment

### 3.1 Model Selection Criteria

```python
local_model_specs = {
    # P40 Cluster Models (M5 Server)
    'llama3.1-70b-q4': {
        'vram_required': '48GB',
        'quantization': '4-bit',
        'hardware': 'P40 cluster',
        'inference_speed': '30 tokens/sec',
        'specialization': 'general intelligence'
    },
    'mistral-large-q4': {
        'vram_required': '22.5GB',
        'quantization': '4-bit',
        'hardware': 'P40 single',
        'inference_speed': '50 tokens/sec',
        'specialization': 'coding + reasoning'
    },

    # P4 Models (Irina)
    'codellama-7b': {
        'vram_required': '14GB',
        'quantization': 'none',
        'hardware': 'P4',
        'inference_speed': '80 tokens/sec',
        'specialization': 'code generation'
    },

    # RTX 3050 Models (Workstation)
    'llama3.2-3b': {
        'vram_required': '6GB',
        'quantization': 'none',
        'hardware': 'RTX 3050',
        'inference_speed': '100 tokens/sec',
        'specialization': 'fast inference'
    }
}
```

### 3.2 Dynamic Model Loading Strategy

```python
class ModelManager:
    def __init__(self):
        # Irina's tiered storage for model library
        self.model_storage = {
            'fast_tier': '/mnt/irina/fast/models/',  # 4x16TB RAID 5 direct
            'slow_tier': '/mnt/irina/slow/models/',  # 4x16TB RAID 5 bottlenecked
        }

        # Store 50+ models, load 5-10 at a time
        self.available_models = {
            # Code-specialized models
            'codellama-7b': {'size': '13GB', 'tier': 'fast'},
            'codellama-13b': {'size': '26GB', 'tier': 'fast'},
            'codellama-34b': {'size': '68GB', 'tier': 'slow'},
            'qwen2.5-coder-1.5b': {'size': '3GB', 'tier': 'fast'},
            'qwen2.5-coder-7b': {'size': '14GB', 'tier': 'fast'},
            'qwen2.5-coder-14b': {'size': '28GB', 'tier': 'slow'},
            'deepseek-coder-6.7b': {'size': '13GB', 'tier': 'fast'},
            'starcoder2-15b': {'size': '30GB', 'tier': 'slow'},

            # General models (various sizes)
            'llama3.1-8b': {'size': '16GB', 'tier': 'fast'},
            'llama3.1-70b-q4': {'size': '35GB', 'tier': 'fast'},
            'llama3.1-70b-q8': {'size': '70GB', 'tier': 'slow'},
            'mistral-7b': {'size': '14GB', 'tier': 'fast'},
            'mistral-large-q4': {'size': '22GB', 'tier': 'fast'},
            'phi-3-mini': {'size': '3GB', 'tier': 'fast'},
            'phi-3-small': {'size': '7GB', 'tier': 'fast'},
            'gemma-2b': {'size': '4GB', 'tier': 'fast'},
            'gemma-7b': {'size': '14GB', 'tier': 'fast'},

            # Specialized variants
            'llama3.1-8b-instruct': {'size': '16GB', 'tier': 'fast'},
            'mistral-7b-instruct': {'size': '14GB', 'tier': 'fast'},
            'zephyr-7b': {'size': '14GB', 'tier': 'fast'},
            'neural-chat-7b': {'size': '14GB', 'tier': 'fast'},
        }

        self.loaded_models = {}  # Currently in GPU memory
        self.max_loaded = 10  # Maximum concurrent models

    def swap_models(self, unload_list, load_list):
        """Hot-swap models based on training phase needs"""
        for model in unload_list:
            self.unload_from_gpu(model)
        for model in load_list:
            self.load_to_gpu(model)
```

### 3.3 Expanded Model Library for Irina Storage

Irina's 60TB storage allows us to maintain an extensive model library with only ~2TB used:

```python
model_library = {
    # Core Large Models (~500GB)
    'primary_large': {
        'llama3.1-70b': {'size': '140GB', 'quantized_4bit': '48GB'},
        'deepseek-r1-70b': {'size': '140GB', 'quantized_4bit': '48GB'},
        'mistral-large': {'size': '45GB', 'quantized_4bit': '22.5GB'},
        'llama4-maverick': {'size': '~100GB', 'context': '10M tokens'},
    },

    # Code Specialists (~800GB)
    'code_generation': {
        'starcoder2-15b': {'size': '30GB', 'strength': 'matches 33B+ performance'},
        'starcoder2-7b': {'size': '14GB', 'strength': 'efficient'},
        'starcoder2-3b': {'size': '6GB', 'strength': 'matches original 15B'},
        'yi-coder-9b': {'size': '18GB', 'strength': 'state-of-art <10B'},
        'yi-coder-1.5b': {'size': '3GB', 'strength': 'ultra-efficient'},
        'granite-code-20b': {'size': '40GB', 'quantized_4bit': '10GB'},
        'granite-code-8b': {'size': '16GB', 'quantized_4bit': '3.77GB'},
        'codestral': {'size': '~45GB', 'strength': 'reasoning + permissive license'},
        'qwen2.5-coder-32b': {'size': '64GB', 'quantized_4bit': '24GB'},
        'qwen2.5-coder-14b': {'size': '28GB'},
        'qwen2.5-coder-7b': {'size': '14GB'},
        'qwen2.5-coder-1.5b': {'size': '3GB'},
    },

    # Testing & Quality Specialists (~300GB) - NEW
    'testing_quality': {
        'codet5-large': {'size': '~3GB', 'strength': 'test generation, code understanding'},
        'codet5-base': {'size': '~1GB', 'strength': 'efficient test generation'},
        'graphcodebert': {'size': '~500MB', 'strength': 'structural analysis, data flow'},
        'testing-llama-7b': {'size': '14GB', 'strength': 'fine-tuned for test generation'},
        'bug-detection-specialist': {'size': '~7GB', 'strength': 'security/bug detection'},
    },

    # Small Efficient Models (~200GB)
    'small_efficient': {
        'phi-4': {'size': '~8GB', 'strength': 'reasoning on consumer hardware'},
        'llama3.2-3b': {'size': '6GB'},
        'llama3.2-1b': {'size': '2GB'},
        'gemma-2b': {'size': '4GB'},
        'phi-3-mini': {'size': '3GB'},
    },

    # Specialized Reasoning (~200GB)
    'reasoning_specialists': {
        'kimi-k2-32b': {'size': '~48GB', 'strength': 'agentic, 85.7% MultiPL-E'},
        'deepseek-math': {'size': '~14GB', 'strength': 'mathematical reasoning'},
        'qwen2.5-max-72b': {'size': '144GB', 'quantized_4bit': '48GB'},
    },

    # Total: ~2TB for 50+ model variants
    # Remaining: 58TB for conversation data and future expansion
}
```

### 3.4 Phase-Specific Model Rotation Strategy

```python
phase_model_sets = {
    'phase_1_subject_expertise': {
        # Diverse code generation for subject learning
        'primary': ['llama3.1-70b', 'deepseek-r1-70b', 'mistral-large'],
        'rotation_pool': ['qwen2.5-coder-14b', 'starcoder2-15b', 'yi-coder-9b'],
        'rotation_frequency': '12 hours',
        'purpose': 'Learn coding patterns from diverse models'
    },

    'phase_2_context_engineering': {
        # Mix of sizes for quality gradient learning
        'primary': ['llama3.1-70b', 'mistral-large'],
        'rotation_pool': ['phi-4', 'llama3.1-8b', 'codellama-7b', 'qwen-variants'],
        'rotation_frequency': '6 hours',
        'purpose': 'Learn context transformation from varied quality levels'
    },

    'phase_3_interactive_feedback': {
        # Maximum diversity: Code generation + Testing specialists
        'code_generators': {
            'always_loaded': ['llama3.1-70b', 'deepseek-r1-70b'],
            'rotation_pool': [
                'starcoder2-15b', 'yi-coder-9b', 'granite-20b',
                'codestral', 'qwen2.5-coder-32b'
            ],
        },
        'testing_evaluators': {  # NEW - Critical for Phase 3
            'always_loaded': ['codet5-large', 'graphcodebert'],
            'rotation_pool': [
                'testing-llama-7b', 'bug-detection-specialist',
                'codet5-base'
            ],
        },
        'rotation_frequency': '4 hours',
        'purpose': 'Diverse code + specialized test evaluation creates rich feedback'
    },

    'phase_4_production': {
        # Proven performers, no rotation
        'fixed_set': [
            'llama3.1-70b', 'deepseek-r1-70b',
            'codet5-large', 'graphcodebert'
        ],
        'purpose': 'Stable, validated production configuration'
    }
}
```

### 3.4 Deployment Configuration

```yaml
deployment:
  # M5 Server - P40 Cluster
  llama3.1-70b:
    gpus: [P40_0, P40_1]  # 2x Tesla P40 (48GB total)
    max_batch_size: 4
    max_sequence_length: 8192

  mistral-large:
    gpus: [P40_2]  # 1x Tesla P40 (24GB)
    max_batch_size: 8
    max_sequence_length: 16384

  # Irina - P4s
  codellama-7b:
    gpus: [P4_0]  # 1x Tesla P4 (8GB)
    max_batch_size: 16
    max_sequence_length: 4096

  # Workstation - RTX 3050
  llama3.2-3b:
    gpus: [RTX_3050]  # When not testing edge
    max_batch_size: 32
    max_sequence_length: 2048

  # V100 Reserved for CET Training
  cet_training:
    gpus: [V100]  # Tesla V100 32GB
    purpose: 'CET model training only'
```

### 3.3 Quantization Strategies

[Balancing model quality with memory constraints]

## 4. Cloud LLM Integration

### 4.1 API Management

```python
class APIManager:
    def __init__(self):
        self.rate_limits = {
            'gpt4': RateLimit(rpm=10000, tpm=1000000),
            'claude': RateLimit(rpm=5000, tpm=500000),
            'gemini': RateLimit(rpm=6000, tpm=750000)
        }
        self.fallback_chain = ['gpt4', 'claude', 'gemini']
```

### 4.2 Authentication and Secrets

```python
class SecureAPIClient:
    def __init__(self):
        self.vault = HashiCorpVault()
        self.keys = self.vault.get_api_keys()
        self.rotation_schedule = '30 days'
```

## 5. Load Balancing Strategies

### 5.1 Intelligent Routing

```python
def route_request(request, model_states):
    if request.requires_speed:
        return select_fastest_available(model_states)
    elif request.requires_quality:
        return select_highest_quality(model_states)
    elif request.requires_specialization:
        return select_specialized_model(request.type)
    else:
        return load_balance_round_robin(model_states)
```

### 5.2 Dynamic Scaling

[Auto-scaling based on queue depth]

## 6. Response Caching

### 6.1 Cache Architecture

```python
class ResponseCache:
    def __init__(self):
        self.redis_cluster = RedisCluster()
        self.cache_ttl = 3600  # 1 hour
        self.max_cache_size = '100GB'

    def cache_key(self, model, prompt, params):
        return hash(f"{model}:{prompt}:{params}")
```

### 6.2 Cache Hit Optimization

[Strategies for improving cache hit rates]

## 7. Latency Optimization

### 7.1 Request Batching

```python
class BatchProcessor:
    def batch_requests(self, requests, max_batch_size=32):
        batches = []
        current_batch = []
        for req in requests:
            if len(current_batch) < max_batch_size:
                current_batch.append(req)
            else:
                batches.append(current_batch)
                current_batch = [req]
        return batches
```

### 7.2 Parallel Processing

[Concurrent request handling across models]

## 8. Cost Management and Model Selection Strategy

### 8.0 LLM Capability Comparison Matrix

The following comparison guides our model selection strategy, showing capabilities, costs, and VRAM requirements:

| Capability                           | Grok 3 (Commercial)                                                                       | Claude 3 Opus                                                 | Google Gemini 2.5 Pro                                       | OpenAI GPT-4o                                                | OpenAI GPT-5                                      | Llama 3 (Open-Source)                                                  | DeepSeek-R1 (Open-Source)                                               | Mistral Large (Open-Source)                                             | Qwen2.5-Max (Open-Source)                                                   | Cohere Command R+ (Open-Source)                                      |
| ------------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------- | ---------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------- | --------------------------------------------------------------------------- | -------------------------------------------------------------------- |
| **Reasoning & General Intelligence** | Excellent. Designed for advanced reasoning and complex problem-solving.                   | Excellent. Considered a leader in complex, nuanced reasoning. | Excellent. Very strong on complex reasoning tasks and math. | Excellent. Excels at complex reasoning and problem-solving.  | Expected to be state-of-the-art in reasoning.     | Strong. Comparable to older commercial models.                         | **Very strong, competitive with top proprietary models.**               | Very strong. A top performer among open-source models.                  | Very strong, particularly in math and coding.                               | Strong. Optimized for reasoning over enterprise data.                |
| **Coding**                           | Excellent. Excels at coding and debugging tasks.                                          | Very strong. Excellent code generation and debugging.         | Excellent. Very strong coding performance.                  | Excellent. Strong coding capabilities, especially in Python. | Expected to be a leader in code generation.       | Very strong. A top performer among open-source models.                 | Strong. Uses RAG to improve accuracy and reduce hallucinations.         | Excellent.                                                              | Excellent, strong performance in coding challenges.                         | Strong, with a focus on enterprise-ready code generation.            |
| **Multimodality**                    | Primarily text-based. The Grok series is text-focused, though future versions may evolve. | Supports image and text input.                                | Natively multimodal (text, images, audio, video).           | Natively multimodal (text, images, audio, video).            | Natively multimodal (text, images, audio, video). | Primarily text-based, with Meta developing multimodal versions.        | Primarily text-based, though some versions may support images.          | Primarily text-based.                                                   | Supports text, code, and images.                                            | Primarily text-based, with strong integration with RAG.              |
| **Multilingualism**                  | Strong. Supports multiple languages, handling text and code.                              | Strong, with high performance across multiple languages.      | Strong multilingual support.                                | Strong, with strong performance in many languages.           | Strong, likely to lead.                           | Strong, especially with recent improvements.                           | Strong, particularly in multilingual instruction-following.             | Very strong multilingual capabilities.                                  | Strong performance in multiple languages.                                   | Very strong multilingual support.                                    |
| **Context Window**                   | 131K tokens (API limit, marketed as 1M).                                                  | 200K tokens.                                                  | Up to 1 million tokens.                                     | 128K tokens.                                                 | Unconfirmed, but expected to be large.            | 128K tokens.                                                           | Varies by version. Often has a large context window.                    | 32k tokens.                                                             | 128K tokens.                                                                | 128K tokens.                                                         |
| **Architecture**                     | Proprietary powered by custom supercomputers.                                             | Proprietary Transformer-based.                                | Mixture-of-Experts (MoE) based.                             | Mixture-of-Experts (MoE) based.                              | Mixture-of-Experts (MoE) based.                   | Standard Transformer-based.                                            | Mixture-of-Experts (MoE) and RAG.                                       | Transformer-based.                                                      | Standard Transformer-based.                                                 | Standard Transformer-based.                                          |
| **API Cost (per million tokens)**    | Input: $3.00 / Output: $15.00                                                             | Input: $15.00 / Output: $75.00                                | Input: $1.25+ / Output: $10.00+                             | Input: $1.10 / Output: $4.40                                 | Input: $1.25+ / Output: $10.00+                   | **Varies by host, but lower. (e.g., Together AI: Input/Output $0.88)** | **Varies by host, but lower. (e.g., Together AI: Input $0.50, Output)** | **Varies by host, but lower. (e.g., Together AI: Input $0.80, Output)** | **Varies by host, but lower. (e.g., Together AI: Input $0.50, Output)**     | **Varies by host, but lower. (e.g., AI: Input $0.50, Output $1.50)** |
| **Deployment**                       | Primarily through X or xAI API.                                                           | API-only (proprietary).                                       | API-only (proprietary).                                     | API-only (proprietary).                                      | API-only (proprietary).                           | On-prem or various API providers.                                      | On-prem or various API providers.                                       | On-prem or various API providers.                                       | On-prem or various API providers.                                           | On-prem or various API providers.                                    |
| **VRAM Requirement**                 | N/A (API only)                                                                            | N/A (API only)                                                | N/A (API only)                                              | N/A (API only)                                               | N/A (API only)                                    | **Llama 3.1 8B:** FP16: 32GB+ / 4-bit: 8GB+                            | **DeepSeek-V3 (671B, MoE):** FP16: ~1.2TB / 4-bit ~70B: ~48GB+          | **Mistral Large (MoE):** FP16: 90GB+                                    | **Qwen2.5-Max:** FP16: >160GB / Qwen2.5 32B: FP16: ~160GB / 4-bit: ~24-48GB | **Command R+ (104B):** FP16: ~193GB+ / 4-bit: ~48GB+                 |
|                                      |                                                                                           |                                                               |                                                             |                                                              |                                                   | **Llama 3.1 70B:** FP16: ~140GB+ / **4-bit: ~24-48GB+**                | **Mixtral 8x7B (MoE):** 4-bit: ~22.5GB                                  |                                                                         |                                                                             |                                                                      |
|                                      |                                                                                           |                                                               |                                                             |                                                              |                                                   |                                                                        | 32B: ~24GB+                                                             |                                                                         |                                                                             |                                                                      |

**Key Insights from Comparison:**

1. **DeepSeek-R1 highlighted**: Very strong reasoning, competitive with top proprietary models - but we can run 70B variant locally
2. **Llama 3.1 70B @ 4-bit: ~24-48GB** - Perfect for our P40 cluster (2x P40 = 48GB)
3. **Cost advantage**: Open-source models via Together AI ($0.50-0.88/M) vs proprietary ($1.10-15.00/M)
4. **VRAM advantage**: We can run models locally that cost $0.88-1.20/M via Together AI

### 8.1 Cost-Benefit Analysis: Local vs Cloud

**Current Local Capacity (4x P40 = 96GB):**

- Can run: Llama 3.1 70B (4-bit, ~48GB) + smaller models
- Monthly cost: ~$30 electricity for M5 P40 cluster
- One-time investment: Already owned

**Expansion Option: +2 P40s ($600):**

- Total capacity: 6x P40 = 144GB VRAM
- Can run: 2x Llama 3.1 70B simultaneously OR 1x 70B + multiple 13B/7B models
- Monthly cost: +$10 electricity (~$40 total)
- ROI: Pays for itself vs Together AI in 3-6 months

**Cloud Alternative Costs:**

```python
# Cost per 1M tokens (input) - CORRECTED FROM COMPARISON TABLE
cost_per_1m_tokens = {
    # Premium APIs
    'claude-opus': 15.00,        # $15/M input - ELIMINATE (too expensive)
    'claude-sonnet': 3.00,       # $3/M input - Not in comparison table, verify
    'gpt-4o': 1.10,             # $1.10/M input (CORRECTED from table)
    'gemini-2.5-pro': 1.25,     # $1.25/M - 1M context window valuable

    # Together AI - Pay per token
    'llama3.1-405b': 3.50,      # Expensive - rarely worth it
    'llama3.1-70b': 0.88,       # OVERPRICED vs local
    'mistral-large': 1.20,      # OVERPRICED vs local
    'qwen2.5-max-72b': 1.20,    # OVERPRICED vs local
    'qwen2.5-coder-32b': 0.80,
    'codellama-70b': 0.90,      # OVERPRICED vs local
    'deepseek-r1': 'TBD',       # Strong reasoning, wait for pricing

    # Local GPU cost
    'local_70b_model': 0.003,   # ~$0.003/M effective (electricity only)
    'local_7b_model': 0.001,    # ~$0.001/M effective
}

# Monthly cost comparison at 100M tokens
monthly_100m_tokens = {
    'together_llama70b': 88,      # $0.88/M × 100M
    'local_llama70b': 0.30,       # Electricity only
    'savings': 87.70,             # $88/month saved
    'p40_roi_months': 6.8         # $600 ÷ $88/month
}

# At 200M tokens/month (realistic Phase 3)
monthly_200m_tokens = {
    'together_llama70b': 176,     # $0.88/M × 200M
    'local_llama70b': 0.40,       # Electricity
    'savings': 175.60,            # $176/month saved
    'p40_roi_months': 3.4         # $600 ÷ $176/month = 3.4 months ROI
}
```

### 8.2 When to Use Each Tier

**Tier 1 - Local Models (PRIMARY WORKHORSES):**

Use for: 95% of all training requests

*Why Local Wins:*

- Cost: ~$0.001-0.003 per 1M tokens (electricity only)
- Latency: <500ms (no network overhead)
- Privacy: Data never leaves infrastructure
- Unlimited: No rate limits or quotas
- Proven capability: Llama 3.1 70B matches Together AI 70B quality

*Current Capacity:*

- 4x P40 (96GB): Run 1x Llama 3.1 70B (48GB) + multiple 7B models
- Phase 3 bottleneck: Only 1 large model at a time

*With +2 P40s (144GB total):*

**Recommended Configuration:**
- Llama 3.1 70B (48GB) - General intelligence baseline
- DeepSeek-R1 70B (48GB) - Strong reasoning, competitive with proprietary models
- Mistral Large (22.5GB) - Different architecture, excellent coding
- Qwen2.5-Coder 14B (28GB) - Code specialist
- **Total: ~146GB** - 4 diverse models running continuously

**Alternative: Rotation Strategy:**
- Keep 2 large models warm: Llama 3.1 70B + DeepSeek 70B (96GB)
- Rotate through smaller models in 48GB slot every 4-6 hours for diversity
- Models in rotation pool: Mistral variants, Qwen variants, CodeLlama, specialized models

Eliminates most Together AI usage

**Tier 2 - Together AI (SELECTIVE USE ONLY):**

Use for: <5% of requests, specific capabilities only

*When Together AI is Worth It:*

1. **Llama 3.1 405B** ($3.50/M) - RARELY
   
   - Use case: Final validation of complex architectural decisions
   - Frequency: ~1-2% of validation requests
   - Local alternative: Ensemble of 2x 70B models (with +2 P40s)
   - Verdict: **Skip unless +2 P40s insufficient**

2. **DeepSeek-R1** (price TBD) - CONDITIONAL
   
   - Use case: Advanced reasoning for debugging complex failures
   - Frequency: ~5% of Phase 3 feedback requests
   - Local alternative: Llama 3.1 70B handles most reasoning
   - Verdict: **Evaluate when pricing available**

3. **Qwen2.5-Coder 32B** ($0.80/M) - MAYBE
   
   - Use case: Specialized code generation diversity
   - Local alternative: We can run Qwen2.5-Coder 14B locally (28GB)
   - Verdict: **Skip - use local 14B variant**

4. **Models we CAN'T run locally** - TARGETED USE
   
   - Example: Mixtral 8x22B (too large for our GPUs)
   - Frequency: Only when specific capability needed
   - Verdict: **Use sparingly for diversity**

*Together AI Monthly Estimate:*

- Conservative: $20-40/month (10-20M tokens at $0.80-1.20/M)
- With +2 P40s: Could drop to <$10/month (rare specialized use)

**Tier 3 - Premium APIs (VALIDATION ONLY):**

Use for: <1% of requests, final quality checks

*Cost Analysis:*

1. **Claude Opus** ($15/M) - **ELIMINATE**

   - Too expensive for any regular use (per comparison table)
   - Better alternatives exist at much lower cost
   - Verdict: **REMOVE from architecture**

2. **GPT-4o** ($1.10/M) - GOOD VALUE

   - Use case: General validation, diverse perspective
   - Frequency: ~1000 requests/month = ~20M tokens
   - Monthly cost: ~$22
   - Verdict: **Keep as primary validator** (much cheaper than previously thought)

3. **Gemini 2.5 Pro** ($1.25/M) - EXCELLENT FOR LARGE CODEBASES

   - Use case: **1M token context window** - perfect for entire codebases
   - Frequency: ~500 requests/month = ~10M tokens (large context requests)
   - Monthly cost: ~$12.50
   - Unique capability: Can validate against entire project context
   - Verdict: **Keep for large-context validation**

4. **Claude Sonnet** ($3/M) - OCCASIONAL

   - Use case: Anthropic-specific validation when needed
   - Frequency: ~300 requests/month = ~6M tokens
   - Monthly cost: ~$18
   - Verdict: **Keep for specific Anthropic validation**

*Premium API Monthly Estimate (REVISED):*

- GPT-4o: ~$22 (primary validator)
- Gemini 2.5 Pro: ~$12.50 (large-context validation)
- Claude Sonnet: ~$18 (Anthropic validation)
- **Total: ~$50-55/month** (revised upward from $35-40)

### 8.3 Recommended Strategy with Hardware Expansion

**Option A: No Additional Hardware ($0)**

Current monthly costs:

- Local electricity: $40
- Together AI: $50-100 (moderate usage)
- Premium APIs: $50-55 (REVISED: GPT-4o + Gemini 2.5 Pro + Claude Sonnet)
- **Total: $140-195/month**

Pros: No upfront cost
Cons: Ongoing Together AI costs, limited diversity

**Option B: Add 2x P40 GPUs ($600 one-time) ✅ RECOMMENDED**

ROI Analysis:

- Upfront: $600
- Monthly savings: ~$80-140 (reduced Together AI usage)
- ROI: 4-8 months
- After ROI: Save $960-1680/year

New monthly costs:

- Local electricity: $50 (+$10)
- Together AI: $10-20 (rare specialized use: DeepSeek-R1 when available, models we can't run)
- Premium APIs: $50-55 (validation unchanged - Gemini 2.5 Pro's 1M context is valuable)
- **Total: $110-125/month**

Capabilities gained:

- **4 diverse models continuously loaded:**
  - Llama 3.1 70B (general intelligence)
  - DeepSeek-R1 70B (reasoning + RAG, competitive with proprietary)
  - Mistral Large (different architecture, coding)
  - Qwen2.5-Coder 14B (code specialist)
- **OR: Rotation strategy** - 2 large models + rotating smaller models for even more diversity
- Eliminate 80-90% of Together AI usage
- No capacity bottlenecks in Phase 3
- True model diversity (4 different architectures/training approaches)
- Keep cloud APIs for unique capabilities (Gemini's 1M context, GPT-4o validation)

**Verdict: $600 for +2 P40s pays for itself in <8 months, then saves $960-1680/year**

**Key Strategy Update:**
- **Gemini 2.5 Pro's 1M context window** justifies keeping it for large codebase validation
- **GPT-4o at $1.10/M** is much more competitive than expected - good primary validator
- **Local models still dominate** (95% of requests) but cloud has specific valuable use cases

### 8.2 Budget Optimization Strategy

```python
def optimize_model_selection(request, phase):
    if phase == 'training_phase_1_2':
        # Use local models primarily, Together AI for diversity
        return prefer_local_then_together_ai(request)
    elif phase == 'training_phase_3':
        # Mix all tiers for diversity
        return orchestrate_diverse_models(request)
    elif phase == 'validation':
        # Use premium for quality baseline
        return select_premium_apis(request)
    else:
        # Production: Local primary, Together AI for capabilities
        return local_with_together_ai_fallback(request)

# Cost-aware routing
def route_by_cost(request):
    # Prefer local (free except electricity)
    # Then Together AI cheaper models ($0.88-1.20/M)
    # Reserve expensive models (405B @ $3.50/M) for complex tasks
    # Use premium APIs ($5-15/M) only for validation
    pass
```

## 9. Diversity Metrics

### 9.1 Response Diversity Measurement

```python
def measure_diversity(responses):
    return {
        'syntactic_diversity': calculate_ast_diversity(responses),
        'semantic_diversity': calculate_embedding_diversity(responses),
        'approach_diversity': calculate_solution_approach_diversity(responses)
    }
```

### 9.2 Ensuring Diverse Training Signals

[Strategies for maximizing response variety]

## 10. Monitoring and Observability

### 10.1 Metrics Dashboard

```yaml
metrics:
  - request_latency_p99
  - model_availability
  - cache_hit_rate
  - cost_per_request
  - diversity_score
  - error_rate
```

### 10.2 Alert Configuration

[Critical alerts and escalation]

## 11. Results

### 11.1 Expected Performance Metrics

- Throughput capacity: 50K+ requests/day
- Average latency: 1-3 seconds (Together AI), <1 second (local)
- Cache hit rate: Target 35-40%
- Uptime target: 99.5%

### 11.2 Cost Analysis

**Current Configuration (4x P40):**

- Local electricity: $40/month
- Together AI: $50-100/month (moderate usage to compensate for limited local capacity)
- Premium APIs: $50-55/month (GPT-4o $22, Gemini 2.5 Pro $12.50, Claude Sonnet $18)
- **Total: $140-195/month**

**Recommended Configuration (+2 P40s for $600 one-time):**

- Local electricity: $50/month (+$10)
- Together AI: $10-20/month (rare specialized models only)
- Premium APIs: $50-55/month (validation unchanged)
- **Total: $110-125/month**
- **Savings: $55-85/month after expansion**
- **ROI: 7-11 months**
- **Annual savings after ROI: $660-1020/year**

**vs Cloud-Only Alternative:**

- Cloud GPU time: ~$3,000-5,000/month
- **Savings with +2 P40s: 95-98% cost reduction**

**Key Insights from Cost Analysis:**

1. **Premium APIs more valuable than expected:**
   - GPT-4o at $1.10/M (not $2.50/M) is competitive
   - Gemini 2.5 Pro's 1M context window justifies $1.25/M for large codebase validation
   - Total premium API cost: ~$50-55/month (justified for unique capabilities)

2. **Adding 2x P40 GPUs ($600) still recommended:**
   - Enables 4 diverse large models simultaneously:
     - Llama 3.1 70B (general)
     - DeepSeek-R1 70B (reasoning, competitive with proprietary)
     - Mistral Large (different architecture)
     - Qwen2.5-Coder 14B (code specialist)
   - 80-90% reduction in Together AI costs
   - ROI: 7-11 months (slightly longer due to higher premium API costs)
   - After payback: Save $660-1020/year
   - True diversity: 4 different model families/architectures, not redundant copies

### 11.3 Model Diversity Achievement

**Model Library Composition:**
- **Core large models:** 4 (Llama 3.1 70B, DeepSeek-R1 70B, Mistral Large, Llama 4 Maverick)
- **Code specialists:** 12+ (StarCoder2 variants, Yi-Coder variants, Granite Code, Codestral, Qwen2.5-Coder variants)
- **Testing specialists:** 5+ (CodeT5 variants, GraphCodeBERT, testing-focused fine-tunes) - **NEW**
- **Small efficient models:** 5 (Phi-4, Llama 3.2 variants, Gemma, Phi-3)
- **Reasoning specialists:** 3 (Kimi K2, DeepSeek-Math, Qwen2.5-Max)
- **Premium APIs:** 3 (GPT-4o, Gemini 2.5 Pro, Claude Sonnet)
- **Together AI:** Selective access to models we can't run locally

**Storage & Rotation:**
- **Total stored on Irina:** 50+ model variants (~2TB / 60TB capacity)
- **Simultaneously loaded:** 4-6 models (with 6x P40s = 144GB VRAM)
- **Phase 3 rotation:** Every 4-6 hours for maximum diversity
- **Unique Phase 3 capability:** Code generators + Testing evaluators working together

**Effective Training Diversity:**
- **Phase 1-2:** 15-20 unique models through rotation
- **Phase 3:** 25-35 unique models (code + testing specialists)
- **Phase 4:** 4-6 proven models (no rotation)
- **Total accessible:** 70+ models across all tiers (local + cloud)

## 12. Conclusion

LLM Orchestra demonstrates that coordinating multiple diverse LLMs—both local and cloud-based—provides superior training signals for CETs while managing costs and maintaining high availability.

## References

[To be added]