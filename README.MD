
# Exploring Communicative Mixture of Experts (MoE) Models


## Background
The concept of a Mixture of Experts (MoE) model can be likened to a system where multiple specialized experts process different aspects of a given input, and their outputs are combined to generate a final prediction. This approach is reminiscent of how customer service functions in an industry setting, where an issue is forwarded to different agents, each analyzing and addressing specific aspects before arriving at a resolution.

A fine-grained MoE model differs from a simpler MoE model in that each token is processed by a subset of experts rather than a single expert. In large-scale MoE architectures, multiple experts exist, but only a few are activated per token, ensuring that processing is distributed efficiently. This idea aligns with a scenario where a problem is assigned to a specialized group of experts, who collaborate to resolve it at a granular level.

One analogy that comes to mind is how a token is processed in a feed-forward network with multiple layers, separated by activation functions. If a token passes through multiple experts—say four—it will generate four distinct activations. Instead of selecting just one, the model could average these activations, thereby leveraging multiple perspectives. This is comparable to a customer service system where various experts collaborate before delivering a final response to the customer.

We would like to extend this concept into a research project, where We conduct an experiment to test the efficacy of a communicative MoE model. My plan is to create a minimal prototype, starting with a basic implementation. We intend to generate around 2 test cases, with one set based on a traditional MoE model and another employing a communicative MoE approach. By running these models on a dataset, We aim to observe how communication among experts influences outcomes.

To control whether experts communicate, We will implement a switch within the model's configuration. This will allow me to compare performance between communicative and non-communicative setups. My hypothesis is that a communicative approach may lead to lower loss values, demonstrating improved efficiency when experts collaborate.

The final deliverable for this project will be a simple, well-documented code repository implementation, including instructions on running the experiment. The results will be presented in a structured format, allowing for further analysis and discussion.


## Configuration
```bash
bash scripts/setup.sh
bash train.sh
```

The r1-hf configuration is directly borrowed from the DeepSeek-R1 repository.

For r1-lite-hf and r1-tiny-hf, we compared the difference between deepseek-v2 and deepseek-v2-lite, summarized their trends, and applied a similar configuration from r1-hf to r1-lite-hf and to r1-tiny-hf.

### Scaling Analysis: DeepSeek-V2 vs V2-Lite

#### Core Model Dimensions
- **Hidden size**: 60% reduction (5120 → 2048)
- **Layer count**: 55% reduction (60 → 27)
- **Attention heads**: 87.5% reduction (128 → 16)

#### MoE Architecture Scaling
- **Expert count**: 60% reduction (160 → 64)
- **Group structure**: Simplified from 8 groups to 1
- **Routing complexity**: Reduced from `group_limited_greedy` to `greedy`
- **Routing scaling factor**: Decreased from 16.0 to 1.0

#### Feed-Forward Network
- **FFN ratio**: Increased in smaller model (2.4× → 5.3×)
- **MoE intermediate size**: Proportionally reduced

### Applied DownScaling: R1 to R1-Lite (Around 30B Parameters)

Using the observed scaling patterns, we transformed the r1-hf configuration to r1-lite-hf by:

1. **Balanced dimension reduction**:
   - Hidden size: 7168 → 2816 (~60% reduction)
   - Layers: 61 → 28 (~55% reduction)
   - Attention heads: 128 → 16 (following V2-Lite pattern)

2. **MoE architecture simplification**:
   - Experts: 256 → 96 (proportional reduction)
   - Groups: 8 → 1 (matching V2-Lite simplification)
   - Routing method: `noaux_tc` → `greedy` (following scaling trend)
   - Topk_group: 4 → 1 (simplification)

3. **FFN scaling**:
   - Maintained higher FFN-to-hidden ratio in the smaller model
   - Intermediate size: 18432 → 15232
   - MoE intermediate: 2048 → 1792

4. **Preserved architectural elements**:
   - Maintained experts-per-token count (8)
   - Preserved attention head dimensions
   - Retained R1-specific parameters like quantization config



### Further DownScaling to R1-Tiny (800M Parameters)

For r1-tiny-hf (800M scale), we've applied more aggressive scaling than the v2-lite pattern to create a truly lightweight model suitable for resource-constrained environments. To scale down the R1 model from ~32B to ~800M parameters, we need to apply substantially more aggressive scaling than the V2 to V2-Lite transition.

## Scaling Rationale

1. **Core Dimensions**:
   - **Hidden size**: 7168 → 1024 (~86% reduction)
   - **Layer count**: 61 → 16 (~74% reduction)
   - **Attention heads**: 128 → 16 (~88% reduction)

2. **MoE Architecture**:
   - **Expert count**: 256 → 16 (~94% reduction)
   - **Experts per token**: 8 → 4 (reduced for efficiency)
   - **Architecture simplification**: Maintained the simpler routing approach

3. **Head Dimensions**:
   - Reduced head dimensions from 128 to 64 for efficiency
   - Maintained equal RoPE and NoRoPE dimensions

4. **Other Optimizations**:
   - **Preserved critical features**: Kept important architectural elements like RoPE scaling


We follow the empirical scaling patterns observed in DeepSeek's V2 series while respecting the unique architectural elements of the R1 model.


https://github.com/Zjh-819/LLMDataHub?tab=readme-ov-file#pretrain
https://huggingface.co/datasets/hoskinson-center/proof-pile
https://huggingface.co/datasets/wangrui6/Zhihu-KOL
https://huggingface.co/datasets/Linly-AI/Chinese-pretraining-dataset