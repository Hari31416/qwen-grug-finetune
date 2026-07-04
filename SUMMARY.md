# Experimental Findings and Next Steps

This document summarizes the findings from the Grug-style reasoning fine-tuning experiments on **DeepSeek-R1-Distill-Qwen-1.5B** and outlines the proposed next steps for the project.

## Key Findings

Across two iterations of design and training, the project achieved the following experimental outcomes:

- **Successful Reasoning Style Transfer**
  - The model successfully learned to internalize a telegraphic, token-efficient reasoning style ("Grug/caveman" style) within its `<think>...</think>` block.
  - In Iteration 2 (Regularized), average thinking tokens dropped by **73.9%** (from 517.4 down to 135.0 tokens), leading to a **52.3% reduction** in end-to-end inference latency (from 1.28s to 0.61s).
- **Prompt Leakage and Regurgitation Mitigated**
  - Iteration 1 and early Iteration 2 (Unregularized) suffered from instruction regurgitation, where the model repeated rules like "Final answer inside the thinking block..." in its outputs.
  - Implementing **SFT Regularization** (20% prompt dropout, 30% negative examples using verbose raw thinking traces, and 50% negative system prompts) in the final Iteration 2 run completely resolved this leakage.
- **Significant Increase in Format Compliance**
  - Fine-tuning dramatically improved format compliance stickiness. Under regularized training, format compliance rose to **98.2%** (compared to **91.1%** for the base model under the style prompt), proving that the model can be strictly aligned to the target delimiters (`<think>...</think>` tags) even under condensed reasoning outputs.
- **The "Alignment Tax" on Reasoning Tasks**
  - Condensing the reasoning block led to an accuracy drop on the Grade School Math (GSM8K) test split, falling from 70.1% (baseline) to 54.6% (fine-tuned).
  - Because the SFT dataset only contained general reasoning datasets (StrategyQA, BoolQ, ReClor, etc.) and lacked math-specific tasks, the model over-compressed mathematical derivations and skipped critical steps.

## Next Steps

To build upon the successful style transfer while minimizing the alignment tax on accuracy, the next phases will focus on the following directions:

### 1. Scaling to Larger Base Models

- **Distill-Qwen-7B or 8B Models**
  - Fine-tune larger distilled reasoning models (e.g. `DeepSeek-R1-Distill-Qwen-7B` or Llama-8B-based distill variants).
  - Larger models possess significantly higher representation capacity. This increased capacity is expected to allow the model to absorb the telegraphic style constraints without degrading its foundational reasoning and calculation accuracy.
- **Improved Alignment Trade-off**
  - A larger model can maintain complex logical chains even when forced into a terse format, minimizing the alignment tax observed in the 1.5B model.

### 2. Scaling SFT Data and Target Tasks

- **Incorporate Task-Specific Training Traces**
  - Generate and compress math-specific (GSM8K) training split traces in the target style and inject them into the SFT dataset.
  - This will directly teach the model how to express mathematical derivations in the telegraphic style without skipping key calculations.
- **Larger General-Purpose Dataset**
  - Scale up the stratified general SFT corpus from 1,700 samples to 5,000+ samples to solidify style internalization across different reasoning domains.

### 3. Calibrating Adapter Capacity

- **Reduce LoRA Rank**
  - Lower the rank from 16 to 8 or 4, and target only specific attention projections (`q_proj` and `v_proj`).
  - This restricts the adapter's capacity, preventing it from overriding pretrained weights too aggressively and acting as an implicit regularizer.
- **Checkpoint Sweep Evaluation**
  - Run evaluations on intermediate training checkpoints (e.g. every 50 steps) to capture the optimal checkpoint where style transfer has converged but task accuracy has not yet degraded.
