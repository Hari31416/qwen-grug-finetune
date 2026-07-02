# Experimental Report: Grade School Math SFT Alignment

This report documents the results of fine-tuning the target reasoning model using telegraphic, token-efficient reasoning traces ("Grug" style). The objective is to achieve a significant reduction in emitted thinking tokens and inference latency without causing catastrophic degradation in Grade School Math (GSM8K) accuracy.

## 1. Experimental Setup

- **Base Model:** `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit` (4-bit quantized DeepSeek-R1 distill variant).
- **SFT Dataset:** 333 high-quality Grade School Math training instances mapped to telegraphic "Grug" style reasoning traces.
- **Fine-Tuning Parameters:**
  - **Optimizer:** AdamW
  - **Learning Rate:** $1\times 10^{-5}$
  - **Batch Size:** 4
  - **Iterations:** 300 (approximately 3.6 Epochs)
  - **Save & Evaluation Frequency:** Every 20 iterations
  - **LoRA Target Layers:** 16 layers (rank=8, alpha=16)

## 2. SFT Training & Convergence

Training was executed on Apple Silicon (GPU via MLX). The cross-entropy loss converged smoothly over the 300 iterations:

- **Starting Loss:** Validation loss of **`3.726`** at iteration 1.
- **Best Validation Step:** Iteration **300** achieved the lowest validation loss of **`2.942`** (with training loss at **`2.598`**).
- **Model Selection:** The script successfully copied iteration 300 weights to `best_adapters.safetensors` and active `adapters.safetensors`.

The complete training loss progression is shown below:

![Training and Validation Loss Curve](loss_curve.png)

## 3. Evaluation Metrics

Evaluations were performed on all 1000 test samples of the Grade School Math (GSM8K) benchmark test split under four configurations:

1. **Base Normal:** Pre-trained model with default prompting.
2. **Base Grug:** Pre-trained model with explicit telegraphic system prompts.
3. **FT Normal:** Fine-tuned model with default prompting (adapter loaded).
4. **FT Grug:** Fine-tuned model with explicit telegraphic system prompts (adapter loaded).

### Summary Statistics

| Metric                       | Base Normal | Base Grug | FT Normal | FT Grug |
| :--------------------------- | :---------: | :-------: | :-------: | :-----: |
| **Accuracy**                 |    64.9%    |   67.2%   |   66.0%   |  45.6%  |
| **Mean Thinking Tokens**     |    219.0    |   512.8   |   156.2   |  120.0  |
| **Mean Total Tokens**        |    477.4    |   581.1   |   389.3   |  229.0  |
| **Mean Latency (s)**         |    0.88s    |   1.21s   |   0.73s   |  0.64s  |
| **Generation Speed (tok/s)** |    541.6    |   479.6   |   535.1   |  360.5  |
| **Format Compliance**        |    96.6%    |   91.5%   |   98.9%   |  95.1%  |

Below is the consolidated performance comparison dashboard:

![Consolidated Comparative Dashboard](dashboard.png)

## 4. Performance & Efficiency Deltas

Direct deltas comparing the fine-tuned model to the baseline models under normal and Grug prompting are shown below:

![Accuracy & Emitted Token Changes](deltas.png)

### Key Findings & Achievements

1. **Massive Token Compression in Reasoning Mode**
   - Under the **Grug Prompt** condition, the fine-tuned model reduced the average emitted thinking tokens by **76.6%** (from `512.8` down to `120.0` tokens).
   - Total token generation was compressed by **60.6%** (from `581.1` down to `229.0` tokens).

2. **Substantial Latency Reductions**
   - Average inference latency for telegraphic reasoning was cut by **47.6%** (from `1.21 seconds` down to `0.64 seconds`).
   - Average generation throughput changed from `479.6 tok/s` to `360.5 tok/s`.

3. **Reasoning Style Transfer vs. Accuracy Trade-Off**
   - When prompted with default instructions (**Normal Prompt**), the fine-tuned model achieved an accuracy of **66.0%** (compared to `64.9%` for Base Normal, representing a **1.1 percentage point increase**) while reducing mean thinking tokens by **28.7%** (from `219.0` to `156.2`).
   - When forced to think in telegraphic fragments (**Grug Prompt**), accuracy dropped from `67.2%` (Base Grug) to `45.6%` (FT Grug). This represents a **21.6 percentage point** trade-off for aggressive reasoning compression on grade-school math.

4. **Format Stickiness**
   - Format compliance rate under Grug prompting improved from **91.5%** to **95.1%** for FT Grug, demonstrating that fine-tuning helped ground the model in the expected output formats despite the condensed reasoning. Under normal prompting, compliance improved from **96.6%** to **98.9%**.
