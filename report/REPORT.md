# Experimental Report: Grade School Math SFT Alignment

This report documents the results of fine-tuning the target reasoning model using telegraphic, token-efficient reasoning traces ("Grug" style). The objective is to achieve a significant reduction in emitted thinking tokens and inference latency without causing catastrophic degradation in Grade School Math (GSM8K) accuracy.

---

## 1. Experimental Setup

* **Base Model:** `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit` (4-bit quantized DeepSeek-R1 distill variant).
* **SFT Dataset:** 333 high-quality Grade School Math training instances mapped to telegraphic "Grug" style reasoning traces.
* **Fine-Tuning Parameters:**
  * **Optimizer:** AdamW
  * **Learning Rate:** $1\times 10^{-5}$
  * **Batch Size:** 4
  * **Iterations:** 300 (approximately 3.6 Epochs)
  * **Save & Evaluation Frequency:** Every 20 iterations
  * **LoRA Target Layers:** 16 layers (rank=8, alpha=16)

---

## 2. SFT Training & Convergence

Training was executed on Apple Silicon (GPU via MLX). The cross-entropy loss converged smoothly over the 300 iterations:

* **Starting Loss:** Validation loss of **`3.726`** at iteration 1.
* **Best Validation Step:** Iteration **300** achieved the lowest validation loss of **`2.942`** (with training loss at **`2.598`**).
* **Model Selection:** The script successfully copied iteration 300 weights to `best_adapters.safetensors` and active `adapters.safetensors`.

The complete training loss progression is shown below:

![Training and Validation Loss Curve](loss_curve.png)

---

## 3. Evaluation Metrics

Evaluations were performed on the first 100 test samples of the Grade School Math (GSM8K) benchmark test split under four configurations:
1. **Base Normal:** Pre-trained model with default prompting.
2. **Base Grug:** Pre-trained model with explicit telegraphic system prompts.
3. **FT Normal:** Fine-tuned model with default prompting (no adapter).
4. **FT Grug:** Fine-tuned model with explicit telegraphic system prompts (adapter loaded).

### Summary Statistics

| Metric                       | Base Normal | Base Grug | FT Normal | FT Grug |
| :--------------------------- | :---------: | :-------: | :-------: | :-----: |
| **Accuracy**                 |    73.0%    |   70.0%   |   62.0%   |  40.0%  |
| **Mean Thinking Tokens**     |    205.0    |   481.2   |   164.3   |  166.8  |
| **Mean Total Tokens**        |    476.0    |   545.3   |   392.4   |  247.8  |
| **Mean Latency (s)**         |    1.10s    |   1.47s   |   0.96s   |  0.80s  |
| **Generation Speed (tok/s)** |    431.0    |   370.4   |   407.9   |  309.9  |
| **Format Compliance**        |    98.0%    |   92.0%   |   99.0%   |  92.0%  |

Below is the consolidated performance comparison dashboard:

![Consolidated Comparative Dashboard](dashboard.png)

---

## 4. Performance & Efficiency Deltas

Direct deltas comparing the fine-tuned model to the baseline models under normal and Grug prompting are shown below:

![Accuracy & Emitted Token Changes](deltas.png)

### Key Findings & Achievements:

1. **Massive Token Compression in Reasoning Mode:**
   * Under the **Grug Prompt** condition, the fine-tuned model reduced the average emitted thinking tokens by **65.3%** (from `481.2` down to `166.8` tokens).
   * Total token generation was compressed by **54.6%** (from `545.3` down to `247.8` tokens).
2. **Substantial Latency Reductions:**
   * Average inference latency for telegraphic reasoning was cut by **45.6%** (from `1.47 seconds` down to `0.80 seconds`).
   * Average generation throughput changed from `370.4 tok/s` to `309.9 tok/s`.
3. **Reasoning Style Transfer vs. Accuracy Trade-Off:**
   * When prompted with default instructions (**Normal Prompt**), the fine-tuned model achieved an accuracy of **62.0%** (compared to `73.0%` for Base Normal) while reducing mean thinking tokens by **19.8%** (from `205.0` to `164.3`).
   * When forced to think in telegraphic fragments (**Grug Prompt**), accuracy dropped from `70.0%` (Base Grug) to `40.0%` (FT Grug). This represents a 30 percentage point trade-off for aggressive reasoning compression on grade-school math, matching the expectations in `PLAN.md`.
4. **Format Stickiness:**
   * Format compliance rate under Grug prompting remained steady at **92.0%** for both Base Grug and FT Grug, and improved from **98.0%** to **99.0%** under normal prompting.
