# The Grug Reasoning Experiment: Can LLMs Think Like Cavemen

This document shares the complete narrative of our experiment: why it was started, the technical pipeline we built, the roadblocks we encountered, the scaling journey from 1.5B to 7B, what failed, and the empirical benchmark results we achieved.

## The Motivation: Replicating Frontier Token Efficiency

The spark for this project came from an intriguing theory regarding the efficiency of frontier reasoning models.

### The Grug Hypothesis

While proprietary models hide their inner monologue behind a generated summary or block it entirely, many developers hypothesized that frontier models do not reason in full, grammatically correct prose internally. Instead, to optimize token generation and minimize latency, they might think in a highly compressed, telegraphic "Grug" or "caveman" style—dropping articles, conjugations, politeness markers, and syntactic filler.

By shrinking the length of the internal monologue, the model saves processing time, bandwidth, and compute cost while preserving the logical structure of its thoughts.

### Why Small Models and Local Compute First

We began this project as an educational exploration into fine-tuning reasoning models with three core learning goals:

- Custom SFT Curation: How to create, clean, validate, and format an end-to-end dataset pipeline.
- Apple Silicon Training: How to use the MLX framework to train and evaluate models locally on a consumer Apple Silicon machine (a Mac M4 GPU).
- Style Internalization: Whether a small LLM could learn to internalize a terse reasoning style without relying on runtime system prompts.

We started with `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B-4bit` on local Apple Silicon before scaling up to `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` on cloud CUDA infrastructure.

## Phase 1: Local Apple Silicon Experiments (1.5B)

To execute the initial experiments, we built a modular pipeline from scratch comprising six distinct stages:

- Stage 1: Prompt Sampling: Stratified general-purpose prompts across six source datasets (StrategyQA, LogiQA, BoolQ, ANLI, PIQA, and ReClor).
- Stage 2: Verbose Trace Generation: Ran the base model locally on the M4 GPU to generate raw reasoning traces, filtering out incorrect conclusions.
- Stage 3: Trace Compression: Synthetically compressed verbose thinking blocks into grammar-stripped, telegraphic bullet points via an LLM compressor.
- Stage 4: Automated Style Validation: Filtered out compressed traces that exceeded 50% length, dropped numeric facts, or retained conversational commentary.
- Stage 5: Chat Template Formatting: Appended the `<think>compressed_thinking</think>\n\nfinal_answer` structure onto formatted prompts.
- Stage 6: SFT LoRA Training: Trained with MLX LoRA (`train.py`) using AdamW, logging validation metrics in real time.

### Iteration 1: The Proof of Concept

- Base Model: `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit`
- Dataset Size: 333 training samples
- Outcome: The model successfully adopted the telegraphic voice, but suffered from severe prompt leakage. When evaluated without the explicit system prompt, the model parroted the training instructions back (*"Think like Grug, keep it brief..."*) inside its thinking block.

### Iteration 2: Scaling Up and Regularizing

- Base Model: `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit`
- Dataset Size: Scaled to 1,530 training samples
- Key Technique: SFT Regularization (20% prompt dropout, 30% uncompressed negative example mixing).
- Outcome: Prompt leakage was completely eliminated, and format compliance reached 98.2%. Thinking tokens dropped by 73.9%, and generation latency was cut in half (from 1.28s to 0.61s).
- The Alignment Tax: Accuracy on math reasoning (GSM8K) dropped from 70.1% to 54.6%. The small 1.5B model lacked the capacity to compress reasoning without dropping critical intermediate calculation steps.

## Phase 2: Cloud CUDA Scaling to 7B and Preference Optimization

To overcome the capacity limits of the 1.5B model and resolve the math alignment tax, we migrated to **DeepSeek-R1-Distill-Qwen-7B** on Kaggle using 2x NVIDIA T4 GPUs.

### Supervised Fine-Tuning at 7B

We applied 4-bit QLoRA fine-tuning using Hugging Face `transformers`, `peft`, and `trl`:

- Base Model: `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` (4-bit NF4 quantized)
- Target Modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` (LoRA rank = 16, alpha = 32)
- Training: Scaled over 1,530 curated demonstrations using standard chat templates.

### The Critical Discovery: SFT Looping Degeneration

When evaluating the 7B SFT model across the full 1,319 GSM8K test split, we discovered a notable edge-case failure mode:

- For well-behaved samples (over 94% of the split), SFT generated exceptionally terse answers: mean answer length dropped from 160.4 tokens (Base) down to 84.7 tokens (a 47% reduction).
- However, on 71 difficult math questions, SFT entered a cyclical repetition loop inside the `<think>` block (e.g. repeatedly recalculating intermediate fractions or regurgitating problem facts).
- Because SFT was stuck looping inside the thinking block, it consumed all 512 generation tokens before ever emitting `</think>`. As a result, format compliance dipped to 94.62%.

Supervised fine-tuning alone learns only from positive demonstrations; when an SFT model becomes confused or out-of-distribution, it has no negative feedback mechanism to break out of cyclical reasoning.

### The DPO Solution: Direct Preference Optimization

To eradicate the looping behavior, we trained a second-stage Direct Preference Optimization (DPO) adapter:

- Preference Dataset Construction: We generated pairs of chosen vs rejected trajectories.
  - Chosen: Concise, well-structured, telegraphic reasoning chains that terminate cleanly with `</think>` and deliver the final answer.
  - Rejected: Overly verbose, rambling, or looping traces that repeat derivations.
- Training Dynamics: DPO trained with `beta = 0.1` and learning rate `5e-6` over 1,530 preference pairs.
- Result: DPO completely cured the looping failure mode. Non-compliant generations dropped from 71 down to just 2 across all 1,319 test questions, restoring format compliance to 99.85% and maintaining 75.44% task accuracy.

## Full GSM8K Benchmark Results (1,319 Samples)

Below are the empirical metrics from the full GSM8K test split evaluated sequentially under unified conditions on NVIDIA T4 hardware:

| Model Variant    | Test Samples | Accuracy | Format Compliance | Mean Thinking Tokens | Mean Answer Tokens | Mean Total Tokens | Mean Latency |
| :--------------- | :----------: | :------: | :---------------: | :------------------: | :----------------: | :---------------: | :----------: |
| Base Model (7B)  |    1,319     |  75.97%  |      99.85%       |        122.5         |       160.4        |       427.4       |    6.09s     |
| SFT Adapter (7B) |    1,319     |  72.18%  |      94.62%       |        107.7         |       107.3        |       434.5       |    6.75s     |
| DPO Adapter (7B) |    1,319     |  75.44%  |      99.85%       |        122.3         |       162.1        |       428.8       |    6.39s     |

## Key Insights and Breakthroughs

### 1. The Power of Preference Optimization Over Pure SFT

SFT is sufficient to teach an LLM a stylistic syntax (such as bullet points or telegraphic shorthand), but it cannot teach the model when to stop when it is uncertain. DPO provides the negative gradient needed to suppress wandering monologues and looping degenerations.

### 2. Resolution of the Math Alignment Tax at 7B

In the 1.5B model, compressing thoughts dropped math accuracy by 15.5 percentage points. At 7B, the model possessed sufficient parameter capacity to maintain mathematical rigor: the DPO adapter achieved 75.44% accuracy on GSM8K, within 0.5% of the uncompressed base model (75.97%).

### 3. Unified Two-Repository Architecture

All deliverables from both phases of the project have been structured into a clean two-repository ecosystem on Hugging Face:

- Model LoRA Adapters: [hari31416/deepseek-r1-grug-adapters](https://huggingface.co/hari31416/deepseek-r1-grug-adapters)
  Houses both 7B PEFT adapters (`sft/`, `dpo/`) and 1.5B Apple Silicon MLX adapters (`it-1/`, `it-2-regularized/`, `it-2-unregularized/`).
- Datasets and Full Benchmarks: [hari31416/grug-reasoning-data-and-benchmarks](https://huggingface.co/datasets/hari31416/grug-reasoning-data-and-benchmarks)
  Houses all SFT splits, DPO preference pairs, full 1,319-sample generation JSON logs, and comparative dashboard figures.
