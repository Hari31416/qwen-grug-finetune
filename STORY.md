# The Grug Reasoning Experiment: Can Small LLMs Think Like Cavemen?

This document shares the complete narrative of our experiment: why it was started, the technical pipeline we built, the roadblocks we encountered, what worked, what failed, and the results we achieved.

## The Motivation: Replicating Frontier Token Efficiency

The spark for this project came from an intriguing theory discussed online regarding the efficiency of frontier reasoning models (such as OpenAI's GPT-5.x variants).

### The "Grug Hypothesis"
While proprietary models hide their inner monologue behind a generated summary or block it entirely, many developers hypothesized that these models do not reason in full, grammatically correct sentences internally. Instead, to optimize token generation and minimize latency, they might think in a highly compressed, telegramic "Grug" or "caveman" style—dropping articles, conjugations, politeness markers, and syntactic filler. 

By shrinking the length of the internal monologue, the model saves processing time and bandwidth while preserving the logical structure of its thoughts. 

### Why Small Models and Local Compute?
At the same time, we wanted to build an educational project around fine-tuning large language models. We wanted to learn:

- **Custom SFT Curation:** How to create, clean, validate, and format an end-to-end dataset pipeline.
- **Apple Silicon Training:** How to use the MLX framework to train and evaluate models locally on a consumer Apple Silicon machine (a Mac M4 GPU).
- **Style Internalization:** Whether a small LLM could learn to internalize a reasoning style without requiring explicit system prompts.

We selected **Qwen-3.5-0.8B-Instruct** (and later pivoted to **DeepSeek-R1-Distill-Qwen-1.5B-4bit**) as our target models, using a local Mac M4 GPU as our compute node.

## The Technical Pipeline: How We Built It

To execute this, we built a modular pipeline from scratch. The pipeline comprises six distinct stages:

- **Stage 1: Prompt Sampling**
  We stratified general-purpose prompts across six source datasets (StrategyQA, LogiQA, BoolQ, ANLI, PIQA, and ReClor) — sampling 1,000 prompts for Iteration 1 and scaling to 4,000 prompts for Iteration 2 to generate enough correct and validated traces.
- **Stage 2: Verbose Trace Generation**
  We ran the base model locally on the M4 GPU to generate raw, verbose reasoning traces. Only prompts where the model answered correctly against the dataset's ground truth were kept, filtering out hallucinated reasoning chains.
- **Stage 3: Trace Compression**
  Correct traces were sent concurrently to an Nvidia NIM API (`glm 5.2`) acting as the "compressor". Using a detailed system prompt loaded from `style_guide.md`, the compressor rewrote the verbose thinking blocks into a grammar-stripped, telegraphic caveman style.
- **Stage 4: Automated Style Validation**
  We built `validate_traces.py` to enforce strict quality filters. It rejected compressed traces that exceeded 50% of the raw trace length, dropped critical numeric facts, omitted multiple-choice option letters, or included meta-commentary (like *"Wait, let me think..."*).
- **Stage 5: Chat Template Formatting**
  Accepted traces were formatted into standard chat templates. Since standard templates (like Jinja) often strip `<think>` tags from assistant messages to save context, we bypassed this by template-formatting only the user prompt and manually appending the `<think>compressed_thinking</think>\n\nfinal_answer` sequence.
- **Stage 6: SFT LoRA Training**
  We wrapped `mlx_lm.lora` in a training script `train.py` to handle AdamW optimization, logging validation loss curves to `metrics.json`, and rendering loss progression plots in real-time.

## Detailed Run Iterations

To test the Grug Reasoning style, we executed three distinct training runs, tweaking the dataset size, training duration, and SFT formatting layout.

### Iteration 1: The Proof of Concept

- **Base Model:** `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit`
- **Dataset Size:** 333 training samples
- **Training Duration:** 300 steps (LoRA SFT)
- **Objective:** Establish whether a small, distilled model could internalize the telegraphic "Grug" reasoning format under normal prompt conditions.
- **Outcome:** The model successfully adopted the "Grug" voice. Its thinking blocks dropped auxiliary words, articles, and verbose commentary.
- **Key Flaw:** Severe prompt leakage. If evaluated without the custom system prompt, the model would parrot the SFT instructions (*"Think like Grug, keep it brief, no articles..."*) inside the thinking block instead of using it to reason.

### Iteration 2 (Unregularized): Scaling Up

- **Base Model:** `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit`
- **Dataset Size:** Scaled to 1,530 training samples (153 validation set)
- **Training Duration:** 2,000 steps (LoRA SFT)
- **Objective:** Scale training duration and samples to cement the formatting constraints and stabilize multi-step derivation.
- **Outcome:** The model achieved extremely clean reasoning formats but suffered from severe overfitting due to the system prompt formatting.
- **Key Flaw:** Prompt regurgitation worsened. The model would repeat the prompt guidelines back verbatim during evaluations, leading to format compliance failures on standard test prompts.

### Iteration 2 (Regularized): The Final Calibration

- **Base Model:** `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit`
- **Dataset Size:** Same 1,530 training samples, but modified using **SFT Regularization** mixtures:
  - 20% system prompt dropout (no system prompt on positive samples).
  - 30% negative (verbose) reasoning example mixture (to teach the model to reason normally when not prompted).
  - 50% negative system prompts (system prompt retained on negative examples to align negative instances).
- **Training Duration:** 1,000 steps (LoRA SFT)
- **Objective:** Resolve prompt regurgitation/leakage and establish stable style boundaries.
- **Outcome:** Complete success. Prompt regurgitation was fully eliminated, and the model learned to apply the style contextually.
- **Key Metrics:** Average thinking block tokens dropped by **73.9%** (from 517.4 to 135.0 tokens), leading to a **52.3% end-to-end speedup** (from 1.28s to 0.61s) with **98.2% format compliance**.

## The Roadblocks: What Did Not Work

Fine-tuning reasoning models is notoriously sensitive. We ran into three major failures:

### 1. Small Standard Instruct Models Fail at Reasoning Loops
Our first choice, Qwen-3.5-0.8B-Instruct, failed completely during the raw trace generation stage. Because standard instruct checkpoints are not aligned using reinforcement learning (RL) specifically for multi-step reasoning, forcing them into a thinking block via prompting caused them to enter infinite self-correcting loops. 

The 0.8B model would get stuck repeating variations of *"Wait, is X correct? No, because Y. Wait..."* until it hit the 1,536 token generator cap, failing to output a final answer. 

**The Fix:** We pivoted to **DeepSeek-R1-Distill-Qwen-1.5B-4bit**. This model was trained using reinforcement learning to think. It natively structures its thoughts inside `<think>...</think>` tags and reliably emits the closing token.

### 2. Overfitting and Prompt Leakage
In Iteration 1 and early Iteration 2 runs, the model suffered from severe prompt leakage. Because it trained only on positive (compressed) traces formatted with the style system prompt, it overfit to the instructions themselves. 

During evaluation, even when prompted normally, the model would regurgitate the system instructions inside its thinking block: *"You must think in short, telegraphic fragments. Final answer must be..."* instead of actually solving the problem.

**The Fix:** We implemented SFT Regularization:
- **System Prompt Dropout:** We omitted the system prompt in 20% of the training examples.
- **Negative Example Mixing:** We mixed in 30% uncompressed, verbose traces (`raw_thinking`) to show the model how to reason normally.
- **Negative System Prompting:** We kept the system prompt in 50% of the negative instances to train the model not to over-compress reasoning unconditionally.

### 3. The Math "Alignment Tax"
While the regularized model successfully learned the telegraphic style, its accuracy on the Grade School Math (GSM8K) test split dropped significantly from 70.1% (base) to 54.6% (fine-tuned). 

Because our SFT dataset consisted only of general reasoning prompts and lacked math-specific tasks, the model learned to compress reasoning by dropping mathematical derivations, intermediate equations, and calculation checks. It over-compressed its logic, leading to calculation errors.

## What Worked: The Breakthroughs

Despite the setbacks, several elements worked exceptionally well:

- **MLX Performance:** Training locally on the Mac M4 GPU using MLX was incredibly efficient. The 1,000-step training runs took less than 30 minutes, and the memory footprint was under 6 GB.
- **Format Stickiness:** The SFT regularization successfully solved prompt leakage. The model achieved a **98.2% format compliance rate**, outputting clean, parseable `<think>` blocks.
- **Substantial Token Savings:** Emitted thinking tokens dropped by **73.9%** (from 517.4 down to 135.0 tokens on average).
- **Latency Reduction:** The average end-to-end latency was cut in half, dropping by **52.3%** (from 1.28 seconds down to 0.61 seconds).

## Experimental Results

Here is the comparison of our baseline vs. fine-tuned models on the GSM8K test split:

| Metric                   | Base Model (Style Prompt) | Fine-Tuned Model (Regularized) |       Performance Change        |
| :----------------------- | :-----------------------: | :----------------------------: | :-----------------------------: |
| **Accuracy**             |           70.1%           |             54.6%              |    -15.5 pp (Alignment Tax)     |
| **Mean Thinking Tokens** |           517.4           |             135.0              |    **-73.9%** (Tokens Saved)    |
| **Mean Total Tokens**    |           582.1           |             214.7              |  **-63.1%** (Bandwidth Saved)   |
| **Mean Latency (s)**     |           1.28s           |             0.61s              |      **-52.3%** (Speedup)       |
| **Format Compliance**    |           91.1%           |             98.2%              | **+7.1 pp** (Format Stickiness) |

## Next Steps: The Future

The results show that while reasoning style transfer is highly effective at saving tokens and latency, the 1.5B model's capacity limit results in a heavy alignment tax. 

To build on these findings, we plan to focus on:

### 1. Scaling to Larger Base Models
We want to transition from the 1.5B model to larger distilled reasoning models, such as **DeepSeek-R1-Distill-Qwen-7B** or Llama-8B-based distill variants. Larger models have significantly higher representation capacity. This increased capacity should allow the model to absorb the telegraphic style constraints and internalize the Grug reasoning format without sacrificing task accuracy.

### 2. Task-Specific SFT Mixing
To eliminate the math alignment tax, we will generate and inject math-specific (GSM8K) SFT training traces into the dataset. This will teach the model how to express mathematical calculations and derivations telegraphically without skipping key intermediate equations.

### 3. Scaling SFT Data
We will scale the SFT dataset from 1,701 to 5,000+ samples.

### 4. Calibrating Adapter Capacity
We will reduce the LoRA rank from 16 to 8 or 4, and restrict target layers to `q_proj` and `v_proj` to act as an implicit regularizer, preventing the adapter from overriding the base weights too aggressively.
