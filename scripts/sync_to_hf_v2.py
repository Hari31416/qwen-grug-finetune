#!/usr/bin/env python3
import os
import sys
import shutil
import logging
from typing import Dict, Any, List, Optional
from huggingface_hub import HfApi

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("sync_hf_v2")

WORKSPACE_ROOT: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASET_REPO_ID: str = "hari31416/grug-reasoning-data-and-benchmarks"
MODEL_REPO_ID: str = "hari31416/deepseek-r1-7b-grug-adapters"


def copy_path(src: str, dst: str) -> None:
    """Copy a file or directory tree, creating destination parents if needed."""
    if not os.path.exists(src):
        logger.warning("Source path does not exist, skipping: %s", src)
        return

    dst_dir: str = os.path.dirname(dst)
    if dst_dir and not os.path.exists(dst_dir):
        os.makedirs(dst_dir, exist_ok=True)

    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        logger.info("Copied directory from %s to %s", src, dst)
    else:
        shutil.copy2(src, dst)
        logger.info("Copied file from %s to %s", src, dst)


def generate_model_card() -> str:
    """Generate the Model Card markdown for the 7B adapters repository."""
    card: str = """---
license: mit
base_model: deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
tags:
- peft
- lora
- trl
- dpo
- sft
- reasoning
- gsm8k
- grug-reasoning
pipeline_tag: text-generation
library_name: peft
---

# DeepSeek-R1-Distill-Qwen-7B Grug Adapters

This repository contains fine-tuned LoRA adapter weights for **DeepSeek-R1-Distill-Qwen-7B** trained to produce compressed, telegraphic reasoning traces inside `<think>...</think>` tags while preserving high mathematical and problem-solving accuracy.

Two adapter variants are provided:
- `sft/`: Supervised Fine-Tuning adapter trained on verified terse chain-of-thought demonstrations.
- `dpo/`: Direct Preference Optimization adapter trained on preference pairs to penalize verbosity and conversational filler while rewarding direct reasoning.

# Benchmark Comparison

Evaluated on the full GSM8K test split (1,319 samples):

| Model Variant | Samples | Accuracy | Format Compliance | Mean Thinking Tokens | Mean Answer Tokens | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Model (7B)** | 1,319 (full test) | 75.97% | 99.85% | 122.5 | 160.4 | 6.09s |
| **SFT Adapter** | 1,319 (full test) | 72.18% | 94.62% | 107.7 | 107.3 | 6.75s |
| **DPO Adapter** | 1,319 (full test) | 75.44% | 99.85% | 122.3 | 162.1 | 6.39s |

### Key Observations
- **Answer Brevity**: The SFT adapter reduced answer length from 160.4 tokens to 107.3 tokens (a 33.1% reduction in output verbosity).
- **Format Compliance**: Both Base and DPO achieved 99.85% format compliance, reliably outputting well-formed reasoning and response blocks.
- **Accuracy Retention**: DPO retained 75.44% accuracy (within 0.5% of uncompressed Base model), while eliminating filler reasoning.

# How to Use

Both adapters can be loaded using Hugging Face `transformers` and `peft`.

## Requirements

```bash
pip install torch transformers peft bitsandbytes accelerate
```

## Loading the DPO Adapter

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
adapter_repo = "hari31416/deepseek-r1-7b-grug-adapters"

# Load base model in 4-bit for memory efficiency
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

# Attach the DPO adapter
model = PeftModel.from_pretrained(base_model, adapter_repo, subfolder="dpo")

# Format prompt with style instructions
system_prompt = (
    "You are a helpful assistant. You must think in short, telegraphic, "
    "bullet-point style fragments inside a <think>...</think> block before answering."
)
messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": "Solve: A store sells notebooks for $3 each and pens for $1.50 each. If Jane buys 4 notebooks and 6 pens, what is her total cost?"}
]

prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.6,
        top_p=0.95,
        do_sample=True,
    )

print(tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True))
```

## Loading the SFT Adapter

To load the SFT variant instead, specify `subfolder="sft"`:

```python
model = PeftModel.from_pretrained(base_model, adapter_repo, subfolder="sft")
```

# Related Links

- Dataset and benchmark repository: [hari31416/grug-reasoning-data-and-benchmarks](https://huggingface.co/datasets/hari31416/grug-reasoning-data-and-benchmarks)
- GitHub repository: [Hari31416/qwen-grug-finetune](https://github.com/Hari31416/qwen-grug-finetune)
"""
    return card.strip() + "\n"


def generate_dataset_card() -> str:
    """Generate the Dataset Card markdown for the datasets and benchmark logs."""
    card: str = """---
license: mit
task_categories:
- text-generation
language:
- en
tags:
- reasoning
- preference-tuning
- dpo
- sft
- gsm8k
pretty_name: Grug Reasoning Datasets and Benchmarks
---

# Grug Reasoning Datasets and Benchmark Results

This repository contains training datasets, preference pairs, raw model responses, and empirical evaluation benchmark logs for the Grug reasoning fine-tuning project.

# Repository Contents

- `data/sft/`: Supervised fine-tuning dataset formatted for chat completion models.
  - `train.jsonl` (1,530 rows)
  - `valid.jsonl` (171 rows)
- `data/dpo/`: Direct preference optimization pairs containing chosen (terse, compressed reasoning) versus rejected (overly verbose, repetitive reasoning) trajectories.
  - `train.jsonl` (1,530 preference pairs)
  - `valid.jsonl` (171 preference pairs)
- `benchmarks/deepseek-r1-7b/`: Complete GSM8K evaluation records (1,319 test samples each) containing full model responses, thinking traces, answers, and metric tags.
  - `baseline/gsm8k.json`
  - `finetuned/gsm8k.json`
  - `dpo/gsm8k.json`
- `benchmarks/reports/`: Comparative dashboard figures, loss curves, and markdown reports.
  - `benchmark_comparison_dashboard.png`
  - `loss_plot.png`
  - `BENCHMARK_REPORT.md`

# Benchmark Results (GSM8K Test Split, 1,319 Samples)

| Model Variant | Samples | Accuracy | Format Compliance | Mean Thinking Tokens | Mean Answer Tokens | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Model (7B)** | 1,319 | 75.97% | 99.85% | 122.5 | 160.4 | 6.09s |
| **SFT Adapter (7B)** | 1,319 | 72.18% | 94.62% | 107.7 | 107.3 | 6.75s |
| **DPO Adapter (7B)** | 1,319 | 75.44% | 99.85% | 122.3 | 162.1 | 6.39s |

### Key Benchmark Insights

- **Answer Brevity**: The SFT model compressed final answer lengths from 160.4 tokens down to 107.3 tokens (a 33.1% reduction in verbosity).
- **Loop Prevention**: SFT without preference tuning occasionally cycled inside the thinking block on out-of-distribution math questions, consuming the token budget before closing `<think>`. DPO preference optimization eliminated this failure mode, recovering format compliance to 99.85%.
- **Accuracy Retention**: DPO retained 75.44% accuracy on GSM8K (within 0.5% of the uncompressed base model) while stripping conversational fluff.

# Visualizations

## Benchmark Comparison Dashboard

![Benchmark Comparison Dashboard](benchmarks/reports/benchmark_comparison_dashboard.png)

## DPO Training Loss Curve

![DPO Loss Curve](benchmarks/reports/loss_plot.png)

# Dataset Schemas and Usage

## 1. SFT Dataset (`data/sft/train.jsonl`)

Formatted for conversational SFT with `<think>...</think>` reasoning tokens:

```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant. You must think in short, telegraphic, bullet-point style fragments inside a <think>...</think> block before answering."},
    {"role": "user", "content": "Question text here..."},
    {"role": "assistant", "content": "<think>\n- fact 1\n- fact 2\n</think>\nFinal answer"}
  ]
}
```

## 2. DPO Preference Dataset (`data/dpo/train.jsonl`)

Contains paired chosen (terse, compressed thinking) versus rejected (unnecessarily verbose, wandering monologue) chains:

```json
{
  "prompt": "Question text here...",
  "chosen": "<think>\n- deduction 1\n- deduction 2\n</think>\nFinal answer",
  "rejected": "<think>\nOkay, let me carefully ponder this problem step by step. First, I should think about all aspects...\n</think>\nFinal answer"
}
```

## 3. Benchmark Records Schema (`benchmarks/deepseek-r1-7b/*/gsm8k.json`)

Each benchmark file contains aggregate summary metrics and full question-by-question generation logs:

```json
{
  "index": 0,
  "question": "Janet's ducks lay 16 eggs per day...",
  "ground_truth_raw": "Janet sells 16 - 3 - 4 = <<16-3-4=9>>9 eggs...",
  "ground_truth_numeric": 18.0,
  "prediction_numeric": 18.0,
  "raw_response": "First, determine how many eggs Janet sells each day...",
  "thinking": "First, calculate the total number of eggs Janet collects...",
  "answer": "Janet sells 9 eggs each day at the farmers market...",
  "is_correct": true,
  "is_format_compliant": true,
  "thinking_tokens": 105,
  "answer_tokens": 42,
  "total_tokens": 147,
  "latency_sec": 4.12
}
```

# How to Load via Hugging Face Datasets

```python
from datasets import load_dataset

# Load SFT training set
sft_train = load_dataset(
    "hari31416/grug-reasoning-data-and-benchmarks",
    data_files="data/sft/train.jsonl",
    split="train",
)

# Load DPO preference pairs
dpo_train = load_dataset(
    "hari31416/grug-reasoning-data-and-benchmarks",
    data_files="data/dpo/train.jsonl",
    split="train",
)
```

# Related Resources

- Model LoRA Adapters: [hari31416/deepseek-r1-7b-grug-adapters](https://huggingface.co/hari31416/deepseek-r1-7b-grug-adapters)
- GitHub Project: [Hari31416/qwen-grug-finetune](https://github.com/Hari31416/qwen-grug-finetune)
"""
    return card.strip() + "\n"


def stage_and_upload_model_repo(api: HfApi) -> None:
    """Stage adapter weights and upload to the model repository."""
    logger.info("Staging model repository...")
    staging_dir: str = os.path.join(WORKSPACE_ROOT, "staging_hf_models")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # 1. Copy SFT adapters
    sft_src: str = os.path.join(
        WORKSPACE_ROOT, "adapters/deepseek-r1-7b/20260804_040058/final_adapters"
    )
    sft_dst: str = os.path.join(staging_dir, "sft")
    copy_path(sft_src, sft_dst)

    # 2. Copy DPO adapters
    dpo_src: str = os.path.join(
        WORKSPACE_ROOT, "adapters/deepseek-r1-7b/dpo/20260805_055634/final_dpo_adapters"
    )
    dpo_dst: str = os.path.join(staging_dir, "dpo")
    copy_path(dpo_src, dpo_dst)

    # 3. Create README.md
    readme_path: str = os.path.join(staging_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(generate_model_card())

    logger.info("Uploading model files to %s...", MODEL_REPO_ID)
    api.upload_folder(
        folder_path=staging_dir,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Add DeepSeek-R1-7B SFT and DPO LoRA adapters",
    )
    logger.info("Model repository upload completed successfully.")

    shutil.rmtree(staging_dir)


def stage_and_upload_dataset_repo(api: HfApi) -> None:
    """Stage datasets and benchmarks and upload to the dataset repository."""
    logger.info("Staging dataset repository...")
    staging_dir: str = os.path.join(WORKSPACE_ROOT, "staging_hf_datasets")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # 1. SFT data
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/train.jsonl"),
        os.path.join(staging_dir, "data/sft/train.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/valid.jsonl"),
        os.path.join(staging_dir, "data/sft/valid.jsonl"),
    )

    # 2. DPO data
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/dpo/train.jsonl"),
        os.path.join(staging_dir, "data/dpo/train.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/dpo/valid.jsonl"),
        os.path.join(staging_dir, "data/dpo/valid.jsonl"),
    )

    # 3. Benchmark logs
    copy_path(
        os.path.join(WORKSPACE_ROOT, "results/deepseek-r1-7b"),
        os.path.join(staging_dir, "benchmarks/deepseek-r1-7b"),
    )

    # 4. Reports & Dashboards
    copy_path(
        os.path.join(WORKSPACE_ROOT, "report/deepseek-r1-7b"),
        os.path.join(staging_dir, "benchmarks/reports"),
    )

    # 5. Create README.md
    readme_path: str = os.path.join(staging_dir, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(generate_dataset_card())

    logger.info("Uploading dataset files to %s...", DATASET_REPO_ID)
    api.upload_folder(
        folder_path=staging_dir,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message="Add SFT datasets, DPO preference pairs, and benchmark logs",
    )
    logger.info("Dataset repository upload completed successfully.")

    shutil.rmtree(staging_dir)


def main() -> None:
    api: HfApi = HfApi()
    stage_and_upload_model_repo(api)
    stage_and_upload_dataset_repo(api)
    logger.info("All repositories synced to Hugging Face successfully.")


if __name__ == "__main__":
    main()
