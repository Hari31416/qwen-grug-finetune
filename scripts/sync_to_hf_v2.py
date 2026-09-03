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

Evaluated on the GSM8K test split:

| Model Variant | Samples | Accuracy | Format Compliance | Mean Thinking Tokens | Mean Answer Tokens | Mean Latency |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Base Model (7B)** | 50 (test) | 86.00% | 100.0% | 143.5 | 119.4 | 19.63s |
| **SFT Adapter** | 50 (test) | 82.00% | 100.0% | 90.4 | 40.2 | 17.42s |
| **DPO Adapter** | 1,319 (full test) | 81.50% | 100.0% | 150.0 | 122.9 | 7.06s |

*Note: A unified 1,319-sample benchmark run across all three variants is currently executing to provide complete full-split comparisons.*

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

This repository contains the training datasets, preference pairs, and empirical evaluation benchmark logs for the Grug reasoning fine-tuning project.

# Repository Contents

- `data/sft/`: Supervised fine-tuning dataset formatted for chat completion models.
  - `train.jsonl` (1,530 rows)
  - `valid.jsonl` (171 rows)
- `data/dpo/`: Direct preference optimization pairs containing chosen (terse, compressed reasoning) versus rejected (overly verbose, repetitive reasoning) trajectories.
  - `train.jsonl` (1,530 preference pairs)
  - `valid.jsonl` (171 preference pairs)
- `benchmarks/`: Evaluation logs on GSM8K containing generation responses, thinking length counts, accuracy, and latency records.
  - `deepseek-r1-7b/`: GSM8K evaluation metrics for Baseline, SFT, and DPO models.
  - `reports/`: Comparative dashboard figures, loss curves, and analysis reports.

# Dataset Schema

## SFT (`data/sft/train.jsonl`)
Each line contains a JSON object with:
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "<think>...</think>..."}
  ]
}
```

## DPO (`data/dpo/train.jsonl`)
Each line contains preference pairs:
```json
{
  "prompt": "...",
  "chosen": "<think>\\n- step 1\\n- step 2\\n</think>\\nFinal answer",
  "rejected": "<think>\\nFirst, let us carefully consider all the various aspects...\\n</think>\\nFinal answer"
}
```

# Related Models

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
