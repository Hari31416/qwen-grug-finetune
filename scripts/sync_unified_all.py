"""Unified synchronization script to push all Grug reasoning models and datasets to Hugging Face.

Organizes:
1. Model Repository: hari31416/deepseek-r1-grug-adapters
   - deepseek-r1-7b/ (sft, dpo) + root sft/ and dpo/ for backward compatibility
   - deepseek-r1-1.5b/ (it-1, it-2-regularized, it-2-unregularized)
2. Dataset & Benchmark Repository: hari31416/grug-reasoning-data-and-benchmarks
   - data/ (1.5b/it-1, 1.5b/it-2, 7b/sft, 7b/dpo)
   - benchmarks/ (deepseek-r1-1.5b/, deepseek-r1-7b/)
   - reports/ (deepseek-r1-1.5b/, deepseek-r1-7b/)
"""

import logging
import os
import shutil
from typing import Optional
from huggingface_hub import HfApi

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("sync_unified_all")

WORKSPACE_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_REPO_ID: str = "hari31416/deepseek-r1-grug-adapters"
DATASET_REPO_ID: str = "hari31416/grug-reasoning-data-and-benchmarks"
LOCAL_15B_ADAPTERS: str = (
    "/Users/hari/.gemini/antigravity-ide/brain/31327625-1eb1-47c2-b7e1-34f8554ad209/scratch/1.5b_adapters/1.5b"
)


def copy_path(src: str, dst: str) -> None:
    """Copy a file or folder safely to destination."""
    if not os.path.exists(src):
        logger.warning("Source path does not exist: %s", src)
        return
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        logger.info("Copied directory from %s to %s", src, dst)
    else:
        shutil.copy2(src, dst)
        logger.info("Copied file from %s to %s", src, dst)


def generate_unified_model_card() -> str:
    """Generate comprehensive Model Card README for all Grug reasoning adapters."""
    return """---
license: apache-2.0
base_model:
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-7B
  - deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
tags:
  - reasoning
  - grug
  - telegraphic
  - lora
  - sft
  - dpo
  - mlx
  - peft
pipeline_tag: text-generation
library_name: peft
---

# DeepSeek-R1 Grug Reasoning Adapters (7B & 1.5B)

This repository houses all trained fine-tuned LoRA adapter weights across both phases of the **Grug Reasoning Project**:
1. **DeepSeek-R1-Distill-Qwen-7B**: 4-bit QLoRA Supervised Fine-Tuned (SFT) and Direct Preference Optimized (DPO) adapters trained on CUDA (NVIDIA T4).
2. **DeepSeek-R1-Distill-Qwen-1.5B**: LoRA adapters trained using the MLX framework on Apple Silicon (M4 GPU).

The goal of Grug reasoning is aligning models to think in compressed, telegraphic fragments inside `<think>...</think>` tags—cutting conversational filler and generation latency while preserving mathematical task accuracy.

# Repository Organization

```text
adapters/
├── deepseek-r1-7b/
│   ├── sft/                      # 7B SFT LoRA adapters (PEFT safetensors)
│   └── dpo/                      # 7B DPO LoRA adapters (PEFT safetensors)
├── deepseek-r1-1.5b/
│   ├── it-1/                     # 1.5B Proof of Concept MLX adapters
│   ├── it-2-regularized/         # 1.5B Regularized MLX adapters (prompt-dropout)
│   └── it-2-unregularized/       # 1.5B Unregularized MLX adapters
├── sft/                          # Backward compatible pointer to 7B SFT
└── dpo/                          # Backward compatible pointer to 7B DPO
```

# Benchmark Evaluation Summary

## 1. DeepSeek-R1-7B (Full GSM8K Test Split, 1,319 Samples)

| Model Variant | Test Samples | Accuracy | Format Compliance | Mean Thinking Tokens | Mean Answer Tokens | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Model (7B)** | 1,319 | 75.97% | 99.85% | 122.5 | 160.4 | 6.09s |
| **SFT Adapter (7B)** | 1,319 | 72.18% | 94.62% | 107.7 | 107.3 | 6.75s |
| **DPO Adapter (7B)** | 1,319 | 75.44% | 99.85% | 122.3 | 162.1 | 6.39s |

- **Answer Brevity**: SFT cut final answer lengths from 160.4 to 107.3 tokens (33.1% reduction).
- **Preference Alignment**: DPO preference pairs eliminated repetitive derivation loops, restoring format compliance to 99.85% and task accuracy to 75.44%.

## 2. DeepSeek-R1-1.5B (Apple Silicon Experiments)

| Configuration | Accuracy | Mean Thinking Tokens | Mean Total Tokens | Mean Latency | Format Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Normal (1.5B)** | 64.9% | 219.0 | 477.4 | 0.88s | 96.6% |
| **FT Normal (1.5B)** | 66.0% | 156.2 | 389.3 | 0.73s | 98.9% |
| **FT Regularized (1.5B)** | 54.6% | 135.0 | 214.7 | 0.61s | 98.2% |

# How to Use

## Loading 7B Adapters with Hugging Face Transformers & PEFT

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
adapter_repo = "hari31416/deepseek-r1-grug-adapters"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)

# Load SFT adapter
sft_model = PeftModel.from_pretrained(base_model, adapter_repo, subfolder="deepseek-r1-7b/sft")

# Or load DPO adapter
# dpo_model = PeftModel.from_pretrained(base_model, adapter_repo, subfolder="deepseek-r1-7b/dpo")
```

## Loading 1.5B Adapters with Apple Silicon MLX

```bash
# Load and generate using MLX LoRA
mlx_lm.generate \\
    --model mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit \\
    --adapter-path <path-to-downloaded-1.5b-adapters> \\
    --prompt "Janet has 16 eggs..."
```

# Related Resources

- Datasets and Benchmark Logs: [hari31416/grug-reasoning-data-and-benchmarks](https://huggingface.co/datasets/hari31416/grug-reasoning-data-and-benchmarks)
- GitHub Project Repository: [Hari31416/qwen-grug-finetune](https://github.com/Hari31416/qwen-grug-finetune)
"""


def generate_unified_dataset_card() -> str:
    """Generate comprehensive Dataset Card README."""
    return """---
license: apache-2.0
tags:
  - reasoning
  - preference-tuning
  - dpo
  - sft
  - gsm8k
pretty_name: Grug Reasoning Datasets and Benchmarks
---

# Grug Reasoning Datasets and Benchmark Results

This repository contains all training datasets, preference pairs, raw model generation logs, and empirical benchmark results for the Grug reasoning research project (spanning both 1.5B and 7B models).

# Repository Contents

```text
├── data/
│   ├── 1.5b/
│   │   ├── it-1/                 # Iteration 1 SFT data and compressed traces
│   │   └── it-2/                 # Iteration 2 scaled SFT data
│   ├── 7b/
│   │   ├── sft/                  # 7B SFT training and validation splits
│   │   └── dpo/                  # 7B DPO preference pairs (chosen vs rejected)
│   ├── sft/                      # Root pointer to 7B SFT
│   └── dpo/                      # Root pointer to 7B DPO
├── benchmarks/
│   ├── deepseek-r1-1.5b/
│   │   ├── it-1/                 # 1.5B Iteration 1 GSM8K evaluation JSONs
│   │   └── it-2/                 # 1.5B Iteration 2 GSM8K evaluation JSONs
│   └── deepseek-r1-7b/
│       ├── baseline/gsm8k.json   # Full 1,319 GSM8K evaluations (Base 7B)
│       ├── finetuned/gsm8k.json  # Full 1,319 GSM8K evaluations (SFT 7B)
│       └── dpo/gsm8k.json        # Full 1,319 GSM8K evaluations (DPO 7B)
└── reports/
    ├── deepseek-r1-1.5b/
    │   ├── it-1/                 # 1.5B Iteration 1 dashboard plots and report PDF
    │   └── it-2/                 # 1.5B Iteration 2 dashboard plots and markdown report
    └── deepseek-r1-7b/
        ├── benchmark_comparison_dashboard.png
        ├── loss_plot.png
        └── BENCHMARK_REPORT.md
```

# Benchmark Results

## DeepSeek-R1-7B (Full GSM8K Test Split, 1,319 Samples)

| Model Variant | Samples | Accuracy | Format Compliance | Mean Thinking Tokens | Mean Answer Tokens | Mean Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Model (7B)** | 1,319 | 75.97% | 99.85% | 122.5 | 160.4 | 6.09s |
| **SFT Adapter (7B)** | 1,319 | 72.18% | 94.62% | 107.7 | 107.3 | 6.75s |
| **DPO Adapter (7B)** | 1,319 | 75.44% | 99.85% | 122.3 | 162.1 | 6.39s |

## DeepSeek-R1-1.5B (Apple Silicon Experiments)

| Configuration | Accuracy | Mean Thinking Tokens | Mean Total Tokens | Mean Latency | Format Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Normal (1.5B)** | 64.9% | 219.0 | 477.4 | 0.88s | 96.6% |
| **FT Normal (1.5B)** | 66.0% | 156.2 | 389.3 | 0.73s | 98.9% |
| **FT Regularized (1.5B)** | 54.6% | 135.0 | 214.7 | 0.61s | 98.2% |

# Visualizations

## 7B Benchmark Comparison Dashboard

![7B Dashboard](reports/deepseek-r1-7b/benchmark_comparison_dashboard.png)

## 7B DPO Training Loss Curve

![7B DPO Loss](reports/deepseek-r1-7b/loss_plot.png)

## 1.5B Iteration 2 Dashboard

![1.5B Dashboard](reports/deepseek-r1-1.5b/it-2/dashboard.png)

# How to Load via Hugging Face Datasets

```python
from datasets import load_dataset

# Load 7B SFT dataset
sft_7b = load_dataset("hari31416/grug-reasoning-data-and-benchmarks", data_files="data/7b/sft/train.jsonl", split="train")

# Load 7B DPO preference pairs
dpo_7b = load_dataset("hari31416/grug-reasoning-data-and-benchmarks", data_files="data/7b/dpo/train.jsonl", split="train")

# Load 1.5B SFT dataset
sft_15b = load_dataset("hari31416/grug-reasoning-data-and-benchmarks", data_files="data/1.5b/it-2/train.jsonl", split="train")
```

# Related Resources

- Model LoRA Adapters: [hari31416/deepseek-r1-grug-adapters](https://huggingface.co/hari31416/deepseek-r1-grug-adapters)
- GitHub Project Repository: [Hari31416/qwen-grug-finetune](https://github.com/Hari31416/qwen-grug-finetune)
"""


def stage_and_upload_models(api: HfApi) -> None:
    """Stage and upload unified model repository."""
    logger.info("Staging unified model repository...")
    staging_dir = os.path.join(WORKSPACE_ROOT, "staging_hf_unified_models")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # 1. Copy 7B SFT adapters (both root and deepseek-r1-7b/sft)
    sft_src = os.path.join(
        WORKSPACE_ROOT, "adapters/deepseek-r1-7b/20260804_040058/final_adapters"
    )
    copy_path(sft_src, os.path.join(staging_dir, "sft"))
    copy_path(sft_src, os.path.join(staging_dir, "deepseek-r1-7b/sft"))

    # 2. Copy 7B DPO adapters (both root and deepseek-r1-7b/dpo)
    dpo_src = os.path.join(
        WORKSPACE_ROOT, "adapters/deepseek-r1-7b/dpo/20260805_055634/final_dpo_adapters"
    )
    copy_path(dpo_src, os.path.join(staging_dir, "dpo"))
    copy_path(dpo_src, os.path.join(staging_dir, "deepseek-r1-7b/dpo"))

    # 3. Copy 1.5B MLX adapters
    copy_path(
        os.path.join(LOCAL_15B_ADAPTERS, "it-1"),
        os.path.join(staging_dir, "deepseek-r1-1.5b/it-1"),
    )
    copy_path(
        os.path.join(LOCAL_15B_ADAPTERS, "it-2-regularized"),
        os.path.join(staging_dir, "deepseek-r1-1.5b/it-2-regularized"),
    )
    copy_path(
        os.path.join(LOCAL_15B_ADAPTERS, "it-2-unregularized"),
        os.path.join(staging_dir, "deepseek-r1-1.5b/it-2-unregularized"),
    )

    # 4. Generate README.md
    with open(os.path.join(staging_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(generate_unified_model_card())

    logger.info("Uploading unified model repository to %s...", MODEL_REPO_ID)
    api.upload_folder(
        folder_path=staging_dir,
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Unify all 7B and 1.5B Grug reasoning adapters",
    )
    logger.info("Unified model upload complete.")
    shutil.rmtree(staging_dir)


def stage_and_upload_datasets(api: HfApi) -> None:
    """Stage and upload unified dataset and benchmark repository."""
    logger.info("Staging unified dataset repository...")
    staging_dir = os.path.join(WORKSPACE_ROOT, "staging_hf_unified_datasets")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # 1. SFT and DPO data (7B)
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/train.jsonl"),
        os.path.join(staging_dir, "data/sft/train.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/valid.jsonl"),
        os.path.join(staging_dir, "data/sft/valid.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/train.jsonl"),
        os.path.join(staging_dir, "data/7b/sft/train.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/valid.jsonl"),
        os.path.join(staging_dir, "data/7b/sft/valid.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/dpo/train.jsonl"),
        os.path.join(staging_dir, "data/dpo/train.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/dpo/valid.jsonl"),
        os.path.join(staging_dir, "data/dpo/valid.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/dpo/train.jsonl"),
        os.path.join(staging_dir, "data/7b/dpo/train.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/dpo/valid.jsonl"),
        os.path.join(staging_dir, "data/7b/dpo/valid.jsonl"),
    )

    # 2. 1.5B data
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/it-1/train.jsonl"),
        os.path.join(staging_dir, "data/1.5b/it-1/train.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/it-1/valid.jsonl"),
        os.path.join(staging_dir, "data/1.5b/it-1/valid.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/it-2-3/train.jsonl"),
        os.path.join(staging_dir, "data/1.5b/it-2/train.jsonl"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "data/it-2-3/valid.jsonl"),
        os.path.join(staging_dir, "data/1.5b/it-2/valid.jsonl"),
    )

    # 3. 7B Benchmarks
    copy_path(
        os.path.join(WORKSPACE_ROOT, "results/deepseek-r1-7b"),
        os.path.join(staging_dir, "benchmarks/deepseek-r1-7b"),
    )

    # 4. 1.5B Benchmarks
    copy_path(
        os.path.join(WORKSPACE_ROOT, "results/it-1/deepseek-r1-1.5b"),
        os.path.join(staging_dir, "benchmarks/deepseek-r1-1.5b/it-1"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "results/it-2/deepseek-r1-1.5b"),
        os.path.join(staging_dir, "benchmarks/deepseek-r1-1.5b/it-2"),
    )

    # 5. Reports and plots
    copy_path(
        os.path.join(WORKSPACE_ROOT, "report/deepseek-r1-7b"),
        os.path.join(staging_dir, "reports/deepseek-r1-7b"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "report/it-1"),
        os.path.join(staging_dir, "reports/deepseek-r1-1.5b/it-1"),
    )
    copy_path(
        os.path.join(WORKSPACE_ROOT, "report/it-2"),
        os.path.join(staging_dir, "reports/deepseek-r1-1.5b/it-2"),
    )

    # 6. Generate README.md
    with open(os.path.join(staging_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(generate_unified_dataset_card())

    logger.info("Uploading unified dataset repository to %s...", DATASET_REPO_ID)
    api.upload_folder(
        folder_path=staging_dir,
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        commit_message="Unify all 7B and 1.5B Grug reasoning data, benchmarks, and reports",
    )
    logger.info("Unified dataset upload complete.")
    shutil.rmtree(staging_dir)


def main() -> None:
    api = HfApi()
    stage_and_upload_models(api)
    stage_and_upload_datasets(api)
    logger.info("All repositories unified and synced successfully!")


if __name__ == "__main__":
    main()
