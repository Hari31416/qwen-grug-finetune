#!/usr/bin/env python3
import os
import sys
import shutil
import subprocess
import argparse
import logging
from typing import Optional

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import config

# Constants
GITHUB_REPO_URL = "https://github.com/Hari31416/qwen-grug-finetune"
HF_REPO_ID = "hari31416/qwen-grug-finetune"

def run_command(command: list[str]) -> bool:
    """Run a system command and return whether it succeeded."""
    try:
        logger.info("Executing command: %s", " ".join(command))
        result = subprocess.run(command, check=True, text=True)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        logger.error("Command failed: %s", e)
        return False
    except Exception as e:
        logger.error("Error executing command: %s", e)
        return False

def copy_file_or_dir(src: str, dst: str) -> None:
    """Copy a file or directory, creating destination directories if needed."""
    if not os.path.exists(src):
        logger.warning("Source path does not exist, skipping: %s", src)
        return

    dst_dir = os.path.dirname(dst)
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

def generate_readme(output_path: str) -> None:
    """Generate a comprehensive README.md for the Hugging Face repository."""
    base_model_name = config.model_mlx_path.split("/")[-1].replace("-4bit", "")
    readme_content = f"""---
license: mit
base_model: {config.model_mlx_path}
tags:
- mlx
- lora
- reasoning
- grug-style
---

# Grug Reasoning Fine-Tune ({base_model_name})

This repository contains the fine-tuning training datasets, adapters (LoRA weights), and experimental results for **DeepSeek-R1-Distill-Qwen-1.5B** to learn a telegraphic, token-efficient reasoning style ("Grug/caveman" style) on Apple Silicon using MLX.

For the full code, training scripts, evaluation pipeline, and development history, visit the GitHub repository:
👉 **[GitHub Repository: Hari31416/qwen-grug-finetune]({GITHUB_REPO_URL})**

---

## 📌 Project Overview
The "Grug Hypothesis" tests whether a small reasoning model can internalize a highly compressed, terse reasoning style (removing articles, fillers, and politeness markers) inside its `<think>...</think>` block to save generation tokens and latency, without severely degrading task accuracy.

The project progressed through three distinct experimental runs/iterations:
1. **Iteration 1**: Initial proof-of-concept using 333 validated SFT traces. Trained for 300 steps.
2. **Iteration 2 (Unregularized)**: Scaled dataset to 1,530 training rows and LoRA rank to 16. Trained for 2,000 steps. Experienced severe prompt leakage and instruction regurgitation due to overfitting.
3. **Iteration 2 (Regularized / Final)**: Applied 20% prompt dropout for positive examples, 30% negative example mixture (uncompressed verbose traces), and 50% negative system prompts. Trained for 1,000 steps. Completely eliminated prompt leakage and achieved robust format compliance.

---

## 📁 Repository Structure
The repository is organized by iteration:
```
.
├── README.md
├── iteration-1/
│   ├── data/             # Training & validation datasets (333 train rows)
│   ├── model/            # LoRA adapters, metrics.json, loss_plot.png
│   └── report/           # Performance reports & evaluation JSON logs
│
├── iteration-2-unregularized/
│   ├── model/            # Overfit adapters, metrics.json, loss_plot.png (2000 steps)
│   └── report/           # Performance reports & evaluation JSON logs
│
└── iteration-2-regularized/
    ├── data/             # Regularized SFT datasets (1,530 train rows)
    ├── model/            # Calibrated LoRA adapters, metrics.json, loss_plot.png (1000 steps)
    └── report/           # Final performance reports & evaluation JSON logs
```

### 📈 Detailed Report Directories
Each iteration includes a dedicated `report/` directory containing detailed analyses, performance graphs, and raw logs:
- **Experimental Writeups (`REPORT.md` / `REPORT.pdf`)**: A comprehensive breakdown of setup parameters, convergence details, evaluation metrics, and key takeaways.
- **Comparison Plots & Images**:
  - `loss_curve.png`: Progression of training and validation loss.
  - `accuracy.png`: Task accuracy comparison between baseline and fine-tuned checkpoints.
  - `tokens.png`: Distribution of emitted reasoning token lengths.
  - `latency_speed.png`: Inference latency and token-per-second generation throughput comparison.
  - `deltas.png`: Exact performance and token saving deltas.
  - `dashboard.png`: Unified dashboard compiling all experimental graphs.
- **Evaluation Logs**: Raw JSON output logs (e.g., `gsm8k_baseline.json`, `gsm8k_finetuned.json`) containing prompt formatting, model responses, parsed final answers, and correctness tags for all test samples.

---

## 📊 Experimental Results & Comparison

### Iteration 1 Metrics (GSM8K Test Split)

| Configuration | Accuracy | Mean Thinking Tokens | Mean Total Tokens | Mean Latency (s) | Format Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Normal** | 64.9% | 219.0 | 477.4 | 0.88s | 96.6% |
| **Base Grug Prompt** | 67.2% | 512.8 | 581.1 | 1.21s | 91.5% |
| **FT Normal** | 66.0% | 156.2 | 389.3 | 0.73s | 98.9% |
| **FT Grug Prompt** | 45.6% | 120.0 | 229.0 | 0.64s | 95.1% |

### Iteration 2 (Unregularized) Metrics (GSM8K Test Split)
Evaluated under the target style system prompt (Base vs. FT):

| Configuration | Accuracy | Mean Thinking Tokens | Mean Total Tokens | Mean Latency (s) | Format Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Model (Style Prompt)** | 71.5% | 504.0 | 568.2 | 1.09s | 92.3% |
| **FT Model (Unregularized)** | 52.7% | 99.6 | 226.0 | 0.59s | 99.2% |

### Iteration 2 (Regularized / Final) Metrics (GSM8K Test Split)
Evaluated under the target style system prompt (Base vs. FT):

| Configuration | Accuracy | Mean Thinking Tokens | Mean Total Tokens | Mean Latency (s) | Format Compliance |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Base Model (Style Prompt)** | 70.1% | 517.4 | 582.1 | 1.28s | 91.1% |
| **FT Model (Regularized)** | 54.6% | 135.0 | 214.7 | 0.61s | 98.2% |

### Key Takeaways
- **Overfitting & Prompt Leakage Mitigated**: The SFT regularization strategy (20% prompt dropout, 30% negative mixture) successfully prevented the model from repeating system prompt rules, ensuring high format compliance (98.2%).
- **Reasoning Compression Achieved**: Fine-tuning achieved a **73.9% reduction** in thinking tokens and a **52.3% reduction** in generation latency compared to the baseline.
- **The "Alignment Tax"**: Accuracy dropped by 15.5 percentage points. Since the SFT dataset only contained general-reasoning tasks, the model lacked task-specific math SFT examples, leading it to over-compress derivations and drop calculations. This forms the basis of **Iteration 3 (Benchmark SFT Mixing)**.

---

## 🚀 How to Use the Adapters

You can load these adapters using the MLX framework on Apple Silicon.

### 1. Install Dependencies
```bash
pip install mlx-lm
```

### 2. Run Inference in Python
```python
from mlx_lm import load, generate

# Path to the downloaded adapter directory
adapter_path = "./iteration-2-regularized/model"

# Load the base model with LoRA adapters
model, tokenizer = load(
    "{config.model_mlx_path}",
    adapter_path=adapter_path
)

# Format target system style prompt
system_prompt = (
    "You are a helpful assistant. You must think in short, telegraphic, "
    "bullet-point style fragments inside a <think>...</think> block before answering."
)
messages = [
    {{"role": "system", "content": system_prompt}},
    {{"role": "user", "content": "If John has 3 apples and buys 2 more, how many does he have?"}}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Generate response
response = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=1000,
    temp=0.6
)
print(response)
```

---

## 🔗 Links & Resources
- **GitHub Repository:** [{GITHUB_REPO_URL}]({GITHUB_REPO_URL})
- **Base Model:** [{config.model_mlx_path}](https://huggingface.co/{config.model_mlx_path})
- **Hugging Face Model Hub:** [{HF_REPO_ID}](https://huggingface.co/{HF_REPO_ID})
"""
    with open(output_path, "w") as f:
        f.write(readme_content.strip() + "\n")
    logger.info("Generated README.md at %s", output_path)

def main() -> None:
    parser = argparse.ArgumentParser(description="Sync and push data & models to Hugging Face")
    parser.add_argument("--push", action="store_true", help="Push to Hugging Face after staging files")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token")
    args = parser.parse_args()

    workspace_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    staging_dir = os.path.join(workspace_root, "data-and-models")

    logger.info("Staging folder: %s", staging_dir)
    os.makedirs(staging_dir, exist_ok=True)

    # 1. Stage Iteration 1
    logger.info("Staging Iteration 1...")
    copy_file_or_dir(os.path.join(workspace_root, "data", "it-1"), os.path.join(staging_dir, "iteration-1", "data"))
    copy_file_or_dir(os.path.join(workspace_root, "adapters", "deepseek-r1-1.5b", "20260701_210744"), os.path.join(staging_dir, "iteration-1", "model"))
    copy_file_or_dir(os.path.join(workspace_root, "report", "it-1"), os.path.join(staging_dir, "iteration-1", "report"))

    # 2. Stage Iteration 2 (Unregularized)
    logger.info("Staging Iteration 2 (Unregularized)...")
    copy_file_or_dir(os.path.join(workspace_root, "adapters", "deepseek-r1-1.5b", "20260703_101243"), os.path.join(staging_dir, "iteration-2-unregularized", "model"))
    copy_file_or_dir(os.path.join(workspace_root, "report", "it-2"), os.path.join(staging_dir, "iteration-2-unregularized", "report"))

    # 3. Stage Iteration 2 (Regularized / Final)
    logger.info("Staging Iteration 2 (Regularized)...")
    copy_file_or_dir(os.path.join(workspace_root, "data", "train.jsonl"), os.path.join(staging_dir, "iteration-2-regularized", "data", "train.jsonl"))
    copy_file_or_dir(os.path.join(workspace_root, "data", "valid.jsonl"), os.path.join(staging_dir, "iteration-2-regularized", "data", "valid.jsonl"))
    copy_file_or_dir(os.path.join(workspace_root, "adapters", "deepseek-r1-1.5b", "20260704_121944"), os.path.join(staging_dir, "iteration-2-regularized", "model"))
    
    # Stage files from root report folder (non-recursive)
    report_src = os.path.join(workspace_root, "report")
    report_dst = os.path.join(staging_dir, "iteration-2-regularized", "report")
    os.makedirs(report_dst, exist_ok=True)
    if os.path.exists(report_src):
        for item in os.listdir(report_src):
            item_path = os.path.join(report_src, item)
            if os.path.isfile(item_path):
                copy_file_or_dir(item_path, os.path.join(report_dst, item))

    # 4. Generate README.md
    logger.info("Generating README.md...")
    generate_readme(os.path.join(staging_dir, "README.md"))

    logger.info("All files successfully staged in 'data-and-models/'!")

    # 5. Push to Hugging Face
    if args.push:
        logger.info("Uploading to Hugging Face...")
        upload_cmd = ["huggingface-cli", "upload", HF_REPO_ID, staging_dir, ".", "--repo-type", "model"]
        if args.token:
            upload_cmd.extend(["--token", args.token])
        
        success = run_command(upload_cmd)
        if success:
            logger.info("Successfully pushed all iterations to Hugging Face: https://huggingface.co/%s", HF_REPO_ID)
        else:
            logger.error("Hugging Face upload failed.")
    else:
        logger.info("Staging complete. Run with '--push' to automatically upload to Hugging Face.")

if __name__ == "__main__":
    main()
