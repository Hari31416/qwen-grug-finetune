#!/usr/bin/env python3
import json
import os

def create_notebook():
    cells = []

    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    # Title
    add_md("""# Grug Reasoning Fine-Tuning: Kaggle / CUDA Interactive Notebook

This notebook performs end-to-end 4-bit QLoRA fine-tuning, inference, evaluation, and loss visualization on **DeepSeek-R1-Distill-Qwen-7B**.

### Notebook Workflow:
1. **Hyperparameters & Config**: Centralized experimental parameters (epochs, batch sizes, learning rates, limits).
2. **Repository Clone & Setup**: Automatically clones repository scripts and installs dependencies.
3. **Base Model Inference**: Load 4-bit quantized base model and run sample generation.
4. **QLoRA Fine-Tuning**: Execute `run_sft_training()` using the pre-loaded base model (saves VRAM).
5. **Loss Visualization**: Plot Loss Curves and Learning Rate schedule using `plot_latest_training_loss()`.
6. **GSM8K Benchmarking**: Benchmark Base Model vs. Fine-Tuned Model using `run_gsm8k_eval()`.
7. **EDA Dashboard**: Plot comparative metrics for Accuracy, Format Compliance, Reasoning Tokens, and Latency.""")

    # Section 1: Hyperparameters
    add_md("## 1. Centralized Hyperparameters & Configuration")

    add_code("""# ==========================================
# ⚙️ EXPERIMENTAL HYPERPARAMETERS & CONFIG
# ==========================================

# Repository & Dataset Paths
REPO_URL = "https://github.com/Hari31416/qwen-grug-finetune.git"
REPO_NAME = "qwen-grug-finetune"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATA_DIR = "data"
ADAPTER_OUTPUT_DIR = "adapters"

# Training Hyperparameters
EPOCHS = 1                # Number of training epochs (e.g. 1 or 3)
TRAIN_BATCH_SIZE = 2      # Per-device training batch size
GRAD_ACCUM = 4            # Gradient accumulation steps (Effective batch size = 2 * 4 * num_gpus)
LEARNING_RATE = 2e-4      # Peak learning rate
MAX_SEQ_LENGTH = 1536     # Maximum token sequence length for training
LORA_R = 16               # LoRA rank dimension
LORA_ALPHA = 32           # LoRA alpha scaling factor

# Benchmark Evaluation Hyperparameters
EVAL_LIMIT = 50           # Limit number of GSM8K test samples for fast evaluation (e.g., 50 or 100)
EVAL_BATCH_SIZE = 2       # Per-device evaluation batch size (keep low to prevent VRAM OOM)
EVAL_MAX_TOKENS = 1024     # Max generation tokens per GSM8K problem""")

    # Section 2: Repository Clone & Setup
    add_md("## 2. Repository Clone & Environment Setup")

    add_code("""# Install required Python packages (omitting -U torch to prevent CUDA driver conflicts)
%pip install -q peft trl bitsandbytes datasets accelerate huggingface_hub matplotlib seaborn pandas""")

    add_code("""import sys
import os

# Automatically clone repo scripts if running in fresh Kaggle / Colab session
if not os.path.exists("scripts") and not os.path.exists(f"{REPO_NAME}/scripts"):
    print(f"Cloning {REPO_URL} into workspace...")
    !git clone {REPO_URL}
    if os.path.exists(REPO_NAME):
        %cd {REPO_NAME}
elif os.path.exists(REPO_NAME) and os.path.exists(f"{REPO_NAME}/scripts"):
    %cd {REPO_NAME}

# Add repository root to python path
sys.path.append(".")

from scripts.cuda.cuda_utils import patch_transformers_lazy_imports
from scripts.cuda.download_data import download_hf_data

# Apply Kaggle/Colab lazy import patches & download dataset
patch_transformers_lazy_imports()
download_hf_data(output_dir=DATA_DIR)""")

    # Section 3: Base Model Inference
    add_md("## 3. Base Model Inference")

    add_code("""import torch
from transformers import BitsAndBytesConfig
from scripts.cuda.cuda_utils import load_causal_lm_model, load_causal_lm_tokenizer
from scripts.cuda.generate_cuda import generate_response

is_cuda = torch.cuda.is_available()

# 4-bit Quantization Config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
) if is_cuda else None

model_kwargs = {"trust_remote_code": True}
if is_cuda:
    model_kwargs["quantization_config"] = bnb_config
    model_kwargs["device_map"] = "auto"
    model_kwargs["torch_dtype"] = torch.float16

print(f"Loading Base Model ({MODEL_ID})...")
model = load_causal_lm_model(MODEL_ID, **model_kwargs)
tokenizer = load_causal_lm_tokenizer(MODEL_ID)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# Test sample inference
prompt = "If a train travels 60 mph for 2.5 hours, how far does it go?"
response = generate_response(model, tokenizer, prompt)
print("=== BASE MODEL RESPONSE ===\\n", response)""")

    # Section 4: Fine-Tuning
    add_md("## 4. Fine-Tuning via `run_sft_training` Import")

    add_code("""from scripts.cuda.train_cuda import run_sft_training

print("Starting SFT Training with pre-loaded model (reusing VRAM)...")
trainer = run_sft_training(
    model_arg=MODEL_ID,
    data_dir=DATA_DIR,
    adapter_path=ADAPTER_OUTPUT_DIR,
    epochs=EPOCHS,
    batch_size=TRAIN_BATCH_SIZE,
    grad_accum=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    max_seq_length=MAX_SEQ_LENGTH,
    lora_r=LORA_R,
    lora_alpha=LORA_ALPHA,
    model=model,
    tokenizer=tokenizer,
)""")

    # Section 5: Plot Loss
    add_md("## 5. Plot Loss Curves & Training Metrics")

    add_code("""from scripts.cuda.plot_loss import plot_latest_training_loss

# Generate loss curves and learning rate schedule plots
plot_latest_training_loss()""")

    # Section 6: GSM8K Evaluation
    add_md("## 6. Benchmarking Base Model vs. Fine-Tuned Model on GSM8K")

    add_code("""import glob
from peft import PeftModel
from scripts.cuda.eval_cuda import run_gsm8k_eval

print(f"1. Evaluating Baseline Model on GSM8K (limit={EVAL_LIMIT}, batch_size={EVAL_BATCH_SIZE})...")
base_eval_results = run_gsm8k_eval(
    model,
    tokenizer,
    limit=EVAL_LIMIT,
    batch_size=EVAL_BATCH_SIZE,
    max_tokens=EVAL_MAX_TOKENS,
    is_adapter=False,
)
print("Baseline Summary:", base_eval_results)

adapter_dirs = glob.glob(f"{ADAPTER_OUTPUT_DIR}/**/final_adapters", recursive=True)
latest_adapter = max(adapter_dirs, key=os.path.getmtime) if adapter_dirs else ""

ft_eval_results = {}
if latest_adapter:
    print(f"2. Loading fine-tuned LoRA model from: {latest_adapter}...")
    ft_model = PeftModel.from_pretrained(model, latest_adapter)
    ft_model.eval()
    
    print(f"3. Evaluating Fine-Tuned Model on GSM8K (limit={EVAL_LIMIT}, batch_size={EVAL_BATCH_SIZE})...")
    ft_eval_results = run_gsm8k_eval(
        ft_model,
        tokenizer,
        limit=EVAL_LIMIT,
        batch_size=EVAL_BATCH_SIZE,
        max_tokens=EVAL_MAX_TOKENS,
        is_adapter=True,
    )
    print("Fine-Tuned Summary:", ft_eval_results)""")

    # Section 7: EDA Dashboard
    add_md("## 7. EDA & Comparison Dashboard")

    add_code("""import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

if base_eval_results and ft_eval_results:
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Accuracy & Format Compliance
    metrics_df = pd.DataFrame([
        {"Model": "Baseline", "Metric": "Accuracy", "Value": base_eval_results["accuracy"] * 100},
        {"Model": "Fine-Tuned", "Metric": "Accuracy", "Value": ft_eval_results["accuracy"] * 100},
        {"Model": "Baseline", "Metric": "Format Compliance", "Value": base_eval_results["format_compliance_rate"] * 100},
        {"Model": "Fine-Tuned", "Metric": "Format Compliance", "Value": ft_eval_results["format_compliance_rate"] * 100},
    ])
    sns.barplot(data=metrics_df, x="Metric", y="Value", hue="Model", ax=axes[0], palette="viridis")
    axes[0].set_title("Accuracy & Format Compliance (%)", fontsize=12, fontweight="bold")
    axes[0].set_ylabel("Percentage (%)")
    for p in axes[0].patches:
        if p.get_height() > 0:
            axes[0].annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)

    # 2. Reasoning Tokens Reduction
    tok_df = pd.DataFrame([
        {"Model": "Baseline", "Mean Thinking Tokens": base_eval_results["mean_thinking_tokens"]},
        {"Model": "Fine-Tuned", "Mean Thinking Tokens": ft_eval_results["mean_thinking_tokens"]}
    ])
    sns.barplot(data=tok_df, x="Model", y="Mean Thinking Tokens", ax=axes[1], palette="mako")
    axes[1].set_title("Mean Thinking Tokens per Problem", fontsize=12, fontweight="bold")
    axes[1].set_ylabel("Tokens")
    for p in axes[1].patches:
        if p.get_height() > 0:
            axes[1].annotate(f"{p.get_height():.1f}", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)

    # 3. Inference Latency
    lat_df = pd.DataFrame([
        {"Model": "Baseline", "Mean Latency (s)": base_eval_results["mean_latency"]},
        {"Model": "Fine-Tuned", "Mean Latency (s)": ft_eval_results["mean_latency"]}
    ])
    sns.barplot(data=lat_df, x="Model", y="Mean Latency (s)", ax=axes[2], palette="rocket")
    axes[2].set_title("Mean Inference Latency (Seconds)", fontsize=12, fontweight="bold")
    axes[2].set_ylabel("Seconds")
    for p in axes[2].patches:
        if p.get_height() > 0:
            axes[2].annotate(f"{p.get_height():.2f}s", (p.get_x() + p.get_width() / 2., p.get_height()), ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plot_save_path = os.path.join(ADAPTER_OUTPUT_DIR, "eda_comparison_dashboard.png")
    plt.savefig(plot_save_path, dpi=150, bbox_inches="tight")
    print(f"Saved dashboard plot to: {plot_save_path}")
    plt.show()""")

    nb = {
        "cells": cells,
        "metadata": {
            "language_info": {
                "name": "python"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 2
    }

    os.makedirs("notebooks", exist_ok=True)
    out_path = "notebooks/kaggle_grug_finetune.ipynb"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully generated notebook with VRAM optimization at: {out_path}")

if __name__ == "__main__":
    create_notebook()
