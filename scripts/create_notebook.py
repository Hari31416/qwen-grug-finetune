#!/usr/bin/env python3
import json
import os


def create_cell_helpers():
    cells = []
    def add_md(text):
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [line + "\n" for line in text.strip().split("\n")]
        })

    def add_code(text):
        cells.append(
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "source": [line + "\n" for line in text.strip().split("\n")],
            }
        )
    return cells, add_md, add_code


def save_notebook(cells, filename):
    nb = {
        "cells": cells,
        "metadata": {"language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    os.makedirs("notebooks", exist_ok=True)
    out_path = os.path.join("notebooks", filename)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=2)
    print(f"✅ Generated notebook: {out_path}")


# ==============================================================================
# NOTEBOOK 1: SFT FINE-TUNING & EVALUATION PIPELINE
# ==============================================================================
def generate_sft_notebook():
    cells, add_md, add_code = create_cell_helpers()

    # Top Standalone Overview Cell
    add_md(
        """# Telegraphic SFT Alignment Pipeline (DeepSeek-R1-7B)

## 🎯 What Are We Doing?
We are performing end-to-end 4-bit QLoRA Supervised Fine-Tuning (SFT) on `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` followed by GSM8K benchmark evaluation, qualitative reasoning trace inspection, and comparative dashboard plotting.

## 💡 Why Are We Doing It?
Standard reasoning models emit verbose, multi-paragraph reasoning traces before giving an answer. This SFT pipeline aligns the model to produce **telegraphic, token-dense `<think>` blocks** (reducing reasoning tokens by ~50%) while maintaining strict `<think>...</think>` format compliance and preserving mathematical accuracy without system prompt regurgitation.

## 🛠️ Code Source & Infrastructure
- **GitHub Repository:** [Hari31416/qwen-grug-finetune](https://github.com/Hari31416/qwen-grug-finetune.git)
- **Frameworks Used:** PyTorch, Hugging Face `transformers`, `peft` (LoRA), `trl` (`SFTTrainer`), `bitsandbytes` (4-bit NF4 quantization).
- **Target Hardware:** 2x NVIDIA T4 GPUs (Kaggle / Google Colab CUDA environment).

## 📊 Data Source
- **Hugging Face Dataset Repository:** [hari31416/qwen-grug-finetune](https://huggingface.co/datasets/hari31416/qwen-grug-finetune) (1,701 stratified reasoning samples across StrategyQA, LogiQA, BoolQ, ANLI, PIQA, and ReClor).
- **Evaluation Benchmark:** [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (Grade School Math reasoning test split).

---

### Notebook Execution Workflow:
1. **Centralized Hyperparameters**: Configure SFT experimental parameters (epochs, batch size, learning rates, limits).
2. **Environment & Data Setup**: Clone repository, install dependencies, and download SFT dataset splits.
3. **Base Model Inference**: Load 4-bit NF4 quantized base model and run sample generation.
4. **SFT Fine-Tuning Execution**: Execute `run_sft_training()` directly in the notebook kernel.
5. **Loss Visualization**: Plot Loss Curves and Learning Rate decay schedule using `plot_latest_training_loss()`.
6. **GSM8K Benchmarking**: Benchmark Base Model vs. Fine-Tuned Model using `run_gsm8k_eval()`.
7. **Qualitative Sample Inspection**: Print side-by-side reasoning traces before and after SFT.
8. **EDA Dashboard**: Plot comparative metrics for Accuracy (%) and Token Count Breakdown.
9. **Export Artifacts Package**: Zip trained adapters, evaluation JSONs, and plot images into a downloadable ZIP file."""
    )

    add_md("## 1. Centralized SFT Hyperparameters & Configuration")
    add_code(
        """# ==========================================
# ⚙️ SFT EXPERIMENTAL HYPERPARAMETERS & CONFIG
# ==========================================

REPO_URL = "https://github.com/Hari31416/qwen-grug-finetune.git"
REPO_NAME = "qwen-grug-finetune"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATA_DIR = "data"
ADAPTER_OUTPUT_DIR = "adapters"

# Training Hyperparameters
EPOCHS = 1                # 1 epoch for optimal style adaptation without overfitting
TRAIN_BATCH_SIZE = 1      # Per-device batch size (1 max VRAM headroom on T4 GPUs)
GRAD_ACCUM = 8            # Gradient accumulation steps (Effective batch size = 1 * 8 = 8)
LEARNING_RATE = 2e-4      # Peak SFT learning rate
MAX_SEQ_LENGTH = 1536     # Max token sequence length for training
LORA_R = 16               # LoRA rank dimension
LORA_ALPHA = 32           # LoRA alpha scaling factor

# Benchmark Evaluation Hyperparameters
EVAL_LIMIT = None         # Set to None for FULL benchmark evaluation (all 1,000 test samples), or set e.g. 50 for quick debugging
EVAL_BATCH_SIZE = 1       # Per-device evaluation batch size
EVAL_MAX_TOKENS = 1024     # Max generation tokens per GSM8K problem"""
    )

    add_md("## 2. Environment Setup & Data Download")
    add_code(
        """import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

%pip install -q peft trl bitsandbytes datasets accelerate huggingface_hub matplotlib seaborn pandas pyyaml 'torchao>=0.16.0'"""
    )

    add_code(
        """import sys
if not os.path.exists("scripts") and not os.path.exists(f"{REPO_NAME}/scripts"):
    print(f"Cloning {REPO_URL} into workspace...")
    !git clone {REPO_URL}
    if os.path.exists(REPO_NAME):
        %cd {REPO_NAME}
elif os.path.exists(REPO_NAME) and os.path.exists(f"{REPO_NAME}/scripts"):
    %cd {REPO_NAME}

sys.path.append(".")
from scripts.cuda.cuda_utils import patch_transformers_lazy_imports
from scripts.cuda.download_data import download_hf_data

patch_transformers_lazy_imports()
download_hf_data(output_dir=DATA_DIR)"""
    )

    add_md("## 3. Base Model Inference")
    add_code(
        """import torch
from transformers import BitsAndBytesConfig
from scripts.cuda.cuda_utils import load_causal_lm_model, load_causal_lm_tokenizer
from scripts.cuda.generate_cuda import generate_response

print("Loading Base Model:", MODEL_ID)
is_cuda = torch.cuda.is_available()
model_kwargs = {}
if is_cuda:
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs["quantization_config"] = bnb_config
    model_kwargs["device_map"] = "auto"
    model_kwargs["torch_dtype"] = torch.float16

tokenizer = load_causal_lm_tokenizer(MODEL_ID)
model = load_causal_lm_model(MODEL_ID, **model_kwargs)

sample_prompt = "Josh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?"
print("\\n--- Base Model Sample Response ---")
print(generate_response(model, tokenizer, sample_prompt, max_new_tokens=512))"""
    )

    add_md("## 4. Execute SFT QLoRA Fine-Tuning")
    add_code(
        """from scripts.cuda.train_cuda import run_sft_training

print("Starting SFT QLoRA Fine-Tuning...")
sft_trainer = run_sft_training(
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
    tokenizer=tokenizer
)"""
    )

    add_md("## 5. Plot Training Loss Curves")
    add_code(
        """from scripts.cuda.plot_loss import plot_latest_training_loss

print("Plotting Training & Validation Loss...")
plot_latest_training_loss()"""
    )

    add_md("## 6. GSM8K Benchmark Evaluation (Base vs. SFT)")
    add_code(
        """import gc
import torch
from scripts.cuda.eval_cuda import run_gsm8k_eval
from peft import PeftModel

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print("Evaluating Base Model...")
base_summary = run_gsm8k_eval(model, tokenizer, limit=EVAL_LIMIT, batch_size=EVAL_BATCH_SIZE)

latest_adapter = os.path.join(ADAPTER_OUTPUT_DIR, "deepseek-r1-7b/20260804_040058/final_adapters")
if os.path.exists(latest_adapter):
    print("\\nEvaluating SFT Fine-Tuned Model...")
    ft_model = PeftModel.from_pretrained(model, latest_adapter)
    if torch.cuda.is_available():
        try:
            ft_model = ft_model.to("cuda")
        except Exception:
            pass
    ft_summary = run_gsm8k_eval(ft_model, tokenizer, limit=EVAL_LIMIT, batch_size=EVAL_BATCH_SIZE, is_adapter=True)"""
    )

    add_md("## 7. Qualitative Reasoning Trace Inspection")
    add_code(
        """import json

b_file = "results/deepseek-r1-7b/baseline/gsm8k.json"
f_file = "results/deepseek-r1-7b/finetuned/gsm8k.json"

if os.path.exists(b_file) and os.path.exists(f_file):
    with open(b_file) as f:
        b_data = json.load(f)["results"]
    with open(f_file) as f:
        f_data = json.load(f)["results"]

    print("=======================================================")
    print("🔍 BEFORE vs AFTER SFT REASONING COMPARISON")
    print("=======================================================")
    for i in range(min(3, len(b_data))):
        b_item, f_item = b_data[i], f_data[i]
        print(f"\\n--- Sample {i+1} ---")
        print("Question:", b_item["question"])
        print(f"[BASE] Think: {b_item['thinking_tokens']} tok | Answer: {b_item['answer_tokens']} tok")
        print("Thinking:", b_item["thinking_content"])
        print(f"[SFT] Think: {f_item['thinking_tokens']} tok | Answer: {f_item['answer_tokens']} tok")
        print("Thinking:", f_item["thinking_content"])
        print("Answer:", f_item["answer_content"])
        print("-" * 55)"""
    )

    add_md("## 8. Comparative Performance Dashboard")
    add_code(
        """import matplotlib.pyplot as plt
import numpy as np

def get_summary(p):
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f).get("summary")

b_s = get_summary(b_file)
f_s = get_summary(f_file)

if b_s and f_s:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cats = ["7B Base", "7B Fine-Tuned"]
    colors = ["#4A90E2", "#50E3C2"]

    # Accuracy
    accs = [b_s["accuracy"] * 100, f_s["accuracy"] * 100]
    bars1 = ax1.bar(cats, accs, color=colors, width=0.45)
    ax1.set_title("GSM8K Accuracy (%)", fontsize=12, fontweight="bold", pad=15)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Tokens
    think_t = [b_s["mean_thinking_tokens"], f_s["mean_thinking_tokens"]]
    ans_t = [b_s["mean_answer_tokens"], f_s["mean_answer_tokens"]]
    ax2.bar(cats, think_t, label="Thinking Tokens", color="#4A90E2", width=0.45)
    ax2.bar(cats, ans_t, bottom=think_t, label="Answer Tokens", color="#B8E986", width=0.45)
    ax2.set_title("Token Count Breakdown", fontsize=12, fontweight="bold", pad=15)
    ax2.set_ylabel("Average Tokens", fontsize=11)
    ax2.set_ylim(0, 320)
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    for idx, (t, a) in enumerate(zip(think_t, ans_t)):
        tot = t + a
        ax2.annotate(f"Total: {int(tot)}", xy=(idx, tot), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plot_p = os.path.join(ADAPTER_OUTPUT_DIR, "eda_sft_dashboard.png")
    plt.savefig(plot_p, dpi=150, bbox_inches="tight")
    print("Saved dashboard to:", plot_p)
    plt.show()"""
    )

    add_md("## 9. Export Artifacts Zip Package")
    add_code(
        """import zipfile

ZIP_FILE = "kaggle_sft_artifacts.zip"
print(f"Creating downloadable artifacts package: {ZIP_FILE}...")
with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zipf:
    for target in ["adapters", "results"]:
        if os.path.exists(target):
            for root, dirs, files in os.walk(target):
                for file in files:
                    fp = os.path.join(root, file)
                    zipf.write(fp, os.path.relpath(fp, "."))

if os.path.exists(ZIP_FILE):
    size_mb = os.path.getsize(ZIP_FILE) / (1024 * 1024)
    print(f"\\n✅ Packaged artifacts into '{ZIP_FILE}' ({size_mb:.2f} MB)")"""
    )

    save_notebook(cells, "kaggle_sft_pipeline.ipynb")


# ==============================================================================
# NOTEBOOK 2: DPO PREFERENCE OPTIMIZATION PIPELINE
# ==============================================================================
def generate_dpo_notebook():
    cells, add_md, add_code = create_cell_helpers()

    # Top Standalone Overview Cell
    add_md(
        """# Direct Preference Optimization (DPO) Pipeline (DeepSeek-R1-7B)

## 🎯 What Are We Doing?
We are running Direct Preference Optimization (DPO) on top of the SFT-aligned `DeepSeek-R1-Distill-Qwen-7B` model using preference pairs (`chosen` vs `rejected`), followed by loss plotting, benchmark evaluation, qualitative sample inspection, comparative performance plotting, and artifact export.

## 💡 Why Are We Doing It?
While SFT teaches formatting and telegraphic syntax, DPO directly optimizes the model's preference margin to reward concise, accurate reasoning (`chosen`) over verbose or error-prone derivations (`rejected`), ensuring maximum brevity without sacrificing math precision.

## 🛠️ Code Source & Infrastructure
- **GitHub Repository:** [Hari31416/qwen-grug-finetune](https://github.com/Hari31416/qwen-grug-finetune.git)
- **Frameworks Used:** PyTorch, Hugging Face `transformers`, `peft` (LoRA), `trl` (`DPOTrainer`), `bitsandbytes` (4-bit NF4 quantization).
- **Target Hardware:** 2x NVIDIA T4 GPUs (Kaggle / Google Colab CUDA environment).

## 📊 Data Source
- **Hugging Face Dataset Repository:** [hari31416/qwen-grug-finetune](https://huggingface.co/datasets/hari31416/qwen-grug-finetune) (Preference dataset splits containing JSONL rows with `prompt`, `chosen`, and `rejected`).
- **Evaluation Benchmark:** [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) (Grade School Math reasoning test split).

---

### Notebook Execution Workflow:
1. **DPO Hyperparameters**: Set preference optimization learning rate (`5e-7`), KL penalty (`beta=0.1`), and batch sizes.
2. **Environment & Dataset Setup**: Clone repository, load preference data (`data/dpo/train.jsonl`).
3. **Execute DPO Training**: Run `run_dpo_training()` using Hugging Face TRL `DPOTrainer`.
4. **Plot Training Loss Curves**: Visualize training loss and learning rate decay using `plot_latest_training_loss()`.
5. **DPO GSM8K Evaluation**: Benchmark DPO model on GSM8K reasoning dataset.
6. **Qualitative Preference Inspection**: Inspect how DPO further refines telegraphic brevity and accuracy.
7. **Comparative Performance Dashboard**: Plot comparative charts for Accuracy (%) and Token Breakdown (Base vs. SFT vs. DPO).
8. **Export DPO Package**: Package DPO adapters and evaluation results into a downloadable ZIP archive."""
    )

    add_md("## 1. Centralized DPO Hyperparameters & Configuration")
    add_code(
        """# ==========================================
# ⚙️ DPO EXPERIMENTAL HYPERPARAMETERS & CONFIG
# ==========================================

REPO_URL = "https://github.com/Hari31416/qwen-grug-finetune.git"
REPO_NAME = "qwen-grug-finetune"
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
SFT_ADAPTER_PATH = "adapters/deepseek-r1-7b/20260804_040058/final_adapters"
DPO_DATA_DIR = "data/dpo"
DPO_OUTPUT_DIR = "adapters/deepseek-r1-7b/dpo"

# DPO Training Hyperparameters
DPO_EPOCHS = 1             # 1-2 epochs for preference optimization
DPO_BATCH_SIZE = 1         # Per-device batch size
DPO_GRAD_ACCUM = 8         # Gradient accumulation steps
DPO_LEARNING_RATE = 5e-7   # Learning rate (100x smaller than SFT)
DPO_BETA = 0.1             # KL divergence penalty weight
MAX_LENGTH = 1536          # Maximum sequence length
MAX_PROMPT_LENGTH = 512    # Maximum prompt length

EVAL_LIMIT = None         # Set to None for FULL benchmark evaluation (all 1,000 test samples), or set e.g. 50 for quick debugging
EVAL_BATCH_SIZE = 1        # Evaluation batch size"""
    )

    add_md("## 2. Environment Setup & Preference Data Verification")
    add_code(
        """import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

%pip install -q peft trl bitsandbytes datasets accelerate huggingface_hub matplotlib seaborn pandas pyyaml 'torchao>=0.16.0'"""
    )

    add_code(
        """import sys
if not os.path.exists("scripts") and not os.path.exists(f"{REPO_NAME}/scripts"):
    print(f"Cloning {REPO_URL} into workspace...")
    !git clone {REPO_URL}
    if os.path.exists(REPO_NAME):
        %cd {REPO_NAME}
elif os.path.exists(REPO_NAME) and os.path.exists(f"{REPO_NAME}/scripts"):
    %cd {REPO_NAME}

sys.path.append(".")
from scripts.cuda.cuda_utils import patch_transformers_lazy_imports
from scripts.cuda.download_data import download_hf_data
from scripts.create_dpo_dataset import generate_dpo_dataset

patch_transformers_lazy_imports()

# 1. Download base dataset from Hugging Face repository
print("Downloading dataset files from Hugging Face (hari31416/qwen-grug-finetune)...")
download_hf_data(output_dir="data")

# 2. Check and auto-generate DPO dataset from downloaded SFT data if missing
train_file = os.path.join(DPO_DATA_DIR, "train.jsonl")
if not os.path.exists(train_file):
    print(f"Generating DPO dataset at '{DPO_DATA_DIR}' from downloaded SFT format data...")
    generate_dpo_dataset(data_dir="data", dpo_dir=DPO_DATA_DIR)
else:
    print(f"✅ Found existing DPO dataset: {train_file}")"""
    )

    add_md("## 3. Execute DPO Fine-Tuning (`DPOTrainer`)")
    add_code(
        """from scripts.cuda.dpo_cuda import run_dpo_training

print("Starting DPO Fine-Tuning...")
dpo_trainer = run_dpo_training(
    model_arg=MODEL_ID,
    adapter_path=SFT_ADAPTER_PATH,
    dpo_data_dir=DPO_DATA_DIR,
    output_dir=DPO_OUTPUT_DIR,
    epochs=DPO_EPOCHS,
    batch_size=DPO_BATCH_SIZE,
    grad_accum=DPO_GRAD_ACCUM,
    learning_rate=DPO_LEARNING_RATE,
    beta=DPO_BETA,
    max_length=MAX_LENGTH,
    max_prompt_length=MAX_PROMPT_LENGTH,
)"""
    )

    add_md("## 4. Plot Training Loss Curves")
    add_code(
        """from scripts.cuda.plot_loss import plot_latest_training_loss

print("Plotting DPO Training Loss & Learning Rate Schedule...")
plot_latest_training_loss()"""
    )

    add_md("## 5. Benchmark Evaluation on DPO Model")
    add_code(
        """import gc
import torch
import glob
from scripts.cuda.cuda_utils import load_causal_lm_model, load_causal_lm_tokenizer
from scripts.cuda.eval_cuda import run_gsm8k_eval
from peft import PeftModel

gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()

is_cuda = torch.cuda.is_available()
model_kwargs = {}
if is_cuda:
    from transformers import BitsAndBytesConfig
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    model_kwargs["quantization_config"] = bnb_config
    model_kwargs["device_map"] = {"": 0}
    model_kwargs["torch_dtype"] = torch.float16

tokenizer = load_causal_lm_tokenizer(MODEL_ID)
model = load_causal_lm_model(MODEL_ID, **model_kwargs)

dpo_matches = glob.glob(os.path.join(DPO_OUTPUT_DIR, "**/final_dpo_adapters"), recursive=True)
if dpo_matches:
    dpo_adapter_path = sorted(dpo_matches)[-1]
    print(f"Evaluating DPO Model on GSM8K Benchmark from '{dpo_adapter_path}'...")
    dpo_model = PeftModel.from_pretrained(model, dpo_adapter_path)
    dpo_summary = run_gsm8k_eval(dpo_model, tokenizer, limit=EVAL_LIMIT, batch_size=EVAL_BATCH_SIZE, is_adapter=True, output_subfolder="dpo")
else:
    print(f"DPO adapter path inside '{DPO_OUTPUT_DIR}' not found.")"""
    )

    add_md("## 6. Qualitative Reasoning Trace Inspection")
    add_code(
        """import json

d_file = "results/deepseek-r1-7b/dpo/gsm8k.json"

if os.path.exists(d_file):
    with open(d_file) as f:
        d_data = json.load(f)["results"]

    print("=======================================================")
    print("🔍 DPO REASONING TRACE INSPECTION")
    print("=======================================================")
    for i in range(min(3, len(d_data))):
        d_item = d_data[i]
        print(f"\\n--- Sample {i+1} ---")
        print("Question:", d_item["question"])
        print(f"[DPO] Think: {d_item['thinking_tokens']} tok | Answer: {d_item['answer_tokens']} tok | Correct: {d_item['correct']}")
        print("Thinking:", d_item["thinking_content"])
        print("Answer:", d_item["answer_content"])
        print("-" * 55)"""
    )

    add_md("## 7. Comparative Performance Dashboard")
    add_code(
        """import matplotlib.pyplot as plt
import numpy as np

def get_summary(p):
    if not os.path.exists(p): return None
    with open(p) as f: return json.load(f).get("summary")

b_s = get_summary("results/deepseek-r1-7b/baseline/gsm8k.json")
s_s = get_summary("results/deepseek-r1-7b/finetuned/gsm8k.json")
d_s = get_summary("results/deepseek-r1-7b/dpo/gsm8k.json")

cats, accs, think_t, ans_t, colors = [], [], [], [], []

if b_s:
    cats.append("7B Base")
    accs.append(b_s["accuracy"] * 100)
    think_t.append(b_s["mean_thinking_tokens"])
    ans_t.append(b_s["mean_answer_tokens"])
    colors.append("#4A90E2")

if s_s:
    cats.append("7B SFT")
    accs.append(s_s["accuracy"] * 100)
    think_t.append(s_s["mean_thinking_tokens"])
    ans_t.append(s_s["mean_answer_tokens"])
    colors.append("#50E3C2")

if d_s:
    cats.append("7B DPO")
    accs.append(d_s["accuracy"] * 100)
    think_t.append(d_s["mean_thinking_tokens"])
    ans_t.append(d_s["mean_answer_tokens"])
    colors.append("#F5A623")

if cats:
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy Chart
    bars1 = ax1.bar(cats, accs, color=colors, width=0.45)
    ax1.set_title("GSM8K Accuracy (%)", fontsize=12, fontweight="bold", pad=15)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Tokens Chart
    ax2.bar(cats, think_t, label="Thinking Tokens", color="#4A90E2", width=0.45)
    ax2.bar(cats, ans_t, bottom=think_t, label="Answer Tokens", color="#B8E986", width=0.45)
    ax2.set_title("Token Count Breakdown", fontsize=12, fontweight="bold", pad=15)
    ax2.set_ylabel("Average Tokens", fontsize=11)
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    for idx, (t, a) in enumerate(zip(think_t, ans_t)):
        tot = t + a
        ax2.annotate(f"Total: {int(tot)}", xy=(idx, tot), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plot_p = os.path.join(DPO_OUTPUT_DIR, "eda_dpo_dashboard.png")
    plt.savefig(plot_p, dpi=150, bbox_inches="tight")
    print("Saved DPO dashboard to:", plot_p)
    plt.show()"""
    )

    add_md("## 8. Export DPO Artifacts Package")
    add_code(
        """import zipfile

ZIP_FILE = "kaggle_dpo_artifacts.zip"
print(f"Creating downloadable DPO artifacts package: {ZIP_FILE}...")
with zipfile.ZipFile(ZIP_FILE, "w", zipfile.ZIP_DEFLATED) as zipf:
    for target in ["adapters", "results"]:
        if os.path.exists(target):
            for root, dirs, files in os.walk(target):
                for file in files:
                    fp = os.path.join(root, file)
                    zipf.write(fp, os.path.relpath(fp, "."))

if os.path.exists(ZIP_FILE):
    size_mb = os.path.getsize(ZIP_FILE) / (1024 * 1024)
    print(f"\\n✅ Packaged DPO artifacts into '{ZIP_FILE}' ({size_mb:.2f} MB)")"""
    )

    save_notebook(cells, "kaggle_dpo_pipeline.ipynb")


if __name__ == "__main__":
    generate_sft_notebook()
    generate_dpo_notebook()
