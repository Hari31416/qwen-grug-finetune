# CUDA / Kaggle / Colab Scripts (2x NVIDIA T4 Support)

This directory contains PyTorch and Hugging Face scripts designed for CUDA environments (such as Kaggle Notebooks, Google Colab, or Cloud GPU instances with 2x NVIDIA T4 GPUs).

Unlike Apple Silicon scripts which use `mlx_lm`, these scripts utilize `torch`, `transformers`, `peft`, `trl`, and `bitsandbytes` (4-bit QLoRA) to train, evaluate, and run generation on NVIDIA GPUs.

## Scripts Overview

- **`download_data.py`**: Downloads dataset splits (`train.jsonl` and `valid.jsonl`) directly from the Hugging Face repository (`hari31416/qwen-grug-finetune`).
- **`train_cuda.py`**: QLoRA SFT fine-tuning on 2x T4 GPUs using Hugging Face `SFTTrainer` and `bitsandbytes`. Exposes `run_sft_training()`.
- **`eval_cuda.py`**: GSM8K benchmark evaluation on base or fine-tuned LoRA models. Exposes `run_gsm8k_eval()`.
- **`generate_cuda.py`**: Single-prompt generation script for quick inference on CUDA GPUs. Exposes `generate_response()`.
- **`plot_loss.py`**: Visualizes loss curves, learning rate schedule, and benchmark performance metrics.

---

## Interactive Jupyter Notebook Workflow (Recommended)

An interactive notebook **[notebooks/kaggle_grug_finetune.ipynb](../../notebooks/kaggle_grug_finetune.ipynb)** is available for execution in Kaggle or Colab without needing shell commands.

### 1. Open Notebook in Kaggle / Colab

1. **Kaggle**: Select Accelerator **GPU T4 x2** and turn Internet **On**.
2. Open `notebooks/kaggle_grug_finetune.ipynb`.

### 2. Centralized Experimental Parameters (Top Cell)

Modify hyperparameters in the first configuration cell:

```python
# Model & Environment Paths
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
DATA_DIR = "data"
ADAPTER_OUTPUT_DIR = "adapters"

# Training Hyperparameters
EPOCHS = 1                # Training epochs
TRAIN_BATCH_SIZE = 2      # Per-device batch size
GRAD_ACCUM = 4            # Gradient accumulation steps
LEARNING_RATE = 2e-4      # Learning rate
MAX_SEQ_LENGTH = 1536     # Max token length
LORA_R = 16               # LoRA rank
LORA_ALPHA = 32           # LoRA alpha

# Evaluation Hyperparameters
EVAL_LIMIT = 50           # Sample limit for fast evaluation
EVAL_BATCH_SIZE = 2       # Per-device evaluation batch size
EVAL_MAX_TOKENS = 1024     # Max generation tokens
```

### 3. Execution via Direct Python Imports

```python
# 1. Download SFT Dataset
from scripts.cuda.download_data import download_hf_data
download_hf_data(output_dir=DATA_DIR)

# 2. SFT Fine-Tuning
from scripts.cuda.train_cuda import run_sft_training
trainer = run_sft_training(
    model_arg=MODEL_ID, data_dir=DATA_DIR, adapter_path=ADAPTER_OUTPUT_DIR,
    epochs=EPOCHS, batch_size=TRAIN_BATCH_SIZE, grad_accum=GRAD_ACCUM,
    learning_rate=LEARNING_RATE, max_seq_length=MAX_SEQ_LENGTH
)

# 3. Plot Training & Validation Loss
from scripts.cuda.plot_loss import plot_latest_training_loss
plot_latest_training_loss()

# 4. GSM8K Benchmark Evaluation
from scripts.cuda.eval_cuda import run_gsm8k_eval
base_metrics = run_gsm8k_eval(model, tokenizer, limit=EVAL_LIMIT, batch_size=EVAL_BATCH_SIZE)
ft_metrics = run_gsm8k_eval(ft_model, tokenizer, limit=EVAL_LIMIT, batch_size=EVAL_BATCH_SIZE, is_adapter=True)
```

---

## Command Line (CLI) Workflow

### 1. Download Dataset

```bash
!python scripts/cuda/download_data.py --output-dir data
```

### 2. Fine-Tune Model on 2x T4 GPUs

```bash
!python scripts/cuda/train_cuda.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --data data \
  --epochs 1 \
  --batch-size 2 \
  --grad-accum 4
```

### 3. Plot Training Loss

```python
!python scripts/cuda/plot_loss.py
```

### 4. Evaluate Base and Fine-Tuned Models

```python
!python scripts/cuda/eval_cuda.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --benchmark gsm8k \
  --limit 50
```

### 5. Compress Adapters for Download

```bash
!zip -r fine_tuned_adapters.zip adapters/
```

```bash
!zip -r fine_tuned_adapters.zip adapters/
```

## CLI Usage Reference

### Download Dataset

```bash
python scripts/cuda/download_data.py --output-dir data/
```

### Training

```bash
python scripts/cuda/train_cuda.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --data data \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4
```

### Base Model Evaluation

```bash
python scripts/cuda/eval_cuda.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --benchmark gsm8k \
  --limit 100 \
  --batch-size 4
```

### Fine-Tuned Model Evaluation

```bash
python scripts/cuda/eval_cuda.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --adapter \
  --adapter-path adapters/20260802_223000/final_adapters \
  --limit 100
```

### Sample Generation

```bash
python scripts/cuda/generate_cuda.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --prompt "If a train travels 60 mph for 2.5 hours, how far does it go?"
```
