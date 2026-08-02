# CUDA / Kaggle / Colab Scripts (2x NVIDIA T4 Support)

This directory contains PyTorch and Hugging Face scripts designed for CUDA environments (such as Kaggle Notebooks, Google Colab, or Cloud GPU instances with 2x NVIDIA T4 GPUs).

Unlike Apple Silicon scripts which use `mlx_lm`, these scripts utilize `torch`, `transformers`, `peft`, `trl`, and `bitsandbytes` (4-bit QLoRA) to train, evaluate, and run generation on NVIDIA GPUs.

## Scripts Overview

- **`download_data.py`**: Downloads dataset splits (`train.jsonl` and `valid.jsonl`) directly from the Hugging Face repository (`hari31416/qwen-grug-finetune`).
- **`train_cuda.py`**: QLoRA SFT fine-tuning on 2x T4 GPUs using Hugging Face `SFTTrainer` and `bitsandbytes`.
- **`eval_cuda.py`**: GSM8K benchmark evaluation on base or fine-tuned LoRA models.
- **`generate_cuda.py`**: Single-prompt generation script for quick inference on CUDA GPUs.

## Running in Kaggle or Colab Notebooks

### 1. Notebook Settings

- **Kaggle**: In the right sidebar, select Accelerator **GPU T4 x2** and set Internet to **On**.
- **Google Colab**: Select **Runtime** > **Change runtime type** > **T4 GPU**.

### 2. Clone Repository and Install Dependencies

Execute in the first notebook cell:

```bash
!git clone https://github.com/Hari31416/qwen-grug-finetune.git
%cd qwen-grug-finetune
!pip install -q -U torch transformers peft trl bitsandbytes datasets accelerate huggingface_hub
```

### 3. Download SFT Dataset

Download training and validation datasets from Hugging Face:

```bash
!python scripts/cuda/download_data.py --output-dir data
```

### 4. Fine-Tune Model on 2x T4 GPUs

Run QLoRA SFT fine-tuning (automatically uses 2x T4 GPUs via `device_map="auto"`):

```bash
!python scripts/cuda/train_cuda.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --data data \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4
```

### 5. Evaluate Base and Fine-Tuned Models

Evaluate accuracy, latency, and format compliance on GSM8K:

```python
import os, glob

# Evaluate Base Model
!python scripts/cuda/eval_cuda.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --benchmark gsm8k \
  --limit 100 \
  --batch-size 4

# Resolve latest trained adapter folder
adapter_dirs = glob.glob("adapters/**/final_adapters", recursive=True)
latest_adapter = max(adapter_dirs, key=os.path.getmtime)
print(f"Latest adapter found: {latest_adapter}")

# Evaluate Fine-Tuned Model
!python scripts/cuda/eval_cuda.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --adapter \
  --adapter-path {latest_adapter} \
  --limit 100
```

### 6. Run Sample Generation

Test reasoning and response generation with the fine-tuned adapter:

```python
!python scripts/cuda/generate_cuda.py \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --adapter-path {latest_adapter} \
  --prompt "Solve: If a train travels 60 mph for 2.5 hours, how far does it go?"
```

### 7. Compress and Save Adapters

Zip adapter weights for download from notebook outputs:

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
