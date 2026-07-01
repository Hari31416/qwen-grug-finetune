# Grug Reasoning Fine-Tune

Fine-tune Qwen 3.5 models on Apple Silicon using MLX to learn telegraphic, token-efficient Grug style chain-of-thought reasoning without sacrificing task accuracy.

> This project is WIP. Right now, we have completed phase 2.

## Project Structure

The project has the following directory layout:

- `data/` - Holds prompts, raw traces, compressed traces, and formatted SFT splits.
- `scripts/` - Pipeline Python scripts for trace generation, compression, training, and evaluation.
- `adapters/` - LoRA adapter weights (gitignored).
- `results/` - Evaluation outputs and reports (gitignored).

## Configuration

The model targets and training hyperparameters are defined in:

- `config.yaml` - Model paths, dataset sizes, and output paths.
- `lora_config.yaml` - Shared LoRA training parameters.
- `style_guide.md` - Rules and before/after examples for Grug-style CoT traces.

## Getting Started

1. Create a Python virtual environment and install requirements:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

2. Verify that MLX has Apple Silicon GPU access:

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
```

3. Setup environment variables by copying `.env.example` to `.env`:

```bash
cp .env.example .env
```

4. Verify path resolution and scaffold directory folders:

```bash
python scripts/config.py
```
