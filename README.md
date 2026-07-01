# Grug Reasoning Fine-Tune

Fine-tune **DeepSeek-R1-Distill-Qwen-1.5B** on Apple Silicon using MLX to learn telegraphic, token-efficient Grug style chain-of-thought reasoning without sacrificing task accuracy.

> Phase 5 pilot pipeline complete. Next: full 1k data pipeline (Phase 6).

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
- `style_guide.md` - Rules and before/after examples for Grug-style CoT traces (loaded by `compress_traces.py`).

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

Set `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL` for the compression API.

4. Verify path resolution and scaffold directory folders:

```bash
python scripts/config.py
```

## Pilot pipeline

```bash
python scripts/generate_traces.py --limit 10
python scripts/compress_traces.py --limit 10
python scripts/validate_traces.py --report
python scripts/format_data.py
```
