# Grug Reasoning Fine-Tune

Fine-tune **DeepSeek-R1-Distill-Qwen-1.5B** on Apple Silicon using MLX to learn telegraphic, token-efficient Grug style chain-of-thought reasoning without sacrificing task accuracy.

The project is structured around iterating on SFT data scale, validation checks, and regularization to find the optimal trade-offs for terse, efficient reasoning blocks.

## Project Structure

The project has the following directory layout:

- `adapters/` - LoRA adapter weights (gitignored). Contains timestamped training run directories (e.g. `20260704_121944/`).
- `data/` - Holds SFT prompts, raw traces, compressed traces, and formatted SFT splits (gitignored).
- `data-and-models/` - Structured staging directory for Hugging Face uploads (gitignored).
- `report/` - Compiled experimental plots, raw evaluation JSON configurations, and markdown write-ups for each iteration.
- `results/` - Detailed evaluation outputs, generated samples, and metrics (gitignored).
- `scripts/` - Pipeline Python scripts for trace generation, compression, training, evaluation, plotting, and syncing.
- `SUMMARY.md` - Comprehensive summary of experimental findings, training strategies, and future scaling plans.
- `STORY.md` - The narrative behind the project, motivating factors, experimental journey, and key takeaways.

> [!NOTE]
> Since the `data/`, `adapters/`, and `data-and-models/` directories are gitignored, you can download all training datasets, adapter weights, and evaluation reports directly from the Hugging Face repository: [hari31416/qwen-grug-finetune](https://huggingface.co/hari31416/qwen-grug-finetune).

## Configuration

The model targets and training hyperparameters are defined in:

- `config.yaml` - Model paths, dataset sizes, and output paths.
- `lora_config.yaml` - Shared LoRA training parameters.
- `style_guide.md` - Rules and before/after examples for Grug-style CoT traces (loaded by `compress_traces.py`).

## Getting Started

### 1. Installation

Initialize and sync the virtual environment using `uv`:

```bash
uv sync
```

Verify that MLX has Apple Silicon GPU access:

```bash
uv run python -c "import mlx.core as mx; print(mx.default_device())"
```

Setup environment variables by copying `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL` for the compression API.

### 2. Staging and Hugging Face Upload

The data, model adapters, and reports across all iterations can be staged and synced to the Hugging Face Hub:

```bash
# Stage files locally inside data-and-models/
uv run python scripts/sync_hf.py

# Stage and upload directly to Hugging Face
uv run python scripts/sync_hf.py --push
```

## Pipeline Execution Reference

You can use target `make` commands to quickly trigger tasks (which allow CLI overrides for `ITERS`, `LIMIT`, `BATCH_SIZE`, etc.):

```bash
# View list of available commands
make help

# 1. Run LoRA SFT Training (Default: 1000 iterations)
make train
make train ITERS=100

# 2. Run Baseline Model Evaluation
make eval-base

# 3. Run Fine-Tuned Model Evaluation (automatically searches latest timestamped adapter folder)
make eval-ft

# 4. Generate comparative plots and copy raw JSON configs to report/
make plot

# 5. Clean Python cache files
make clean
```

## Experimental Findings

The compiled experimental summary write-up and comparison dashboard plots can be viewed under the `report/` directory:

- **Iteration 1 Report**: [report/it-1/REPORT.md](report/it-1/REPORT.md)
- **Iteration 2 Report**: [report/REPORT.md](report/REPORT.md)

### Iteration 1 Metrics (GSM8K Test Split)

| Configuration | Accuracy | Mean Thinking Tokens | Mean Total Tokens | Mean Latency (s) | Format Compliance |
| :------------ | :------: | :------------------: | :---------------: | :--------------: | :---------------: |
| Base Normal   |  64.9%   |        219.0         |       477.4       |      0.88s       |       96.6%       |
| Base Grug     |  67.2%   |        512.8         |       581.1       |      1.21s       |       91.5%       |
| FT Normal     |  66.0%   |        156.2         |       389.3       |      0.73s       |       98.9%       |
| FT Grug       |  45.6%   |        120.0         |       229.0       |      0.64s       |       95.1%       |

### Iteration 2 (Unregularized) Metrics (GSM8K Test Split)

| Configuration             | Accuracy | Mean Thinking Tokens | Mean Total Tokens | Mean Latency (s) | Format Compliance |
| :------------------------ | :------: | :------------------: | :---------------: | :--------------: | :---------------: |
| Base Model (Style Prompt) |  71.5%   |        504.0         |       568.2       |      1.09s       |       92.3%       |
| FT Model (Unregularized)  |  52.7%   |         99.6         |       226.0       |      0.59s       |       99.2%       |

### Iteration 2 (Regularized / Final) Metrics (GSM8K Test Split)

| Configuration             | Accuracy | Mean Thinking Tokens | Mean Total Tokens | Mean Latency (s) | Format Compliance |
| :------------------------ | :------: | :------------------: | :---------------: | :--------------: | :---------------: |
| Base Model (Style Prompt) |  70.1%   |        517.4         |       582.1       |      1.28s       |       91.1%       |
| FT Model (Regularized)    |  54.6%   |        135.0         |       214.7       |      0.61s       |       98.2%       |
