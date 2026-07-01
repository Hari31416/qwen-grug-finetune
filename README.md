# Grug Reasoning Fine-Tune

Fine-tune **DeepSeek-R1-Distill-Qwen-1.5B** on Apple Silicon using MLX to learn telegraphic, token-efficient Grug style chain-of-thought reasoning without sacrificing task accuracy.

> SFT training and baseline/fine-tuned evaluations (Phases 1-9) are complete. Preliminary benchmark plots and experimental findings are compiled in the [report/](./report/) directory.

---

## Project Structure

The project has the following directory layout:

- `data/` - Holds prompts, raw traces, compressed traces, and formatted SFT splits.
- `scripts/` - Pipeline Python scripts for trace generation, compression, training, evaluation, and plotting.
- `adapters/` - LoRA adapter weights (gitignored). Contains timestamped training run directories (e.g. `20260701_210744/`).
- `results/` - Evaluation outputs and reports (gitignored).
- `report/` - Compiled experimental plots, raw evaluation JSON configurations, and the markdown write-up.

---

## Configuration

The model targets and training hyperparameters are defined in:

- `config.yaml` - Model paths, dataset sizes, and output paths.
- `lora_config.yaml` - Shared LoRA training parameters.
- `style_guide.md` - Rules and before/after examples for Grug-style CoT traces (loaded by `compress_traces.py`).

---

## Getting Started

### 1. Installation

Create a Python virtual environment and install requirements:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

Verify that MLX has Apple Silicon GPU access:

```bash
python -c "import mlx.core as mx; print(mx.default_device())"
```

Setup environment variables by copying `.env.example` to `.env`:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY`, `OPENAI_API_BASE`, and `OPENAI_MODEL` for the compression API.

---

## Pipeline Execution Reference

You can use target `make` commands to quickly trigger tasks (which allow CLI overrides for `ITERS`, `LIMIT`, `BATCH_SIZE`, etc.):

```bash
# View list of available commands
make help

# 1. Run LoRA SFT Training (Default: 300 iterations)
make train
make train ITERS=100

# 2. Run Baseline Model Evaluation
make eval-base-normal
make eval-base-grug

# 3. Run Fine-Tuned Model Evaluation (automatically searches/resolves latest timestamped adapter folder)
make eval-ft-normal
make eval-ft-grug
make eval-ft-grug LIMIT=200 BATCH_SIZE=32   # custom override example

# 4. Run all evaluations sequentially
make eval-all

# 5. Generate comparative plots and copy raw JSON configs to report/
make plot

# 6. Clean Python cache files
make clean
```

---

## Experimental Findings

The compiled experimental summary write-up and comparison dashboard plots can be viewed under the [report/](./report/) directory:
* **Write-up:** [report/REPORT.md](./report/REPORT.md)
* **Deltas Plot:** [report/deltas.png](./report/deltas.png)
* **Dashboard Plot:** [report/dashboard.png](./report/dashboard.png)
