# Grug Reasoning Fine-Tune Execution Results

## Phase 2 — Local Inference Smoke Test

### What We Did

- Configured and executed a local generation test using Qwen 3.5 0.8B to verify Apple Silicon GPU support and reasoning capabilities.
- Created a temporary script (`scripts/prepare_smoke_data.py`) to generate a small mock SFT dataset with 5 training and validation rows in JSONL format using the `text` field.
- Ran a 10-iteration LoRA training job using `mlx_lm.lora` on the mock SFT dataset.
- Verified that the fine-tuning adapter weights (`adapters.safetensors`) were successfully generated.
- Cleaned up the temporary datasets, helper scripts, and temporary adapter directory.

### Key Commands Run

- Verify MLX device is active:

  ```bash
  ./.venv/bin/python -c "import mlx.core as mx; print(mx.default_device())"
  ```

- Run local generation inference test (with repetition & presence penalties):

  ```bash
  ./.venv/bin/python scripts/generate.py \
    --prompt "If John has 3 apples and buys 2 more, how many does he have?" \
    --repetition-penalty 1.1 \
    --presence-penalty 0.2 \
    --max-tokens 500
  ```

- Create mock SFT dataset:

  ```bash
  ./.venv/bin/python scripts/prepare_smoke_data.py
  ```

- Run tiny LoRA training smoke test:

  ```bash
  ./.venv/bin/mlx_lm.lora \
    --model mlx-community/Qwen3.5-0.8B-OptiQ-4bit \
    --config lora_config.yaml \
    --data data/temp_smoke_test \
    --train \
    --iters 10 \
    --batch-size 2 \
    --adapter-path adapters/temp_smoke_test
  ```

### What Worked

- Automated download and caching of the `mlx-community/Qwen3.5-0.8B-OptiQ-4bit` model from Hugging Face.
- Generation of the reasoning/thinking block (`<think>...</think>`) and final answer.
- Generation of `adapters.safetensors` on the M4 GPU.

### Issues Faced and Resolutions

- **Inference repetition/looping:** When running with temperature `0` or default parameters, the model got stuck in a repetitive loop inside the thinking process block. Running the inference with standard temperature (`0.6`) and `top_p` (`0.95`) combined with `repetition_penalty: 1.1` and `presence_penalty: 0.2` completely resolved this issue, yielding a clean CoT trace and correct final answer. We built a custom inference runner (`scripts/generate.py`) supporting these parameters.
- **LoRA configuration scale KeyError:** Executing the training command resulted in a traceback (`KeyError: 'scale'`). Newer versions of `mlx_lm` require the `scale` parameter in the `lora_parameters` section of `lora_config.yaml`. Adding `scale: 2.0` (computed as `alpha / rank` = `16 / 8`) resolved the training crash.

## Phase 4 — Sample SFT Prompts

### What We Did

- Compiled a benchmark question blocklist from the test split of `openai/gsm8k` and validation/test splits of `allenai/ai2_arc` to prevent data leakage.
- Created the dataset sampling and validation script `scripts/sample_sft_prompts.py`.
- Downloaded and processed six source datasets (StrategyQA, LogiQA, BoolQ, ANLI, PIQA, and ReClor), normalizing questions and applying a Jaccard similarity filter (threshold of 0.85) against the benchmark blocklist.
- Sampled exactly 1,000 prompts stratified according to the target dataset sizes defined in `config.yaml`.
- Verified the generated prompts file (`data/sft/prompts.jsonl`) for schema compliance, exact sample counts, and formatting style.

### Key Commands Run

- Execute the prompt sampling script:

  ```bash
  ./.venv/bin/python scripts/sample_sft_prompts.py
  ```

### What Worked

- Successful integration with the `datasets` 5.0.0 library using `refs/convert/parquet` to bypass script deprecations for LogiQA.
- Jaccard similarity token verification successfully protected the SFT dataset from any benchmark contamination.
- Full count verification and formatting checks succeeded, generating exactly 1,000 valid SFT rows conforming to the unified schema.

### Issues Faced and Resolutions

- **Dataset scripts deprecated in datasets 5.0.0:** Standard repositories for LogiQA, PIQA, and ReClor failed to load due to security restrictions on remote Python script execution in datasets 5.0.0. To resolve this, we used community-converted Parquet datasets (`hadithya369/ReClor`, `baber/piqa`) or targeted the auto-converted parquet branch (`revision="refs/convert/parquet"`) for `lucasmccabe/logiqa`.
- **ANLI split name mapping:** The standard `train` split is not available for ANLI, which uses specific round splits. We resolved this by loading and concatenating `train_r1`, `train_r2`, and `train_r3` splits into a unified training pool.
