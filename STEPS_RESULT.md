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

- Run local generation inference test:

  ```bash
  ./.venv/bin/mlx_lm.generate \
    --model mlx-community/Qwen3.5-0.8B-OptiQ-4bit \
    --prompt "If John has 3 apples and buys 2 more, how many does he have?" \
    --max-tokens 500 \
    --temp 0.6 \
    --top-p 0.95
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

- **Inference repetition/looping:** When running with temperature `0` or default parameters, the model got stuck in a repetitive loop inside the thinking process block. Running the inference with the default config settings (temperature `0.6`, `top_p 0.95`) resolved this issue, yielding a clean CoT trace and correct final answer.
- **LoRA configuration scale KeyError:** Executing the training command resulted in a traceback (`KeyError: 'scale'`). Newer versions of `mlx_lm` require the `scale` parameter in the `lora_parameters` section of `lora_config.yaml`. Adding `scale: 2.0` (computed as `alpha / rank` = `16 / 8`) resolved the training crash.
