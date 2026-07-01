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

## Phase 5 — Pilot Pipeline (10 examples)

### What We Did

- Switched the base model from `Qwen3.5-0.8B-OptiQ-4bit` to `DeepSeek-R1-Distill-Qwen-1.5B-4bit` as the target model to tune.
- Implemented `scripts/generate_traces.py` to run local inference, extract the reasoning chain, and validate correctness.
- Implemented `scripts/compress_traces.py` to execute asynchronous, concurrent Grug-style trace compression using the Nvidia NIM Integrate API (`openai/gpt-oss-120b`).
- Implemented `scripts/validate_traces.py` to enforce style-guide validation checks (rejecting traces exceeding 50% length or containing answer restatements).
- Implemented `scripts/format_data.py` to format accepted records into train/valid SFT JSONL splits while preserving thinking tags.
- Verified the end-to-end pipeline by running it on the first 10 StrategyQA prompts, yielding 4 accepted, fully formatted training SFT rows.

### Model Change Logic

Initially, we ran the raw trace generation using the target `Qwen3.5-0.8B-OptiQ-4bit` (and the `2B` fallback). Because these are standard instruct models rather than RL-aligned reasoning models, forcing them into a thinking mode via the tokenizer chat template caused them to fail on general knowledge reasoning questions (StrategyQA, LogiQA, etc.). Specifically:
- They lacked the knowledge to resolve the queries, causing them to enter infinite self-correcting loops (e.g. *"Wait, is X the producer? No. Wait..."*) and run out of tokens (hitting the 1024 limit) without ever closing the `<think>` tag. Bulk testing yielded a **0/30 success rate** across all datasets.
- Switching to `DeepSeek-R1-Distill-Qwen-1.5B-4bit` (which was explicitly RL-trained to reason) completely resolved this loop behavior. It natively structures thoughts inside `<think>` tags and reliably emits the closing `</think>` token while maintaining high reasoning accuracy.

### Key Commands Run

- Run local trace generation:

  ```bash
  ./.venv/bin/python scripts/generate_traces.py --limit 10
  ```

- Run asynchronous compression:

  ```bash
  ./.venv/bin/python scripts/compress_traces.py
  ```

- Validate compressed traces:

  ```bash
  ./.venv/bin/python scripts/validate_traces.py --report
  ```

- Format accepted traces into SFT train/valid JSONL files:

  ```bash
  ./.venv/bin/python scripts/format_data.py
  ```

### What Worked

- Appending format constraints (e.g., *"\nAnswer in exactly one word: yes or no."*) dynamically to the user prompts forced clean, single-word final outputs (`Yes`, `No`, option letters) that matched the ground truth perfectly.
- Nvidia NIM API returned clean compressed reasoning in the `content` field which we successfully isolated.
- Bypassing the default tokenizer template's stripping behavior in `format_data.py` by using `add_generation_prompt=True` and manually concatenating the reasoning block ensured the reasoning channel was successfully saved.

### Issues Faced and Resolutions

- **DeepSeek Template Strips `<think>` tags**: By default, the official DeepSeek-R1 JINJA template strips `<think>` blocks from assistant messages in multi-turn chat templates to conserve context length. We resolved this by using the template only up to the generation prompt prefix, manually appending the thinking block, and terminating with `tokenizer.eos_token`.
- **NIM Reasoning Blocks**: When calling NIM, the model's output resides in the `content` field, while its meta-thinking is stored in the `reasoning` and `reasoning_content` fields. We updated our extraction in `compress_traces.py` to prioritize `msg.content` directly to prevent meta-commentary from bleeding into the target.

### Phase 5 follow-up — style guide, validation, and docs

- Wired `style_guide.md` into `compress_traces.py` as the runtime compressor system prompt (replaces hardcoded rules).
- Aligned API env vars with `.env.example`: `OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL` (with `LLM_*` fallback for compatibility).
- Expanded `validate_traces.py` auto-rejection policy:
  - Truncated or incomplete compressions
  - Key-value / label format anti-patterns
  - Dropped numeric facts and multiple-choice option letters
  - Meta filler phrases in compressed output
  - Logic-fragment preservation (filler-stripped raw trace vs compressed)
- Added `validated_traces` path to `config.yaml` / `scripts/config.py`.
- Updated `STEPS.md`, `PLAN.md`, and `README.md` to reflect DeepSeek-R1-Distill-Qwen-1.5B as the permanent target model for CoT generation, SFT, and LoRA.
- Re-ran validation on pilot data: 3/4 accepted, 1 rejected (truncated compression for `strategyqa-0004`).
