# Step-by-Step Execution Plan

Actionable checklist for the Grug reasoning fine-tune project. Background and rationale live in [PLAN.md](./PLAN.md).

**Target model:** DeepSeek-R1-Distill-Qwen-1.5B (`mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit`)  
**Training:** LoRA on MLX (Mac M4)  
**SFT:** 1,000 general-purpose prompts → 900 train / 100 valid  
**Eval:** GSM8K test (Week 1), ARC-Challenge eval (Week 2) — never used for training

---

## Overview

```txt
Scaffold → Inference + tiny LoRA smoke test → Style guide → Sample prompts
    → Pilot (10) → Full pipeline (1k correct rows) → Baseline eval → LoRA → Finetuned eval
    → ARC eval → Ablations → Write-up
```

---

## Phase 0 — Prerequisites

- [x] Python 3.10+ installed
- [x] ~10 GB free disk (model weights + data)
- [x] OpenAI-compatible API credentials ready (`OPENAI_API_KEY`, `OPENAI_API_BASE`, `OPENAI_MODEL`)
- [x] Hugging Face access for datasets (`huggingface-cli login` if needed)

**Verify Apple Silicon MLX:**

```bash
pip install mlx-lm
python -c "import mlx.core as mx; print(mx.default_device())"
```

---

## Phase 1 — Scaffold the repo

**Goal:** Config, deps, and directory layout so every script shares one source of truth.

- [x] Create `config.yaml` (model path, paths, dataset sample sizes)
- [x] Add reproducibility fields to `config.yaml` (`seed`, sampling params, max generation tokens)
- [x] Create `.gitignore` (`data/`, `adapters/`, `results/`, `.env`)
- [x] Create `requirements.txt` or `pyproject.toml` (`mlx-lm`, `datasets`, `openai`, `pyyaml`, `python-dotenv`)
- [x] Create `.env.example`
- [x] Create `scripts/config.py` — load config, resolve `{name}` path templates
- [x] Create empty dirs: `data/sft/`, `data/raw/`, `data/compressed/`, `adapters/`, `results/`

**Milestone:** `python scripts/config.py` prints model name and resolved paths.

**Files to create:**

```txt
config.yaml
.env.example
.gitignore
requirements.txt
scripts/config.py
lora_config.yaml
```

---

## Phase 2 — Local inference smoke test

**Goal:** Confirm Qwen 3.5 0.8B runs on M4 with thinking mode before building the pipeline.

- [x] Download / cache model: `mlx-community/Qwen3.5-0.8B-OptiQ-4bit`
- [x] Run a single generation with a reasoning prompt
- [x] Confirm output contains a thinking block + final answer
- [x] Confirm the exact configured checkpoint supports a tiny LoRA run before full data generation

```bash
mlx_lm.generate \
  --model mlx-community/Qwen3.5-0.8B-OptiQ-4bit \
  --prompt "If John has 3 apples and buys 2 more, how many does he have?" \
  --max-tokens 500
```

**Tiny LoRA smoke test:** create 5–10 temporary training rows and run a very short adapter job (for example 5–10 iterations). Delete the temporary adapter after confirming `mlx_lm.lora` works with the configured OptiQ checkpoint.

**Milestone:** One verbose CoT trace saved manually; note `<think>...</think>` vs final-answer token boundaries; tiny LoRA run succeeds.

---

## Phase 3 — Grug style guide

**Goal:** Fixed rules for compression prompt and manual spot-checks.

- [x] Create `style_guide.md`
- [x] Include 5–10 before/after examples (math, logic, NLI, commonsense)
- [x] Document moderate terse rules (fragments, continuous paragraph format, no meta, keep logic steps)
- [x] Document compression quality bar (logic preserved, fewer tokens, reasoning-only output)

**Milestone:** `style_guide.md` is loaded at runtime by `compress_traces.py` as the compressor system prompt.

---

## Phase 4 — Sample SFT prompts

**Goal:** 1,000 prompts from six datasets, zero benchmark overlap.

- [x] Implement `scripts/sample_sft_prompts.py`
- [x] Sample per source:

  | Source     | Count |
  | ---------- | ----- |
  | StrategyQA | 170   |
  | LogiQA     | 150   |
  | BoolQ      | 130   |
  | ANLI       | 200   |
  | PIQA       | 150   |
  | ReClor     | 200   |
  | **Total**  | 1,000 |

- [x] Blocklist `gsm8k`, `ai2_arc` dataset IDs
- [x] Tag each row with `source`, `prompt`, `choices` where applicable, and `ground_truth`
- [x] Write `data/sft/prompts.jsonl` (1,000 rows)
- [x] Keep unsplit prompt pool; split train/valid only after filtering for correct model answers

**Milestone:** `data/sft/prompts.jsonl` exists; spot-check 10 rows for format and no benchmark leakage.

---

## Phase 5 — Pilot pipeline (10 examples)

**Goal:** End-to-end dry run before spending time/API on 1k rows.

Build scripts in this order:

| Step | Script               | Input → Output                                     |
| ---- | -------------------- | -------------------------------------------------- |
| 5a   | `generate_traces.py` | prompts → `data/raw/deepseek-r1-1.5b/traces.jsonl` |
| 5b   | `compress_traces.py` | raw traces → `data/compressed/traces.jsonl`        |
| 5c   | `validate_traces.py` | compressed → pass/reject + stats                   |
| 5d   | `format_data.py`     | validated → `data/train.jsonl`, `data/valid.jsonl` |

**Pilot run:**

- [x] Take first 10 BoolQ prompts from `data/sft/prompts.jsonl` (`--source boolq --limit 10`)
- [x] Generate raw CoT locally (DeepSeek-R1 1.5B, thinking on)
- [x] Parse raw thinking and final answer
- [x] Validate raw final answer against the source dataset's ground truth
- [x] Compress via OpenAI-compatible API using `style_guide.md`
- [x] Auto-reject raw answer mismatches, compressions that drop logic steps or restate the final answer, and unparseable rows
- [ ] Manually review 5 compressions — check logic preserved, Grug style consistent
- [x] Format to MLX chat template with visible `<think>...</think>` tokens and required chat delimiters
- [x] Fix any bugs in parsing, chat template, or API prompt

**Milestone:** Pipeline runs end-to-end without errors; accepted rows written to `data/train.jsonl` / `data/valid.jsonl`.

---

## Phase 6 — Full data pipeline (1,000 examples)

**Goal:** Complete SFT dataset with 1,000 accepted, ground-truth-correct rows.

- [ ] Run `generate_traces.py` on prompt pool (long-running; resumable checkpoints recommended)
- [ ] Filter raw traces against dataset ground truth; refill from source datasets if needed
- [ ] Run `compress_traces.py` on accepted raw traces (~1k OpenAI calls)
- [ ] Run `validate_traces.py` — log rejection rate and reasons
- [ ] Spot-check ~100 accepted compressions manually
- [ ] If rejection rate > 15%, inspect failures and tune compression prompt
- [ ] Run `format_data.py` → `data/train.jsonl` (900) + `data/valid.jsonl` (100)

**Milestone:** Final train/valid JSONL ready; document raw-answer rejection, compression rejection, and spot-check stats in `data/compressed/validation_report.json`.

---

## Phase 7 — GSM8K baseline evaluation

**Goal:** Pre-training numbers for base model (no adapter).

- [ ] Implement `scripts/eval.py` for GSM8K test split
- [ ] Metrics per problem:
  - Accuracy (exact match on numeric answer)
  - Thinking tokens
  - Answer tokens
  - Total tokens
  - Latency / tok/s
  - Format compliance (parseable `<think>` block + final answer)
- [ ] Run base model with normal prompt (adapter off)
- [ ] Run base model with explicit Grug prompt (adapter off)
- [ ] Save results → `results/deepseek-r1-1.5b/baseline/gsm8k_normal.json` and `gsm8k_grug_prompt.json`

```bash
python scripts/eval.py --benchmark gsm8k --split test
```

**Milestone:** Baseline JSON saved; note accuracy, mean emitted thinking tokens, and whether prompt-only Grug compression already works.

---

## Phase 8 — LoRA training

**Goal:** Fine-tune adapter on Grug-style SFT data.

- [ ] Implement `scripts/train.py` (wrapper around `mlx_lm.lora` reading `config.yaml`)
- [ ] Train with defaults:

  ```bash
  mlx_lm.lora \
    --model mlx-community/Qwen3.5-0.8B-OptiQ-4bit \
    --adapter-path adapters/deepseek-r1-1.5b \
    --train \
    --data ./data \
    --iters 800 \
    --batch-size 2 \
    --lora-layers 16
  ```

- [ ] Watch valid loss; stop early if overfitting
- [ ] Optional: `mlx_lm.fuse` for easier inference

**Milestone:** Adapter saved in `adapters/deepseek-r1-1.5b/`.

---

## Phase 9 — GSM8K fine-tuned evaluation

**Goal:** Compare base vs fine-tuned on held-out math benchmark.

- [ ] Run `eval.py` with adapter loaded
- [ ] Evaluate both prompt modes:
  - fine-tuned / normal prompt
  - fine-tuned / Grug prompt
- [ ] Save → `results/deepseek-r1-1.5b/finetuned/gsm8k_normal.json` and `gsm8k_grug_prompt.json`
- [ ] Compare side by side:

  | Metric                    | Base normal | Base Grug prompt | FT normal | FT Grug prompt |
  | ------------------------- | ----------- | ---------------- | --------- | -------------- |
  | Accuracy                  |             |                  |           |                |
  | Mean thinking tokens      |             |                  |           |                |
  | Tokens per correct answer |             |                  |           |                |
  | Format compliance         |             |                  |           |                |

**Milestone:** Answer — did Grug style transfer without hurting GSM8K accuracy?

---

## Phase 10 — Week 2 (after GSM8K pipeline works)

### 10a — ARC-Challenge eval

- [ ] Extend `eval.py` for ARC eval split (multiple-choice letter accuracy)
- [ ] Baseline + fine-tuned runs, each with normal and Grug prompt modes
- [ ] Save → `results/.../baseline/arc.json` and `finetuned/arc.json`

### 10b — Token budget ablations

- [ ] Re-run eval with thinking caps: 256, 512 tokens
- [ ] Compare accuracy vs efficiency trade-off

### 10c — Optional ablations

- [ ] LoRA layers: 8 vs 16 vs all
- [ ] Answer-only terse fine-tune (isolate mouth vs brain)
- [ ] Stage 2 data (stronger model raw CoT) — if stage 1 underwhelms

### 10d — 2B fallback (if needed)

Trigger if (see [PLAN.md](./PLAN.md)):

- GSM8K baseline accuracy too low to measure
- Grug style not sticking at inference
- Large accuracy drop after fine-tune

Steps:

- [ ] Update `config.yaml` → 2B
- [ ] Regenerate raw traces for 2B
- [ ] Re-compress (or reuse compressed if same prompts + answers match)
- [ ] Re-train LoRA
- [ ] Re-run benchmarks

### 10e — Write-up

- [ ] Plots: accuracy vs thinking tokens (GSM8K, ARC)
- [ ] Summary: did telegraphic visible CoT improve tokens-per-correct-answer?
- [ ] Lessons learned (data pipeline, compression quality, MLX, eval design)

---

## Quick reference — commands (in order)

```bash
# 1. Setup
pip install -r requirements.txt
cp .env.example .env   # fill in OpenAI vars

# 2. Sample prompts
python scripts/sample_sft_prompts.py

# 3. Pilot (10 BoolQ rows)
python scripts/generate_traces.py --source boolq --limit 10
python scripts/compress_traces.py --limit 10
python scripts/validate_traces.py
python scripts/format_data.py

# 4. Full pipeline
python scripts/generate_traces.py
python scripts/compress_traces.py
python scripts/validate_traces.py
python scripts/format_data.py

# 5. Baseline
python scripts/eval.py --benchmark gsm8k --split test
python scripts/eval.py --benchmark gsm8k --split test --prompt-style grug

# 6. Train
python scripts/train.py

# 7. Fine-tuned eval
python scripts/eval.py --benchmark gsm8k --split test --adapter
python scripts/eval.py --benchmark gsm8k --split test --adapter --prompt-style grug
```

---

## Master checklist

| #   | Phase                             | Status |
| --- | --------------------------------- | ------ |
| 0   | Prerequisites                     | ☑      |
| 1   | Scaffold repo                     | ☑      |
| 2   | MLX smoke test                    | ☑      |
| 3   | Style guide                       | ☑      |
| 4   | Sample 1k prompts                 | ☑      |
| 5   | Pilot pipeline (10)               | ☑      |
| 6   | Full pipeline (1k)                | ☐      |
| 7   | GSM8K baseline                    | ☐      |
| 8   | LoRA training                     | ☐      |
| 9   | GSM8K fine-tuned eval             | ☐      |
| 10  | Week 2 (ARC, ablations, write-up) | ☐      |

---

## What to do right now

Start with **Phase 6 full pipeline (1k rows)**. Manual spot-check ~100 accepted compressions before first LoRA training run.
