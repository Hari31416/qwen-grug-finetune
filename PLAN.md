# Grug Reasoning Fine-Tune — Project Plan

Fine-tune **DeepSeek-R1-Distill-Qwen-1.5B** on Apple Silicon (M4) to learn whether telegraphic "Grug/caveman" chain-of-thought can improve **token efficiency** without sacrificing **accuracy**.

**Target model:** `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit` — used for raw CoT generation, compression SFT, LoRA training, and eval.

All scripts and configs are **model-agnostic** — swap via `config.yaml`, not code changes.

**Goal:** Personal learning exercise in data curation, trace compression, LoRA fine-tuning, and benchmark evaluation — not building a tutoring product or education-domain dataset.

**Training objective:** Align reasoning **style** to Grug (token-efficient thinking) — not learn GSM8K or ARC task distributions.

**Evaluation objective:** Measure whether style alignment helps or hurts on **held-out benchmarks** (GSM8K, ARC-Challenge) that the model never trained on.

---

## Background: The "Grug Hypothesis"

There are two different ideas often mixed together:

| Claim                                                             | Evidence                                                                                                                                                                                   |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Terse output** (drop filler, articles, politeness) saves tokens | Strong — tools like [caveman](https://github.com/JuliusBrussee/caveman) / [grug](https://github.com/maxPiroddi/grug) report ~65–75% fewer **output** tokens with similar technical content |
| **GPT-5 reasons internally in "caveman"**                         | Weak / unverified — we cannot see proprietary reasoning traces; public APIs expose `reasoning_content` separately from the answer                                                          |

Important nuance from those projects:

> Grug mode shrinks the **mouth**, not necessarily the **brain**. Output compression ≠ internal reasoning compression.

This project tests a more interesting question: **can a small model learn compressed visible reasoning traces that still solve problems** — not just shorter final answers?

Research is mixed on aggressive reasoning compression:

- [Broken Chains (2026)](https://arxiv.org/pdf/2602.14444) — truncated CoT can **hurt** some models
- [Brevity constraints](https://arxiv.org/abs/2604.00025) — can sometimes **help** accuracy

Worth testing empirically on small reasoning models.

---

## Model Selection & Swappability

The pipeline treats the **target model** as a config parameter. The project uses **DeepSeek-R1-Distill-Qwen-1.5B** for all stages (raw CoT, compression SFT, LoRA, eval).

### Supported variant

| Variant            | MLX model ID                                        | Peak RAM (LoRA) | Role                                        |
| ------------------ | --------------------------------------------------- | --------------- | ------------------------------------------- |
| **1.5B** (default) | `mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit` | ~5 GB           | Primary — RL-trained reasoning, native `<think>` |

**Why not Qwen 3.5 0.8B/2B?** Pilot testing showed standard Qwen 3.5 instruct checkpoints could not reliably complete reasoning traces on general-knowledge SFT prompts (infinite self-correction loops, unclosed thinking tags). DeepSeek-R1-Distill-Qwen was chosen because it natively emits structured reasoning traces with high accuracy on the SFT corpus.

### Central config (`config.yaml`)

One file drives all scripts:

```yaml
target_model:
  name: deepseek-r1-1.5b
  mlx_path: mlx-community/DeepSeek-R1-Distill-Qwen-1.5B-4bit
  size: 1.5b

run:
  seed: 42
  temperature: 0.6
  top_p: 0.95
  max_generation_tokens: 1536
  eval_max_generation_tokens: 1536

paths:
  adapters: adapters/{name}/   # LoRA weights — per model, not shared
  results: results/{name}/     # baseline + finetuned eval — per model
  raw_traces: data/raw/{name}/ # stage-1 self-traces — per model
```

Every script (`generate_traces`, `eval`, `train`) reads `target_model` from config. CLI override optional: `--model qwen3.5-2b`.

**Smoke-test requirement:** Before building the full pipeline, verify that the exact configured MLX checkpoint works for both inference and a tiny LoRA run. OptiQ checkpoints are standard MLX checkpoints, but training support should be confirmed with 5–10 rows before spending time on data generation.

### What is shared vs model-specific

| Artifact                    | Shared across models? | Notes                                                   |
| --------------------------- | --------------------- | ------------------------------------------------------- |
| Grug style guide            | Yes                   | Same compression rules for all                          |
| SFT prompt corpus           | Yes                   | General-purpose sources (6 datasets, see below)         |
| Compressed traces (stage 1) | **No**                | Self-traces depend on target model — regenerate on swap |
| Compressed traces (stage 2) | Partially             | Same prompts possible; raw CoT from stronger model      |
| LoRA adapters               | **No**                | Tied to base weights — train fresh per variant          |
| Benchmark eval splits       | Yes                   | GSM8K test / ARC eval — never used in SFT               |
| Eval scripts                | Yes                   | Same code, different `mlx_path` + adapter dir           |

### When to switch to 2B

Move to 2B if any of these hold after a full 0.8B iteration:

- Baseline accuracy too low on GSM8K to measure improvement (e.g. below 15%)
- Fine-tuned model learns Grug style but accuracy drops sharply
- Compressed-trace validation fails often because 0.8B raw CoT is frequently wrong
- LoRA loss converges but eval shows no token reduction at same accuracy

Switching cost: update config → regenerate stage-1 traces → re-run baseline → re-train LoRA. Stage-2 data and style guide carry over.

---

## Feasibility on M4

| Resource     | Qwen 3.5 0.8B                                               | Qwen 3.5 2B |
| ------------ | ----------------------------------------------------------- | ----------- |
| Peak RAM     | ~4 GB                                                       | ~6 GB       |
| Train time   | ~15–30 min                                                  | ~20–40 min  |
| Adapter size | ~14 MB                                                      | ~25 MB      |
| Framework    | [MLX](https://github.com/ml-explore/mlx-lm) (`mlx_lm.lora`) | same        |

Reference: [mlx-lora-finetune](https://github.com/sciences44/mlx-lora-finetune) fine-tuned both 0.8B and 2B on Mac; 2B reached higher task accuracy at modest RAM cost.

**Caveat:** Small models will not match frontier models on hard tasks. Treat this as a **learning + compression** experiment, not frontier-model replication.

## SFT Data vs Benchmarks (critical separation)

| Role               | Source                                        | Overlap with eval?                                          |
| ------------------ | --------------------------------------------- | ----------------------------------------------------------- |
| **SFT training**   | General-purpose reasoning prompts (see below) | **None** — never use GSM8K or ARC train/eval splits         |
| **Benchmark eval** | GSM8K test + ARC-Challenge eval               | Held-out — both base and fine-tuned see identical questions |

This avoids distribution bias: fine-tune teaches *how to think* (Grug), not *how to ace GSM8K*. Comparing original vs fine-tuned on GSM8K/ARC is a fair test of whether style transfer preserves task performance.

```txt
SFT prompts (StrategyQA, LogiQA, BoolQ, ANLI, PIQA, ReClor)
    → target model raw CoT → ground-truth filter → OpenAI compress → Grug traces → LoRA

GSM8K test / ARC eval  →  baseline + fine-tuned eval only (never in SFT)
```

Critical guardrail: keep only SFT rows where the target model's generated final answer matches the dataset's ground-truth label. The compressor shortens the reasoning trace only; it never sees or rewrites the answer field.

---

## Domains & Benchmarks (eval only)

Benchmarks are **evaluation-only**. No benchmark train split is used for SFT.

### Benchmark 1 — Grade-school math

| Item          | Choice                                                                                |
| ------------- | ------------------------------------------------------------------------------------- |
| **Benchmark** | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) **test split only**             |
| **Metric**    | Exact match on final numeric answer                                                   |
| **Why**       | Well-studied, clear pass/fail, tests whether Grug style hurts/help math reasoning OOD |

### Benchmark 2 — Commonsense / logical reasoning

| Item          | Choice                                                                               |
| ------------- | ------------------------------------------------------------------------------------ |
| **Benchmark** | [ARC-Challenge](https://huggingface.co/datasets/allenai/ai2_arc) **eval split only** |
| **Metric**    | Accuracy on answer letter (A/B/C/D)                                                  |
| **Why**       | Different domain than math; tests cross-domain style transfer                        |
| **Timing**    | Week 2 (after GSM8K eval pipeline validated)                                         |

### SFT training corpus (general-purpose, no benchmark overlap)

Prompts should require multi-step reasoning but come from datasets **unrelated** to GSM8K and ARC-Challenge.

| Source                                                           | Why include                                 | v1 sample |
| ---------------------------------------------------------------- | ------------------------------------------- | --------- |
| [StrategyQA](https://huggingface.co/datasets/ChilleD/StrategyQA) | Yes/no questions needing implicit reasoning | 170       |
| [LogiQA](https://huggingface.co/datasets/lucasmccabe/logiqa)     | Formal logic reading comprehension          | 150       |
| [BoolQ](https://huggingface.co/datasets/google/boolq)            | Boolean reasoning over short passages       | 130       |
| [ANLI](https://huggingface.co/datasets/anli)                     | Natural language inference (entailment)     | 200       |
| [PIQA](https://huggingface.co/datasets/piqa)                     | Physical commonsense (two choices)          | 150       |
| [ReClor](https://huggingface.co/datasets/re-clor)                | Logical reasoning (GMAT/LSAT style)         | 200       |

**v1 target:** 1,000 prompts total, stratified across sources (table above).

**SFT split:** 900 → `train.jsonl`, 100 → `valid.jsonl` (10% held-out for training loss monitoring).

**Expand later:** Scale to 2k+ from the same six sources (or add BBH slices, HotpotQA).

**Data prep guardrail:** Script must blocklist dataset IDs / splits for `gsm8k`, `ai2_arc`, and any row that fuzzy-matches benchmark question text.

### Evaluation protocol (both benchmarks)

For each benchmark, run **base target model** and **fine-tuned model** (same `config.yaml` entry) with identical inference settings:

| Metric                        | Definition                                                   |
| ----------------------------- | ------------------------------------------------------------ |
| **Accuracy**                  | Task-specific (exact match / MC accuracy)                    |
| **Thinking tokens / problem** | Token count of visible `<think>` block only                  |
| **Answer tokens / problem**   | Token count of final answer                                  |
| **Total tokens / problem**    | Thinking + answer                                            |
| **Tokens per correct answer** | Total tokens ÷ number correct (efficiency given correctness) |
| **Latency**                   | Wall-clock and tok/s on M4                                   |
| **Format compliance**         | Parseable thinking block + final answer / choice format      |

Report results per benchmark. Primary success criterion: **similar or better accuracy with meaningfully fewer emitted thinking tokens** — on data the model never trained on.

### Domains deferred (optional later)

- **Code** — HumanEval / MBPP (harder for 0.8B; add in stage 2 if math + ARC show signal)
- **Hard math** — MATH subset (likely too hard for baseline 0.8B)

---

## Project Overview

```mermaid
flowchart LR
    A[Define Grug spec] --> B[Sample SFT prompts]
    B --> C[Generate raw CoT on target model]
    C --> V[Filter by dataset ground truth]
    V --> D[Compress traces via OpenAI]
    D --> E[Baseline eval on GSM8K / ARC]
    E --> F[LoRA SFT on Mac]
    F --> G[Compare accuracy + tokens on benchmarks]
    G --> H[Iterate / stage 2 data]
```

---

## Phase 1 — Define "Grug Reasoning" Precisely

Write a style guide with **before/after** examples. This guide is used both for the compression prompt and as a quality checklist during spot-review.

**Verbose CoT:**

> First, I need to identify the variables. Let me set up an equation. The problem states that...

**Grug CoT:**

> vars x,y. eq 2x+3=9. x=3. check 2*3+3=9 ok.

### Style rules

- Drop articles ("the", "a")
- Use fragments, not full sentences
- Keep symbols, numbers, code tokens intact
- One continuous paragraph of fragments (period-separated steps, not key-value lines)
- No meta ("let me think", "I should consider")
- Preserve logical steps — compression removes words, not reasoning steps

Grug style applies to the **thinking block only**. Final answers in SFT data stay clear and unchanged — benchmarks are unaffected by answer formatting.

### Compression quality bar

A compressed trace is valid only if:

1. It reaches the **same final answer** as the raw trace
2. No intermediate step is dropped that is needed for correctness
3. Token count is meaningfully lower (target: ≤40–50% of raw thinking tokens)

---

## Phase 2 — Build SFT Data via Trace Compression

**Primary approach: compress existing traces** — not teacher distillation. No existing model is trained in Grug style. Instead: generate verbose CoT from the target model on **general-purpose prompts**, then compress with OpenAI following the style guide.

**Not used for SFT:** GSM8K and ARC-Challenge (any split). Those are eval-only.

### Stage 1 — Self-traces from target model (first iteration)

Use the same model we will fine-tune as the trace generator (`config.yaml` → `target_model`).

```txt
General-purpose prompt (StrategyQA / LogiQA / BoolQ / ANLI / PIQA / ReClor)
    → Target Qwen 3.5 model (thinking mode) generates raw CoT + answer
    → OpenAI compressor compresses CoT to Grug style (answer unchanged)
    → Auto-filter + spot-check bad compressions
    → SFT training example
```

**Why start here:**

- Traces match what the base model actually produces — fine-tune teaches compression of *its own* reasoning habit
- No API cost for trace generation (runs locally on M4)
- SFT is style-only — benchmarks stay unbiased
- Regenerate traces when swapping models (0.8B → 2B) — raw CoT distribution differs

**Compressor:** OpenAI API (user-provided key, base URL, model name via env/config).

**Pipeline per example:**

1. Feed general-purpose prompt to target model with thinking enabled
2. Capture `raw_thinking`, `answer`
3. Compare `answer` against the dataset's ground-truth label; reject wrong or unparseable rows
4. Prompt compressor: given `raw_thinking` + Grug style guide → output `grug_thinking` (reasoning only; question and answer not sent)
5. Validate: logic steps preserved, token budget met, no final-answer restatement in `grug_thinking`; reject or flag failures
6. Format as MLX training example

### Stage 2 — Higher-quality raw CoT (later iteration)

Once stage 1 pipeline works end-to-end, optionally upgrade the trace source:

```txt
General-purpose prompt
    → Stronger reasoning model generates raw CoT + answer
    → Same OpenAI compressor → Grug style
    → SFT training example
```

**Why add this:** Target model raw traces may be weak on harder prompts. Stronger CoT gives richer content to compress.

**Trade-off:** Stage 2 traces may not match base model's native style. Compare stage 1 vs stage 2 fine-tunes in eval.

**Decision:** Defer choice of stage-2 raw trace model until stage 1 results are in.

### Stored row schema

Keep structured JSONL before producing MLX `text` rows:

```json
{
  "id": "strategyqa-0001",
  "source": "strategyqa",
  "prompt": "...",
  "choices": ["yes", "no"],
  "ground_truth": "yes",
  "raw_thinking": "...",
  "raw_answer": "yes",
  "grug_thinking": "Route Chicago to New Orleans. Chicago Canal to Illinois River, Mississippi to New Orleans. Continuous waterway. Sailing possible.",
  "accepted": true,
  "accepted_reason": "answer_matches_ground_truth"
}
```

Split train/valid after filtering, or refill rejected rows so the final dataset still has 900 train and 100 valid examples.

### Spot-checking

Manually review ~100 compressed examples before first training run:

- Did compression drop a necessary step?
- Is the answer still correct against the dataset label?
- Is Grug style consistent?

Discard or fix failures before training.

**Validation policy:** Auto-reject rows where raw answer mismatches ground truth, auto-reject compressions that drop logic steps or restate the final answer, then manual spot-check (no LLM-judge on every row for v1).

### Qwen 3.5 chat format

```txt
<|im_start|>user
{question}
<|im_end|>

<|im_start|>assistant
<think>
{grug_reasoning}
</think>

{final_answer}
<|im_end|>
```

For MLX, store as JSONL with a `text` field containing the full formatted conversation. Prefer the tokenizer's chat template when available, then verify it preserves the visible `<think>...</think>` reasoning channel.

### Data volume & splits

| Purpose            | Source                                             | Size (v1)                   |
| ------------------ | -------------------------------------------------- | --------------------------- |
| **SFT train**      | StrategyQA + LogiQA + BoolQ + ANLI + PIQA + ReClor | 900 prompts → `train.jsonl` |
| **SFT valid**      | Same sources, held-out rows                        | 100 prompts → `valid.jsonl` |
| **Benchmark eval** | GSM8K test split                                   | ~1.3k (eval only)           |
| **Benchmark eval** | ARC-Challenge eval split                           | Week 2 (eval only)          |

- **Blocklist:** No GSM8K or ARC rows in SFT pipeline
- **Correctness filter:** raw answer must match the source dataset label before compression
- SFT valid split is for training loss monitoring — not the same as benchmark eval

---

## Phase 3 — Baseline Before Fine-Tuning

Run **base target model** (from config) on held-out benchmark splits and record:

- Accuracy per domain (GSM8K exact match, ARC letter accuracy)
- Thinking tokens per problem
- Total tokens and latency on M4
- Format compliance (% parseable thinking block and final answer)
- Failure modes (truncated logic, arithmetic slips, wrong format)

Save baseline results to `results/{model_name}/baseline/` — all post-training comparisons reference this. Keeps 0.8B and 2B runs separate if both are tried.

Run two baseline prompt modes:

1. **Base / normal prompt** — no Grug instruction
2. **Base / Grug prompt** — explicit Grug-style instruction, no adapter

These show whether LoRA improves beyond prompt-only style control.

---

## Phase 4 — Fine-Tune on Mac (LoRA)

**Decision: LoRA, not full fine-tuning.** See [LoRA vs full fine-tuning](#lora-vs-full-fine-tuning) below.

Model path and adapter output dir come from `config.yaml`. Example below uses 0.8B; swap config for 2B.

### Setup

```bash
# Install
pip install mlx-lm

# Prepare data in ./data/train.jsonl and ./data/valid.jsonl
# (stage-1 data should match target model in config)

# Train — model path read from config or passed explicitly
mlx_lm.lora \
  --model mlx-community/Qwen3.5-0.8B-OptiQ-4bit \
  --adapter-path adapters/qwen3.5-0.8b \
  --train \
  --data ./data \
  --iters 800 \
  --batch-size 2 \
  --lora-layers 16

# Optional: fuse adapter for easier inference
mlx_lm.fuse \
  --model mlx-community/Qwen3.5-0.8B-OptiQ-4bit \
  --adapter-path adapters/qwen3.5-0.8b
```

### Hyperparameters to try

| Parameter     | Range    | Notes                                       |
| ------------- | -------- | ------------------------------------------- |
| `iters`       | 600–1200 | ~1k examples; watch valid loss              |
| `lora-layers` | 8–16     | More layers = more capacity                 |
| `batch-size`  | 1–4      | Limited by RAM                              |
| Base quant    | 4-bit    | Default; 8-bit if 16GB+ and quality matters |

---

## LoRA vs Full Fine-Tuning

### Recommendation: **LoRA** (primary)

| Factor          | LoRA                                          | Full fine-tuning                                               |
| --------------- | --------------------------------------------- | -------------------------------------------------------------- |
| **RAM on M4**   | ~4 GB (0.8B), ~6 GB (2B)                      | ~8–12 GB (0.8B), ~16–24 GB (2B) — tight or infeasible on 16 GB |
| **Train time**  | ~15–30 min                                    | Hours; optimizer state over all weights                        |
| **Goal fit**    | Style/format alignment is what LoRA excels at | Overkill for narrow style change                               |
| **Comparison**  | Toggle adapter off → instant A/B vs base      | Must keep separate checkpoint; harder to isolate style effect  |
| **Risk**        | Lower catastrophic-forgetting risk            | Higher — 1k examples can shift general behavior                |
| **MLX support** | First-class (`mlx_lm.lora`)                   | Not practical on consumer Mac for multi-B models               |

**Why LoRA fits this project:**

1. **Narrow objective** — we are aligning thinking *style* (shorter, telegraphic CoT), not retraining world knowledge or task distributions.
2. **SFT set size** — 1k examples is appropriate for LoRA; full FT on that data tends to overfit or destabilize the base model.
3. **Clean eval story** — same base weights, adapter on/off: any accuracy or token change is attributable to the adaptation.
4. **Hardware** — the plan already assumes M4 + MLX LoRA; full FT of even 0.8B is possible in theory but offers little upside for much more RAM and complexity.

### When full FT might make sense (not v1)

- LoRA converges but model **ignores** Grug style at inference (format not sticking)
- You upgrade to **2B** and have **32 GB+** unified memory
- You scale SFT to **10k+** diverse examples and want maximum style absorption

Even then, try first: more LoRA layers, more iters, or **DoRA** (`--fine-tune-type dora` in mlx-lm) before full FT.

### Optional ablation (Week 2+)

If LoRA shows weak style adoption, compare:

- LoRA (16 layers) vs LoRA (all layers) vs DoRA

Skip full FT unless LoRA/DoRA clearly fail and hardware allows it.

---

## Phase 5 — Evaluate: Accuracy vs Token Efficiency

Compare **base vs fine-tuned** on the same held-out benchmarks (GSM8K test, ARC eval):

| Metric                            | What it tells you                |
| --------------------------------- | -------------------------------- |
| **Accuracy**                      | Did compression break reasoning? |
| **Thinking tokens / problem**     | Emitted Grug efficiency metric   |
| **Total tokens (think + answer)** | End-to-end cost                  |
| **Tokens per correct answer**     | Efficiency *given* correctness   |
| **Latency (tok/s on M4)**         | Practical edge benefit           |
| **Format compliance**             | Whether outputs stay parseable   |
| **Error type breakdown**          | Logic vs arithmetic vs format    |

### Required comparison modes

Run each benchmark with identical sampling, seed, and max-generation settings:

| Mode                       | Adapter | Prompt style     | Purpose                                  |
| -------------------------- | ------- | ---------------- | ---------------------------------------- |
| Base / normal              | Off     | Normal reasoning | True baseline                            |
| Base / Grug prompt         | Off     | Explicit Grug    | Prompt-only compression baseline         |
| Fine-tuned / normal prompt | On      | Normal reasoning | Tests whether LoRA internalized style    |
| Fine-tuned / Grug prompt   | On      | Explicit Grug    | Best-case style adherence / sanity check |

### Ablations

- **Stage 1 vs stage 2 training data** — self-traces vs stronger-model traces
- **Fixed thinking token budget** (Week 2) — 256 / 512 caps; primary eval has no cap
- **Answer-only terse fine-tune** (no Grug thinking) — isolates shorter final output vs shorter visible reasoning
- **Per-benchmark breakdown** — GSM8K vs ARC

### Deliverable

Plots and table: **accuracy vs thinking tokens** per domain, base vs fine-tuned. This is the core answer to the Grug hypothesis for a small model.

---

## What You'll Learn

This project is structured to build hands-on skill in:

1. **Benchmark-driven ML** — picking domains, metrics, and held-out eval before training
2. **Data pipelines** — generate → compress → validate → format
3. **Fine-tuning on Apple Silicon** — MLX LoRA workflow
4. **Evaluation design** — accuracy and efficiency as joint objectives
5. **Iterative experimentation** — stage 1 vs stage 2 data, ablations, model fallback (0.8B → 2B)

---

## Realistic Expectations

### Likely achievable

- 30–60% shorter emitted thinking traces on benchmark prompts (if style transfers)
- Faster inference on device
- Solid end-to-end fine-tuning workflow on Apple Silicon
- Clear benchmark comparison (base vs fine-tuned)

### Hard / unlikely

- Matching frontier-model reasoning quality at 0.8B
- Proving proprietary models use internal "Grug" (we cannot observe hidden reasoning)
- Large accuracy gains — expect trade-offs, not free lunch
- Very hard multi-step problems — model capacity is the ceiling

### Model fallback (0.8B → 2B)

See [When to switch to 2B](#when-to-switch-to-2b) above. The 2B variant fits comfortably on M4 (~6 GB) and often learns style/format better. Because adapters and stage-1 traces are model-specific, a fallback is a **config change + re-run**, not a fork of the codebase.

---

## Suggested Timeline (2 weeks)

| Week       | Focus                                                                                  |
| ---------- | -------------------------------------------------------------------------------------- |
| **Week 1** | Config + style guide → SFT data (1k prompts) → compress → GSM8K baseline → first LoRA  |
| **Week 2** | GSM8K + ARC benchmark eval → token-budget ablations → 2B fallback if needed → write up |

---

## Decisions (closed)

| Decision                    | Choice                                                                          |
| --------------------------- | ------------------------------------------------------------------------------- |
| **SFT data**                | StrategyQA + LogiQA + BoolQ + ANLI + PIQA + ReClor — **not** GSM8K or ARC       |
| **Benchmarks**              | GSM8K test + ARC-Challenge eval only — never used for training                  |
| **SFT size (v1)**           | 1,000 prompts (900 train / 100 valid)                                           |
| **ARC timing**              | Week 2 eval (defer until pipeline validated)                                    |
| **Grug tereness**           | Moderate — telegraphic fragments, all logic steps preserved                     |
| **Compression validation**  | Ground-truth filter + logic/token checks on compressed reasoning + manual spot-check ~100 examples |
| **Compressor**              | OpenAI API — user provides API key, base URL, and model name via env            |
| **Stage 2 raw trace model** | Decide after stage 1 results                                                    |
| **Thinking token budget**   | No cap in primary eval; capped ablations (256/512) in Week 2                    |
| **Target model**            | DeepSeek-R1-Distill-Qwen-1.5B via `config.yaml`                                 |
| **Training method**         | **LoRA** (4-bit QLoRA via MLX) — not full fine-tuning                           |
| **Config schema**           | Adopt `config.yaml` layout in [Model Selection](#model-selection--swappability) |

---

## Repo Structure (proposed)

```txt
qwen-finetune/
├── PLAN.md                      # this file
├── config.yaml                  # target model + paths — swap model here
├── style_guide.md               # Grug compression rules + examples
├── data/
│   ├── sft/                     # general-purpose prompts (6 datasets)
│   ├── raw/
│   │   └── deepseek-r1-1.5b/    # stage-1 self-traces (per model)
│   ├── compressed/              # Grug-compressed traces
│   ├── validated/               # accepted traces after validation
│   ├── train.jsonl              # MLX-ready SFT data
│   └── valid.jsonl
├── scripts/
│   ├── config.py                # load config.yaml, resolve paths
│   ├── sample_sft_prompts.py    # sample general corpus, blocklist benchmarks
│   ├── generate_traces.py       # target model → raw CoT (stage 1)
│   ├── compress_traces.py       # correct raw CoT → Grug via OpenAI
│   ├── validate_traces.py       # ground-truth, answer match + spot-check helpers
│   ├── format_data.py           # validated traces → MLX JSONL
│   ├── train.py                 # wrapper around mlx_lm.lora using config
│   └── eval.py                  # benchmark eval: accuracy + token counts
├── adapters/
│   ├── deepseek-r1-1.5b/        # LoRA per model (gitignored)
├── results/
│   └── deepseek-r1-1.5b/
│       ├── baseline/
│       └── finetuned/
└── lora_config.yaml             # training hyperparams (shared; model path from config)
```

---

## Bottom Line

- **Feasible?** Yes. M4 + DeepSeek-R1-Distill-Qwen-1.5B + MLX LoRA is well-trodden; expect sub-hour training runs.
- **Swappable?** Yes. One `config.yaml` change swaps model, adapters, traces, and results — no script rewrites.
- **Worth doing?** Yes as a learning project — covers data pipelines, compression, SFT, and benchmark eval.
- **Fair comparison?** Yes — SFT is style-only on unrelated data; GSM8K/ARC eval is OOD for both base and fine-tuned.
