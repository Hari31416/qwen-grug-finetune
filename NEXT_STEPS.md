# Next Steps and Iteration Plans

This document defines the second experimental iteration after the results in [`report/`](./report/). The goal is to improve **telegraphic reasoning style transfer** while preserving benchmark accuracy — without re-introducing the Grug system prompt or Grug-prompt eval modes.

## Context from Iteration 1

| Result                                           | Takeaway                                                     |
| ------------------------------------------------ | ------------------------------------------------------------ |
| FT Normal: +1.1pp accuracy, −29% thinking tokens | Style partially internalized without a special prompt        |
| FT Grug: −21.6pp accuracy, −77% thinking tokens  | Over-compression hurt reasoning; Grug prompt eval is retired |
| SFT corpus: 370 validated rows (333 train)       | Too small; inconsistent compression quality                  |
| Base Grug prompt increased tokens (219 → 513)    | Prompt-only Grug control is unreliable — drop it             |

## Scope

### In scope

- Scale SFT data from the **six existing general-purpose sources** (no benchmark train splits for now)
- Automated **Grug adherence scoring** during validation
- Stricter logic-preservation checks in `validate_traces.py`
- SFT rows formatted with a **new system prompt** (style-guide summary, no "Grug" branding)
- **10% negative examples** (uncompressed `raw_thinking` rows)
- Final answers stay **unchanged** from the generation step (`raw_answer` as-is)
- LoRA **rank 16**, lower learning rate, more iterations
- Eval: **Base + system prompt** vs **FT + system prompt** on GSM8K and ARC-Challenge

### Out of scope (deferred)

- Benchmark-sourced SFT rows (GSM8K train, ARC train, etc.) — may revisit later
- `GRUG_SYSTEM_PROMPT` and all `--prompt-style grug` eval modes
- Curriculum training (multi-phase compression tiers)
- Stage 2 stronger-model raw traces
- Inference-time tricks (token caps, two-pass generation, penalty tuning)
- Making final answers terse

---

## Phase 1 — Expand and harden SFT data

### 1.1 Increase sample sizes (six sources only)

Keep the existing blocklist: **GSM8K test** and **ARC-Challenge validation/test** splits must never enter SFT.

Update `config.yaml` `dataset.sample_sizes` to target **~4,000 raw prompts** before filtering (same v1 stratification ratios, 4× scale). Expect ~15% rejection → **~3,400 accepted rows** (~3,060 train / ~340 valid at 90/10).

| Source     | Current (v1) | Target (v2) |
| ---------- | ------------ | ----------- |
| StrategyQA | 170          | 680         |
| LogiQA     | 150          | 600         |
| BoolQ      | 130          | 520         |
| ANLI       | 200          | 800         |
| PIQA       | 150          | 600         |
| ReClor     | 200          | 800         |
| **Total**  | **1,000**    | **4,000**   |

**Files:** `config.yaml`, `scripts/sample_sft_prompts.py` (re-run sampling)

```bash
./.venv/bin/python scripts/sample_sft_prompts.py
./.venv/bin/python scripts/generate_traces.py
./.venv/bin/python scripts/compress_traces.py
```

### 1.2 Keep compression target at ≤50%

No change to `style_guide.md` compression bar:

> Compressed thinking block uses **50% or fewer tokens** than the verbose thinking block.

Reject rows that exceed 50% during validation (enforce programmatically, not only via compressor prompt).

### 1.3 Re-enable strict logic checks

In `scripts/validate_traces.py`, set:

```python
SKIP_LOW_NUM_FRAGMEMENTS = False
SKIP_KEY_VALUE_PATTERN = False
```

This prevents over-compressed traces that drop intermediate reasoning steps from entering SFT.

### 1.4 Add automated Grug adherence scoring

Add a scoring module (e.g. `scripts/grug_score.py`) called from `validate_traces.py`. Each compressed trace gets a `grug_score` (0.0–1.0) and component metrics.

**Proposed metrics:**

| Metric                 | How to compute                                                      | Pass threshold (initial) |
| ---------------------- | ------------------------------------------------------------------- | ------------------------ |
| `compression_ratio`    | `len(compressed_tokens) / len(raw_tokens)`                          | ≤ 0.50                   |
| `article_density`      | count of `the`/`a`/`an` per 100 words in compressed text            | ≤ 3.0                    |
| `meta_commentary_hits` | regex hits for `wait`, `okay`, `let me`, `let's`, `hmm`, `actually` | 0                        |
| `avg_fragment_words`   | mean word count per period-separated fragment                       | ≤ 12                     |
| `repetition_score`     | duplicate sentence/fragment detection                               | no duplicates            |

**Composite score (example):**

```text
grug_score = mean([
  1.0 if compression_ratio <= 0.50 else 0.0,
  1.0 if article_density <= 3.0 else max(0, 1 - (article_density - 3) / 5),
  1.0 if meta_commentary_hits == 0 else 0.0,
  1.0 if avg_fragment_words <= 12 else max(0, 1 - (avg_fragment_words - 12) / 8),
  1.0 if no_repetition else 0.0,
])
```

**Rejection policy:** auto-reject if `grug_score < 0.70` or any hard-fail metric triggers. Log all scores to `data/compressed/{model}/validation_report.json` for distribution analysis before training.

**Files to create/modify:**

- `scripts/grug_score.py` (new)
- `scripts/validate_traces.py` (integrate scoring + rejection)
- `scripts/compress_traces.py` (optional: log score on re-validation pass)

### 1.5 Spot-check before training

Manually review ~50 accepted rows stratified by source and `grug_score` decile. Confirm:

- Logic steps preserved
- Style is telegraphic, not just shorter prose
- `raw_answer` unchanged and still correct vs ground truth

---

## Phase 2 — SFT formatting with system prompt and negative examples

### 2.1 New system prompt (replaces `GRUG_SYSTEM_PROMPT`)

Define a single constant used in **both** `format_data.py` and `eval.py`. Derived from `style_guide.md`; no "Grug" branding.

**Draft `STYLE_SYSTEM_PROMPT`:**

```text
Write your reasoning in a concise, telegraphic style inside the thinking block.

Rules:
- Use short sentence fragments separated by periods, not full prose paragraphs.
- Drop articles ("the", "a") where possible.
- Keep numbers, equations, symbols, variables, and option letters intact.
- Preserve every logical step from the problem; shorten phrasing only, never skip steps.
- No meta-commentary, filler, or self-corrections (e.g. "wait", "okay", "let me think").
- Do not repeat the same calculation or assertion.
- Do not restate the final answer inside the thinking block.
```

**Files:** `scripts/prompt_utils.py` (add constant), `scripts/format_data.py`, `scripts/eval.py`

### 2.2 Format positive examples (90%)

For accepted compressed traces, build chat messages as:

```text
system → STYLE_SYSTEM_PROMPT
user   → question (via build_user_prompt)
assistant → <think>{compressed_thinking}</think>\n\n{raw_answer}
```

Use `tokenizer.apply_chat_template()` with the system message included so training matches inference.

### 2.3 Format negative examples (10%)

After shuffling validated records, mark ~10% as `style: normal`. For these rows:

- Use the **same** `STYLE_SYSTEM_PROMPT` (the model must learn style is conditional on training signal, not prompt alone)
- Assistant thinking block uses **`raw_thinking`** (uncompressed) instead of `compressed_thinking`
- `raw_answer` unchanged

Store the style label in validated JSONL (`"sft_style": "compressed" | "normal"`) so `format_data.py` can branch.

### 2.4 Do not change final answers

Keep `raw_answer` from the generation step exactly as-is. No answer compression, no rewrites.

**Files:** `scripts/format_data.py`, validated row schema in `scripts/validate_traces.py`

---

## Phase 3 — Training recipe changes

### 3.1 LoRA config updates (`lora_config.yaml`)

| Parameter                       | v1   | v2                                         |
| ------------------------------- | ---- | ------------------------------------------ |
| `rank`                          | 8    | **16**                                     |
| `alpha`                         | 16   | **32** (keep `scale = alpha / rank = 2.0`) |
| `learning_rate`                 | 1e-5 | **5e-6**                                   |
| `iters`                         | 300  | **800**                                    |
| `batch_size`                    | 4    | 4 (adjust down to 2 if OOM)                |
| `save_every` / `steps_per_eval` | 20   | 20                                         |

No curriculum: single training run on the full mixed dataset.

### 3.2 Train command

```bash
./.venv/bin/python scripts/train.py --iters 800 --batch-size 4 --learning-rate 5e-6
```

Select best checkpoint by lowest validation loss (same as v1).

**Files:** `lora_config.yaml`, `scripts/train.py` (defaults optional)

---

## Phase 4 — Eval protocol (no Grug prompt)

### 4.1 Retire Grug eval modes

Remove or deprecate:

- `--prompt-style grug` in `scripts/eval.py`
- `GRUG_SYSTEM_PROMPT` constant
- `gsm8k_grug_*` result files and plot variants

### 4.2 New comparison matrix

Run each benchmark with **only** the style system prompt:

| Mode     | Adapter           | System prompt         |
| -------- | ----------------- | --------------------- |
| **Base** | Off               | `STYLE_SYSTEM_PROMPT` |
| **FT**   | On (best adapter) | `STYLE_SYSTEM_PROMPT` |

**Benchmarks:**

1. GSM8K test split (1,000 samples)
2. ARC-Challenge test split (1,172 samples)

```bash
# GSM8K
./.venv/bin/python scripts/eval.py --benchmark gsm8k --split test --limit 1000 --batch-size 16
./.venv/bin/python scripts/eval.py --benchmark gsm8k --split test --limit 1000 --adapter --batch-size 16

# ARC-Challenge
./.venv/bin/python scripts/eval.py --benchmark arc --split test --limit 1172 --batch-size 16
./.venv/bin/python scripts/eval.py --benchmark arc --split test --limit 1172 --adapter --batch-size 16
```

(`eval.py` should inject `STYLE_SYSTEM_PROMPT` by default; add `--no-system-prompt` only if needed for ablation.)

### 4.3 Metrics to report

Per benchmark, per mode (Base vs FT):

| Metric                       | Notes                                     |
| ---------------------------- | ----------------------------------------- |
| Accuracy                     | Task-specific (exact match / MC letter)   |
| Mean thinking tokens         | Primary efficiency metric                 |
| Mean total tokens            | Thinking + answer                         |
| Tokens per correct answer    | Efficiency given correctness              |
| Mean latency / tok/s         | M4 wall-clock                             |
| Format compliance            | Parseable thinking block + answer         |
| Mean `grug_score` on outputs | New: measure style adherence at inference |

### 4.4 Success criteria (iteration 2)

Primary:

> **FT matches or beats Base accuracy (within ±2pp) while reducing mean thinking tokens by ≥25%.**

Stretch:

> Mean output `grug_score` ≥ 0.75 on both benchmarks.

Secondary:

> ARC-Challenge eval completes end-to-end (validates cross-domain style transfer).

### 4.5 Update reporting

- Regenerate plots via `scripts/plot_results.py` (simplify to Base vs FT, drop Grug variants)
- Write `report/REPORT_v2.md` with GSM8K + ARC tables and delta charts

---

## Implementation checklist

| #   | Task                                                | File(s)                      | Status |
| --- | --------------------------------------------------- | ---------------------------- | ------ |
| 1   | Bump `dataset.sample_sizes` in config               | `config.yaml`                | ☐      |
| 2   | Re-sample, generate, compress traces                | pipeline scripts             | ☐      |
| 3   | Add `grug_score.py`                                 | `scripts/grug_score.py`      | ☐      |
| 4   | Integrate scoring + re-enable strict validation     | `scripts/validate_traces.py` | ☐      |
| 5   | Add `STYLE_SYSTEM_PROMPT`                           | `scripts/prompt_utils.py`    | ☐      |
| 6   | Format with system prompt + 10% negatives           | `scripts/format_data.py`     | ☐      |
| 7   | Update LoRA rank/alpha/LR/iters                     | `lora_config.yaml`           | ☐      |
| 8   | Remove Grug prompt from eval; default system prompt | `scripts/eval.py`            | ☐      |
| 9   | Simplify plots (Base vs FT)                         | `scripts/plot_results.py`    | ☐      |
| 10  | Train + eval + report                               | —                            | ☐      |

---

## Deferred for a future iteration

Documented here so we do not scope-creep now:

- **Benchmark train splits in SFT** — add ARC-Challenge train first, GSM8K train second, only after confirming no leakage and measuring contamination risk
- **Curriculum training** — moderate compression first, aggressive second
- **Stage 2 traces** — stronger model as raw CoT source
- **Inference-time techniques** — token caps, two-pass verify, penalty sweeps
- **DoRA / extra LoRA layers** — only if rank-16 LoRA still shows weak style adoption

## Iteration 3 — Mitigating Alignment Tax and Preserving Reasoning

Following the Iteration 2 training run (which successfully mitigated prompt leakage and instruction regurgitation), we evaluated the model and observed an accuracy of **53%** (baseline is **78%** on 100 samples). This indicates a significant alignment tax where the model over-compresses math steps and drops crucial logical steps.

### 3.1 Goals

- Close the accuracy gap to ±3pp of the baseline (GSM8K target accuracy: **≥75%**).
- Maintain style compression with a reduction in thinking tokens by **≥20%** compared to the baseline.
- Prevent instruction regurgitation.

### 3.2 Proposed Experiments

- **Benchmark SFT Mixing (Task-Specific Data)**
  - **Why:** The SFT dataset currently contains only general-reasoning datasets. The model has never seen math-specific (GSM8K) SFT examples in the target style.
  - **What:** Add a portion of GSM8K training split traces to the SFT corpus (e.g., 500 GSM8K training questions generated and compressed in the Grug style) while strictly ensuring they do not overlap with the GSM8K test split.
- **LoRA Capacity Reduction**
  - **Why:** A LoRA rank of 16 targets all projection layers, allowing the adapter to override pre-trained attention weights too easily.
  - **What:** Reduce rank to 8 or 4, and/or restrict LoRA target layers to only `q_proj` and `v_proj` to act as a stronger regularizer.
- **Relaxed Dynamic Compression**
  - **Why:** The 50% hard cutoff on thinking tokens in `validate_traces.py` forces extreme over-compression on complex math questions.
  - **What:** Relax the compression threshold dynamically based on prompt complexity (e.g., allow up to 60-70% token length for long math steps, or use a multi-stage curriculum).
- **Checkpoint Sweep Evaluation**
  - **Why:** The model may overfit to style as epoch count increases.
  - **What:** Evaluate intermediate checkpoints (e.g., every 50 steps from step 100 to 800) to identify the step where style transfer is learned before accuracy deteriorates.

