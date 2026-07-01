# Makefile for Qwen SFT Fine-Tuning, Evaluation, and Visualizations

PYTHON = .venv/bin/python
LIMIT = 100
BATCH_SIZE = 16
ITERS = 300
SAVE_EVERY = 20

.PHONY: help train eval-base-normal eval-base-grug eval-ft-normal eval-ft-grug eval-all plot clean

help:
	@echo "Available commands:"
	@echo "  make train               Run SFT training with default settings (300 iters)"
	@echo "  make eval-base-normal    Evaluate base model with normal prompt (default: LIMIT=100, BATCH_SIZE=16)"
	@echo "  make eval-base-grug      Evaluate base model with Grug prompt"
	@echo "  make eval-ft-normal      Evaluate fine-tuned model with normal prompt"
	@echo "  make eval-ft-grug        Evaluate fine-tuned model with Grug prompt"
	@echo "  make eval-all            Run all baseline & fine-tuned evaluations sequentially"
	@echo "  make plot                Generate comparison dashboard and separate plots"
	@echo "  make clean               Clean Python cache files"

train:
	$(PYTHON) scripts/train.py --iters $(ITERS) --save-every $(SAVE_EVERY) --batch-size 4

eval-base-normal:
	$(PYTHON) scripts/eval.py --benchmark gsm8k --split test --limit $(LIMIT) --batch-size $(BATCH_SIZE)

eval-base-grug:
	$(PYTHON) scripts/eval.py --benchmark gsm8k --split test --limit $(LIMIT) --batch-size $(BATCH_SIZE) --prompt-style grug

eval-ft-normal:
	$(PYTHON) scripts/eval.py --benchmark gsm8k --split test --limit $(LIMIT) --batch-size $(BATCH_SIZE) --adapter

eval-ft-grug:
	$(PYTHON) scripts/eval.py --benchmark gsm8k --split test --limit $(LIMIT) --batch-size $(BATCH_SIZE) --prompt-style grug --adapter

eval-all: eval-base-normal eval-base-grug eval-ft-normal eval-ft-grug

plot:
	$(PYTHON) scripts/plot_results.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
