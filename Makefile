# Makefile for Qwen SFT Fine-Tuning, Evaluation, and Visualizations

PYTHON = uv run python
LIMIT = 100
BATCH_SIZE = 4
SAVE_EVERY = 20

.PHONY: help train eval-base eval-ft eval-all plot clean

help:
	@echo "Available commands:"
	@echo "  make train          Run SFT training (hyperparams from lora_config.yaml)"
	@echo "  make eval-base      Evaluate base model with style system prompt (default: LIMIT=100, BATCH_SIZE=16)"
	@echo "  make eval-ft        Evaluate fine-tuned model with style system prompt"
	@echo "  make eval-all       Run base and fine-tuned evaluations sequentially"
	@echo "  make plot           Generate comparison dashboard and separate plots"
	@echo "  make clean          Clean Python cache files"

train:
	$(PYTHON) scripts/train.py --save-every $(SAVE_EVERY)

eval-base:
	$(PYTHON) scripts/eval.py --benchmark gsm8k --split test --limit $(LIMIT) --batch-size $(BATCH_SIZE)

eval-ft:
	$(PYTHON) scripts/eval.py --benchmark gsm8k --split test --limit $(LIMIT) --batch-size $(BATCH_SIZE) --adapter

eval-all: eval-base eval-ft

plot:
	$(PYTHON) scripts/plot_results.py

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
