import os
import sys

# Set PyTorch allocator settings BEFORE importing torch to prevent memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import argparse
import logging
import datetime
import json
from typing import Any, Optional

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.config import config
from scripts.cuda.cuda_utils import (
    resolve_hf_model_id,
    patch_transformers_lazy_imports,
    load_causal_lm_model,
    load_causal_lm_tokenizer,
)
from transformers import TrainerCallback

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("dpo_cuda")


class ClearCacheCallback(TrainerCallback):
    """Frees cached VRAM after evaluation and periodically during DPO training steps."""

    def on_evaluate(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
        if state.global_step % 20 == 0:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def run_dpo_training(
    model_arg: str = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    adapter_path: str = "adapters/deepseek-r1-7b/20260804_040058/final_adapters",
    dpo_data_dir: str = "data/dpo",
    output_dir: str = "adapters/deepseek-r1-7b/dpo",
    epochs: int = 1,
    batch_size: int = 1,
    grad_accum: int = 8,
    learning_rate: float = 5e-7,
    beta: float = 0.1,
    max_length: int = 1536,
    max_prompt_length: int = 512,
) -> Any:
    """Executes DPO (Direct Preference Optimization) QLoRA training using Hugging Face TRL DPOTrainer."""
    patch_transformers_lazy_imports()

    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from peft import PeftModel, LoraConfig, prepare_model_for_kbit_training
    try:
        from trl import DPOConfig, DPOTrainer
        DPO_CONFIG_CLASS = DPOConfig
    except ImportError:
        from trl import DPOTrainer
        DPO_CONFIG_CLASS = TrainingArguments

    hf_model_id = resolve_hf_model_id(model_arg)
    logger.info("Base Hugging Face Model ID: %s", hf_model_id)

    is_cuda = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if is_cuda else 0
    logger.info("Target Device: CUDA (%d GPUs)" if is_cuda else "Target Device: CPU/MPS", num_gpus)

    # 1. Check for DPO datasets (or auto-generate from available SFT dataset)
    train_file = os.path.join(dpo_data_dir, "train.jsonl")
    valid_file = os.path.join(dpo_data_dir, "valid.jsonl")

    if not os.path.exists(train_file):
        logger.info("DPO dataset not found at %s. Attempting to auto-generate from downloaded SFT data...", train_file)
        from scripts.create_dpo_dataset import generate_dpo_dataset
        success = generate_dpo_dataset(data_dir=config.data_dir, dpo_dir=dpo_data_dir)
        if not success or not os.path.exists(train_file):
            logger.error("Failed to generate DPO dataset at %s", train_file)
            sys.exit(1)

    # Load datasets
    train_dataset = load_dataset("json", data_files=train_file, split="train")
    eval_dataset = load_dataset("json", data_files=valid_file, split="train") if os.path.exists(valid_file) else None

    # 2. Setup Quantization Config for Base Model
    model_kwargs = {}
    if is_cuda:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16

    logger.info("Loading Base Model...")
    model = load_causal_lm_model(hf_model_id, **model_kwargs)
    tokenizer = load_causal_lm_tokenizer(hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. Load Initial SFT LoRA Adapter if available, or create new LoraConfig
    if os.path.exists(adapter_path):
        logger.info("Loading baseline SFT LoRA Adapters from %s...", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path, is_trainable=True)
        peft_config = None
    else:
        logger.info("No adapter path found at %s. Creating new LoRA config...", adapter_path)
        model = prepare_model_for_kbit_training(model)
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            task_type="CAUSAL_LM",
        )

    # 4. Define DPO Training Arguments
    is_dpo_config = False
    try:
        from trl import DPOConfig
        training_args = DPOConfig(
            output_dir=final_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=5,
            save_strategy="steps",
            save_steps=20,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=20 if eval_dataset else None,
            fp16=is_cuda,
            optim="adamw_torch",
            remove_unused_columns=False,
            report_to="none",
            beta=beta,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )
        is_dpo_config = True
        logger.info("Using TRL DPOConfig for training setup.")
    except Exception as cfg_err:
        logger.info("Using standard TrainingArguments fallback: %s", cfg_err)
        training_args = TrainingArguments(
            output_dir=final_output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.1,
            logging_steps=5,
            save_strategy="steps",
            save_steps=20,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=20 if eval_dataset else None,
            fp16=is_cuda,
            optim="adamw_torch",
            remove_unused_columns=False,
            report_to="none",
        )

    # 5. Initialize DPOTrainer
    logger.info("Initializing DPOTrainer (beta=%.3f, lr=%.2e)...", beta, learning_rate)
    dpo_kwargs = {
        "model": model,
        "ref_model": None,  # PEFT uses adapter-disabled base model as reference implicitly
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": eval_dataset,
        "peft_config": peft_config,
        "callbacks": [ClearCacheCallback()],
    }

    if not is_dpo_config:
        dpo_kwargs["beta"] = beta
        dpo_kwargs["max_length"] = max_length
        dpo_kwargs["max_prompt_length"] = max_prompt_length

    # Inspect signature of DPOTrainer.__init__ dynamically for version compatibility
    import inspect
    sig = inspect.signature(DPOTrainer.__init__)
    param_names = set(sig.parameters.keys())

    if "processing_class" in param_names:
        dpo_kwargs["processing_class"] = tokenizer
    else:
        dpo_kwargs["tokenizer"] = tokenizer

    has_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
    )
    if not has_var_kwargs:
        dpo_kwargs = {k: v for k, v in dpo_kwargs.items() if k in param_names}

    dpo_trainer = DPOTrainer(**dpo_kwargs)

    logger.info("Starting DPO Fine-Tuning...")
    dpo_trainer.train()

    # 6. Save final DPO adapters
    final_adapter_dir = os.path.join(final_output_dir, "final_dpo_adapters")
    dpo_trainer.model.save_pretrained(final_adapter_dir)
    tokenizer.save_pretrained(final_adapter_dir)
    logger.info("DPO Fine-Tuning complete! Adapters saved to: %s", final_adapter_dir)

    return dpo_trainer


def main():
    parser = argparse.ArgumentParser(description="Run DPO Fine-Tuning on CUDA / MPS")
    parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B")
    parser.add_argument("--adapter-path", type=str, default="adapters/deepseek-r1-7b/20260804_040058/final_adapters")
    parser.add_argument("--dpo-data", type=str, default="data/dpo")
    parser.add_argument("--output-dir", type=str, default="adapters/deepseek-r1-7b/dpo")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=5e-7)
    parser.add_argument("--beta", type=float, default=0.1)

    args = parser.parse_args()

    run_dpo_training(
        model_arg=args.model,
        adapter_path=args.adapter_path,
        dpo_data_dir=args.dpo_data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
    )


if __name__ == "__main__":
    main()
