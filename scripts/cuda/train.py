import os
import sys
import argparse
import logging
import datetime
import yaml
import json
from typing import List, Optional, Dict, Any
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import LoraConfig, get_peft_model, TaskType

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train_cuda")


def run_training(
    model_id: str,
    adapter_path: str,
    data_dir: str,
    config_file: str,
    iters: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lora_layers: Optional[int] = None,
    save_every: int = 20,
) -> None:
    """Loads model and runs native Hugging Face SFT fine-tuning with PEFT LoRA."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"LoRA config file not found: {config_file}")

    # Load configuration parameters
    with open(config_file, "r") as f:
        config_data = yaml.safe_load(f)

    # Extract LoRA settings
    lora_params = config_data.get("lora_parameters", {})
    rank = lora_params.get("rank", 8)
    alpha = lora_params.get("alpha", 16)
    dropout = lora_params.get("dropout", 0.05)
    keys = lora_params.get("keys", ["q_proj", "v_proj"])
    
    # Map MLX style key names (e.g. self_attn.q_proj) to leaf module names for PEFT
    target_modules = [k.split(".")[-1] for k in keys]

    # Hyperparameters
    train_batch_size = batch_size if batch_size is not None else config_data.get("batch_size", 2)
    max_steps = iters if iters is not None else config_data.get("iters", 800)
    lr = learning_rate if learning_rate is not None else float(config_data.get("learning_rate", 1e-5))

    logger.info("Initializing Hugging Face model and tokenizer...")
    from scripts.cuda.generation_utils import load_model_and_tokenizer
    model, tokenizer = load_model_and_tokenizer(model_id, quantized=config.model_quantized)
    tokenizer.padding_side = "right"  # Padding side right is standard for SFT packing/training

    peft_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    if config.model_quantized:
        from peft import prepare_model_for_kbit_training
        logger.info("Preparing quantized model for k-bit training...")
        model = prepare_model_for_kbit_training(model)

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load SFT datasets (formatted as {"text": "..."})
    train_file = os.path.join(data_dir, "train.jsonl")
    valid_file = os.path.join(data_dir, "valid.jsonl")

    logger.info("Loading SFT datasets: %s and %s", train_file, valid_file)
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": valid_file,
        },
    )

    def tokenize_function(examples):
        inputs = tokenizer(examples["text"], truncation=True, max_length=2048)
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    tokenized_datasets = dataset.map(
        tokenize_function, remove_columns=["text"], batched=True
    )

    # Configure training arguments
    training_args = TrainingArguments(
        output_dir=adapter_path,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=train_batch_size,
        learning_rate=lr,
        max_steps=max_steps,
        logging_steps=max(1, save_every // 2),
        eval_strategy="steps",
        eval_steps=save_every,
        save_strategy="steps",
        save_steps=save_every,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=cuda_available and torch.cuda.is_bf16_supported(),
        fp16=cuda_available and not torch.cuda.is_bf16_supported(),
        report_to="none",
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        data_collator=DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt"),
    )

    trainer.train()

    logger.info("Saving best adapters to: %s", adapter_path)
    model.save_pretrained(adapter_path)
    tokenizer.save_pretrained(adapter_path)

    # Extract log history to reconstruct training metrics
    train_steps: List[int] = []
    train_losses: List[float] = []
    val_steps: List[int] = []
    val_losses: List[float] = []

    for entry in trainer.state.log_history:
        step = entry.get("step")
        if "loss" in entry:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        elif "eval_loss" in entry:
            val_steps.append(step)
            val_losses.append(entry["eval_loss"])

    # Save metrics JSON file
    metrics = {
        "train_steps": train_steps,
        "train_losses": train_losses,
        "val_steps": val_steps,
        "val_losses": val_losses,
    }
    metrics_file = os.path.join(adapter_path, "metrics.json")
    try:
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info("Saved training metrics to: %s", metrics_file)
    except Exception as e:
        logger.error("Failed to save metrics JSON: %s", e)

    # Replicate plotting logic
    if train_losses or val_losses:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            if train_losses:
                plt.plot(train_steps, train_losses, label="Train Loss", marker="o", linestyle="-", color="#1f77b4")
            if val_losses:
                plt.plot(val_steps, val_losses, label="Validation Loss", marker="s", linestyle="--", color="#ff7f0e")
            
            plt.xlabel("Iteration")
            plt.ylabel("Loss")
            plt.title("LoRA SFT Training & Validation Loss (CUDA)")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.6)
            
            plot_file = os.path.join(adapter_path, "loss_plot.png")
            plt.savefig(plot_file, dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Saved loss plot to: %s", plot_file)
        except Exception as pe:
            logger.error("Failed to generate and save loss plot: %s", pe)


def main() -> None:
    workspace_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    default_config_file: str = os.path.join(workspace_root, "lora_config.yaml")

    parser = argparse.ArgumentParser(description="Native SFT LoRA Training on CUDA")
    parser.add_argument(
        "--model",
        type=str,
        default=config.model_hf_path,
        help="Path/repo name of the base Hugging Face model",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=config.adapters,
        help="Directory to save the fine-tuned adapter weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=config.data_dir,
        help="Directory containing train.jsonl and valid.jsonl",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=default_config_file,
        help="Path to the LoRA configuration YAML file",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override number of training iterations (steps)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=None,
        help="Layers override (unused in this HF config)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=50,
        help="Interval in steps to run evaluation and checkpoint save",
    )

    args = parser.parse_args()

    # Pre-checks
    if not os.path.exists(args.data):
        logger.error("Dataset directory does not exist: %s", args.data)
        sys.exit(1)

    train_file = os.path.join(args.data, "train.jsonl")
    valid_file = os.path.join(args.data, "valid.jsonl")
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        logger.error("Required datasets train.jsonl and valid.jsonl not found in data directory: %s", args.data)
        sys.exit(1)

    # Resolve output directory with timestamp to avoid overwriting previous runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.adapter_path = os.path.join(args.adapter_path, timestamp)
    os.makedirs(args.adapter_path, exist_ok=True)

    logger.info("Initializing native CUDA SFT LoRA fine-tuning...")
    logger.info("  Base Model:        %s", args.model)
    logger.info("  Adapter Path:      %s", args.adapter_path)
    logger.info("  Data Directory:    %s", args.data)
    logger.info("  Config File:       %s", args.config_file)
    logger.info("  Save & Eval Every: %d", args.save_every)

    run_training(
        model_id=args.model,
        adapter_path=args.adapter_path,
        data_dir=args.data,
        config_file=args.config_file,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_layers=args.lora_layers,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
