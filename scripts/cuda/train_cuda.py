import os
import sys
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

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("train_cuda")


def run_sft_training(
    model_arg: str = config.model_mlx_path,
    data_dir: str = config.data_dir,
    adapter_path: str = config.adapters,
    epochs: int = 3,
    batch_size: int = 2,
    grad_accum: int = 4,
    learning_rate: float = 2e-4,
    max_seq_length: int = 1536,
    lora_r: int = 16,
    lora_alpha: int = 32,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
) -> Any:
    """Executes SFT QLoRA fine-tuning training and returns the trainer instance."""
    patch_transformers_lazy_imports()

    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer

    hf_model_id = resolve_hf_model_id(model_arg)
    logger.info("Using Hugging Face Base Model ID: %s", hf_model_id)

    is_cuda = torch.cuda.is_available()
    num_gpus = torch.cuda.device_count() if is_cuda else 0
    if is_cuda:
        logger.info("CUDA Devices Available: %d", num_gpus)
        for g in range(num_gpus):
            logger.info("  GPU %d: %s", g, torch.cuda.get_device_name(g))
    else:
        logger.warning(
            "CUDA is not available on this system. Running in fallback mode (MPS/CPU) for testing."
        )

    train_file = os.path.join(data_dir, "train.jsonl")
    valid_file = os.path.join(data_dir, "valid.jsonl")

    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        logger.error("Required dataset files train.jsonl / valid.jsonl not found in %s", data_dir)
        sys.exit(1)

    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_adapter_dir = os.path.join(adapter_path, timestamp)
    os.makedirs(output_adapter_dir, exist_ok=True)
    logger.info("Output Adapter Directory: %s", output_adapter_dir)

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    is_ddp = local_rank != -1

    if model is None or tokenizer is None:
        logger.info("Loading Base Model...")
        bnb_config = None
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
            model_kwargs["torch_dtype"] = torch.float16

            if is_ddp:
                logger.info("DDP Active: Binding Process to GPU %d", local_rank)
                torch.cuda.set_device(local_rank)
                model_kwargs["device_map"] = {"": local_rank}
            else:
                model_kwargs["device_map"] = "auto"

        model = load_causal_lm_model(hf_model_id, **model_kwargs)
        tokenizer = load_causal_lm_tokenizer(hf_model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if is_cuda:
        model = prepare_model_for_kbit_training(model)
        # Cast any bfloat16 parameters to float16 to prevent dtype mismatch on T4 GPUs
        for name, param in model.named_parameters():
            if param.dtype == torch.bfloat16:
                param.data = param.data.to(torch.float16)

    # LoRA Config
    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    logger.info("Loading dataset splits...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": train_file,
            "validation": valid_file,
        },
    )

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    try:
        from trl import SFTConfig, SFTTrainer

        sft_config_kwargs = {
            "output_dir": output_adapter_dir,
            "per_device_train_batch_size": batch_size,
            "per_device_eval_batch_size": batch_size,
            "eval_accumulation_steps": 2,
            "gradient_accumulation_steps": grad_accum,
            "learning_rate": learning_rate,
            "lr_scheduler_type": "cosine",
            "warmup_ratio": 0.03,
            "fp16": False,  # BitsAndBytes handles FP16 compute natively; disable AMP GradScaler to prevent unscale_ error
            "bf16": False,
            "gradient_checkpointing": True,
            "logging_steps": 10,
            "eval_strategy": "steps",
            "eval_steps": 50,
            "save_steps": 50,
            "num_train_epochs": epochs,
            "save_total_limit": 2,
            "report_to": "none",
            "dataset_text_field": "text",
            "max_length": max_seq_length,
        }
        # Use standard 'nll' loss to prevent TRL from defaulting to 'chunked_nll' (which crashes on 4-bit PEFT models)
        try:
            sft_config_kwargs["loss_type"] = "nll"
            sft_config = SFTConfig(**sft_config_kwargs)
        except Exception:
            sft_config_kwargs.pop("loss_type", None)
            sft_config = SFTConfig(**sft_config_kwargs)

        try:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                peft_config=peft_config,
                processing_class=tokenizer,
                args=sft_config,
            )
        except TypeError:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                peft_config=peft_config,
                tokenizer=tokenizer,
                args=sft_config,
            )
    except Exception as exc:
        logger.warning("Initializing with SFTConfig failed (%s). Falling back to TrainingArguments...", exc)
        training_args = TrainingArguments(
            output_dir=output_adapter_dir,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            eval_accumulation_steps=2,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            lr_scheduler_type="cosine",
            warmup_ratio=0.03,
            fp16=False,
            bf16=False,
            gradient_checkpointing=True,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=50,
            save_steps=50,
            num_train_epochs=epochs,
            save_total_limit=2,
            report_to="none",
        )
        try:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                peft_config=peft_config,
                processing_class=tokenizer,
                args=training_args,
            )
        except TypeError:
            trainer = SFTTrainer(
                model=model,
                train_dataset=dataset["train"],
                eval_dataset=dataset["validation"],
                peft_config=peft_config,
                tokenizer=tokenizer,
                args=training_args,
            )

    logger.info("Starting SFT Training...")
    train_result = trainer.train()

    # Save log history to metrics.json
    metrics_path = os.path.join(output_adapter_dir, "metrics.json")
    try:
        log_history = getattr(trainer.state, "log_history", [])
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump({"log_history": log_history}, f, indent=2)
        logger.info("Saved training metrics to: %s", metrics_path)
    except Exception as me:
        logger.warning("Could not save metrics.json: %s", me)

    # Plot loss curve
    try:
        from scripts.cuda.plot_loss import plot_latest_training_loss
        plot_latest_training_loss()
    except Exception as pe:
        logger.warning("Could not generate loss plot: %s", pe)

    final_dir = os.path.join(output_adapter_dir, "final_adapters")
    logger.info("Saving fine-tuned LoRA adapters to %s...", final_dir)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Training Completed Successfully!")
    return trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="CUDA / Cross-Platform QLoRA Fine-Tuning script")
    parser.add_argument("--model", type=str, default=config.model_mlx_path)
    parser.add_argument("--data", type=str, default=config.data_dir)
    parser.add_argument("--adapter-path", type=str, default=config.adapters)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    args = parser.parse_args()

    run_sft_training(
        model_arg=args.model,
        data_dir=args.data,
        adapter_path=args.adapter_path,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
    )


if __name__ == "__main__":
    main()
