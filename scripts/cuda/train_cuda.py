import os
import sys
import argparse
import logging
import datetime
import json
from typing import Optional

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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CUDA / Cross-Platform QLoRA Fine-Tuning script using Hugging Face transformers/trl"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.model_mlx_path,
        help="Hugging Face model ID (e.g. deepseek-ai/DeepSeek-R1-Distill-Qwen-7B)",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=config.data_dir,
        help="Directory containing train.jsonl and valid.jsonl",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=config.adapters,
        help="Base directory to save output LoRA adapters",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Per-device train batch size (default: 2)",
    )
    parser.add_argument(
        "--grad-accum",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=1536,
        help="Maximum sequence length for training",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="LoRA rank (default: 16)",
    )
    parser.add_argument(
        "--lora-alpha",
        type=int,
        default=32,
        help="LoRA alpha (default: 32)",
    )

    args = parser.parse_args()

    # Apply patch to prevent BloomPreTrainedModel ModuleNotFoundError in Kaggle/Colab
    patch_transformers_lazy_imports()

    import torch
    from datasets import load_dataset
    from transformers import TrainingArguments
    from peft import LoraConfig, prepare_model_for_kbit_training
    from trl import SFTTrainer

    hf_model_id = resolve_hf_model_id(args.model)
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

    train_file = os.path.join(args.data, "train.jsonl")
    valid_file = os.path.join(args.data, "valid.jsonl")

    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        logger.error("Required dataset files train.jsonl / valid.jsonl not found in %s", args.data)
        sys.exit(1)

    # Setup output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_adapter_dir = os.path.join(args.adapter_path, timestamp)
    os.makedirs(output_adapter_dir, exist_ok=True)
    logger.info("Output Adapter Directory: %s", output_adapter_dir)

    # Configure 4-bit Quantization (QLoRA) if CUDA is available
    bnb_config = None
    if is_cuda:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    logger.info("Loading Base Model...")
    model_kwargs = {}
    if is_cuda:
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16

    model = load_causal_lm_model(hf_model_id, **model_kwargs)
    tokenizer = load_causal_lm_tokenizer(hf_model_id)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if is_cuda:
        model = prepare_model_for_kbit_training(model)

    # LoRA Config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
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

    training_args = TrainingArguments(
        output_dir=output_adapter_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        fp16=is_cuda,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=50,
        save_steps=50,
        num_train_epochs=args.epochs,
        save_total_limit=2,
        report_to="none",
        use_mps_device=bool(torch.backends.mps.is_available() and not is_cuda),
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        tokenizer=tokenizer,
        args=training_args,
    )

    logger.info("Starting SFT Training...")
    trainer.train()

    final_dir = os.path.join(output_adapter_dir, "final_adapters")
    logger.info("Saving fine-tuned LoRA adapters to %s...", final_dir)
    trainer.model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    logger.info("Training Completed Successfully!")


if __name__ == "__main__":
    main()
