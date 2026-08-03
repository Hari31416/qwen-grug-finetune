import os
import sys
import json
import time
import argparse
import logging
from typing import Dict, Any, List, Optional

# Add workspace root to Python path to import config & utilities
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.config import config
from scripts.prompt_utils import build_user_prompt, STYLE_SYSTEM_PROMPT
from scripts.generation_utils import parse_thinking_and_answer
from scripts.eval import extract_numeric_answer
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
logger = logging.getLogger("eval_cuda")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model on GSM8K using CUDA / MPS / CPU"
    )
    parser.add_argument(
        "--model",
        type=str,
        default=config.model_mlx_path,
        help="Hugging Face model ID or path",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="gsm8k",
        choices=["gsm8k"],
        help="The benchmark to run evaluation on",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="The dataset split to evaluate (e.g. test)",
    )
    parser.add_argument(
        "--adapter",
        action="store_true",
        help="Whether to load the LoRA adapter",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="",
        help="Path to fine-tuned LoRA adapter directory",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation samples",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for batch inference",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.eval_max_generation_tokens,
        help="Max tokens to generate per problem",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=config.temperature,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=config.top_p,
        help="Sampling top-p",
    )
    parser.add_argument(
        "--no-system-prompt",
        action="store_true",
        help="Disable style system prompt",
    )

    args = parser.parse_args()

    # Apply patch to prevent BloomPreTrainedModel ModuleNotFoundError in Kaggle/Colab
    patch_transformers_lazy_imports()

    import torch
    from datasets import load_dataset

    hf_model_id = resolve_hf_model_id(args.model)
    logger.info("Base Hugging Face Model ID: %s", hf_model_id)

    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Target Device: %s", device)

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
    if not is_cuda:
        model.to(device)

    tokenizer = load_causal_lm_tokenizer(hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.adapter:
        adapter_path = args.adapter_path
        if not adapter_path:
            logger.error("--adapter flag requires --adapter-path specified")
            sys.exit(1)
        logger.info("Loading PEFT LoRA Adapter from: %s", adapter_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)


def run_gsm8k_eval(
    model: Any,
    tokenizer: Any,
    limit: Optional[int] = 50,
    batch_size: int = 4,
    max_tokens: int = 1024,
    temp: float = 0.0,
    top_p: float = 1.0,
    no_system_prompt: bool = False,
    split: str = "test",
    is_adapter: bool = False,
) -> Dict[str, Any]:
    """Runs GSM8K evaluation on a given model and tokenizer and returns summary metrics."""
    import torch
    from datasets import load_dataset

    is_cuda = torch.cuda.is_available()
    device = (
        "cuda" if is_cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    model.eval()

    logger.info("Loading Benchmark Dataset: gsm8k (split='%s')...", split)
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    samples = list(dataset)
    if limit is not None:
        logger.info("Limiting evaluation to first %d samples", limit)
        samples = samples[:limit]

    results: List[Dict[str, Any]] = []
    total_count = len(samples)
    logger.info("Starting Evaluation of %d samples...", total_count)

    for i in range(0, total_count, batch_size):
        batch_samples = samples[i : i + batch_size]
        batch_prompts = []
        batch_gts = []

        for row in batch_samples:
            question = row["question"]
            raw_gt = row["answer"]
            gt = raw_gt.split("####")[-1].strip()
            batch_gts.append(gt)

            full_prompt_text = build_user_prompt(question, "gsm8k")
            messages = []
            if not no_system_prompt:
                messages.append({"role": "system", "content": STYLE_SYSTEM_PROMPT})
            messages.append({"role": "user", "content": full_prompt_text})

            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            batch_prompts.append(formatted_prompt)

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        start_time = time.perf_counter()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temp if temp > 0 else None,
                top_p=top_p if temp > 0 else None,
                do_sample=(temp > 0),
                pad_token_id=tokenizer.pad_token_id,
            )

        latency = time.perf_counter() - start_time
        avg_latency = latency / len(batch_samples)

        input_len = inputs["input_ids"].shape[1]
        for b_idx, output_tokens in enumerate(outputs):
            gen_tokens = output_tokens[input_len:]
            output_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

            gt = batch_gts[b_idx]
            sample_idx = i + b_idx + 1

            thinking_content, answer_content = parse_thinking_and_answer(
                output_text, strip_prefix=False
            )
            format_compliance = len(answer_content) > 0

            thinking_tokens = len(tokenizer.encode(thinking_content))
            answer_tokens = len(tokenizer.encode(answer_content))
            total_tokens = thinking_tokens + answer_tokens

            predicted_answer = extract_numeric_answer(answer_content)
            correct = (predicted_answer == gt) if (predicted_answer and gt) else False

            record = {
                "id": sample_idx,
                "question": batch_samples[b_idx]["question"],
                "ground_truth": gt,
                "output": output_text,
                "thinking_content": thinking_content,
                "answer_content": answer_content,
                "predicted_answer": predicted_answer,
                "correct": correct,
                "thinking_tokens": thinking_tokens,
                "answer_tokens": answer_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": avg_latency,
                "tokens_per_second": (
                    total_tokens / avg_latency if avg_latency > 0 else 0.0
                ),
                "format_compliance": format_compliance,
            }
            results.append(record)

    # Compute Summary Statistics
    if total_count > 0:
        correct_count = sum(1 for r in results if r["correct"])
        format_compliant_count = sum(1 for r in results if r["format_compliance"])
        accuracy = correct_count / total_count
        format_compliance_rate = format_compliant_count / total_count
        mean_thinking_tokens = sum(r["thinking_tokens"] for r in results) / total_count
        mean_answer_tokens = sum(r["answer_tokens"] for r in results) / total_count
        mean_total_tokens = sum(r["total_tokens"] for r in results) / total_count
        mean_latency = sum(r["latency_seconds"] for r in results) / total_count
        mean_tokens_per_second = (
            sum(r["tokens_per_second"] for r in results) / total_count
        )
    else:
        accuracy = format_compliance_rate = mean_thinking_tokens = (
            mean_answer_tokens
        ) = 0.0
        mean_total_tokens = mean_latency = mean_tokens_per_second = correct_count = (
            format_compliant_count
        ) = 0

    subfolder = "finetuned" if is_adapter else "baseline"
    output_dir = os.path.join(config.results, subfolder)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gsm8k.json")

    output_data = {
        "summary": {
            "accuracy": accuracy,
            "format_compliance_rate": format_compliance_rate,
            "mean_thinking_tokens": mean_thinking_tokens,
            "mean_answer_tokens": mean_answer_tokens,
            "mean_total_tokens": mean_total_tokens,
            "mean_latency": mean_latency,
            "mean_tokens_per_second": mean_tokens_per_second,
            "sample_count": total_count,
            "correct_count": correct_count,
            "format_compliant_count": format_compliant_count,
        },
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    logger.info("Results saved to: %s", output_path)
    return output_data["summary"]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate model on GSM8K using CUDA / MPS / CPU"
    )
    parser.add_argument("--model", type=str, default=config.model_mlx_path)
    parser.add_argument("--benchmark", type=str, default="gsm8k")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--adapter", action="store_true")
    parser.add_argument("--adapter-path", type=str, default="")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--max-tokens", type=int, default=config.eval_max_generation_tokens
    )
    parser.add_argument("--temp", type=float, default=config.temperature)
    parser.add_argument("--top-p", type=float, default=config.top_p)
    parser.add_argument("--no-system-prompt", action="store_true")
    args = parser.parse_args()

    patch_transformers_lazy_imports()

    import torch

    hf_model_id = resolve_hf_model_id(args.model)
    is_cuda = torch.cuda.is_available()
    device = (
        "cuda" if is_cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

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

    model = load_causal_lm_model(hf_model_id, **model_kwargs)
    if not is_cuda:
        model.to(device)

    tokenizer = load_causal_lm_tokenizer(hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    if args.adapter:
        if not args.adapter_path:
            logger.error("--adapter flag requires --adapter-path specified")
            sys.exit(1)
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path)

    run_gsm8k_eval(
        model=model,
        tokenizer=tokenizer,
        limit=args.limit,
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        temp=args.temp,
        top_p=args.top_p,
        no_system_prompt=args.no_system_prompt,
        split=args.split,
        is_adapter=args.adapter,
    )


if __name__ == "__main__":
    main()
