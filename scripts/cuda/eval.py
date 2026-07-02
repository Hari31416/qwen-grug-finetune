import os
import sys
import json
import time
import argparse
import logging
import re
from typing import Dict, Any, List, Optional
from threading import Thread
import torch
from datasets import load_dataset
from transformers import TextIteratorStreamer

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.config import config
from scripts.prompt_utils import build_user_prompt
from scripts.cuda.generation_utils import (
    load_model_and_tokenizer,
    get_generation_parameters,
    parse_thinking_and_answer,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("eval_cuda")


GRUG_SYSTEM_PROMPT: str = (
    "You must write your thinking process in a token-efficient, telegraphic \"Grug\" style using short, sentence-fragment-based prose.\n"
    "Follow these style rules exactly:\n"
    "- Drop articles like \"the\" and \"a\" where possible.\n"
    "- Use telegraphic fragments rather than complete sentences.\n"
    "- Keep numbers, equations, math symbols, variables, and code tokens exactly intact.\n"
    "- Avoid any meta-commentary, filler phrasing, self-corrections, or back-tracking markers (e.g., \"wait...\", \"okay...\", \"let us see\").\n"
    "- Keep logical transitions and step-by-step intermediate derivations. Never skip steps to make reasoning shorter; only make the phrasing of those steps shorter.\n"
    "- Do not repeat statements.\n"
    "- Output only the thinking process in this style, and then end the thinking block with </think>."
)


def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract the normalized numeric answer from a model's final response text."""
    if not text:
        return None

    # Check LaTeX boxed format first (take the last boxed match)
    boxed_matches = re.findall(r"\\boxed\{([^\}]+)\}", text)
    if boxed_matches:
        val = boxed_matches[-1]
        # Clean LaTeX formatting and spaces specifically inside the boxed value
        val = val.replace("\\$", "").replace("$", "")
        val = val.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
        val = val.replace(" ", "")
        val = re.sub(r"(\d),(\d)", r"\1\2", val)

        num_matches = re.findall(r"-?\d+(?:\.\d+)?", val)
        if num_matches:
            ans = num_matches[-1]
            if ans.endswith(".0"):
                ans = ans[:-2]
            return ans

    # Fallback to the last number in the text
    cleaned_text = text.replace("\\$", "").replace("$", "")
    cleaned_text = cleaned_text.replace("\\!", "").replace("\\,", "").replace("\\ ", "")
    cleaned_text = re.sub(r"(\d),(\d)", r"\1\2", cleaned_text)

    num_matches = re.findall(r"-?\d+(?:\.\d+)?", cleaned_text)
    if num_matches:
        ans = num_matches[-1]
        if ans.endswith(".0"):
            ans = ans[:-2]
        return ans

    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate model on benchmarks using PyTorch/CUDA")
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
        "--prompt-style",
        type=str,
        default="normal",
        choices=["normal", "grug"],
        help="Style of system instructions (normal or grug)",
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
        help="Custom path to LoRA adapter (defaults to config.adapters)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of evaluation samples",
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
        "--max-tokens",
        type=int,
        default=config.eval_max_generation_tokens,
        help="Max tokens to generate per problem",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation. If 1, runs sequential generation (real-time progress, individual latency). If >1, runs batch generation for speed.",
    )

    args = parser.parse_args()

    # Determine adapter path if adapter is enabled
    adapter_path: Optional[str] = None
    if args.adapter:
        if args.adapter_path:
            adapter_path = args.adapter_path
        else:
            # Look for the latest timestamped subdirectory under config.adapters
            base_adapter_dir = config.adapters
            resolved_latest = None
            if os.path.exists(base_adapter_dir):
                subdirs = [
                    os.path.join(base_adapter_dir, d)
                    for d in os.listdir(base_adapter_dir)
                    if os.path.isdir(os.path.join(base_adapter_dir, d))
                ]
                # Filter directories containing adapter files
                valid_subdirs = [
                    sd for sd in subdirs if os.path.exists(os.path.join(sd, "adapter_model.safetensors")) or os.path.exists(os.path.join(sd, "adapters.safetensors"))
                ]
                if valid_subdirs:
                    resolved_latest = max(valid_subdirs)
            
            if resolved_latest:
                adapter_path = resolved_latest
                logger.info("Automatically resolved latest adapter path: %s", adapter_path)
            else:
                adapter_path = base_adapter_dir
                logger.info("No timestamped adapter subdirectories found. Using default adapter path: %s", adapter_path)
    else:
        logger.info("Loading base model: %s", config.model_hf_path)

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model_hf_path, adapter_path=adapter_path, quantized=config.model_quantized)

    # Load benchmark dataset
    if args.benchmark == "gsm8k":
        logger.info("Loading GSM8K split='%s'...", args.split)
        dataset = load_dataset("openai/gsm8k", "main", split=args.split)
    else:
        logger.error("Unsupported benchmark: %s", args.benchmark)
        sys.exit(1)

    # Apply limit if specified
    samples = list(dataset)
    if args.limit is not None:
        logger.info("Limiting evaluation to first %d samples", args.limit)
        samples = samples[: args.limit]

    # Setup generation configuration
    generation_params = get_generation_parameters(
        temp=args.temp,
        top_p=args.top_p,
        repetition_penalty=1.1,
        presence_penalty=0.2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results: List[Dict[str, Any]] = []

    logger.info("Starting evaluation of %d samples...", len(samples))
    if args.batch_size > 1:
        logger.info("Building formatted prompts for batch generation...")
        formatted_prompts = []
        for row in samples:
            question = row["question"]
            full_prompt_text = build_user_prompt(question, "gsm8k")
            messages = []
            if args.prompt_style == "grug":
                messages.append({"role": "system", "content": GRUG_SYSTEM_PROMPT})
            messages.append({"role": "user", "content": full_prompt_text})
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            formatted_prompts.append(formatted_prompt)

        logger.info("Running batch generation (batch_size=%d)...", args.batch_size)
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token

        start_time = time.perf_counter()
        
        # Tokenize with padding
        inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generation_params
            )
            
        total_latency = time.perf_counter() - start_time
        avg_latency = total_latency / len(samples) if len(samples) > 0 else 0.0

        logger.info("Parsing batch results...")
        for i, row in enumerate(samples, 1):
            question = row["question"]
            raw_ground_truth = row["answer"]
            ground_truth = raw_ground_truth.split("####")[-1].strip()
            
            # Extract generated portion
            input_len = inputs.input_ids.shape[-1]
            generated_ids = outputs[i - 1][input_len:]
            output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Parse thinking block
            thinking_content, answer_content = parse_thinking_and_answer(output_text, strip_prefix=False)
            format_compliance = len(answer_content) > 0

            thinking_tokens = len(tokenizer.encode(thinking_content))
            answer_tokens = len(tokenizer.encode(answer_content))
            total_tokens = thinking_tokens + answer_tokens

            predicted_answer = extract_numeric_answer(answer_content)
            correct = (
                (predicted_answer == ground_truth)
                if (predicted_answer is not None and ground_truth is not None)
                else False
            )

            record = {
                "id": i,
                "question": question,
                "ground_truth": ground_truth,
                "output": output_text,
                "thinking_content": thinking_content,
                "answer_content": answer_content,
                "predicted_answer": predicted_answer,
                "correct": correct,
                "thinking_tokens": thinking_tokens,
                "answer_tokens": answer_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": avg_latency,
                "tokens_per_second": total_tokens / avg_latency if avg_latency > 0 else 0.0,
                "format_compliance": format_compliance,
            }
            results.append(record)

            logger.info(
                "[%d/%d] Acc: %s | Think Tok: %d | Ans Tok: %d | Compl: %s | Pred: %s | GT: %s",
                i,
                len(samples),
                "✓" if correct else "✗",
                thinking_tokens,
                answer_tokens,
                "✓" if format_compliance else "✗",
                predicted_answer,
                ground_truth,
            )
    else:
        for i, row in enumerate(samples, 1):
            question = row["question"]
            raw_ground_truth = row["answer"]
            ground_truth = raw_ground_truth.split("####")[-1].strip()

            # Build user prompt with final answer formatting instruction
            full_prompt_text = build_user_prompt(question, "gsm8k")

            # Construct messages using tokenizer's template
            messages = []
            if args.prompt_style == "grug":
                messages.append({"role": "system", "content": GRUG_SYSTEM_PROMPT})
            messages.append({"role": "user", "content": full_prompt_text})

            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            start_time = time.perf_counter()
            
            inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
            streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
            
            generation_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=args.max_tokens,
                pad_token_id=tokenizer.eos_token_id,
                **generation_params
            )
            
            thread = Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()
            
            output_text = ""
            for new_text in streamer:
                output_text += new_text
            
            thread.join()

            latency_seconds = time.perf_counter() - start_time

            # Parse thinking block
            thinking_content, answer_content = parse_thinking_and_answer(output_text, strip_prefix=False)
            format_compliance = len(answer_content) > 0

            # Token counts
            thinking_tokens = len(tokenizer.encode(thinking_content))
            answer_tokens = len(tokenizer.encode(answer_content))
            total_tokens = thinking_tokens + answer_tokens

            # Speed calculation
            tokens_per_second = (
                total_tokens / latency_seconds if latency_seconds > 0 else 0.0
            )

            # Extract answer and check correctness
            predicted_answer = extract_numeric_answer(answer_content)
            correct = (
                (predicted_answer == ground_truth)
                if (predicted_answer is not None and ground_truth is not None)
                else False
            )

            record = {
                "id": i,
                "question": question,
                "ground_truth": ground_truth,
                "output": output_text,
                "thinking_content": thinking_content,
                "answer_content": answer_content,
                "predicted_answer": predicted_answer,
                "correct": correct,
                "thinking_tokens": thinking_tokens,
                "answer_tokens": answer_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": latency_seconds,
                "tokens_per_second": tokens_per_second,
                "format_compliance": format_compliance,
            }
            results.append(record)

            logger.info(
                "[%d/%d] Acc: %s | Think Tok: %d | Ans Tok: %d | Compl: %s | Pred: %s | GT: %s",
                i,
                len(samples),
                "✓" if correct else "✗",
                thinking_tokens,
                answer_tokens,
                "✓" if format_compliance else "✗",
                predicted_answer,
                ground_truth,
            )

    # Compute summary statistics
    total_count = len(results)
    if total_count > 0:
        correct_count = sum(1 for r in results if r["correct"])
        format_compliant_count = sum(1 for r in results if r["format_compliance"])

        accuracy = correct_count / total_count
        format_compliance_rate = format_compliant_count / total_count

        mean_thinking_tokens = sum(r["thinking_tokens"] for r in results) / total_count
        mean_answer_tokens = sum(r["answer_tokens"] for r in results) / total_count
        mean_total_tokens = sum(r["total_tokens"] for r in results) / total_count
        mean_latency = sum(r["latency_seconds"] for r in results) / total_count
        mean_tokens_per_second = sum(r["tokens_per_second"] for r in results) / total_count
    else:
        accuracy = 0.0
        format_compliance_rate = 0.0
        mean_thinking_tokens = 0.0
        mean_answer_tokens = 0.0
        mean_total_tokens = 0.0
        mean_latency = 0.0
        mean_tokens_per_second = 0.0
        correct_count = 0
        format_compliant_count = 0

    logger.info("=== Evaluation Summary ===")
    logger.info("Sample Count:           %d", total_count)
    logger.info("Accuracy:               %.4f (%d/%d)", accuracy, correct_count, total_count)
    logger.info("Format Compliance:      %.4f (%d/%d)", format_compliance_rate, format_compliant_count, total_count)
    logger.info("Mean Thinking Tokens:   %.2f", mean_thinking_tokens)
    logger.info("Mean Answer Tokens:     %.2f", mean_answer_tokens)
    logger.info("Mean Total Tokens:      %.2f", mean_total_tokens)
    logger.info("Mean Latency (s):       %.2f", mean_latency)
    logger.info("Mean Speed (tok/s):     %.2f", mean_tokens_per_second)

    # Setup results output directory
    subfolder = "finetuned" if args.adapter else "baseline"
    output_dir = os.path.join(config.results, subfolder)
    os.makedirs(output_dir, exist_ok=True)

    filename = f"{args.benchmark}_{args.prompt_style}.json"
    if args.prompt_style == "grug":
        filename = f"{args.benchmark}_grug_prompt.json"

    output_path = os.path.join(output_dir, filename)

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


if __name__ == "__main__":
    main()
