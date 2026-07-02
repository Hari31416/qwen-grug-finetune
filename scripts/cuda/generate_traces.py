import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Set, List
import torch

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.config import config
from scripts.prompt_utils import build_user_prompt, extract_predicted_answer, is_correct_answer
from scripts.cuda.generation_utils import (
    load_model_and_tokenizer,
    get_generation_parameters,
    parse_thinking_and_answer,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generate_traces")


VALID_SOURCES = ("strategyqa", "logiqa", "boolq", "anli", "piqa", "reclor")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw CoT traces using target HF model on CUDA")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of prompts to process (useful for pilot runs)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        choices=VALID_SOURCES,
        help="Only process prompts from this dataset source (e.g. boolq for pilot)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generation. If 1, runs sequential generation (real-time progress). If >1, runs batch generation for speed.",
    )
    args = parser.parse_args()

    # Load prompts
    if not os.path.exists(config.sft_prompts):
        logger.error("Prompts file not found at: %s. Please run sample_sft_prompts.py first.", config.sft_prompts)
        sys.exit(1)

    prompts: List[Dict[str, Any]] = []
    with open(config.sft_prompts, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                prompts.append(json.loads(line.strip()))

    logger.info("Loaded %d prompts from %s.", len(prompts), config.sft_prompts)

    if args.source is not None:
        prompts = [p for p in prompts if p["source"] == args.source]
        logger.info("Filtered to %d prompts from source=%s.", len(prompts), args.source)
        if not prompts:
            logger.error("No prompts found for source=%s.", args.source)
            sys.exit(1)

    if args.limit is not None:
        prompts = prompts[: args.limit]
        logger.info("Limiting execution to the first %d prompts for this run.", args.limit)

    # Setup directories and determine output file path
    config.setup_directories()
    output_file = os.path.join(config.raw_traces, "traces.jsonl")
    logger.info("Output traces file will be saved at: %s", output_file)

    # Load existing generated prompts to support resume capability
    existing_ids: Set[str] = set()
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        record = json.loads(line.strip())
                        existing_ids.add(record["id"])
                    except json.JSONDecodeError:
                        continue
        logger.info("Found %d already generated traces. Skipping them (resume mode).", len(existing_ids))

    # Filter out already processed prompts
    prompts_to_process = [p for p in prompts if p["id"] not in existing_ids]
    if not prompts_to_process:
        logger.info("All prompts in target set have already been generated. Nothing to do!")
        return

    logger.info("Processing %d prompts...", len(prompts_to_process))

    # Load HF model and tokenizer
    model, tokenizer = load_model_and_tokenizer(config.model_hf_path, quantized=config.model_quantized)

    # Create generation configuration
    generation_params = get_generation_parameters(
        temp=config.temperature,
        top_p=config.top_p,
        repetition_penalty=1.1,
        presence_penalty=0.2,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Open output file in append mode
    with open(output_file, "a", encoding="utf-8") as out_f:
        if args.batch_size > 1:
            logger.info("Running batch generation (batch_size=%d)...", args.batch_size)
            
            # Configure left padding and pad token for batching
            tokenizer.padding_side = "left"
            tokenizer.pad_token = tokenizer.eos_token

            for i in range(0, len(prompts_to_process), args.batch_size):
                chunk = prompts_to_process[i : i + args.batch_size]
                logger.info(
                    "Processing batch %d-%d/%d...",
                    i + 1,
                    i + len(chunk),
                    len(prompts_to_process),
                )

                formatted_prompts = []
                for prompt_item in chunk:
                    prompt_text = prompt_item["prompt"]
                    source = prompt_item["source"]
                    choices = prompt_item.get("choices")
                    full_prompt_text = build_user_prompt(prompt_text, source, choices)
                    messages = [{"role": "user", "content": full_prompt_text}]
                    formatted_prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    formatted_prompts.append(formatted_prompt)

                try:
                    # Tokenize with padding and move to correct device
                    inputs = tokenizer(formatted_prompts, return_tensors="pt", padding=True).to(device)

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=config.max_generation_tokens,
                            pad_token_id=tokenizer.eos_token_id,
                            **generation_params
                        )

                    # Extract generated responses
                    for j, prompt_item in enumerate(chunk):
                        prompt_id = prompt_item["id"]
                        source = prompt_item["source"]
                        prompt_text = prompt_item["prompt"]
                        choices = prompt_item.get("choices")
                        ground_truth = prompt_item["ground_truth"]
                        
                        # Extract the output ids excluding input ids
                        input_len = inputs.input_ids.shape[-1]
                        generated_ids = outputs[j][input_len:]
                        output = tokenizer.decode(generated_ids, skip_special_tokens=True)

                        raw_thinking, raw_answer = parse_thinking_and_answer(output, strip_prefix=True)

                        is_correct = is_correct_answer(raw_answer, ground_truth, source, choices)
                        extracted = extract_predicted_answer(raw_answer, source, choices)
                        logger.info(
                            "ID=%s -> Correct = %s (Predicted: %r, Extracted: %r, Ground Truth: %r)",
                            prompt_id,
                            is_correct,
                            raw_answer,
                            extracted,
                            ground_truth,
                        )

                        # Save record
                        record = {
                            "id": prompt_id,
                            "source": source,
                            "prompt": prompt_text,
                            "choices": choices,
                            "ground_truth": ground_truth,
                            "raw_thinking": raw_thinking,
                            "raw_answer": raw_answer,
                            "raw_answer_correct": is_correct,
                        }

                        out_f.write(json.dumps(record) + "\n")
                    out_f.flush()

                except Exception as e:
                    logger.error("Batch generation failed for batch starting at index %d: %s", i, e)
                    continue
        else:
            for i, prompt_item in enumerate(prompts_to_process, 1):
                prompt_id = prompt_item["id"]
                source = prompt_item["source"]
                prompt_text = prompt_item["prompt"]
                choices = prompt_item.get("choices")
                ground_truth = prompt_item["ground_truth"]

                logger.info(
                    "[%d/%d] Generating trace for ID=%s (Source=%s)...",
                    i,
                    len(prompts_to_process),
                    prompt_id,
                    source,
                )

                full_prompt_text = build_user_prompt(prompt_text, source, choices)

                # Format using tokenizer chat template
                messages = [{"role": "user", "content": full_prompt_text}]
                formatted_prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # Generate output from model
                try:
                    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)
                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=config.max_generation_tokens,
                            pad_token_id=tokenizer.eos_token_id,
                            **generation_params
                        )
                    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
                    output = tokenizer.decode(generated_ids, skip_special_tokens=True)
                except Exception as e:
                    logger.error("Generation failed for prompt %s: %s", prompt_id, e)
                    continue

                raw_thinking, raw_answer = parse_thinking_and_answer(output, strip_prefix=True)

                is_correct = is_correct_answer(raw_answer, ground_truth, source, choices)
                extracted = extract_predicted_answer(raw_answer, source, choices)
                logger.info(
                    "ID=%s -> Correct = %s (Predicted: %r, Extracted: %r, Ground Truth: %r)",
                    prompt_id,
                    is_correct,
                    raw_answer,
                    extracted,
                    ground_truth,
                )

                # Save record
                record = {
                    "id": prompt_id,
                    "source": source,
                    "prompt": prompt_text,
                    "choices": choices,
                    "ground_truth": ground_truth,
                    "raw_thinking": raw_thinking,
                    "raw_answer": raw_answer,
                    "raw_answer_correct": is_correct,
                }

                out_f.write(json.dumps(record) + "\n")
                out_f.flush()

    logger.info("Trace generation complete. Results saved/appended to %s.", output_file)


if __name__ == "__main__":
    main()
