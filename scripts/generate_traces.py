import os
import sys
import json
import argparse
import logging
import re
from typing import Dict, Any, Set, List

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generate_traces")


def is_correct_answer(raw_answer: str, ground_truth: str, source: str) -> bool:
    """Normalize and verify if the raw answer matches the ground truth for different sources."""
    if not raw_answer:
        return False
    
    ans_clean = raw_answer.strip().lower()
    gt_clean = ground_truth.strip().lower()
    
    # Remove punctuation that might warp exact word comparisons
    ans_clean = re.sub(r'[.,\(\):"\'\?]', ' ', ans_clean).strip()
    words = ans_clean.split()
    if not words:
        return False
        
    if source in ["strategyqa", "boolq"]:
        # yes / no questions
        if ans_clean == gt_clean:
            return True
        if words[0] == gt_clean or words[-1] == gt_clean:
            return True
        if gt_clean in words:
            return True
            
    elif source in ["logiqa", "reclor", "piqa"]:
        # Multiple choice options (A, B, C, D)
        if ans_clean == gt_clean:
            return True
        if words[0] == gt_clean or words[-1] == gt_clean:
            return True
        if gt_clean in words:
            return True
        if ans_clean.startswith(gt_clean):
            return True
            
    elif source == "anli":
        # textual entailment (entailment, neutral, contradiction)
        if ans_clean == gt_clean:
            return True
        if gt_clean in words:
            return True
            
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw CoT traces using target MLX model")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of prompts to process (useful for pilot runs)"
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

    # Determine target limit
    if args.limit is not None:
        prompts = prompts[:args.limit]
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

    # Load MLX model and tokenizer
    logger.info("Loading MLX model from: %s", config.model_mlx_path)
    from mlx_lm import load, generate
    from mlx_lm.sample_utils import make_sampler, make_logits_processors

    model, tokenizer = load(config.model_mlx_path)

    # Create sampler and logits processors
    sampler = make_sampler(temp=config.temperature, top_p=config.top_p)
    logits_processors = make_logits_processors(
        repetition_penalty=1.1,
        presence_penalty=0.2,
    )

    # Open output file in append mode
    with open(output_file, "a", encoding="utf-8") as out_f:
        for i, prompt_item in enumerate(prompts_to_process, 1):
            prompt_id = prompt_item["id"]
            source = prompt_item["source"]
            prompt_text = prompt_item["prompt"]
            choices = prompt_item.get("choices")
            ground_truth = prompt_item["ground_truth"]

            logger.info("[%d/%d] Generating trace for ID=%s (Source=%s)...", i, len(prompts_to_process), prompt_id, source)

            # Append source-specific answer format constraints to the prompt
            suffix = ""
            if source in ["strategyqa", "boolq"]:
                suffix = "\nAnswer in exactly one word: yes or no."
            elif source in ["logiqa", "reclor", "piqa"]:
                suffix = "\nState only the correct option letter corresponding to the answer."
            elif source == "anli":
                suffix = "\nState only the relation (entailment, neutral, or contradiction) corresponding to the answer."

            full_prompt_text = prompt_text.strip() + suffix

            # Format using tokenizer chat template with thinking enabled
            messages = [{"role": "user", "content": full_prompt_text}]
            formatted_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True, enable_thinking=True
            )

            # Generate output from model
            try:
                output = generate(
                    model,
                    tokenizer,
                    prompt=formatted_prompt,
                    max_tokens=config.max_generation_tokens,
                    sampler=sampler,
                    logits_processors=logits_processors,
                    verbose=False
                )
            except Exception as e:
                logger.error("Generation failed for prompt %s: %s", prompt_id, e)
                continue

            # Parse thinking block and answer
            if "</think>" in output:
                parts = output.split("</think>", 1)
                raw_thinking = parts[0]
                raw_answer = parts[1].strip()
            else:
                raw_thinking = output
                raw_answer = ""
                logger.warning("Prompt %s did not emit closing </think> tag.", prompt_id)

            # Strip default Qwen thinking prefix if present
            if raw_thinking.startswith("Thinking Process:\n\n"):
                raw_thinking = raw_thinking[len("Thinking Process:\n\n"):]
            elif raw_thinking.startswith("Thinking Process:"):
                raw_thinking = raw_thinking[len("Thinking Process:"):]

            # Validate answer
            is_correct = is_correct_answer(raw_answer, ground_truth, source)
            logger.info("ID=%s -> Correct = %s (Predicted: %r, Ground Truth: %r)", prompt_id, is_correct, raw_answer, ground_truth)

            # Save record
            record = {
                "id": prompt_id,
                "source": source,
                "prompt": prompt_text,
                "choices": choices,
                "ground_truth": ground_truth,
                "raw_thinking": raw_thinking,
                "raw_answer": raw_answer,
                "raw_answer_correct": is_correct
            }

            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    logger.info("Trace generation complete. Results saved/appended to %s.", output_file)


if __name__ == "__main__":
    main()
