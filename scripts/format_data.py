import os
import sys
import json
import random
import logging
import argparse
from typing import Dict, Any, List

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("format_data")


def main() -> None:
    parser = argparse.ArgumentParser(description="Format validated traces into MLX SFT train/validation datasets.")
    args = parser.parse_args()

    validated_file = os.path.join(config.validated_traces, "traces.jsonl")

    if not os.path.exists(validated_file):
        logger.error("Validated traces file not found at: %s. Please run validate_traces.py first.", validated_file)
        sys.exit(1)

    # Load validated records
    records: List[Dict[str, Any]] = []
    with open(validated_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    logger.info("Loaded %d validated traces.", len(records))

    if not records:
        logger.warning("No validated records to format. Exiting.")
        return

    # Load tokenizer to apply chat template
    logger.info("Loading tokenizer for model: %s", config.model_mlx_path)
    from mlx_lm import load
    _, tokenizer = load(config.model_mlx_path)

    # Format records
    formatted_data: List[Dict[str, str]] = []
    for record in records:
        prompt_text = record["prompt"].strip()
        compressed_thinking = record["compressed_thinking"].strip()
        raw_answer = record["raw_answer"].strip()
        
        # Suffix the generation constraint to user prompt so training mirrors inference setup
        source = record["source"]
        suffix = ""
        if source in ["strategyqa", "boolq"]:
            suffix = "\nAnswer in exactly one word: yes or no."
        elif source in ["logiqa", "reclor", "piqa"]:
            suffix = "\nState only the correct option letter corresponding to the answer."
        elif source == "anli":
            suffix = "\nState only the relation (entailment, neutral, or contradiction) corresponding to the answer."
            
        full_prompt = prompt_text + suffix

        # First format the user message with add_generation_prompt=True
        # This gives us the correct template prefix ending in "<think>\n"
        user_messages = [{"role": "user", "content": full_prompt}]
        try:
            prefix = tokenizer.apply_chat_template(user_messages, tokenize=False, add_generation_prompt=True)
            # Ensure it ends with <think>\n, otherwise append it
            if not prefix.endswith("<think>\n"):
                prefix = prefix + "<think>\n"
            
            # Construct the full training sequence preserving the reasoning block
            formatted_text = prefix + f"{compressed_thinking}\n</think>\n\n{raw_answer}" + tokenizer.eos_token
            formatted_data.append({"text": formatted_text})
        except Exception as e:
            logger.error("Failed to apply chat template for ID=%s: %s", record["id"], e)
            continue

    logger.info("Successfully formatted %d records.", len(formatted_data))

    # Shuffle with seed for reproducibility
    random.seed(config.seed)
    random.shuffle(formatted_data)

    # Split into train/validation sets
    split_idx = int(len(formatted_data) * config.train_split_ratio)
    train_set = formatted_data[:split_idx]
    valid_set = formatted_data[split_idx:]

    # For tiny pilot datasets, make sure we have at least 1 validation example if total >= 2
    if len(formatted_data) >= 2 and not valid_set:
        train_set = formatted_data[:-1]
        valid_set = formatted_data[-1:]

    # Resolve SFT paths
    os.makedirs(os.path.dirname(config.train_data), exist_ok=True)
    os.makedirs(os.path.dirname(config.valid_data), exist_ok=True)

    # Save SFT data
    with open(config.train_data, "w", encoding="utf-8") as f:
        for item in train_set:
            f.write(json.dumps(item) + "\n")
            
    with open(config.valid_data, "w", encoding="utf-8") as f:
        for item in valid_set:
            f.write(json.dumps(item) + "\n")

    logger.info("Saved %d training examples to: %s", len(train_set), config.train_data)
    logger.info("Saved %d validation examples to: %s", len(valid_set), config.valid_data)
    logger.info("SFT formatting complete.")


if __name__ == "__main__":
    main()
