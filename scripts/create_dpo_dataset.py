import os
import sys
import json
import re
import logging
from typing import Optional, Dict, Any, List

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("create_dpo_dataset")

def parse_sft_text(text: str):
    """Parses SFT formatted text into system prompt, user prompt, and assistant response."""
    norm_text = text.replace("\uff5c", "|").replace("\u2581", " ")
    m = re.search(
        r"(?:<\|begin_of_sentence\|>)?(.*?)(?:<\|User\|>|<\|user\|>)(.*?)(?:<\|Assistant\|>|<\|assistant\|>)(.*)",
        norm_text,
        re.DOTALL,
    )
    if m:
        sys_prompt = m.group(1).replace("<|begin of sentence|>", "").replace("<|begin_of_sentence|>", "").strip()
        user_prompt = m.group(2).strip()
        resp = m.group(3).replace("<|end_of_sentence|>", "").replace("<|end of sentence|>", "").strip()
        return sys_prompt, user_prompt, resp
    return None, None, None

def make_verbose_rejected(chosen_resp: str) -> str:
    """Generates a verbose, un-telegraphic rejected response from a concise chosen response."""
    m = re.search(r"<think>(.*?)</think>(.*)", chosen_resp, re.DOTALL)
    if m:
        think_text, answer_text = m.group(1).strip(), m.group(2).strip()
        verbose_think = (
            "First, let us carefully break down the question step-by-step. "
            "We need to evaluate all the rules and conditions provided in the prompt. "
            + think_text.replace(". ", ". Furthermore, ").replace(" → ", " which implies that ")
            + " Therefore, after analyzing all details, we arrive at the final conclusion."
        )
        verbose_ans = f"{verbose_think}\n\nFinal Answer: {answer_text}"
        return f"<think>\n{verbose_think}\n</think>\n\n{verbose_ans}"
    return chosen_resp

def generate_dpo_dataset(
    data_dir: str = "data",
    dpo_dir: str = "data/dpo",
    force_recreate: bool = False,
) -> bool:
    """Generates DPO preference data (data/dpo/train.jsonl & data/dpo/valid.jsonl) from available SFT data."""
    train_dpo_path = os.path.join(dpo_dir, "train.jsonl")
    valid_dpo_path = os.path.join(dpo_dir, "valid.jsonl")

    if os.path.exists(train_dpo_path) and not force_recreate:
        logger.info("DPO dataset already exists at '%s'. Skipping generation.", train_dpo_path)
        return True

    sft_train_path = os.path.join(data_dir, "train.jsonl")
    sft_valid_path = os.path.join(data_dir, "valid.jsonl")

    if not os.path.exists(sft_train_path):
        logger.error("Base SFT dataset file '%s' not found.", sft_train_path)
        return False

    os.makedirs(dpo_dir, exist_ok=True)
    logger.info("Generating DPO dataset from available SFT format data ('%s')...", sft_train_path)

    # Process train dataset split
    dpo_train_records: List[Dict[str, Any]] = []
    with open(sft_train_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            sys_p, user_q, chosen = parse_sft_text(item["text"])
            if user_q and chosen:
                prompt_text = f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{user_q}<|im_end|>\n<|im_start|>assistant\n"
                rejected = make_verbose_rejected(chosen)
                dpo_train_records.append({
                    "prompt": prompt_text,
                    "chosen": chosen,
                    "rejected": rejected,
                })

    with open(train_dpo_path, "w", encoding="utf-8") as f:
        for rec in dpo_train_records:
            f.write(json.dumps(rec) + "\n")

    logger.info("✅ Successfully generated DPO train split: %s (%d preference pairs)", train_dpo_path, len(dpo_train_records))

    # Process valid dataset split if available
    if os.path.exists(sft_valid_path):
        dpo_valid_records: List[Dict[str, Any]] = []
        with open(sft_valid_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                sys_p, user_q, chosen = parse_sft_text(item["text"])
                if user_q and chosen:
                    prompt_text = f"<|im_start|>system\n{sys_p}<|im_end|>\n<|im_start|>user\n{user_q}<|im_end|>\n<|im_start|>assistant\n"
                    rejected = make_verbose_rejected(chosen)
                    dpo_valid_records.append({
                        "prompt": prompt_text,
                        "chosen": chosen,
                        "rejected": rejected,
                    })

        with open(valid_dpo_path, "w", encoding="utf-8") as f:
            for rec in dpo_valid_records:
                f.write(json.dumps(rec) + "\n")

        logger.info("✅ Successfully generated DPO valid split: %s (%d preference pairs)", valid_dpo_path, len(dpo_valid_records))

    return True

if __name__ == "__main__":
    generate_dpo_dataset()
