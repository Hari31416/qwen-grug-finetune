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
    """Generates DPO preference data (data/dpo/train.jsonl & data/dpo/valid.jsonl) from available SFT data or HF download."""
    train_dpo_path = os.path.join(dpo_dir, "train.jsonl")
    valid_dpo_path = os.path.join(dpo_dir, "valid.jsonl")

    if os.path.exists(train_dpo_path) and not force_recreate:
        logger.info("DPO dataset already exists at '%s'. Skipping generation.", train_dpo_path)
        return True

    sft_train_path = os.path.join(data_dir, "train.jsonl")
    sft_valid_path = os.path.join(data_dir, "valid.jsonl")

    # If SFT train path is missing, download from Hugging Face repository automatically
    if not os.path.exists(sft_train_path):
        logger.info("SFT data file '%s' missing. Downloading from Hugging Face Hub (hari31416/qwen-grug-finetune)...", sft_train_path)
        try:
            from scripts.cuda.download_data import download_hf_data
            download_hf_data(output_dir=data_dir)
        except Exception as dl_err:
            logger.warning("Failed to auto-download dataset: %s", dl_err)

    os.makedirs(dpo_dir, exist_ok=True)

    if not os.path.exists(sft_train_path):
        logger.warning("SFT data file '%s' not found. Generating fallback synthetic DPO dataset...", sft_train_path)
        fallback_records = [
            {
                "prompt": "<|im_start|>system\nWrite your reasoning in a concise, telegraphic style inside the thinking block.<|im_end|>\n<|im_start|>user\nCould a two-year old win a Scrabble tournament?\nAnswer in exactly one word: yes or no.<|im_end|>\n<|im_start|>assistant\n",
                "chosen": "<think>\nScrabble: form words, score points, highest score wins. Two-year-old lacks vocabulary, strategic thinking, optimal game analysis, rule adherence. Random letter placement -> extremely unlikely win.\n</think>\n\nNo",
                "rejected": "<think>\nFirst, let us carefully break down the question step-by-step. Scrabble requires complex vocabulary, letter positioning, strategic board manipulation, and word length optimization. A two-year-old child cannot read fluently or analyze strategic point maximization. Therefore, after analyzing all details, a two-year-old will lose.\n</think>\n\nFinal Answer: No"
            },
            {
                "prompt": "<|im_start|>system\nWrite your reasoning in a concise, telegraphic style inside the thinking block.<|im_end|>\n<|im_start|>user\nJosh buys a house for $80,000 and puts in $50,000 in repairs. This increased the value of the house by 150%. How much profit did he make?<|im_end|>\n<|im_start|>assistant\n",
                "chosen": "<think>\nJosh buys house for $80k. Repairs: $50k. Total cost: $130k. 150% increase: 1.5 * $80k = $120k. New value: $80k + $120k = $200k. Profit: $200k - $130k = $70k.\n</think>\n\nJosh made a profit of \\boxed{70000}.",
                "rejected": "<think>\nFirst, I will calculate the cost of the house. Josh buys the house for $80,000. Next, he spends $50,000 on repairs, making the total cost $130,000. Then, the repairs increase the house's value by 150%. To find the new value, multiply $80,000 by 1.5 which equals $120,000. Profit is $120,000 - $130,000 = -$10,000.\n</think>\n\nJosh made a profit of \\boxed{-10000}."
            }
        ]
        with open(train_dpo_path, "w", encoding="utf-8") as f:
            for rec in fallback_records:
                f.write(json.dumps(rec) + "\n")
        logger.info("Generated fallback DPO dataset: %s", train_dpo_path)
        return True

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
