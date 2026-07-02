import os
import re
import sys
import json
import random
import logging
from typing import Dict, Any, List, Set
from datasets import load_dataset, concatenate_datasets, Dataset

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("sample_sft_prompts")


def normalize_text(text: str) -> str:
    """Normalize text by converting to lowercase, removing punctuation, and squeezing whitespace."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return " ".join(text.split())


def build_benchmark_blocklist() -> Set[str]:
    """Fetch and compile the benchmark questions blocklist from GSM8K and ARC-Challenge."""
    logger.info("Building benchmark question blocklist to prevent data leakage...")
    blocklist: Set[str] = set()

    # 1. Load GSM8K test split
    try:
        gsm8k = load_dataset("openai/gsm8k", "main", split="test")
        for row in gsm8k:
            q = row.get("question", "")
            if q:
                blocklist.add(normalize_text(q))
        logger.info("Loaded %d questions from GSM8K test split.", len(gsm8k))
    except Exception as e:
        logger.error("Failed to load GSM8K test split: %s", e)
        raise e

    # 2. Load ARC-Challenge test and validation splits
    try:
        arc_test = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
        arc_val = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="validation")
        
        for row in arc_test:
            q = row.get("question", "")
            if q:
                blocklist.add(normalize_text(q))
                
        for row in arc_val:
            q = row.get("question", "")
            if q:
                blocklist.add(normalize_text(q))
                
        logger.info("Loaded %d questions from ARC-Challenge test+val splits.", len(arc_test) + len(arc_val))
    except Exception as e:
        logger.error("Failed to load ARC-Challenge splits: %s", e)
        raise e

    logger.info("Total unique normalized questions in blocklist: %d", len(blocklist))
    return blocklist


def is_leaking(prompt_text: str, blocklist_set: Set[str], threshold: float = 0.85) -> bool:
    """Check if the prompt_text leaks benchmark questions using exact match or Jaccard similarity."""
    norm_prompt = normalize_text(prompt_text)
    if not norm_prompt:
        return False
        
    if norm_prompt in blocklist_set:
        return True

    # Token-based Jaccard similarity for fuzzy matches
    prompt_tokens = set(norm_prompt.split())
    if not prompt_tokens:
        return False

    for block_norm in blocklist_set:
        block_tokens = set(block_norm.split())
        if not block_tokens:
            continue
        intersection = prompt_tokens.intersection(block_tokens)
        union = prompt_tokens.union(block_tokens)
        similarity = len(intersection) / len(union)
        if similarity > threshold:
            return True

    return False


def sample_strategyqa(blocklist: Set[str], target_count: int) -> List[Dict[str, Any]]:
    """Sample from StrategyQA dataset."""
    logger.info("Loading and processing StrategyQA...")
    ds = load_dataset("ChilleD/StrategyQA", split="train")
    
    # Shuffle indices
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    sampled: List[Dict[str, Any]] = []
    skipped_leak = 0
    
    for idx in indices:
        if len(sampled) >= target_count:
            break
            
        row = ds[idx]
        question = row.get("question", "")
        if not question:
            continue
            
        if is_leaking(question, blocklist):
            skipped_leak += 1
            continue
            
        answer_val = row.get("answer")
        ground_truth = "yes" if answer_val is True else "no"
        
        sampled.append({
            "id": f"strategyqa-{len(sampled) + 1:04d}",
            "source": "strategyqa",
            "prompt": question,
            "choices": ["yes", "no"],
            "ground_truth": ground_truth
        })
        
    logger.info("StrategyQA: sampled %d/%d (skipped %d leaks).", len(sampled), target_count, skipped_leak)
    if len(sampled) < target_count:
        raise ValueError(f"Not enough clean rows in StrategyQA. Needed {target_count}, got {len(sampled)}")
    return sampled


def sample_logiqa(blocklist: Set[str], target_count: int) -> List[Dict[str, Any]]:
    """Sample from LogiQA dataset."""
    logger.info("Loading and processing LogiQA...")
    ds = load_dataset("lucasmccabe/logiqa", revision="refs/convert/parquet", split="train")
    
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    sampled: List[Dict[str, Any]] = []
    skipped_leak = 0
    
    for idx in indices:
        if len(sampled) >= target_count:
            break
            
        row = ds[idx]
        context = row.get("context", "")
        query = row.get("query", "")
        options = row.get("options", [])
        correct_option_idx = row.get("correct_option")
        
        if not context or not query or len(options) < 4 or correct_option_idx is None:
            continue
            
        # Combine context and query for leakage checking
        check_text = f"{context} {query}"
        if is_leaking(check_text, blocklist):
            skipped_leak += 1
            continue
            
        formatted_prompt = (
            f"Passage: {context}\n"
            f"Question: {query}\n"
            f"A) {options[0]}\n"
            f"B) {options[1]}\n"
            f"C) {options[2]}\n"
            f"D) {options[3]}\n"
            f"Answer:"
        )
        
        ground_truth = ["A", "B", "C", "D"][correct_option_idx]
        
        sampled.append({
            "id": f"logiqa-{len(sampled) + 1:04d}",
            "source": "logiqa",
            "prompt": formatted_prompt,
            "choices": ["A", "B", "C", "D"],
            "ground_truth": ground_truth
        })
        
    logger.info("LogiQA: sampled %d/%d (skipped %d leaks).", len(sampled), target_count, skipped_leak)
    if len(sampled) < target_count:
        raise ValueError(f"Not enough clean rows in LogiQA. Needed {target_count}, got {len(sampled)}")
    return sampled


def sample_boolq(blocklist: Set[str], target_count: int) -> List[Dict[str, Any]]:
    """Sample from BoolQ dataset."""
    logger.info("Loading and processing BoolQ...")
    ds = load_dataset("google/boolq", split="train")
    
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    sampled: List[Dict[str, Any]] = []
    skipped_leak = 0
    
    for idx in indices:
        if len(sampled) >= target_count:
            break
            
        row = ds[idx]
        passage = row.get("passage", "")
        question = row.get("question", "")
        answer_val = row.get("answer")
        
        if not passage or not question or answer_val is None:
            continue
            
        # Check leakage on question + passage
        check_text = f"{passage} {question}"
        if is_leaking(check_text, blocklist):
            skipped_leak += 1
            continue
            
        # Ensure question ends with a question mark properly formatted
        q_str = question.strip()
        if not q_str.endswith("?"):
            q_str += "?"
            
        formatted_prompt = (
            f"Passage: {passage}\n"
            f"Question: {q_str}\n"
            f"Answer:"
        )
        
        ground_truth = "yes" if answer_val is True else "no"
        
        sampled.append({
            "id": f"boolq-{len(sampled) + 1:04d}",
            "source": "boolq",
            "prompt": formatted_prompt,
            "choices": ["yes", "no"],
            "ground_truth": ground_truth
        })
        
    logger.info("BoolQ: sampled %d/%d (skipped %d leaks).", len(sampled), target_count, skipped_leak)
    if len(sampled) < target_count:
        raise ValueError(f"Not enough clean rows in BoolQ. Needed {target_count}, got {len(sampled)}")
    return sampled


def sample_anli(blocklist: Set[str], target_count: int) -> List[Dict[str, Any]]:
    """Sample from ANLI dataset (merged train_r1, train_r2, train_r3)."""
    logger.info("Loading and processing ANLI (r1, r2, r3)...")
    anli_r1 = load_dataset("facebook/anli", split="train_r1")
    anli_r2 = load_dataset("facebook/anli", split="train_r2")
    anli_r3 = load_dataset("facebook/anli", split="train_r3")
    ds = concatenate_datasets([anli_r1, anli_r2, anli_r3])
    
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    sampled: List[Dict[str, Any]] = []
    skipped_leak = 0
    
    label_map = ["entailment", "neutral", "contradiction"]
    
    for idx in indices:
        if len(sampled) >= target_count:
            break
            
        row = ds[idx]
        premise = row.get("premise", "")
        hypothesis = row.get("hypothesis", "")
        label_idx = row.get("label")
        
        if not premise or not hypothesis or label_idx is None or label_idx < 0 or label_idx >= len(label_map):
            continue
            
        check_text = f"{premise} {hypothesis}"
        if is_leaking(check_text, blocklist):
            skipped_leak += 1
            continue
            
        formatted_prompt = (
            f"Context: {premise}\n"
            f"Hypothesis: {hypothesis}\n"
            f"Is the hypothesis entailment, neutral, or contradiction?\n"
            f"Answer:"
        )
        
        ground_truth = label_map[label_idx]
        
        sampled.append({
            "id": f"anli-{len(sampled) + 1:04d}",
            "source": "anli",
            "prompt": formatted_prompt,
            "choices": label_map,
            "ground_truth": ground_truth
        })
        
    logger.info("ANLI: sampled %d/%d (skipped %d leaks).", len(sampled), target_count, skipped_leak)
    if len(sampled) < target_count:
        raise ValueError(f"Not enough clean rows in ANLI. Needed {target_count}, got {len(sampled)}")
    return sampled


def sample_piqa(blocklist: Set[str], target_count: int) -> List[Dict[str, Any]]:
    """Sample from PIQA dataset."""
    logger.info("Loading and processing PIQA...")
    ds = load_dataset("baber/piqa", split="train")
    
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    sampled: List[Dict[str, Any]] = []
    skipped_leak = 0
    
    for idx in indices:
        if len(sampled) >= target_count:
            break
            
        row = ds[idx]
        goal = row.get("goal", "")
        sol1 = row.get("sol1", "")
        sol2 = row.get("sol2", "")
        label_idx = row.get("label")
        
        if not goal or not sol1 or not sol2 or label_idx is None or label_idx not in (0, 1):
            continue
            
        check_text = f"{goal} {sol1} {sol2}"
        if is_leaking(check_text, blocklist):
            skipped_leak += 1
            continue
            
        formatted_prompt = (
            f"Goal: {goal}\n"
            f"Which option is more sensible?\n"
            f"A) {sol1}\n"
            f"B) {sol2}\n"
            f"Answer:"
        )
        
        ground_truth = ["A", "B"][label_idx]
        
        sampled.append({
            "id": f"piqa-{len(sampled) + 1:04d}",
            "source": "piqa",
            "prompt": formatted_prompt,
            "choices": ["A", "B"],
            "ground_truth": ground_truth
        })
        
    logger.info("PIQA: sampled %d/%d (skipped %d leaks).", len(sampled), target_count, skipped_leak)
    if len(sampled) < target_count:
        raise ValueError(f"Not enough clean rows in PIQA. Needed {target_count}, got {len(sampled)}")
    return sampled


def sample_reclor(blocklist: Set[str], target_count: int) -> List[Dict[str, Any]]:
    """Sample from ReClor dataset."""
    logger.info("Loading and processing ReClor...")
    ds = load_dataset("hadithya369/ReClor", split="train")
    
    indices = list(range(len(ds)))
    random.shuffle(indices)
    
    sampled: List[Dict[str, Any]] = []
    skipped_leak = 0
    
    for idx in indices:
        if len(sampled) >= target_count:
            break
            
        row = ds[idx]
        context = row.get("context", "")
        question = row.get("question", "")
        answers = row.get("answers", [])
        label_idx = row.get("label")
        
        if not context or not question or len(answers) < 4 or label_idx is None or label_idx < 0 or label_idx >= len(answers):
            continue
            
        check_text = f"{context} {question}"
        if is_leaking(check_text, blocklist):
            skipped_leak += 1
            continue
            
        formatted_prompt = (
            f"Context: {context}\n"
            f"Question: {question}\n"
            f"A) {answers[0]}\n"
            f"B) {answers[1]}\n"
            f"C) {answers[2]}\n"
            f"D) {answers[3]}\n"
            f"Answer:"
        )
        
        ground_truth = ["A", "B", "C", "D"][label_idx]
        
        sampled.append({
            "id": f"reclor-{len(sampled) + 1:04d}",
            "source": "reclor",
            "prompt": formatted_prompt,
            "choices": ["A", "B", "C", "D"],
            "ground_truth": ground_truth
        })
        
    logger.info("ReClor: sampled %d/%d (skipped %d leaks).", len(sampled), target_count, skipped_leak)
    if len(sampled) < target_count:
        raise ValueError(f"Not enough clean rows in ReClor. Needed {target_count}, got {len(sampled)}")
    return sampled


def main() -> None:
    # 1. Setup seed and directories
    random.seed(config.seed)
    config.setup_directories()
    
    logger.info("Using random seed: %d", config.seed)
    
    # 2. Compile benchmark blocklist
    blocklist = build_benchmark_blocklist()
    
    # 3. Pull target sample sizes
    sample_sizes = config.dataset_sample_sizes
    logger.info("Target sample sizes: %s", sample_sizes)
    
    all_prompts: List[Dict[str, Any]] = []
    
    # 4. Extract samples
    all_prompts.extend(sample_strategyqa(blocklist, sample_sizes.get("strategyqa", 680)))
    all_prompts.extend(sample_logiqa(blocklist, sample_sizes.get("logiqa", 600)))
    all_prompts.extend(sample_boolq(blocklist, sample_sizes.get("boolq", 520)))
    all_prompts.extend(sample_anli(blocklist, sample_sizes.get("anli", 800)))
    all_prompts.extend(sample_piqa(blocklist, sample_sizes.get("piqa", 600)))
    all_prompts.extend(sample_reclor(blocklist, sample_sizes.get("reclor", 800)))
    
    # Shuffle the final compiled dataset pool to keep it source-stratified but randomized
    # Wait, the instruction says: "Keep unsplit prompt pool; split train/valid only after filtering for correct model answers"
    # To keep it standard, we write the entire pool to config.sft_prompts
    logger.info("Total prompts compiled: %d", len(all_prompts))
    
    # Verify exact count is 1,000
    expected_total = sum(sample_sizes.values())
    if len(all_prompts) != expected_total:
        logger.warning("Total prompts count (%d) does not match expected total (%d)", len(all_prompts), expected_total)
        
    logger.info("Saving SFT prompts to: %s", config.sft_prompts)
    os.makedirs(os.path.dirname(config.sft_prompts), exist_ok=True)
    with open(config.sft_prompts, "w", encoding="utf-8") as f:
        for prompt_row in all_prompts:
            f.write(json.dumps(prompt_row) + "\n")
            
    logger.info("Successfully generated %d prompts in %s!", len(all_prompts), config.sft_prompts)


if __name__ == "__main__":
    main()
