import os
import sys
import json
import logging
import argparse
import re
from typing import Dict, Any, List, Tuple

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("validate_traces")

KEY_VALUE_PATTERN = re.compile(r"^\s*\w+\s*:", re.MULTILINE)
NUMBER_PATTERN = re.compile(r"\b\d+\.?\d*\b")
OPTION_LETTER_PATTERN = re.compile(r"\b[ABCD]\b")

FORBIDDEN_ANSWER_PHRASES = [
    "therefore the answer is",
    "so the answer is",
    "conclude that the answer",
    "the correct option is",
    "therefore, the answer",
    "so the answer",
    "the answer is",
]

FORBIDDEN_META_PHRASES = [
    "let's see",
    "let me think",
    "wait,",
    "wait...",
    "hmm,",
    "okay,",
    "alright,",
]

FILLER_FRAGMENT_PREFIXES = (
    "wait",
    "okay",
    "hmm",
    "alright",
    "let me",
    "so ",
    "yes",
    "no",
    "i think",
    "i need",
)

INCOMPLETE_TAIL_WORDS = {
    "between",
    "of",
    "to",
    "for",
    "with",
    "and",
    "or",
    "the",
    "a",
    "an",
    "that",
    "which",
    "if",
    "whether",
}


def count_fragments(text: str, *, skip_filler: bool = False) -> int:
    """Count period-separated reasoning fragments, ignoring very short segments."""
    normalized = text.replace("\n", " ")
    parts = [part for part in normalized.split(".") if len(part.strip()) > 3]
    if not skip_filler:
        return len(parts)

    meaningful = []
    for part in parts:
        lower = part.strip().lower()
        if any(lower.startswith(prefix) for prefix in FILLER_FRAGMENT_PREFIXES):
            continue
        meaningful.append(part)
    return len(meaningful)


def extract_numbers(text: str) -> set[str]:
    return set(NUMBER_PATTERN.findall(text))


def extract_option_letters(text: str) -> set[str]:
    return set(OPTION_LETTER_PATTERN.findall(text.upper()))


def is_incomplete_compression(compressed_thinking: str) -> bool:
    """Detect truncated compressions that end mid-phrase without a conclusion."""
    stripped = compressed_thinking.strip()
    if not stripped:
        return True

    if stripped[-1] in ".!?":
        return False

    words = stripped.split()
    if not words:
        return True

    if words[-1].lower().rstrip(",:;") in INCOMPLETE_TAIL_WORDS:
        return True

    # Very short compressions without terminal punctuation are likely truncated API output
    return len(words) < 12


def is_parseable(compressed_thinking: str) -> Tuple[bool, str]:
    if not compressed_thinking or not compressed_thinking.strip():
        return False, "Compressed thinking block is empty"

    stripped = compressed_thinking.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        return False, "Compressed thinking is only a markdown code block"

    if stripped.lower().startswith("compressed reasoning:") or stripped.lower().startswith("grug reasoning:"):
        return False, "Compressed thinking contains wrapper labels"

    if is_incomplete_compression(stripped):
        return False, "Compressed thinking appears truncated or incomplete"

    return True, ""


def check_logic_steps_preserved(
    raw_thinking: str,
    compressed_thinking: str,
    source: str,
) -> Tuple[bool, str]:
    if KEY_VALUE_PATTERN.search(compressed_thinking):
        return False, "Compressed thinking uses key-value or label format"

    raw_fragments = count_fragments(raw_thinking, skip_filler=True)
    comp_fragments = count_fragments(compressed_thinking)
    if raw_fragments >= 6 and comp_fragments < max(2, int(raw_fragments * 0.2)):
        return (
            False,
            f"Too few logic fragments preserved: {comp_fragments} vs raw {raw_fragments}",
        )

    raw_words = len(raw_thinking.split())
    comp_words = len(compressed_thinking.split())
    if raw_words >= 50 and comp_words < raw_words * 0.08:
        return False, "Compression likely dropped logic steps (too short)"

    raw_numbers = extract_numbers(raw_thinking)
    if raw_numbers:
        comp_numbers = extract_numbers(compressed_thinking)
        preserved = len(raw_numbers & comp_numbers)
        if len(raw_numbers) <= 2:
            if preserved == 0:
                return False, "Numeric facts dropped from compression"
        elif preserved / len(raw_numbers) < 0.6:
            return (
                False,
                f"Numeric facts dropped from compression (preserved {preserved / len(raw_numbers):.0%})",
            )

    if source in ["logiqa", "reclor", "piqa"]:
        raw_options = extract_option_letters(raw_thinking)
        if raw_options:
            comp_options = extract_option_letters(compressed_thinking)
            if not raw_options & comp_options:
                return False, "Multiple-choice option letters dropped from compression"

    comp_lower = compressed_thinking.lower()
    for phrase in FORBIDDEN_META_PHRASES:
        if phrase in comp_lower:
            return False, f"Compressed thinking contains meta filler phrase: '{phrase}'"

    return True, ""


def validate_record(record: Dict[str, Any]) -> Tuple[bool, str]:
    """Validates a single compressed trace against the style guide validation policy."""
    if not record.get("raw_answer_correct"):
        return False, "Raw answer was incorrect"

    compressed_thinking = record.get("compressed_thinking", "").strip()
    raw_thinking = record.get("raw_thinking", "").strip()
    source = record.get("source", "")

    is_valid, reason = is_parseable(compressed_thinking)
    if not is_valid:
        return False, reason

    raw_words = len(raw_thinking.split())
    comp_words = len(compressed_thinking.split())

    if raw_words == 0:
        return False, "Raw thinking block has 0 words"

    ratio = comp_words / raw_words
    if ratio > 0.55:
        return (
            False,
            f"Compression ratio too high: {ratio:.2f} (raw words: {raw_words}, compressed words: {comp_words})",
        )

    comp_lower = compressed_thinking.lower()
    for phrase in FORBIDDEN_ANSWER_PHRASES:
        if phrase in comp_lower:
            return False, f"Compressed thinking contains answer restatement phrase: '{phrase}'"

    raw_ans_clean = record.get("raw_answer", "").strip().lower()
    if raw_ans_clean and comp_lower.endswith(f"answer is {raw_ans_clean}"):
        return False, "Compressed thinking ends with answer restatement"

    logic_ok, logic_reason = check_logic_steps_preserved(raw_thinking, compressed_thinking, source)
    if not logic_ok:
        return False, logic_reason

    return True, "Passed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate compressed CoT traces against style guide rules.")
    parser.add_argument("--report", action="store_true", help="Print detailed validation report")
    args = parser.parse_args()

    compressed_file = os.path.join(config.compressed_traces, "traces.jsonl")
    validated_file = os.path.join(config.validated_traces, "traces.jsonl")

    if not os.path.exists(compressed_file):
        logger.error("Compressed traces file not found at: %s. Please run compress_traces.py first.", compressed_file)
        sys.exit(1)

    os.makedirs(config.validated_traces, exist_ok=True)

    records: List[Dict[str, Any]] = []
    with open(compressed_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    logger.info("Loaded %d compressed traces for validation.", len(records))

    accepted_records: List[Dict[str, Any]] = []
    rejected_stats: Dict[str, int] = {}

    for record in records:
        is_valid, reason = validate_record(record)
        if is_valid:
            accepted_records.append(record)
        else:
            rejected_stats[reason] = rejected_stats.get(reason, 0) + 1
            if args.report:
                logger.info("Rejected ID=%s. Reason: %s", record["id"], reason)

    with open(validated_file, "w", encoding="utf-8") as out_f:
        for record in accepted_records:
            out_f.write(json.dumps(record) + "\n")

    total = len(records)
    accepted_count = len(accepted_records)
    rejection_rate = (total - accepted_count) / total if total > 0 else 0

    logger.info("=== VALIDATION STATS ===")
    logger.info("Total Checked: %d", total)
    logger.info("Accepted: %d", accepted_count)
    logger.info("Rejected: %d", total - accepted_count)
    logger.info("Rejection Rate: %.2f%%", rejection_rate * 100)
    logger.info("Rejection Reasons breakdown:")
    for reason, count in rejected_stats.items():
        logger.info("  - %s: %d", reason, count)
    logger.info("========================")

    report_path = os.path.join(config.compressed_traces, "validation_report.json")
    report_data = {
        "total_checked": total,
        "accepted": accepted_count,
        "rejected": total - accepted_count,
        "rejection_rate": rejection_rate,
        "rejection_reasons": rejected_stats,
    }
    with open(report_path, "w", encoding="utf-8") as rf:
        json.dump(report_data, rf, indent=2)
    logger.info("Validation report saved to: %s", report_path)


if __name__ == "__main__":
    main()
