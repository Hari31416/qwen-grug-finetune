import re
from typing import List, Optional, Set


def get_answer_format_suffix(source: str, choices: Optional[List[str]] = None) -> str:
    """Return a strict final-answer instruction appended to the user prompt."""
    if source in ("strategyqa", "boolq"):
        return "\nAnswer in exactly one word: yes or no."

    if source == "piqa":
        return "\nAnswer in exactly one letter: A or B. Do not include any explanation."

    if source in ("logiqa", "reclor"):
        return "\nAnswer in exactly one letter: A, B, C, or D. Do not include any explanation."

    if source == "anli":
        return (
            "\nAnswer in exactly one word: entailment, neutral, or contradiction. "
            "Do not include any explanation."
        )

    if choices:
        if len(choices) == 1:
            options = choices[0]
        elif len(choices) == 2:
            options = f"{choices[0]} or {choices[1]}"
        else:
            options = ", ".join(choices[:-1]) + f", or {choices[-1]}"
        return f"\nAnswer in exactly one token: {options}. Do not include any explanation."

    return "\nAnswer in exactly one word. Do not include any explanation."


def build_user_prompt(prompt_text: str, source: str, choices: Optional[List[str]] = None) -> str:
    """Strip a trailing Answer: line and append the strict format suffix."""
    text = prompt_text.strip()
    if text.endswith("Answer:"):
        text = text[: -len("Answer:")].rstrip()
    return text + get_answer_format_suffix(source, choices)


def _valid_mc_choices(choices: Optional[List[str]]) -> Set[str]:
    if not choices:
        return {"A", "B", "C", "D"}
    return {choice.strip().upper() for choice in choices}


def _extract_yes_no(raw_answer: str) -> str:
    text = re.sub(r"[.,():\"'?\[\]]", " ", raw_answer.strip().lower())
    words = [word for word in text.split() if word]
    if not words:
        return ""

    for word in reversed(words):
        if word in ("yes", "no"):
            return word
    return ""


def _extract_anli_label(raw_answer: str) -> str:
    labels = ("entailment", "neutral", "contradiction")
    text = raw_answer.lower()
    best_pos = -1
    best_label = ""
    for label in labels:
        pos = text.rfind(label)
        if pos > best_pos:
            best_pos = pos
            best_label = label
    return best_label


def _extract_mc_letter(raw_answer: str, valid_choices: Set[str]) -> str:
    text = raw_answer.strip()
    if not text:
        return ""

    patterns = [
        r"(?:the\s+)?(?:correct\s+)?(?:answer|option|choice)\s+is\s+([A-D])\b",
        r"(?:answer|option|choice)\s*:\s*([A-D])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        if matches:
            letter = matches[-1].upper()
            if letter in valid_choices:
                return letter

    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines:
        last_line = re.sub(r"[^A-Za-z]", "", lines[-1]).upper()
        if last_line in valid_choices:
            return last_line

    words = re.sub(r"[.,():\"'?\[\]]", " ", text).split()
    if words:
        last_word = re.sub(r"[^A-Za-z]", "", words[-1]).upper()
        if last_word in valid_choices:
            return last_word

    return ""


def extract_predicted_answer(
    raw_answer: str,
    source: str,
    choices: Optional[List[str]] = None,
) -> str:
    """Extract the normalized predicted label from a possibly verbose model answer."""
    if not raw_answer or not raw_answer.strip():
        return ""

    if source in ("strategyqa", "boolq"):
        return _extract_yes_no(raw_answer)

    if source in ("logiqa", "reclor", "piqa"):
        return _extract_mc_letter(raw_answer, _valid_mc_choices(choices))

    if source == "anli":
        return _extract_anli_label(raw_answer)

    return raw_answer.strip().lower()


def is_correct_answer(
    raw_answer: str,
    ground_truth: str,
    source: str,
    choices: Optional[List[str]] = None,
) -> bool:
    """Normalize and verify if the raw answer matches the ground truth."""
    predicted = extract_predicted_answer(raw_answer, source, choices)
    if not predicted:
        return False
    return predicted.strip().lower() == ground_truth.strip().lower()
