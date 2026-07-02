"""Scoring module for Grug-adherence of compressed chain-of-thought traces."""

import re
import logging
from typing import Dict, Any, Tuple
from transformers import AutoTokenizer

logger = logging.getLogger("grug_score")

_tokenizer = None


def get_tokenizer(model_mlx_path: str) -> AutoTokenizer:
    """Lazy loads the tokenizer to avoid loading overhead when not needed."""
    global _tokenizer
    if _tokenizer is None:
        logger.info("Lazy loading tokenizer from %s...", model_mlx_path)
        _tokenizer = AutoTokenizer.from_pretrained(model_mlx_path)
    return _tokenizer


def compute_compression_ratio(
    raw_text: str, compressed_text: str, tokenizer: AutoTokenizer
) -> float:
    """Calculates compression ratio on token level using the model's tokenizer."""
    raw_tokens = len(tokenizer.encode(raw_text))
    comp_tokens = len(tokenizer.encode(compressed_text))
    if raw_tokens == 0:
        return 0.0
    return comp_tokens / raw_tokens


def compute_article_density(text: str) -> float:
    """Counts occurrences of case-insensitive whole words 'the', 'a', 'an' per 100 words."""
    # Strip basic punctuation and split by whitespace
    words = [
        w.strip(".,!?;:()[]\"'-").lower()
        for w in text.split()
        if w.strip()
    ]
    if not words:
        return 0.0
    articles = sum(1 for w in words if w in ("the", "a", "an"))
    return (articles / len(words)) * 100.0


def count_meta_commentary(text: str) -> int:
    """Returns number of regex hits for forbidden meta words/phrases."""
    patterns = [
        re.compile(r"\bwait\b", re.IGNORECASE),
        re.compile(r"\bokay\b", re.IGNORECASE),
        re.compile(r"\blet me\b", re.IGNORECASE),
        re.compile(r"\blet's\b", re.IGNORECASE),
        re.compile(r"\bhmm\b", re.IGNORECASE),
        re.compile(r"\bactually\b", re.IGNORECASE),
    ]
    hits = 0
    for pattern in patterns:
        hits += len(pattern.findall(text))
    return hits


def compute_avg_fragment_words(text: str) -> float:
    """Computes mean word count per period-separated reasoning fragment."""
    fragments = [f.strip() for f in text.split(".") if f.strip()]
    if not fragments:
        return 0.0
    word_counts = [len(f.split()) for f in fragments]
    return sum(word_counts) / len(fragments)


def check_no_repetition(text: str) -> bool:
    """Returns True if there are no duplicate sentences or reasoning fragments in the text."""
    # Split by common sentence/fragment delimiters
    fragments = re.split(r"[.!?\n]", text)
    seen = set()
    for frag in fragments:
        normalized = " ".join(
            re.sub(r"[^a-zA-Z0-9\s]", "", frag).lower().split()
        )
        # Only count fragments with > 2 words to avoid tiny fragments or punctuation repeating
        if len(normalized.split()) > 2:
            if normalized in seen:
                return False
            seen.add(normalized)
    return True


def calculate_grug_score(
    raw_text: str, compressed_text: str, model_mlx_path: str
) -> Tuple[float, Dict[str, Any]]:
    """Computes the final Grug score and returns it with component metrics."""
    tokenizer = get_tokenizer(model_mlx_path)

    ratio = compute_compression_ratio(raw_text, compressed_text, tokenizer)
    density = compute_article_density(compressed_text)
    meta_hits = count_meta_commentary(compressed_text)
    avg_words = compute_avg_fragment_words(compressed_text)
    no_rep = check_no_repetition(compressed_text)

    # Calculate score component values
    comp_ratio_score = 1.0 if ratio <= 0.50 else 0.0
    density_score = (
        1.0
        if density <= 3.0
        else max(0.0, 1.0 - (density - 3.0) / 5.0)
    )
    meta_score = 1.0 if meta_hits == 0 else 0.0
    fragment_score = (
        1.0
        if avg_words <= 12.0
        else max(0.0, 1.0 - (avg_words - 12.0) / 8.0)
    )
    rep_score = 1.0 if no_rep else 0.0

    # Average the scores of the 5 component metrics
    grug_score = (
        comp_ratio_score
        + density_score
        + meta_score
        + fragment_score
        + rep_score
    ) / 5.0

    metrics = {
        "compression_ratio": ratio,
        "article_density": density,
        "meta_commentary_hits": meta_hits,
        "avg_fragment_words": avg_words,
        "no_repetition": no_rep,
        "scores": {
            "compression_ratio_score": comp_ratio_score,
            "article_density_score": density_score,
            "meta_commentary_score": meta_score,
            "avg_fragment_words_score": fragment_score,
            "repetition_score": rep_score,
        },
    }

    return grug_score, metrics
