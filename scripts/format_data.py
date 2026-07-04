import os
import sys
import json
import random
import logging
import argparse
from typing import Dict, Any, List

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.config import config
from scripts.prompt_utils import build_user_prompt, STYLE_SYSTEM_PROMPT

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('format_data')

# Fraction of records to mark as negative examples (raw/uncompressed thinking)
NEGATIVE_EXAMPLE_FRACTION = 0.30
# Fraction of positive examples where system prompt is omitted
SYSTEM_PROMPT_DROPOUT = 0.20
# Fraction of negative examples that include the system prompt
NEGATIVE_SYSTEM_PROMPT_FRACTION = 0.50


def format_record(
    record: Dict[str, Any],
    tokenizer: Any,
    sft_style: str,
    include_system_prompt: bool,
) -> str | None:
    """Apply chat template to a single record and return the formatted training text.

    Args:
        record: Validated trace record containing prompt, thinking, answer fields.
        tokenizer: Loaded tokenizer with apply_chat_template support.
        sft_style: Either 'compressed' (positive) or 'normal' (negative example).
        include_system_prompt: Whether to prepend the STYLE_SYSTEM_PROMPT to the message history.

    Returns:
        Formatted text string ready for SFT, or None on failure.
    """
    prompt_text = record['prompt'].strip()
    raw_answer = record['raw_answer'].strip()
    source = record['source']
    full_prompt = build_user_prompt(prompt_text, source, record.get('choices'))

    # Select thinking block based on style
    if sft_style == 'compressed':
        thinking_block = record.get('compressed_thinking', '').strip()
        if not thinking_block:
            logger.warning('Record ID=%s has no compressed_thinking; skipping.', record.get('id'))
            return None
    else:
        # Negative example: use raw (uncompressed) thinking
        thinking_block = record.get('raw_thinking', '').strip()
        if not thinking_block:
            logger.warning('Record ID=%s has no raw_thinking; skipping.', record.get('id'))
            return None

    messages = [
        {'role': 'user', 'content': full_prompt},
    ]
    # Prepend the style system prompt if required
    if include_system_prompt:
        messages.insert(0, {'role': 'system', 'content': STYLE_SYSTEM_PROMPT})

    try:
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Ensure the generation prompt ends with <think>\n as the model expects
        if not prefix.endswith('<think>\n'):
            prefix = prefix + '<think>\n'
        formatted_text = prefix + f'{thinking_block}\n</think>\n\n{raw_answer}' + tokenizer.eos_token
        return formatted_text
    except Exception as e:
        logger.error('Failed to apply chat template for ID=%s: %s', record.get('id'), e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Format validated traces into MLX SFT train/validation datasets.'
    )
    parser.add_argument(
        '--negative-fraction',
        type=float,
        default=NEGATIVE_EXAMPLE_FRACTION,
        help='Fraction of records to use as negative (raw thinking) examples (default: 0.30)',
    )
    parser.add_argument(
        '--system-prompt-dropout',
        type=float,
        default=SYSTEM_PROMPT_DROPOUT,
        help='Fraction of positive examples where system prompt is omitted (default: 0.20)',
    )
    parser.add_argument(
        '--negative-system-prompt-fraction',
        type=float,
        default=NEGATIVE_SYSTEM_PROMPT_FRACTION,
        help='Fraction of negative examples that include the system prompt (default: 0.50)',
    )
    args = parser.parse_args()

    validated_file = os.path.join(config.validated_traces, 'traces.jsonl')

    if not os.path.exists(validated_file):
        logger.error(
            'Validated traces file not found at: %s. Please run validate_traces.py first.',
            validated_file,
        )
        sys.exit(1)

    # Load validated records
    records: List[Dict[str, Any]] = []
    with open(validated_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line.strip()))

    logger.info('Loaded %d validated traces.', len(records))

    if not records:
        logger.warning('No validated records to format. Exiting.')
        return

    # Load tokenizer to apply chat template
    logger.info('Loading tokenizer for model: %s', config.model_mlx_path)
    from mlx_lm import load
    _, tokenizer = load(config.model_mlx_path)

    # Shuffle for reproducibility before sampling negatives
    random.seed(config.seed)
    shuffled = records[:]
    random.shuffle(shuffled)

    # Format ALL records as compressed positives first
    formatted_data: List[Dict[str, str]] = []
    for record in shuffled:
        # 1. System prompt dropout for positive examples
        include_sys = random.random() >= args.system_prompt_dropout
        text = format_record(record, tokenizer, 'compressed', include_system_prompt=include_sys)
        if text is not None:
            formatted_data.append({'text': text})

    n_positive = len(formatted_data)
    logger.info('Formatted %d positive (compressed) examples.', n_positive)

    # Sample negative (raw thinking) examples on top
    n_negative = max(1, int(len(shuffled) * args.negative_fraction)) if len(shuffled) > 1 else 0
    negative_pool = random.sample(shuffled, n_negative)
    n_added = 0
    for record in negative_pool:
        # 2. Negative system prompt fraction
        include_sys = random.random() < args.negative_system_prompt_fraction
        text = format_record(record, tokenizer, 'normal', include_system_prompt=include_sys)
        if text is not None:
            formatted_data.append({'text': text})
            n_added += 1

    logger.info(
        'Added %d negative (raw thinking) examples on top.',
        n_added,
    )
    logger.info('Total formatted examples: %d', len(formatted_data))


    # Shuffle again for train/valid split
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
    with open(config.train_data, 'w', encoding='utf-8') as f:
        for item in train_set:
            f.write(json.dumps(item) + '\n')

    with open(config.valid_data, 'w', encoding='utf-8') as f:
        for item in valid_set:
            f.write(json.dumps(item) + '\n')

    logger.info('Saved %d training examples to: %s', len(train_set), config.train_data)
    logger.info('Saved %d validation examples to: %s', len(valid_set), config.valid_data)
    logger.info('SFT formatting complete.')


if __name__ == '__main__':
    main()
