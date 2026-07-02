import logging
from typing import Tuple, Any, List, Optional
from mlx_lm import load
from mlx_lm.sample_utils import make_sampler, make_logits_processors

logger = logging.getLogger("generation_utils")


def load_model_and_tokenizer(model_path: str, adapter_path: Optional[str] = None) -> Tuple[Any, Any]:
    """Loads the model and tokenizer from the specified path, optionally with an adapter."""
    logger.info("Loading model and tokenizer from: %s", model_path)
    if adapter_path:
        logger.info("Using adapter path: %s", adapter_path)
    return load(model_path, adapter_path=adapter_path)


def get_generation_parameters(
    temp: float,
    top_p: float,
    repetition_penalty: float = 1.1,
    presence_penalty: float = 0.2,
) -> Tuple[Any, List[Any]]:
    """Creates a sampler and logits processors for MLX generation.

    If repetition_penalty is 1.0 or presence_penalty is 0.0, they will be disabled.
    """
    sampler = make_sampler(temp=temp, top_p=top_p)
    logits_processors = make_logits_processors(
        repetition_penalty=repetition_penalty if repetition_penalty != 1.0 else None,
        presence_penalty=presence_penalty if presence_penalty != 0.0 else None,
    )
    return sampler, logits_processors


def parse_thinking_and_answer(output: str, strip_prefix: bool = True) -> Tuple[str, str]:
    """Parses thinking block and final answer from model output."""
    if "</think>" in output:
        parts = output.split("</think>", 1)
        raw_thinking = parts[0]
        raw_answer = parts[1].strip()
    else:
        raw_thinking = output
        raw_answer = ""

    if strip_prefix:
        # Strip default Qwen thinking prefix if present
        for prefix in ("Thinking Process:\n\n", "Thinking Process:"):
            if raw_thinking.startswith(prefix):
                raw_thinking = raw_thinking[len(prefix) :]
                break

    return raw_thinking, raw_answer
