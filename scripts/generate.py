import os
import sys
import argparse
import logging
from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler, make_logits_processors

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model generation with MLX-LM")
    parser.add_argument(
        "--prompt",
        type=str,
        default="If John has 3 apples and buys 2 more, how many does he have?",
        help="The prompt for generation",
    )
    parser.add_argument(
        "--temp", type=float, default=config.temperature, help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p", type=float, default=config.top_p, help="Sampling top-p"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.max_generation_tokens,
        help="Max tokens to generate",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (1.0 to disable)",
    )
    parser.add_argument(
        "--presence-penalty",
        type=float,
        default=0.2,
        help="Presence penalty (0.0 to disable)",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system prompt to guide model behavior",
    )

    args = parser.parse_args()

    logger.info("Loading model from: %s", config.model_mlx_path)
    model, tokenizer = load(config.model_mlx_path)

    # Format the prompt using the model's tokenizer chat template
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})

    formatted_prompt: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    logger.info(
        "Creating sampler and logits processors (temp=%.2f, top_p=%.2f, rep_penalty=%.2f, pres_penalty=%.2f)",
        args.temp,
        args.top_p,
        args.repetition_penalty,
        args.presence_penalty,
    )

    # 1. Create native MLX-LM sampler
    sampler = make_sampler(temp=args.temp, top_p=args.top_p)

    # 2. Create logits processors for repetition / presence penalty
    logits_processors = make_logits_processors(
        repetition_penalty=args.repetition_penalty if args.repetition_penalty != 1.0 else None,
        presence_penalty=args.presence_penalty if args.presence_penalty != 0.0 else None,
    )

    # Generate response
    generate(
        model,
        tokenizer,
        prompt=formatted_prompt,
        max_tokens=args.max_tokens,
        sampler=sampler,
        logits_processors=logits_processors,
        verbose=True,
    )


if __name__ == "__main__":
    main()
