import os
import sys
import argparse
import logging
import torch

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.config import config
from scripts.cuda.generation_utils import (
    load_model_and_tokenizer,
    get_generation_parameters,
)

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("generate")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model generation with PyTorch/CUDA")
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

    model, tokenizer = load_model_and_tokenizer(config.model_hf_path, quantized=config.model_quantized)

    # Format the prompt using the model's tokenizer chat template
    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})

    formatted_prompt: str = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    logger.info(
        "Tokenizing prompt (temp=%.2f, top_p=%.2f, rep_penalty=%.2f)",
        args.temp,
        args.top_p,
        args.repetition_penalty,
    )

    # Move inputs to CUDA if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    generation_params = get_generation_parameters(
        temp=args.temp,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        presence_penalty=args.presence_penalty,
    )

    logger.info("Generating response on device: %s", device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            pad_token_id=tokenizer.eos_token_id,
            **generation_params
        )

    # Decode and print generated answer only (excluding input tokens)
    generated_ids = outputs[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(response)


if __name__ == "__main__":
    main()
