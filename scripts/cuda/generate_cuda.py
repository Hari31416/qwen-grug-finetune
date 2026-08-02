import os
import sys
import argparse
import logging

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from scripts.config import config
from scripts.cuda.cuda_utils import (
    resolve_hf_model_id,
    patch_transformers_lazy_imports,
    load_causal_lm_model,
    load_causal_lm_tokenizer,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("generate_cuda")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run model generation on CUDA / MPS / CPU")
    parser.add_argument(
        "--model",
        type=str,
        default=config.model_mlx_path,
        help="Hugging Face model ID or local directory",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default="",
        help="Optional path to fine-tuned LoRA adapter",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="If John has 3 apples and buys 2 more, how many does he have?",
        help="The prompt for generation",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="",
        help="Optional system prompt",
    )
    parser.add_argument(
        "--temp",
        type=float,
        default=config.temperature,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=config.top_p,
        help="Sampling top-p",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=config.max_generation_tokens,
        help="Max generation tokens",
    )

    args = parser.parse_args()

    # Apply patch to prevent BloomPreTrainedModel ModuleNotFoundError in Kaggle/Colab
    patch_transformers_lazy_imports()

    import torch

    hf_model_id = resolve_hf_model_id(args.model)
    logger.info("Loading Base Model: %s", hf_model_id)

    is_cuda = torch.cuda.is_available()
    device = "cuda" if is_cuda else ("mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Target Device: %s", device)

    model_kwargs = {}
    if is_cuda:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = bnb_config
        model_kwargs["device_map"] = "auto"
        model_kwargs["torch_dtype"] = torch.float16

    model = load_causal_lm_model(hf_model_id, **model_kwargs)
    if not is_cuda:
        model.to(device)

    tokenizer = load_causal_lm_tokenizer(hf_model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.adapter_path:
        logger.info("Loading LoRA adapter from: %s", args.adapter_path)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)

    model.eval()

    messages = []
    if args.system_prompt:
        messages.append({"role": "system", "content": args.system_prompt})
    messages.append({"role": "user", "content": args.prompt})

    formatted_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(device)

    logger.info("Generating response on %s...", device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            temperature=args.temp if args.temp > 0 else None,
            top_p=args.top_p if args.temp > 0 else None,
            do_sample=(args.temp > 0),
            pad_token_id=tokenizer.pad_token_id,
        )

    input_len = inputs["input_ids"].shape[1]
    gen_tokens = outputs[0][input_len:]
    response_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

    print("\n========== MODEL RESPONSE ==========")
    print(response_text)
    print("===================================\n")


if __name__ == "__main__":
    main()
