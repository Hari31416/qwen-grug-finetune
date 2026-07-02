import logging
from typing import Tuple, Any, Optional, Dict
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger("generation_utils")


def load_model_and_tokenizer(
    model_path: str,
    adapter_path: Optional[str] = None,
    quantized: bool = False,
) -> Tuple[Any, Any]:
    """Loads the model and tokenizer from Hugging Face on CUDA, optionally loading LoRA adapters and/or quantizing to 4-bit."""
    logger.info("Loading tokenizer from: %s", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Configure padding side to left for batch generation
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    logger.info("Loading model from: %s (quantized=%s)", model_path, quantized)
    cuda_available = torch.cuda.is_available()
    
    if quantized:
        from transformers import BitsAndBytesConfig
        # NF4 quantization config for QLoRA
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if cuda_available and torch.cuda.is_bf16_supported() else torch.float16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quantization_config,
            device_map="auto" if cuda_available else None,
        )
    else:
        dtype = torch.float16
        if cuda_available and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if cuda_available else None,
        )

    if adapter_path:
        logger.info("Loading LoRA adapter from: %s", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)

    return model, tokenizer


def get_generation_parameters(
    temp: float,
    top_p: float,
    repetition_penalty: float = 1.1,
    presence_penalty: float = 0.2,
) -> Dict[str, Any]:
    """Creates a dictionary of generation parameters for Hugging Face generation."""
    params = {
        "temperature": temp,
        "top_p": top_p,
        "repetition_penalty": repetition_penalty if repetition_penalty != 1.0 else None,
        "do_sample": True if temp > 0.0 else False,
    }
    # Presence penalty is not directly supported in standard HF generate without custom logits processors,
    # but repetition_penalty is usually sufficient.
    return {k: v for k, v in params.items() if v is not None}


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
