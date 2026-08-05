import os
import sys
import logging
from typing import Any

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

logger = logging.getLogger("cuda_utils")


def resolve_hf_model_id(model_arg: str) -> str:
    """Resolves an MLX model path into standard Hugging Face repo ID if necessary."""
    if "mlx-community" in model_arg:
        if "DeepSeek-R1-Distill-Qwen-7B" in model_arg:
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"
        if "DeepSeek-R1-Distill-Qwen-1.5B" in model_arg:
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
        if "DeepSeek-R1-Distill-Qwen-14B" in model_arg:
            return "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
    return model_arg


def patch_transformers_lazy_imports() -> None:
    """Patches broken lazy imports and torchao version check in transformers in Kaggle/Colab environments."""
    try:
        import transformers
        for mod_name in ["BloomPreTrainedModel", "BloomForCausalLM", "BloomModel"]:
            try:
                getattr(transformers, mod_name)
            except Exception:
                class DummyBloomClass:
                    pass
                setattr(transformers, mod_name, DummyBloomClass)
                logger.debug("Patched transformers.%s with DummyClass", mod_name)

        # Patch incompatible torchao version error in transformers
        try:
            import transformers.utils.import_utils as import_utils
            if hasattr(import_utils, "is_torchao_available"):
                import importlib.metadata
                try:
                    v = importlib.metadata.version("torchao")
                    from packaging.version import parse
                    if parse(v) < parse("0.16.0"):
                        logger.info("Found old torchao version %s (< 0.16.0). Disabling torchao in transformers.", v)
                        import_utils.is_torchao_available = lambda: False
                except Exception:
                    pass
        except Exception as e:
            logger.debug("Could not patch torchao check: %s", e)

    except Exception as e:
        logger.warning("Could not patch transformers lazy imports: %s", e)


def load_causal_lm_model(hf_model_id: str, **kwargs: Any) -> Any:
    """Robustly loads a CausalLM model with fallback to Qwen2ForCausalLM if AutoModel mapping fails."""
    try:
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(hf_model_id, **kwargs)
    except Exception as e:
        logger.warning(
            "AutoModelForCausalLM failed (%s). Falling back directly to Qwen2ForCausalLM...", e
        )
        from transformers import Qwen2ForCausalLM
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "trust_remote_code"}
        return Qwen2ForCausalLM.from_pretrained(hf_model_id, **kwargs_clean)


def load_causal_lm_tokenizer(hf_model_id: str, **kwargs: Any) -> Any:
    """Robustly loads tokenizer with fallback to Qwen2TokenizerFast if AutoTokenizer fails."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(hf_model_id, **kwargs)
    except Exception as e:
        logger.warning(
            "AutoTokenizer failed (%s). Falling back directly to Qwen2TokenizerFast...", e
        )
        from transformers import Qwen2TokenizerFast
        kwargs_clean = {k: v for k, v in kwargs.items() if k != "trust_remote_code"}
        return Qwen2TokenizerFast.from_pretrained(hf_model_id, **kwargs_clean)
