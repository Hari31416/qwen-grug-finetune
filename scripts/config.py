import os
import logging
from typing import Dict, Any
import yaml

# Configure logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("config")


class AppConfig:
    """Manages application configurations, path resolutions, and environment setups."""

    def __init__(self, config_path: str) -> None:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        logger.info("Loading config file from: %s", config_path)
        with open(config_path, "r") as f:
            try:
                self._raw_config: Dict[str, Any] = yaml.safe_load(f)
            except yaml.YAMLError as e:
                logger.error("Failed to parse YAML configuration: %s", e)
                raise e

        # Extract Target Model settings
        target_model: Dict[str, Any] = self._raw_config.get("target_model", {})
        self.model_name: str = target_model.get("name", "")
        self.model_mlx_path: str = target_model.get("mlx_path", "")
        self.model_size: str = target_model.get("size", "")

        if not self.model_name or not self.model_mlx_path:
            raise ValueError("Target model 'name' and 'mlx_path' must be specified in config.yaml")

        # Extract Run parameters
        run: Dict[str, Any] = self._raw_config.get("run", {})
        self.seed: int = run.get("seed", 42)
        self.temperature: float = run.get("temperature", 0.6)
        self.top_p: float = run.get("top_p", 0.95)
        self.max_generation_tokens: int = run.get("max_generation_tokens", 1024)
        self.eval_max_generation_tokens: int = run.get("eval_max_generation_tokens", 1024)

        # Extract Dataset settings
        dataset: Dict[str, Any] = self._raw_config.get("dataset", {})
        self.train_split_ratio: float = dataset.get("train_split_ratio", 0.9)
        self.dataset_sample_sizes: Dict[str, int] = dataset.get("sample_sizes", {})

        # Resolve paths relative to workspace root
        self.workspace_root: str = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..")
        )
        paths: Dict[str, str] = self._raw_config.get("paths", {})

        # Resolve static paths
        self.data_dir: str = os.path.join(
            self.workspace_root, paths.get("data_dir", "data")
        )
        self.sft_dir: str = os.path.join(
            self.workspace_root, paths.get("sft_dir", "data/sft")
        )
        self.sft_prompts: str = os.path.join(
            self.workspace_root, paths.get("sft_prompts", "data/sft/prompts.jsonl")
        )
        self.train_data: str = os.path.join(
            self.workspace_root, paths.get("train_data", "data/train.jsonl")
        )
        self.valid_data: str = os.path.join(
            self.workspace_root, paths.get("valid_data", "data/valid.jsonl")
        )

        # Resolve templated paths replacing {name}
        self.adapters: str = os.path.join(
            self.workspace_root,
            paths.get("adapters", "adapters/{name}/").format(name=self.model_name),
        )
        self.results: str = os.path.join(
            self.workspace_root,
            paths.get("results", "results/{name}/").format(name=self.model_name),
        )
        self.raw_traces: str = os.path.join(
            self.workspace_root,
            paths.get("raw_traces", "data/raw/{name}/").format(
                name=self.model_name
            ),
        )
        self.compressed_traces: str = os.path.join(
            self.workspace_root,
            paths.get("compressed_traces", "data/compressed/{name}/").format(
                name=self.model_name
            ),
        )
        self.validated_traces: str = os.path.join(
            self.workspace_root,
            paths.get("validated_traces", "data/validated/{name}/").format(
                name=self.model_name
            ),
        )

    def setup_directories(self) -> None:
        """Create empty directories required for the pipeline if they do not exist."""
        dirs_to_create = [
            self.data_dir,
            self.sft_dir,
            os.path.dirname(self.sft_prompts),
            os.path.dirname(self.train_data),
            os.path.dirname(self.valid_data),
            self.adapters,
            self.results,
            self.raw_traces,
            self.compressed_traces,
            self.validated_traces,
        ]

        for directory in dirs_to_create:
            if directory and not os.path.exists(directory):
                logger.info("Creating directory: %s", directory)
                os.makedirs(directory, exist_ok=True)


# Load configuration as a singleton
config_file_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yaml")
)
try:
    config = AppConfig(config_file_path)
except Exception as exc:
    logger.exception("Failed to initialize configuration")
    raise exc

if __name__ == "__main__":
    logger.info("--- Configuration Setup Report ---")
    logger.info("Target Model Name: %s", config.model_name)
    logger.info("MLX Model Path:    %s", config.model_mlx_path)
    logger.info("Model Size:        %s", config.model_size)
    logger.info("Dataset Sizes:     %s", config.dataset_sample_sizes)
    logger.info("Run parameters:    Seed=%d, Temp=%.2f, TopP=%.2f", config.seed, config.temperature, config.top_p)

    logger.info("Scaffolding empty directories...")
    config.setup_directories()

    logger.info("Resolved Paths:")
    logger.info("  Workspace Root:    %s", config.workspace_root)
    logger.info("  Data Dir:          %s", config.data_dir)
    logger.info("  SFT Dir:           %s", config.sft_dir)
    logger.info("  SFT Prompts:       %s", config.sft_prompts)
    logger.info("  Train Data:        %s", config.train_data)
    logger.info("  Valid Data:        %s", config.valid_data)
    logger.info("  Adapters Dir:      %s", config.adapters)
    logger.info("  Results Dir:       %s", config.results)
    logger.info("  Raw Traces Dir:    %s", config.raw_traces)
    logger.info("  Compressed Traces: %s", config.compressed_traces)
    logger.info("  Validated Traces:  %s", config.validated_traces)
    logger.info("----------------------------------")
