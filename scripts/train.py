import os
import sys
import argparse
import logging
import subprocess
import re
import json
import datetime
from typing import List, Optional

# Add workspace root to Python path to import config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("train")


def run_training(
    model: str,
    adapter_path: str,
    data: str,
    config_file: str,
    iters: Optional[int] = None,
    batch_size: Optional[int] = None,
    learning_rate: Optional[float] = None,
    lora_layers: Optional[int] = 16,
    train: bool = True,
    test: bool = False,
    save_every: int = 20,
) -> None:
    """Invokes mlx_lm.lora finetuning script as a subprocess with dynamically resolved parameters."""
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"LoRA config file not found: {config_file}")

    # Build command list
    cmd: List[str] = [
        sys.executable,
        "-m",
        "mlx_lm",
        "lora",
        "--model",
        model,
        "--data",
        data,
        "--adapter-path",
        adapter_path,
        "-c",
        config_file,
        "--save-every",
        str(save_every),
        "--steps-per-eval",
        str(save_every),
    ]

    if train:
        cmd.append("--train")
    if test:
        cmd.append("--test")

    if iters is not None:
        cmd.extend(["--iters", str(iters)])
    if batch_size is not None:
        cmd.extend(["--batch-size", str(batch_size)])
    if learning_rate is not None:
        cmd.extend(["--learning-rate", str(learning_rate)])
    if lora_layers is not None:
        # Maps to mlx_lm.lora's --num-layers option
        cmd.extend(["--num-layers", str(lora_layers)])

    logger.info("Executing training command: %s", " ".join(cmd))

    # Initialize lists to capture training metrics
    train_steps: List[int] = []
    train_losses: List[float] = []
    val_steps: List[int] = []
    val_losses: List[float] = []

    # Compile regexes to parse metrics from output
    train_pattern = re.compile(r"Iter (\d+): Train loss ([\d\.]+)")
    val_pattern = re.compile(r"Iter (\d+): Val loss ([\d\.]+)")

    # Execute subprocess and capture output in real-time
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Print training output as it comes
        if process.stdout:
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()

                # Parse training loss
                train_match = train_pattern.search(line)
                if train_match:
                    try:
                        step = int(train_match.group(1))
                        loss = float(train_match.group(2))
                        train_steps.append(step)
                        train_losses.append(loss)
                    except ValueError:
                        pass

                # Parse validation loss
                val_match = val_pattern.search(line)
                if val_match:
                    try:
                        step = int(val_match.group(1))
                        loss = float(val_match.group(2))
                        val_steps.append(step)
                        val_losses.append(loss)
                    except ValueError:
                        pass

        process.wait()
        if process.returncode != 0:
            logger.error("Training command exited with non-zero code: %d", process.returncode)
            sys.exit(process.returncode)
        else:
            logger.info("Training completed successfully. Adapters saved to: %s", adapter_path)

            # Determine best step from validation losses
            best_step = None
            best_val_loss = float("inf")
            for step, val_loss in zip(val_steps, val_losses):
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_step = step

            if best_step is not None:
                best_file_name = f"{best_step:07d}_adapters.safetensors"
                src_path = os.path.join(adapter_path, best_file_name)
                best_dest = os.path.join(adapter_path, "best_adapters.safetensors")
                default_dest = os.path.join(adapter_path, "adapters.safetensors")
                
                if os.path.exists(src_path):
                    import shutil
                    try:
                        shutil.copy2(src_path, best_dest)
                        logger.info("Saved best checkpoint copy (%s) to %s (Val loss: %.4f)", best_file_name, best_dest, best_val_loss)
                        shutil.copy2(src_path, default_dest)
                        logger.info("Overwrote default adapters.safetensors with best checkpoint.")
                    except Exception as copy_err:
                        logger.error("Failed to copy best adapter checkpoint: %s", copy_err)
                else:
                    logger.warning("Expected best checkpoint file not found at %s", src_path)

            # Save metrics to JSON file
            metrics = {
                "train_steps": train_steps,
                "train_losses": train_losses,
                "val_steps": val_steps,
                "val_losses": val_losses,
            }
            metrics_file = os.path.join(adapter_path, "metrics.json")
            try:
                with open(metrics_file, "w") as f:
                    json.dump(metrics, f, indent=2)
                logger.info("Saved training metrics to: %s", metrics_file)
            except Exception as e:
                logger.error("Failed to save metrics JSON: %s", e)

            # Generate loss plot
            if train_losses or val_losses:
                try:
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    if train_losses:
                        plt.plot(train_steps, train_losses, label="Train Loss", marker="o", linestyle="-", color="#1f77b4")
                    if val_losses:
                        plt.plot(val_steps, val_losses, label="Validation Loss", marker="s", linestyle="--", color="#ff7f0e")
                    
                    plt.xlabel("Iteration")
                    plt.ylabel("Loss")
                    plt.title("LoRA SFT Training & Validation Loss")
                    plt.legend()
                    plt.grid(True, linestyle=":", alpha=0.6)
                    
                    plot_file = os.path.join(adapter_path, "loss_plot.png")
                    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
                    plt.close()
                    logger.info("Saved loss plot to: %s", plot_file)
                except Exception as pe:
                    logger.error("Failed to generate and save loss plot: %s", pe)

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user. Cleaning up process...")
        process.terminate()
        sys.exit(1)
    except Exception as e:
        logger.exception("An error occurred during training execution: %s", e)
        sys.exit(1)


def main() -> None:
    workspace_root: str = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    default_config_file: str = os.path.join(workspace_root, "lora_config.yaml")

    parser = argparse.ArgumentParser(description="Wrapper around mlx_lm.lora reading config.yaml and lora_config.yaml")
    parser.add_argument(
        "--model",
        type=str,
        default=config.model_mlx_path,
        help="Path/repo name of the base model",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=config.adapters,
        help="Directory to save the fine-tuned adapter weights",
    )
    parser.add_argument(
        "--data",
        type=str,
        default=config.data_dir,
        help="Directory containing train.jsonl and valid.jsonl",
    )
    parser.add_argument(
        "--config-file",
        type=str,
        default=default_config_file,
        help="Path to the LoRA configuration YAML file",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=None,
        help="Override number of training iterations",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--lora-layers",
        type=int,
        default=16,
        help="Number of layers to fine-tune (maps to --num-layers in mlx_lm.lora)",
    )
    parser.add_argument(
        "--no-train",
        dest="train",
        action="store_false",
        help="Do not perform training (e.g. if testing only)",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run evaluation on the test set after training",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=20,
        help="Interval in iterations to save snapshots and run validation (maps to --save-every and --steps-per-eval)",
    )

    parser.set_defaults(train=True)
    args = parser.parse_args()

    # Pre-checks
    if not os.path.exists(args.data):
        logger.error("Dataset directory does not exist: %s", args.data)
        sys.exit(1)

    train_file = os.path.join(args.data, "train.jsonl")
    valid_file = os.path.join(args.data, "valid.jsonl")
    if not os.path.exists(train_file) or not os.path.exists(valid_file):
        logger.error("Required datasets train.jsonl and valid.jsonl not found in data directory: %s", args.data)
        sys.exit(1)

    # Resolve output directory with timestamp to avoid overwriting previous runs
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.adapter_path = os.path.join(args.adapter_path, timestamp)
    os.makedirs(args.adapter_path, exist_ok=True)

    logger.info("Initializing LoRA training wrapper...")
    logger.info("  Base Model:        %s", args.model)
    logger.info("  Adapter Path:      %s", args.adapter_path)
    logger.info("  Data Directory:    %s", args.data)
    logger.info("  Config File:       %s", args.config_file)
    logger.info("  Num Layers:        %d", args.lora_layers)
    logger.info("  Save & Eval Every: %d", args.save_every)
    if args.iters:
        logger.info("  Iters Override:    %d", args.iters)
    if args.batch_size:
        logger.info("  Batch-Size Overr:  %d", args.batch_size)
    if args.learning_rate:
        logger.info("  LR Override:       %s", args.learning_rate)

    run_training(
        model=args.model,
        adapter_path=args.adapter_path,
        data=args.data,
        config_file=args.config_file,
        iters=args.iters,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lora_layers=args.lora_layers,
        train=args.train,
        test=args.test,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
