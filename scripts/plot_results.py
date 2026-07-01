import os
import sys
import json
import argparse
import logging
from typing import Dict, Any, Optional

# Add workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("plot_results")


def load_json_summary(path: str) -> Optional[Dict[str, Any]]:
    """Loads and returns the 'summary' block from an evaluation JSON file."""
    if not os.path.exists(path):
        logger.warning("File not found: %s", path)
        return None
    try:
        with open(path, "r") as f:
            data = json.load(f)
            return data.get("summary")
    except Exception as e:
        logger.error("Failed to parse JSON file %s: %s", path, e)
        return None


def get_latest_metrics() -> Optional[Dict[str, Any]]:
    """Finds the latest metrics.json in timestamped adapter directories."""
    base_adapter_dir = config.adapters
    if not os.path.exists(base_adapter_dir):
        return None
    
    subdirs = [
        os.path.join(base_adapter_dir, d)
        for d in os.listdir(base_adapter_dir)
        if os.path.isdir(os.path.join(base_adapter_dir, d))
    ]
    valid_subdirs = [
        sd for sd in subdirs if os.path.exists(os.path.join(sd, "metrics.json"))
    ]
    if not valid_subdirs:
        return None
    
    # Sort and pick the latest run
    latest_dir = max(valid_subdirs)
    logger.info("Found latest training metrics in: %s", latest_dir)
    try:
        with open(os.path.join(latest_dir, "metrics.json"), "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to read metrics from %s: %s", latest_dir, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Create comparison dashboard plots from training logs and evaluation results")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the generated dashboard plot (defaults to results/{model}/plots/)",
    )
    args = parser.parse_args()

    # Determine paths
    model_results_dir = config.results
    plots_dir = args.output_dir if args.output_dir else os.path.join(model_results_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load evaluation summaries
    eval_files = {
        "Base Normal": os.path.join(model_results_dir, "baseline", "gsm8k_normal.json"),
        "Base Grug": os.path.join(model_results_dir, "baseline", "gsm8k_grug_prompt.json"),
        "FT Normal": os.path.join(model_results_dir, "finetuned", "gsm8k_normal.json"),
        "FT Grug": os.path.join(model_results_dir, "finetuned", "gsm8k_grug_prompt.json"),
    }

    summaries = {}
    for name, path in eval_files.items():
        summary = load_json_summary(path)
        if summary:
            summaries[name] = summary

    # 2. Load training metrics
    train_metrics = get_latest_metrics()

    if not summaries and not train_metrics:
        logger.error("No metrics or evaluation summaries found. Cannot generate plot.")
        sys.exit(1)

    import matplotlib.pyplot as plt
    import numpy as np

    # Set style/colors
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    
    # Create 2x2 figure layout
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    
    # --- SUBPLOT 1: Training Loss Curve ---
    ax1 = axs[0, 0]
    if train_metrics and train_metrics.get("train_losses"):
        train_steps = train_metrics.get("train_steps", [])
        train_losses = train_metrics.get("train_losses", [])
        val_steps = train_metrics.get("val_steps", [])
        val_losses = train_metrics.get("val_losses", [])
        
        ax1.plot(train_steps, train_losses, label="Train Loss", marker="o", color="#1f77b4", alpha=0.8, linewidth=2)
        if val_losses:
            ax1.plot(val_steps, val_losses, label="Validation Loss", marker="s", color="#ff7f0e", linestyle="--", linewidth=2)
        ax1.set_title("LoRA SFT Training & Validation Loss", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Cross Entropy Loss")
        ax1.legend()
        ax1.grid(True, linestyle=":", alpha=0.6)
    else:
        ax1.text(0.5, 0.5, "No Training Loss Data Available", ha="center", va="center", fontsize=12)
        ax1.set_title("LoRA SFT Training & Validation Loss", fontsize=12, fontweight="bold")

    # Variants present in summaries
    variants = [v for v in ["Base Normal", "Base Grug", "FT Normal", "FT Grug"] if v in summaries]

    # --- SUBPLOT 2: Accuracy Comparison ---
    ax2 = axs[0, 1]
    if variants:
        accuracies = [summaries[v]["accuracy"] * 100 for v in variants]
        bars = ax2.bar(variants, accuracies, color=[colors[i] for i, v in enumerate(["Base Normal", "Base Grug", "FT Normal", "FT Grug"]) if v in summaries], width=0.5)
        ax2.set_title("GSM8K Accuracy Comparison", fontsize=12, fontweight="bold")
        ax2.set_ylabel("Accuracy (%)")
        ax2.set_ylim(0, 100)
        ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
        
        # Add labels on top of bars
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(f"{height:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),  # 3 points vertical offset
                         textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")
    else:
        ax2.text(0.5, 0.5, "No Evaluation Data Available", ha="center", va="center", fontsize=12)
        ax2.set_title("GSM8K Accuracy Comparison", fontsize=12, fontweight="bold")

    # --- SUBPLOT 3: Mean Reasoning Tokens ---
    ax3 = axs[1, 0]
    if variants:
        x = np.arange(len(variants))
        width = 0.35
        
        think_tokens = [summaries[v].get("mean_thinking_tokens", 0) for v in variants]
        ans_tokens = [summaries[v].get("mean_answer_tokens", 0) for v in variants]
        
        rects1 = ax3.bar(x - width/2, think_tokens, width, label="Thinking Tokens", color="#1f77b4")
        rects2 = ax3.bar(x + width/2, ans_tokens, width, label="Answer Tokens", color="#aec7e8")
        
        ax3.set_title("Emitted Tokens per Problem", fontsize=12, fontweight="bold")
        ax3.set_xticks(x)
        ax3.set_xticklabels(variants)
        ax3.set_ylabel("Average Tokens")
        ax3.legend()
        ax3.grid(True, axis="y", linestyle=":", alpha=0.6)
        
        # Add values inside or on top of bars
        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                ax3.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                ax3.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "No Evaluation Data Available", ha="center", va="center", fontsize=12)
        ax3.set_title("Emitted Tokens per Problem", fontsize=12, fontweight="bold")

    # --- SUBPLOT 4: Inference Latency and Speed ---
    ax4 = axs[1, 1]
    if variants:
        x = np.arange(len(variants))
        width = 0.35
        
        latencies = [summaries[v].get("mean_latency", 0) for v in variants]
        speeds = [summaries[v].get("mean_tokens_per_second", 0) for v in variants]
        
        color_lat = "#d62728"
        color_spd = "#2ca02c"
        
        rects_lat = ax4.bar(x - width/2, latencies, width, label="Latency (s)", color=color_lat, alpha=0.8)
        
        ax4_right = ax4.twinx()
        rects_spd = ax4_right.bar(x + width/2, speeds, width, label="Speed (tok/s)", color=color_spd, alpha=0.8)
        
        ax4.set_title("Latency vs Generation Speed", fontsize=12, fontweight="bold")
        ax4.set_xticks(x)
        ax4.set_xticklabels(variants)
        
        ax4.set_ylabel("Inference Latency (seconds)", color=color_lat)
        ax4.tick_params(axis="y", labelcolor=color_lat)
        
        ax4_right.set_ylabel("Token Generation Speed (tok/s)", color=color_spd)
        ax4_right.tick_params(axis="y", labelcolor=color_spd)
        ax4_right.grid(False)
        
        # Annotate
        for rect in rects_lat:
            h = rect.get_height()
            if h > 0:
                ax4.annotate(f"{h:.2f}s", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=color_lat, fontweight="bold")
        for rect in rects_spd:
            h = rect.get_height()
            if h > 0:
                ax4_right.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=color_spd, fontweight="bold")
    else:
        ax4.text(0.5, 0.5, "No Evaluation Data Available", ha="center", va="center", fontsize=12)
        ax4.set_title("Latency vs Generation Speed", fontsize=12, fontweight="bold")

    fig.tight_layout()
    plot_file = os.path.join(plots_dir, "dashboard.png")
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Successfully generated dashboard plot at: %s", plot_file)

    # --- SAVE SEPARATE PLOTS ---
    logger.info("Generating and saving individual plots...")

    # Plot 1: Training Loss Curve
    if train_metrics and train_metrics.get("train_losses"):
        plt.figure(figsize=(8, 6))
        train_steps = train_metrics.get("train_steps", [])
        train_losses = train_metrics.get("train_losses", [])
        val_steps = train_metrics.get("val_steps", [])
        val_losses = train_metrics.get("val_losses", [])
        
        plt.plot(train_steps, train_losses, label="Train Loss", marker="o", color="#1f77b4", alpha=0.8, linewidth=2)
        if val_losses:
            plt.plot(val_steps, val_losses, label="Validation Loss", marker="s", color="#ff7f0e", linestyle="--", linewidth=2)
        plt.title("LoRA SFT Training & Validation Loss", fontsize=12, fontweight="bold")
        plt.xlabel("Iteration")
        plt.ylabel("Cross Entropy Loss")
        plt.legend()
        plt.grid(True, linestyle=":", alpha=0.6)
        
        loss_curve_file = os.path.join(plots_dir, "loss_curve.png")
        plt.savefig(loss_curve_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", loss_curve_file)

    # Plot 2: Accuracy Comparison
    if variants:
        plt.figure(figsize=(8, 6))
        accuracies = [summaries[v]["accuracy"] * 100 for v in variants]
        bars = plt.bar(variants, accuracies, color=[colors[i] for i, v in enumerate(["Base Normal", "Base Grug", "FT Normal", "FT Grug"]) if v in summaries], width=0.5)
        plt.title("GSM8K Accuracy Comparison", fontsize=12, fontweight="bold")
        plt.ylabel("Accuracy (%)")
        plt.ylim(0, 100)
        plt.grid(True, axis="y", linestyle=":", alpha=0.6)
        
        for bar in bars:
            height = bar.get_height()
            plt.annotate(f"{height:.1f}%",
                         xy=(bar.get_x() + bar.get_width() / 2, height),
                         xytext=(0, 3),
                         textcoords="offset points",
                         ha="center", va="bottom", fontweight="bold")
                         
        accuracy_file = os.path.join(plots_dir, "accuracy.png")
        plt.savefig(accuracy_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", accuracy_file)

    # Plot 3: Mean Reasoning Tokens
    if variants:
        plt.figure(figsize=(8, 6))
        x = np.arange(len(variants))
        width = 0.35
        think_tokens = [summaries[v].get("mean_thinking_tokens", 0) for v in variants]
        ans_tokens = [summaries[v].get("mean_answer_tokens", 0) for v in variants]
        
        rects1 = plt.bar(x - width/2, think_tokens, width, label="Thinking Tokens", color="#1f77b4")
        rects2 = plt.bar(x + width/2, ans_tokens, width, label="Answer Tokens", color="#aec7e8")
        
        plt.title("Emitted Tokens per Problem", fontsize=12, fontweight="bold")
        plt.xticks(x, variants)
        plt.ylabel("Average Tokens")
        plt.legend()
        plt.grid(True, axis="y", linestyle=":", alpha=0.6)
        
        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                plt.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                plt.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9)
                
        tokens_file = os.path.join(plots_dir, "tokens.png")
        plt.savefig(tokens_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", tokens_file)

    # Plot 4: Latency vs Speed
    if variants:
        fig_ind, ax1_ind = plt.subplots(figsize=(8, 6))
        x = np.arange(len(variants))
        width = 0.35
        latencies = [summaries[v].get("mean_latency", 0) for v in variants]
        speeds = [summaries[v].get("mean_tokens_per_second", 0) for v in variants]
        
        color_lat = "#d62728"
        color_spd = "#2ca02c"
        
        rects_lat = ax1_ind.bar(x - width/2, latencies, width, label="Latency (s)", color=color_lat, alpha=0.8)
        ax2_ind = ax1_ind.twinx()
        rects_spd = ax2_ind.bar(x + width/2, speeds, width, label="Speed (tok/s)", color=color_spd, alpha=0.8)
        
        plt.title("Latency vs Generation Speed", fontsize=12, fontweight="bold")
        ax1_ind.set_xticks(x)
        ax1_ind.set_xticklabels(variants)
        
        ax1_ind.set_ylabel("Inference Latency (seconds)", color=color_lat)
        ax1_ind.tick_params(axis="y", labelcolor=color_lat)
        
        ax2_ind.set_ylabel("Token Generation Speed (tok/s)", color=color_spd)
        ax2_ind.tick_params(axis="y", labelcolor=color_spd)
        ax2_ind.grid(False)
        
        for rect in rects_lat:
            h = rect.get_height()
            if h > 0:
                ax1_ind.annotate(f"{h:.2f}s", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=color_lat, fontweight="bold")
        for rect in rects_spd:
            h = rect.get_height()
            if h > 0:
                ax2_ind.annotate(f"{int(h)}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=(0, 2), textcoords="offset points", ha="center", va="bottom", fontsize=9, color=color_spd, fontweight="bold")
                
        latency_spd_file = os.path.join(plots_dir, "latency_speed.png")
        plt.savefig(latency_spd_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", latency_spd_file)

    # Plot 5: Deltas (Gain/Loss in Accuracy and Tokens)
    if "Base Normal" in summaries and "FT Normal" in summaries and "Base Grug" in summaries and "FT Grug" in summaries:
        fig_delta, (ax1_d, ax2_d) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Calculate deltas
        acc_delta_normal = (summaries["FT Normal"]["accuracy"] - summaries["Base Normal"]["accuracy"]) * 100
        acc_delta_grug = (summaries["FT Grug"]["accuracy"] - summaries["Base Grug"]["accuracy"]) * 100
        
        think_delta_normal = summaries["FT Normal"].get("mean_thinking_tokens", 0) - summaries["Base Normal"].get("mean_thinking_tokens", 0)
        think_delta_grug = summaries["FT Grug"].get("mean_thinking_tokens", 0) - summaries["Base Grug"].get("mean_thinking_tokens", 0)
        
        total_delta_normal = summaries["FT Normal"].get("mean_total_tokens", 0) - summaries["Base Normal"].get("mean_total_tokens", 0)
        total_delta_grug = summaries["FT Grug"].get("mean_total_tokens", 0) - summaries["Base Grug"].get("mean_total_tokens", 0)
        
        # Left Panel: Accuracy Delta (Percentage Points)
        labels = ["Normal Prompt", "Grug Prompt"]
        acc_deltas = [acc_delta_normal, acc_delta_grug]
        
        # Color: red for loss, green for gain
        colors_acc = ["#d62728" if d < 0 else "#2ca02c" for d in acc_deltas]
        bars_acc = ax1_d.bar(labels, acc_deltas, color=colors_acc, width=0.4)
        ax1_d.axhline(0, color="black", linestyle="-", linewidth=1)
        ax1_d.set_title("Accuracy Change (Fine-Tuned vs Base)", fontsize=12, fontweight="bold")
        ax1_d.set_ylabel("Accuracy Delta (percentage points)")
        # Set limits with some padding
        min_acc = min(acc_deltas)
        max_acc = max(acc_deltas)
        ax1_d.set_ylim(min(min_acc * 1.3, -5), max(max_acc * 1.3, 5))
        ax1_d.grid(True, linestyle=":", alpha=0.6)
        
        for bar in bars_acc:
            h = bar.get_height()
            va_pos = "bottom" if h >= 0 else "top"
            xy_pos = (bar.get_x() + bar.get_width()/2, h)
            offset = (0, 3) if h >= 0 else (0, -12)
            ax1_d.annotate(f"{h:+.1f}%", xy=xy_pos, xytext=offset, textcoords="offset points", ha="center", va=va_pos, fontweight="bold")

        # Right Panel: Token Deltas (Average Tokens Saved/Added)
        x_delta = np.arange(len(labels))
        width_d = 0.35
        
        think_deltas = [think_delta_normal, think_delta_grug]
        total_deltas = [total_delta_normal, total_delta_grug]
        
        rects_think = ax2_d.bar(x_delta - width_d/2, think_deltas, width_d, label="Thinking Tokens Delta", color="#e377c2")
        rects_total = ax2_d.bar(x_delta + width_d/2, total_deltas, width_d, label="Total Tokens Delta", color="#7f7f7f")
        
        ax2_d.axhline(0, color="black", linestyle="-", linewidth=1)
        ax2_d.set_title("Emitted Tokens Change (Fine-Tuned vs Base)", fontsize=12, fontweight="bold")
        ax2_d.set_xticks(x_delta)
        ax2_d.set_xticklabels(labels)
        ax2_d.set_ylabel("Token Delta (negative is fewer tokens)")
        ax2_d.legend()
        ax2_d.grid(True, linestyle=":", alpha=0.6)
        
        # Set limits with some padding
        min_tok = min(min(think_deltas), min(total_deltas))
        max_tok = max(max(think_deltas), max(total_deltas))
        ax2_d.set_ylim(min(min_tok * 1.3, -50), max(max_tok * 1.3, 50))
        
        for rect in rects_think:
            h = rect.get_height()
            va_pos = "bottom" if h >= 0 else "top"
            offset = (0, 2) if h >= 0 else (0, -12)
            ax2_d.annotate(f"{int(h):+d}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=offset, textcoords="offset points", ha="center", va=va_pos, fontsize=9, fontweight="bold")
            
        for rect in rects_total:
            h = rect.get_height()
            va_pos = "bottom" if h >= 0 else "top"
            offset = (0, 2) if h >= 0 else (0, -12)
            ax2_d.annotate(f"{int(h):+d}", xy=(rect.get_x() + rect.get_width()/2, h), xytext=offset, textcoords="offset points", ha="center", va=va_pos, fontsize=9, fontweight="bold")
            
        plt.tight_layout()
        delta_file = os.path.join(plots_dir, "deltas.png")
        plt.savefig(delta_file, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info("Saved: %s", delta_file)


if __name__ == "__main__":
    main()
