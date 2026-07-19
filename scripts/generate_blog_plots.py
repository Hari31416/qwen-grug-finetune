import os
import json
import matplotlib.pyplot as plt
import numpy as np


def load_summary(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data["summary"]


def main():
    workspace_root = "/Users/hari/Desktop/sandbox/qwen-finetune"
    report_dir = os.path.join(workspace_root, "report")

    baseline_path = os.path.join(report_dir, "gsm8k_baseline.json")
    finetuned_path = os.path.join(report_dir, "gsm8k_finetuned.json")

    base_data = load_summary(baseline_path)
    ft_data = load_summary(finetuned_path)

    # Set clean design style
    plt.style.use(
        "seaborn-v0_8-whitegrid"
        if "seaborn-v0_8-whitegrid" in plt.style.available
        else "default"
    )

    # Create 1x3 side-by-side subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    categories = ["Base (1.5B)", "Fine-Tuned (1.5B)"]

    # ----------------------------------------------------
    # Subplot 1: GSM8K Math Accuracy
    # ----------------------------------------------------
    accuracies = [base_data["accuracy"] * 100, ft_data["accuracy"] * 100]
    bars_acc = ax1.bar(categories, accuracies, color=["#1f77b4", "#2ca02c"], width=0.4)
    ax1.set_title("GSM8K Math Accuracy", fontsize=12, fontweight="bold", pad=15)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars_acc:
        height = bar.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # ----------------------------------------------------
    # Subplot 2: Stacked Tokens (Thinking + Answer)
    # ----------------------------------------------------
    thinking = [base_data["mean_thinking_tokens"], ft_data["mean_thinking_tokens"]]
    answers = [base_data["mean_answer_tokens"], ft_data["mean_answer_tokens"]]

    # Stacked bars
    bar_think = ax2.bar(
        categories, thinking, label="Thinking Tokens", color="#1f77b4", width=0.4
    )
    bar_ans = ax2.bar(
        categories,
        answers,
        bottom=thinking,
        label="Answer Tokens",
        color="#aec7e8",
        width=0.4,
    )

    ax2.set_title(
        "Token Breakdown per Response", fontsize=12, fontweight="bold", pad=15
    )
    ax2.set_ylabel("Average Tokens", fontsize=11)
    ax2.set_ylim(0, 700)
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)

    # Annotate totals and segments
    for idx, (t, a) in enumerate(zip(thinking, answers)):
        total = t + a
        # Total annotation at the top
        ax2.annotate(
            f"{int(total)}",
            xy=(idx, total),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
        # Segments annotation
        ax2.annotate(
            f"{int(t)}",
            xy=(idx, t / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=9,
        )
        ax2.annotate(
            f"{int(a)}",
            xy=(idx, t + a / 2),
            ha="center",
            va="center",
            color="#333333",
            fontweight="bold",
            fontsize=9,
        )

    # ----------------------------------------------------
    # Subplot 3: Average Inference Latency
    # ----------------------------------------------------
    latencies = [base_data["mean_latency"], ft_data["mean_latency"]]
    bars_lat = ax3.bar(categories, latencies, color=["#1f77b4", "#2ca02c"], width=0.4)
    ax3.set_title("Average Inference Latency", fontsize=12, fontweight="bold", pad=15)
    ax3.set_ylabel("Latency (seconds)", fontsize=11)
    ax3.set_ylim(0, 1.6)
    ax3.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars_lat:
        height = bar.get_height()
        ax3.annotate(
            f"{height:.2f}s",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(
        os.path.join(report_dir, "evaluation_metrics_comparison.png"),
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()
    print("Generated: evaluation_metrics_comparison.png")


if __name__ == "__main__":
    main()
