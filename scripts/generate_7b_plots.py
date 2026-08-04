import os
import json
import matplotlib.pyplot as plt
import numpy as np

def load_summary(path):
    if not os.path.exists(path):
        return None
    with open(path, "r") as f:
        data = json.load(f)
    return data.get("summary")

def main():
    workspace_root = "/Users/hari/Desktop/sandbox/qwen-finetune"
    report_dir = os.path.join(workspace_root, "report")
    os.makedirs(report_dir, exist_ok=True)

    paths = {
        "1.5b_base": os.path.join(workspace_root, "results/deepseek-r1-1.5b/baseline/gsm8k.json"),
        "1.5b_ft": os.path.join(workspace_root, "results/deepseek-r1-1.5b/finetuned/gsm8k.json"),
        "7b_base": os.path.join(workspace_root, "results/deepseek-r1-7b/baseline/gsm8k.json"),
        "7b_ft": os.path.join(workspace_root, "results/deepseek-r1-7b/finetuned/gsm8k.json"),
    }

    data = {k: load_summary(v) for k, v in paths.items()}

    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")

    # ----------------------------------------------------
    # Plot 1: DeepSeek-R1-7B Base vs Fine-Tuned Dashboard (Accuracy & Tokens)
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    cats_7b = ["7B Base", "7B Fine-Tuned"]
    colors_7b = ["#4A90E2", "#50E3C2"]

    # 1. GSM8K Accuracy
    acc_7b = [data["7b_base"]["accuracy"] * 100, data["7b_ft"]["accuracy"] * 100]
    bars1 = ax1.bar(cats_7b, acc_7b, color=colors_7b, width=0.45)
    ax1.set_title("GSM8K Accuracy (%)", fontsize=12, fontweight="bold", pad=15)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # 2. Token Breakdown
    think_7b = [data["7b_base"]["mean_thinking_tokens"], data["7b_ft"]["mean_thinking_tokens"]]
    ans_7b = [data["7b_base"]["mean_answer_tokens"], data["7b_ft"]["mean_answer_tokens"]]
    ax2.bar(cats_7b, think_7b, label="Thinking Tokens", color="#4A90E2", width=0.45)
    ax2.bar(cats_7b, ans_7b, bottom=think_7b, label="Answer Tokens", color="#B8E986", width=0.45)
    ax2.set_title("Token Count Breakdown", fontsize=12, fontweight="bold", pad=15)
    ax2.set_ylabel("Average Tokens", fontsize=11)
    ax2.set_ylim(0, 320)
    ax2.legend(loc="upper right")
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    for idx, (t, a) in enumerate(zip(think_7b, ans_7b)):
        tot = t + a
        ax2.annotate(f"Total: {int(tot)}", xy=(idx, tot), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")
        ax2.annotate(f"{int(t)}", xy=(idx, t/2), ha="center", va="center", color="white", fontweight="bold")
        ax2.annotate(f"{int(a)}", xy=(idx, t + a/2), ha="center", va="center", color="#333333", fontweight="bold")

    plt.tight_layout()
    plot7b_path = os.path.join(report_dir, "7b_comparison_dashboard.png")
    plt.savefig(plot7b_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", plot7b_path)

    # ----------------------------------------------------
    # Plot 2: 1.5B vs 7B Comparative Cross-Model Overview
    # ----------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    all_cats = ["1.5B Base", "1.5B FT", "7B Base", "7B FT"]
    all_colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    # Accuracy Comparison
    all_acc = [
        data["1.5b_base"]["accuracy"] * 100,
        data["1.5b_ft"]["accuracy"] * 100,
        data["7b_base"]["accuracy"] * 100,
        data["7b_ft"]["accuracy"] * 100
    ]
    bars_acc_all = ax1.bar(all_cats, all_acc, color=all_colors, width=0.5)
    ax1.set_title("GSM8K Accuracy Comparison Across Models", fontsize=12, fontweight="bold", pad=15)
    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars_acc_all:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    # Mean Thinking Tokens Comparison
    all_think = [
        data["1.5b_base"]["mean_thinking_tokens"],
        data["1.5b_ft"]["mean_thinking_tokens"],
        data["7b_base"]["mean_thinking_tokens"],
        data["7b_ft"]["mean_thinking_tokens"]
    ]
    bars_think_all = ax2.bar(all_cats, all_think, color=all_colors, width=0.5)
    ax2.set_title("Mean Thinking Tokens Comparison", fontsize=12, fontweight="bold", pad=15)
    ax2.set_ylabel("Thinking Tokens", fontsize=11)
    ax2.set_ylim(0, 600)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.6)
    for bar in bars_think_all:
        h = bar.get_height()
        ax2.annotate(f"{int(h)}", xy=(bar.get_x() + bar.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plot_cross_path = os.path.join(report_dir, "cross_model_comparison.png")
    plt.savefig(plot_cross_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", plot_cross_path)

if __name__ == "__main__":
    main()
