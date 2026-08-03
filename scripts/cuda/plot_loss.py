#!/usr/bin/env python3
import os
import glob
import json
import matplotlib.pyplot as plt


def plot_latest_training_loss():
    """Finds latest trained adapter run and plots Train Loss, Validation Loss, and Learning Rate."""
    metrics_files = glob.glob("adapters/**/metrics.json", recursive=True) + glob.glob("adapters/**/trainer_state.json", recursive=True)
    if not metrics_files:
        print("No training metrics found under adapters/ directory.")
        return

    latest_metrics_file = max(metrics_files, key=os.path.getmtime)
    adapter_dir = os.path.dirname(latest_metrics_file)
    print(f"Plotting metrics from: {latest_metrics_file}")

    with open(latest_metrics_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Support log_history from trainer_state.json or custom metrics.json
    log_history = data.get("log_history", data if isinstance(data, list) else [])

    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    lr_steps, learning_rates = [], []

    for entry in log_history:
        step = entry.get("step")
        if "loss" in entry and step is not None:
            train_steps.append(step)
            train_losses.append(entry["loss"])
        if "eval_loss" in entry and step is not None:
            val_steps.append(step)
            val_losses.append(entry["eval_loss"])
        if "learning_rate" in entry and step is not None:
            lr_steps.append(step)
            learning_rates.append(entry["learning_rate"])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Plot Loss
    if train_losses:
        ax1.plot(train_steps, train_losses, label="Train Loss", marker="o", color="#1f77b4")
    if val_losses:
        ax1.plot(val_steps, val_losses, label="Validation Loss", marker="s", linestyle="--", color="#ff7f0e")

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend()
    ax1.grid(True, linestyle=":", alpha=0.6)

    # 2. Plot Learning Rate
    if learning_rates:
        ax2.plot(lr_steps, learning_rates, label="Learning Rate", color="#2ca02c", linestyle="-")
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Learning Rate")
        ax2.set_title("Learning Rate Schedule")
        ax2.legend()
        ax2.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    save_path = os.path.join(adapter_dir, "loss_plot.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot image to: {save_path}")
    plt.show()


if __name__ == "__main__":
    plot_latest_training_loss()
