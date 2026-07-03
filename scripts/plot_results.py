import os
import sys
import json
import shutil
import argparse
import logging
from typing import Dict, Any, Optional

# Add workspace root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scripts.config import config

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('plot_results')

# Ordered variant display names and their colours
VARIANTS = ['Base', 'FT']
VARIANT_COLORS = {'Base': '#1f77b4', 'FT': '#2ca02c'}


def load_json_summary(path: str) -> Optional[Dict[str, Any]]:
    """Loads and returns the 'summary' block from an evaluation JSON file."""
    if not os.path.exists(path):
        logger.warning('File not found: %s', path)
        return None
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            return data.get('summary')
    except Exception as e:
        logger.error('Failed to parse JSON file %s: %s', path, e)
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
        sd for sd in subdirs if os.path.exists(os.path.join(sd, 'metrics.json'))
    ]
    if not valid_subdirs:
        return None

    latest_dir = max(valid_subdirs)
    logger.info('Found latest training metrics in: %s', latest_dir)
    try:
        with open(os.path.join(latest_dir, 'metrics.json'), 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error('Failed to read metrics from %s: %s', latest_dir, e)
        return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Create comparison dashboard plots from training logs and evaluation results'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Directory to save the generated dashboard plot (defaults to report/)',
    )
    args = parser.parse_args()

    model_results_dir = config.results
    report_dir = os.path.join(config.workspace_root, 'report')
    plots_dir = args.output_dir if args.output_dir else report_dir
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Load evaluation summaries (Base vs FT, style system prompt)
    eval_files = {
        'Base': os.path.join(model_results_dir, 'baseline', 'gsm8k.json'),
        'FT': os.path.join(model_results_dir, 'finetuned', 'gsm8k.json'),
    }

    summaries: Dict[str, Any] = {}
    for name, path in eval_files.items():
        summary = load_json_summary(path)
        if summary:
            summaries[name] = summary

    # 2. Load training metrics
    train_metrics = get_latest_metrics()

    if not summaries and not train_metrics:
        logger.error('No metrics or evaluation summaries found. Cannot generate plot.')
        sys.exit(1)

    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

    # Variants that actually have data
    variants = [v for v in VARIANTS if v in summaries]
    bar_colors = [VARIANT_COLORS[v] for v in variants]

    # Create 2x2 figure layout
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))

    # --- SUBPLOT 1: Training Loss Curve ---
    ax1 = axs[0, 0]
    if train_metrics and train_metrics.get('train_losses'):
        train_steps = train_metrics.get('train_steps', [])
        train_losses = train_metrics.get('train_losses', [])
        val_steps = train_metrics.get('val_steps', [])
        val_losses = train_metrics.get('val_losses', [])

        ax1.plot(train_steps, train_losses, label='Train Loss', marker='o', color='#1f77b4', alpha=0.8, linewidth=2)
        if val_losses:
            ax1.plot(val_steps, val_losses, label='Validation Loss', marker='s', color='#ff7f0e', linestyle='--', linewidth=2)
        ax1.set_title('LoRA SFT Training & Validation Loss', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Cross Entropy Loss')
        ax1.legend()
        ax1.grid(True, linestyle=':', alpha=0.6)
    else:
        ax1.text(0.5, 0.5, 'No Training Loss Data Available', ha='center', va='center', fontsize=12)
        ax1.set_title('LoRA SFT Training & Validation Loss', fontsize=12, fontweight='bold')

    # --- SUBPLOT 2: Accuracy Comparison ---
    ax2 = axs[0, 1]
    if variants:
        accuracies = [summaries[v]['accuracy'] * 100 for v in variants]
        bars = ax2.bar(variants, accuracies, color=bar_colors, width=0.4)
        ax2.set_title('GSM8K Accuracy — Base vs FT', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_ylim(0, 100)
        ax2.grid(True, axis='y', linestyle=':', alpha=0.6)
        for bar in bars:
            height = bar.get_height()
            ax2.annotate(
                f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom', fontweight='bold',
            )
    else:
        ax2.text(0.5, 0.5, 'No Evaluation Data Available', ha='center', va='center', fontsize=12)
        ax2.set_title('GSM8K Accuracy — Base vs FT', fontsize=12, fontweight='bold')

    # --- SUBPLOT 3: Mean Reasoning Tokens ---
    ax3 = axs[1, 0]
    if variants:
        x = np.arange(len(variants))
        width = 0.35
        think_tokens = [summaries[v].get('mean_thinking_tokens', 0) for v in variants]
        ans_tokens = [summaries[v].get('mean_answer_tokens', 0) for v in variants]

        rects1 = ax3.bar(x - width / 2, think_tokens, width, label='Thinking Tokens', color='#1f77b4')
        rects2 = ax3.bar(x + width / 2, ans_tokens, width, label='Answer Tokens', color='#aec7e8')

        ax3.set_title('Emitted Tokens per Problem', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(variants)
        ax3.set_ylabel('Average Tokens')
        ax3.legend()
        ax3.grid(True, axis='y', linestyle=':', alpha=0.6)

        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                ax3.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                ax3.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    else:
        ax3.text(0.5, 0.5, 'No Evaluation Data Available', ha='center', va='center', fontsize=12)
        ax3.set_title('Emitted Tokens per Problem', fontsize=12, fontweight='bold')

    # --- SUBPLOT 4: Inference Latency and Speed ---
    ax4 = axs[1, 1]
    if variants:
        x = np.arange(len(variants))
        width = 0.35
        latencies = [summaries[v].get('mean_latency', 0) for v in variants]
        speeds = [summaries[v].get('mean_tokens_per_second', 0) for v in variants]

        color_lat = '#d62728'
        color_spd = '#2ca02c'

        rects_lat = ax4.bar(x - width / 2, latencies, width, label='Latency (s)', color=color_lat, alpha=0.8)
        ax4_right = ax4.twinx()
        rects_spd = ax4_right.bar(x + width / 2, speeds, width, label='Speed (tok/s)', color=color_spd, alpha=0.8)

        ax4.set_title('Latency vs Generation Speed', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(variants)
        ax4.set_ylabel('Inference Latency (seconds)', color=color_lat)
        ax4.tick_params(axis='y', labelcolor=color_lat)
        ax4_right.set_ylabel('Token Generation Speed (tok/s)', color=color_spd)
        ax4_right.tick_params(axis='y', labelcolor=color_spd)
        ax4_right.grid(False)

        for rect in rects_lat:
            h = rect.get_height()
            if h > 0:
                ax4.annotate(f'{h:.2f}s', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=9, color=color_lat, fontweight='bold')
        for rect in rects_spd:
            h = rect.get_height()
            if h > 0:
                ax4_right.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=9, color=color_spd, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No Evaluation Data Available', ha='center', va='center', fontsize=12)
        ax4.set_title('Latency vs Generation Speed', fontsize=12, fontweight='bold')

    fig.tight_layout()
    plot_file = os.path.join(plots_dir, 'dashboard.png')
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info('Successfully generated dashboard plot at: %s', plot_file)

    # --- SAVE SEPARATE PLOTS ---
    logger.info('Generating and saving individual plots...')

    # Plot 1: Training Loss Curve
    if train_metrics and train_metrics.get('train_losses'):
        plt.figure(figsize=(8, 6))
        train_steps = train_metrics.get('train_steps', [])
        train_losses = train_metrics.get('train_losses', [])
        val_steps = train_metrics.get('val_steps', [])
        val_losses = train_metrics.get('val_losses', [])

        plt.plot(train_steps, train_losses, label='Train Loss', marker='o', color='#1f77b4', alpha=0.8, linewidth=2)
        if val_losses:
            plt.plot(val_steps, val_losses, label='Validation Loss', marker='s', color='#ff7f0e', linestyle='--', linewidth=2)
        plt.title('LoRA SFT Training & Validation Loss', fontsize=12, fontweight='bold')
        plt.xlabel('Iteration')
        plt.ylabel('Cross Entropy Loss')
        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)

        loss_curve_file = os.path.join(plots_dir, 'loss_curve.png')
        plt.savefig(loss_curve_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info('Saved: %s', loss_curve_file)

    # Plot 2: Accuracy Comparison
    if variants:
        plt.figure(figsize=(7, 6))
        accuracies = [summaries[v]['accuracy'] * 100 for v in variants]
        bars = plt.bar(variants, accuracies, color=bar_colors, width=0.4)
        plt.title('GSM8K Accuracy — Base vs FT', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy (%)')
        plt.ylim(0, 100)
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)
        for bar in bars:
            height = bar.get_height()
            plt.annotate(
                f'{height:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center', va='bottom', fontweight='bold',
            )
        accuracy_file = os.path.join(plots_dir, 'accuracy.png')
        plt.savefig(accuracy_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info('Saved: %s', accuracy_file)

    # Plot 3: Mean Reasoning Tokens
    if variants:
        plt.figure(figsize=(7, 6))
        x = np.arange(len(variants))
        width = 0.35
        think_tokens = [summaries[v].get('mean_thinking_tokens', 0) for v in variants]
        ans_tokens = [summaries[v].get('mean_answer_tokens', 0) for v in variants]

        rects1 = plt.bar(x - width / 2, think_tokens, width, label='Thinking Tokens', color='#1f77b4')
        rects2 = plt.bar(x + width / 2, ans_tokens, width, label='Answer Tokens', color='#aec7e8')

        plt.title('Emitted Tokens per Problem', fontsize=12, fontweight='bold')
        plt.xticks(x, variants)
        plt.ylabel('Average Tokens')
        plt.legend()
        plt.grid(True, axis='y', linestyle=':', alpha=0.6)

        for rect in rects1:
            h = rect.get_height()
            if h > 0:
                plt.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=9)
        for rect in rects2:
            h = rect.get_height()
            if h > 0:
                plt.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=9)

        tokens_file = os.path.join(plots_dir, 'tokens.png')
        plt.savefig(tokens_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info('Saved: %s', tokens_file)

    # Plot 4: Latency vs Speed
    if variants:
        fig_ind, ax1_ind = plt.subplots(figsize=(7, 6))
        x = np.arange(len(variants))
        width = 0.35
        latencies = [summaries[v].get('mean_latency', 0) for v in variants]
        speeds = [summaries[v].get('mean_tokens_per_second', 0) for v in variants]

        color_lat = '#d62728'
        color_spd = '#2ca02c'

        rects_lat = ax1_ind.bar(x - width / 2, latencies, width, label='Latency (s)', color=color_lat, alpha=0.8)
        ax2_ind = ax1_ind.twinx()
        rects_spd = ax2_ind.bar(x + width / 2, speeds, width, label='Speed (tok/s)', color=color_spd, alpha=0.8)

        ax1_ind.set_title('Latency vs Generation Speed', fontsize=12, fontweight='bold')
        ax1_ind.set_xticks(x)
        ax1_ind.set_xticklabels(variants)
        ax1_ind.set_ylabel('Inference Latency (seconds)', color=color_lat)
        ax1_ind.tick_params(axis='y', labelcolor=color_lat)
        ax2_ind.set_ylabel('Token Generation Speed (tok/s)', color=color_spd)
        ax2_ind.tick_params(axis='y', labelcolor=color_spd)
        ax2_ind.grid(False)

        for rect in rects_lat:
            h = rect.get_height()
            if h > 0:
                ax1_ind.annotate(f'{h:.2f}s', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=9, color=color_lat, fontweight='bold')
        for rect in rects_spd:
            h = rect.get_height()
            if h > 0:
                ax2_ind.annotate(f'{int(h)}', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2), textcoords='offset points', ha='center', va='bottom', fontsize=9, color=color_spd, fontweight='bold')

        latency_spd_file = os.path.join(plots_dir, 'latency_speed.png')
        plt.savefig(latency_spd_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info('Saved: %s', latency_spd_file)

    # Plot 5: Deltas (Base vs FT)
    if 'Base' in summaries and 'FT' in summaries:
        fig_delta, (ax1_d, ax2_d) = plt.subplots(1, 2, figsize=(12, 6))

        acc_delta = (summaries['FT']['accuracy'] - summaries['Base']['accuracy']) * 100
        think_delta = summaries['FT'].get('mean_thinking_tokens', 0) - summaries['Base'].get('mean_thinking_tokens', 0)
        total_delta = summaries['FT'].get('mean_total_tokens', 0) - summaries['Base'].get('mean_total_tokens', 0)

        # Left: Accuracy delta
        color_acc = '#d62728' if acc_delta < 0 else '#2ca02c'
        ax1_d.bar(['FT vs Base'], [acc_delta], color=color_acc, width=0.3)
        ax1_d.axhline(0, color='black', linestyle='-', linewidth=1)
        ax1_d.set_title('Accuracy Change (FT vs Base)', fontsize=12, fontweight='bold')
        ax1_d.set_ylabel('Accuracy Delta (percentage points)')
        pad = max(abs(acc_delta) * 0.3, 2)
        ax1_d.set_ylim(min(acc_delta - pad, -3), max(acc_delta + pad, 3))
        ax1_d.grid(True, linestyle=':', alpha=0.6)
        va_pos = 'bottom' if acc_delta >= 0 else 'top'
        offset = (0, 4) if acc_delta >= 0 else (0, -14)
        ax1_d.annotate(f'{acc_delta:+.1f}pp', xy=(0, acc_delta), xytext=offset, textcoords='offset points', ha='center', va=va_pos, fontweight='bold', fontsize=13)

        # Right: Token deltas
        labels_tok = ['Thinking Tokens', 'Total Tokens']
        deltas_tok = [think_delta, total_delta]
        colors_tok = ['#d62728' if d > 0 else '#2ca02c' for d in deltas_tok]
        bars_tok = ax2_d.bar(labels_tok, deltas_tok, color=colors_tok, width=0.4)
        ax2_d.axhline(0, color='black', linestyle='-', linewidth=1)
        ax2_d.set_title('Token Change (FT vs Base)', fontsize=12, fontweight='bold')
        ax2_d.set_ylabel('Token Delta (negative = fewer tokens)')
        ax2_d.grid(True, linestyle=':', alpha=0.6)
        for bar in bars_tok:
            h = bar.get_height()
            va = 'bottom' if h >= 0 else 'top'
            off = (0, 3) if h >= 0 else (0, -12)
            ax2_d.annotate(f'{int(h):+d}', xy=(bar.get_x() + bar.get_width() / 2, h), xytext=off, textcoords='offset points', ha='center', va=va, fontsize=11, fontweight='bold')

        plt.tight_layout()
        delta_file = os.path.join(plots_dir, 'deltas.png')
        plt.savefig(delta_file, dpi=150, bbox_inches='tight')
        plt.close()
        logger.info('Saved: %s', delta_file)

    # --- COPY RAW JSON FILES ---
    logger.info('Copying raw evaluation JSON files to output directory...')
    json_mapping = {
        'Base': (os.path.join(config.results, 'baseline', 'gsm8k.json'), 'gsm8k_baseline.json'),
        'FT': (os.path.join(config.results, 'finetuned', 'gsm8k.json'), 'gsm8k_finetuned.json'),
    }
    for name, (src_path, dest_name) in json_mapping.items():
        if os.path.exists(src_path):
            dest_path = os.path.join(plots_dir, dest_name)
            try:
                shutil.copy2(src_path, dest_path)
                logger.info('Copied raw JSON %s to %s', name, dest_path)
            except Exception as copy_err:
                logger.error('Failed to copy %s: %s', name, copy_err)
        else:
            logger.warning('Raw JSON file not found: %s', src_path)

    # Copy latest metrics.json
    base_adapter_dir = config.adapters
    if os.path.exists(base_adapter_dir):
        subdirs = [
            os.path.join(base_adapter_dir, d)
            for d in os.listdir(base_adapter_dir)
            if os.path.isdir(os.path.join(base_adapter_dir, d))
        ]
        valid_subdirs = [sd for sd in subdirs if os.path.exists(os.path.join(sd, 'metrics.json'))]
        if valid_subdirs:
            latest_dir = max(valid_subdirs)
            src_metrics = os.path.join(latest_dir, 'metrics.json')
            dest_metrics = os.path.join(plots_dir, 'metrics.json')
            try:
                shutil.copy2(src_metrics, dest_metrics)
                logger.info('Copied raw training metrics from %s to %s', latest_dir, dest_metrics)
            except Exception as copy_err:
                logger.error('Failed to copy training metrics: %s', copy_err)


if __name__ == '__main__':
    main()
