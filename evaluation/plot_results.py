"""
evaluation/plot_results.py — Visualize Recovery Results

Generates publication-quality plots:
1. ASR before vs after DPO recovery (bar chart)
2. β ablation curve (ASR vs β)
3. Reward curves during attack training
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("dark_background")

from config import RESULTS_DIR, BETA_VALUES


def moving_average(data, window=5):
    if len(data) < window:
        return data
    return np.convolve(data, np.ones(window) / window, mode="valid")


def plot_asr_comparison(results: dict):
    """Bar chart: ASR before vs after DPO recovery."""
    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#141414")

    labels = ["Poisoned Model\n(Before DPO)"]
    values = [results.get("poisoned", {}).get("asr", 0)]
    colors = ["#ff4466"]

    if "recovered" in results:
        labels.append("Recovered Model\n(After DPO)")
        values.append(results["recovered"]["asr"])
        colors.append("#00ff88")

    bars = ax.bar(labels, [v * 100 for v in values], color=colors,
                  width=0.4, edgecolor="none")

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1%}", ha="center", va="bottom",
            color="white", fontsize=13, fontweight="bold"
        )

    if "recovered" in results:
        recovery = results["recovered"].get("recovery_rate", 0)
        ax.text(0.5, 0.92, f"Recovery Rate: {recovery:.1%}",
                transform=ax.transAxes, ha="center", color="#ffaa00",
                fontsize=11, fontweight="bold")

    ax.set_ylabel("Attack Success Rate (%)", fontsize=11)
    ax.set_title("DPO Recovery: Backdoor ASR Before vs After", fontsize=13)
    ax.set_ylim(0, 45)
    ax.grid(axis="y", alpha=0.2)
    ax.axhline(y=values[0] * 100, color="#ff4466", linestyle="--", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "asr_before_after.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_beta_ablation(results: dict):
    """Line plot: ASR vs β value."""
    ablation = results.get("beta_ablation", {})
    if not ablation:
        print("No ablation results found. Run evaluation first.")
        return

    betas = sorted([float(k) for k in ablation.keys()])
    asrs = [ablation[str(b)]["asr"] * 100 for b in betas]
    recoveries = [ablation[str(b)]["recovery_rate"] * 100 for b in betas]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d0d0d")

    for ax in [ax1, ax2]:
        ax.set_facecolor("#141414")

    # ASR vs β
    ax1.plot(betas, asrs, color="#ff4466", marker="o", linewidth=2, markersize=8)
    ax1.axhline(
        y=results.get("poisoned", {}).get("asr", 0.33) * 100,
        color="#ff4466", linestyle="--", alpha=0.4, label="Poisoned baseline"
    )
    ax1.set_xlabel("β (DPO temperature)", fontsize=11)
    ax1.set_ylabel("ASR (%)", fontsize=11)
    ax1.set_title("ASR vs β Value", fontsize=12)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.2)
    ax1.set_xscale("log")

    # Recovery rate vs β
    ax2.plot(betas, recoveries, color="#00ff88", marker="s", linewidth=2, markersize=8)
    ax2.set_xlabel("β (DPO temperature)", fontsize=11)
    ax2.set_ylabel("Recovery Rate (%)", fontsize=11)
    ax2.set_title("Recovery Rate vs β Value", fontsize=12)
    ax2.grid(alpha=0.2)
    ax2.set_xscale("log")

    plt.suptitle("β Sensitivity Analysis for DPO Recovery", fontsize=13, y=1.02)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "beta_ablation.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_attack_rewards():
    """Plot RL attack training reward curves."""
    reward_file = os.path.join(RESULTS_DIR, "attack_rewards_ppo.npy")
    if not os.path.exists(reward_file):
        print("No attack reward data found.")
        return

    rewards = np.load(reward_file)
    ma = moving_average(rewards, window=10)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0d0d0d")
    ax.set_facecolor("#141414")

    episodes = np.arange(1, len(rewards) + 1)
    ax.plot(episodes, rewards, alpha=0.3, color="#00aaff", linewidth=0.8, label="Episode reward")
    ma_ep = np.arange(10, len(rewards) + 1)
    ax.plot(ma_ep, ma, color="#00aaff", linewidth=2.0, label="Moving avg (window=10)")
    ax.axhline(y=0, color="white", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("Reward", fontsize=11)
    ax.set_title("PPO Attack Agent — Training Rewards", fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.2)

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "attack_rewards.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close()


def plot_all():
    results_file = os.path.join(RESULTS_DIR, "full_evaluation.json")
    if not os.path.exists(results_file):
        print(f"No evaluation results at {results_file}")
        print("Run: python evaluation/evaluate_recovery.py first")
        return

    with open(results_file) as f:
        results = json.load(f)

    os.makedirs(RESULTS_DIR, exist_ok=True)
    plot_asr_comparison(results)
    plot_beta_ablation(results)
    plot_attack_rewards()
    print(f"\nAll plots saved to {RESULTS_DIR}/")


if __name__ == "__main__":
    plot_all()
