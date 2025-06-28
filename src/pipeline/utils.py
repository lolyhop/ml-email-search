import typing as tp

import numpy as np
import matplotlib.pyplot as plt


def plot_recall_comparison(comparison_results: tp.Dict[str, tp.Any]) -> None:
    """Plot recall@k comparison for different algorithms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo_name, results in comparison_results.items():
        recalls = results["recall"]
        k_values = sorted(recalls.keys())
        recall_values = [recalls[k] for k in k_values]
        ax.plot(
            k_values, recall_values, "o-", label=algo_name, linewidth=2, markersize=6
        )
    ax.set_xlabel("k")
    ax.set_ylabel("Recall@k")
    ax.set_title("Recall@k Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim(0, 1.05)
    plt.savefig("recall_comparison.png", dpi=300, bbox_inches="tight")


def plot_time_comparison(comparison_results: tp.Dict[str, tp.Any], ax) -> None:
    """Plot time per query comparison for different algorithms."""
    algo_names = list(comparison_results.keys())
    times = [
        results["time_per_query"] * 1000 for results in comparison_results.values()
    ]

    bars = ax.bar(algo_names, times, alpha=0.7)
    ax.set_ylabel("Time per Query (ms)")
    ax.set_title("Query Time Comparison")
    ax.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.01,
            f"{time_val:.1f}ms",
            ha="center",
            va="bottom",
        )


def plot_speed_accuracy_tradeoff(comparison_results: tp.Dict[str, tp.Any], ax) -> None:
    """Plot speed vs accuracy tradeoff for different algorithms."""
    algo_names = list(comparison_results.keys())
    times = [
        results["time_per_query"] * 1000 for results in comparison_results.values()
    ]
    recall_100_values = [
        results["recall"].get(100, 0) for results in comparison_results.values()
    ]

    colors = plt.cm.viridis(np.linspace(0, 1, len(algo_names)))
    scatter = ax.scatter(times, recall_100_values, s=100, c=colors, alpha=0.7)

    # Add algorithm labels
    for i, algo in enumerate(algo_names):
        ax.annotate(
            algo.upper(),
            (times[i], recall_100_values[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    ax.set_xlabel("Time per Query (ms)")
    ax.set_ylabel("Recall@100")
    ax.set_title("Speed vs Accuracy Tradeoff")
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
