import typing as tp

import numpy as np
import matplotlib.pyplot as plt


def visualize_comparison(
    comparison_results: tp.Dict[str, tp.Annotated], save_path: str = None
) -> None:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5), dpi=200)

    # Recall@k
    for algo_name, results in comparison_results.items():
        recalls = results["recall"]
        k_values = sorted(recalls.keys())
        recall_values = [recalls[k] for k in k_values]
        ax1.plot(
            k_values, recall_values, "o-", label=algo_name, linewidth=2, markersize=6
        )

    ax1.set_xlabel("k")
    ax1.set_ylabel("Recall@k")
    ax1.set_title("Recall@k Comparison")
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim(0, 1.05)

    # Time per query
    algo_names = list(comparison_results.keys())
    times = [
        results["time_per_query"] * 1000 for results in comparison_results.values()
    ]

    bars = ax2.bar(algo_names, times, alpha=0.7)
    ax2.set_ylabel("Time per Query")
    ax2.set_title("Query Time Comparison")
    ax2.tick_params(axis="x", rotation=45)

    # Add value labels on bars
    for bar, time_val in zip(bars, times):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(times) * 0.01,
            f"{time_val:.1f}ms",
            ha="center",
            va="bottom",
        )

    # Recall@100 vs Time
    recall_10_values = [
        results["recall"].get(100, 0) for results in comparison_results.values()
    ]
    time_values = [i * 1000 for i in times]

    colors = plt.cm.viridis(np.linspace(0, 1, len(algo_names)))
    scatter = ax3.scatter(times, recall_10_values, s=100, c=colors, alpha=0.7)

    # Add algorithm labels
    for i, algo in enumerate(algo_names):
        ax3.annotate(
            algo.upper(),
            (times[i], recall_10_values[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
            fontweight="bold",
        )

    ax3.set_xlabel("Time per Query (ms)")
    ax3.set_ylabel("Recall@100")
    ax3.set_title("Speed vs Accuracy Tradeoff")
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
