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


def plot_embedder_timings(timings: tp.Dict[str, tp.Dict[str, float]]) -> None:
    """Plot embedder timings."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for head_size, slice_sizes in timings.items():
        slice_sizes = sorted(slice_sizes.keys())
        timings = [slice_sizes[k] for k in slice_sizes]
        ax.plot(slice_sizes, timings, "o-", label=head_size, linewidth=2, markersize=6)
    ax.set_xlabel("Slice Size")
    ax.set_ylabel("Time (s)")
    ax.set_title("microsoft/mpnet-base-109M Timings")
    ax.grid(True, alpha=0.3)
    plt.savefig("embedder_timings.png", dpi=300, bbox_inches="tight")
