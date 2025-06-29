import typing as tp

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
    for model_name, slice_timings in timings.items():
        slice_sizes = sorted([int(k) for k in slice_timings.keys()])
        timing_values = [slice_timings[str(k)] for k in slice_sizes]
        ax.plot(
            slice_sizes,
            timing_values,
            "o-",
            label=model_name,
            linewidth=2,
            markersize=6,
        )
    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Time to embed corpus (s)")
    ax.set_title("Embedder Timings")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig("embedder_timings.png", dpi=300, bbox_inches="tight")


def plot_build_time_comparison(timings: tp.Dict[str, float]) -> None:
    """Plot build time comparison."""
    index_data = {}
    for key, timing in timings.items():
        parts = key.rsplit("_", 1)
        index_name = parts[0]
        corpus_size = int(parts[1])

        if index_name not in index_data:
            index_data[index_name] = {}
        index_data[index_name][corpus_size] = timing

    fig, ax = plt.subplots(figsize=(10, 6))
    for index_name, size_timings in index_data.items():
        corpus_sizes = sorted(size_timings.keys())
        timing_values = [size_timings[size] for size in corpus_sizes]
        ax.plot(
            corpus_sizes,
            timing_values,
            "o-",
            label=index_name,
            linewidth=2,
            markersize=6,
        )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Time (s)")
    ax.set_title("Build Time Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.savefig("build_time_comparison.png", dpi=300, bbox_inches="tight")


def plot_search_time_comparison(timings: tp.Dict[str, float]) -> None:
    """Plot search time comparison in queries per second."""
    index_data = {}
    for key, timing in timings.items():
        parts = key.split("_")
        index_name = parts[0].split(":")[1]
        corpus_size = int(parts[1].split(":")[1])
        k = int(parts[2].split(":")[1])
        n_queries = int(parts[3].split(":")[1])

        queries_per_second = n_queries / timing if timing > 0 else 0

        if index_name not in index_data:
            index_data[index_name] = {}
        if corpus_size not in index_data[index_name]:
            index_data[index_name][corpus_size] = {}
        index_data[index_name][corpus_size][k] = queries_per_second

    for index_name, size_timings in index_data.items():
        fig, ax = plt.subplots(figsize=(10, 6))
        for corpus_size, k_timings in size_timings.items():
            k_values = sorted(k_timings.keys())
            qps_values = [k_timings[k] for k in k_values]
            ax.plot(
                k_values,
                qps_values,
                "o-",
                label=f"Corpus Size: {corpus_size}",
                linewidth=2,
                markersize=6,
            )
        ax.set_xlabel("k")
        ax.set_ylabel("Queries per Second")
        ax.set_title(f"Search Performance - {index_name}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.savefig(
            f"search_time_comparison_{index_name}.png", dpi=300, bbox_inches="tight"
        )
        plt.show()
