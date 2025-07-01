import typing as tp

import matplotlib.pyplot as plt


def plot_recall_comparison(comparison_results: tp.Dict[str, tp.Any]) -> None:
    """Plot recall@k comparison for different algorithms."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for algo_name, results in comparison_results.items():
        recalls = results["recall"]
        k_values = sorted([int(k) for k in recalls.keys()])
        recall_values = [recalls[str(k)] for k in k_values]
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
    fig, ax = plt.subplots(figsize=(12, 6))

    model_names = list(timings.keys())
    baseline_model = model_names[0] if model_names else None
    baseline_timings = timings.get(baseline_model, {}) if baseline_model else {}

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

        largest_corpus = max(slice_sizes)
        largest_time = slice_timings[str(largest_corpus)]

        if model_name != baseline_model and baseline_timings:
            if str(largest_corpus) in baseline_timings:
                baseline_time = baseline_timings[str(largest_corpus)]
                degradation = ((largest_time - baseline_time) / baseline_time) * 100
                speed_ratio = largest_time / baseline_time

                ax.text(
                    largest_corpus * 1.02,
                    largest_time,
                    f"{speed_ratio:.1f}x slower",
                    fontsize=9,
                    color="red" if speed_ratio > 5 else "orange",
                    fontweight="bold",
                    va="center",
                )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Time to embed corpus (s)")
    ax.set_title("Embedder Timings")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
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

    fig, ax = plt.subplots(figsize=(12, 6))

    largest_corpus = max(
        [max(size_timings.keys()) for size_timings in index_data.values()]
    )
    baseline_time = float("inf")
    baseline_index = None

    for index_name, size_timings in index_data.items():
        if largest_corpus in size_timings:
            time = size_timings[largest_corpus]
            if time < baseline_time:
                baseline_time = time
                baseline_index = index_name

    for index_name, size_timings in index_data.items():
        corpus_sizes = sorted(size_timings.keys())
        timing_values = [size_timings[size] for size in corpus_sizes]

        ax.plot(
            corpus_sizes,
            timing_values,
            "o-",
            label=f"{index_name}",
            linewidth=2,
            markersize=6,
        )

        if index_name != baseline_index and baseline_index:
            largest_corpus = max(corpus_sizes)
            current_time = size_timings[largest_corpus]
            baseline_data = index_data[baseline_index]

            if largest_corpus in baseline_data:
                baseline_time = baseline_data[largest_corpus]
                speed_ratio = current_time / baseline_time

                ax.text(
                    largest_corpus * 1.02,
                    current_time,
                    f"{speed_ratio:.1f}x slower",
                    fontsize=9,
                    color="red" if speed_ratio > 2 else "orange",
                    fontweight="bold",
                    verticalalignment="center",
                )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Build Time (s)")
    ax.set_title("Build Time Comparison")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("build_time_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_search_time_comparison(timings: tp.Dict[str, float]) -> None:
    """Plot search time comparison in queries per second."""
    index_data = {}
    for key, timing in timings.items():
        parts = key.split("_")
        index_name = parts[0].split(":")[1]
        corpus_size = int(parts[1].split(":")[1])
        k = int(parts[2].split(":")[1])
        n_queries = int(parts[3].split(":")[1])

        if len(parts) > 4 and "dimension:" in parts[4]:
            dimension = int(parts[4].split(":")[1])
        else:
            dimension = 384

        queries_per_second = n_queries / timing if timing > 0 else 0

        if index_name not in index_data:
            index_data[index_name] = {}
        if corpus_size not in index_data[index_name]:
            index_data[index_name][corpus_size] = {}
        index_data[index_name][corpus_size][k] = queries_per_second

    fig, ax = plt.subplots(figsize=(12, 6))

    baseline_qps = 0
    baseline_index = None

    for index_name, size_data in index_data.items():
        all_qps = []
        for corpus_size, k_data in size_data.items():
            all_qps.extend(k_data.values())
        avg_qps = sum(all_qps) / len(all_qps) if all_qps else 0

        if avg_qps > baseline_qps:
            baseline_qps = avg_qps
            baseline_index = index_name

    for index_name, size_data in index_data.items():
        corpus_sizes = sorted(size_data.keys())
        k100_qps_values = []

        for corpus_size in corpus_sizes:
            k_timings = size_data[corpus_size]
            if k_timings and 100 in k_timings:
                k100_qps = k_timings[100]
                k100_qps_values.append(k100_qps)
            else:
                k100_qps_values.append(0)

        ax.plot(
            corpus_sizes,
            k100_qps_values,
            "o-",
            label=f"{index_name.upper()}",
            linewidth=2,
            markersize=6,
        )

        if corpus_sizes and 50000 in corpus_sizes:
            corpus_50k_idx = corpus_sizes.index(50000)
            current_qps = k100_qps_values[corpus_50k_idx]

            if current_qps > 0:
                latency_ms = (1 / current_qps) * 1000

                ax.text(
                    50000 * 1.02,
                    current_qps,
                    f"{latency_ms:.1f}ms",
                    fontsize=9,
                    color="blue",
                    fontweight="bold",
                    verticalalignment="center",
                )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Queries per Second")
    ax.set_title("Search Time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig("search_time_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def plot_quantization_comparison(timings: tp.Dict[str, float]) -> None:
    """Plot quantization comparison."""
    index_data = {}
    for key, timing in timings.items():
        parts = key.split("_")
        index_name = parts[0].split(":")[1]
        quantization = parts[1].split(":")[1]
        corpus_size = int(parts[2].split(":")[1])
        k = int(parts[3].split(":")[1])
        n_queries = int(parts[4].split(":")[1])

        queries_per_second = n_queries / timing if timing > 0 else 0

        if quantization not in index_data:
            index_data[quantization] = {}
        if corpus_size not in index_data[quantization]:
            index_data[quantization][corpus_size] = {}
        index_data[quantization][corpus_size][k] = queries_per_second

    fig, ax = plt.subplots(figsize=(10, 6))

    quantization_methods = list(index_data.keys())
    corpus_sizes = set()
    for quantization in index_data:
        corpus_sizes.update(index_data[quantization].keys())
    corpus_sizes = sorted(corpus_sizes)

    baseline_qps = {}
    if "none" in index_data:
        for corpus_size in corpus_sizes:
            if corpus_size in index_data["none"]:
                qps_for_size = list(index_data["none"][corpus_size].values())
                baseline_qps[corpus_size] = (
                    sum(qps_for_size) / len(qps_for_size) if qps_for_size else 0
                )

    for quantization in quantization_methods:
        avg_qps_by_size = []
        for corpus_size in corpus_sizes:
            if corpus_size in index_data[quantization]:
                qps_for_size = list(index_data[quantization][corpus_size].values())
                avg_qps_by_size.append(
                    sum(qps_for_size) / len(qps_for_size) if qps_for_size else 0
                )
            else:
                avg_qps_by_size.append(0)

        ax.plot(
            corpus_sizes,
            avg_qps_by_size,
            marker="s",
            linewidth=2,
            markersize=6,
            label=quantization,
        )

        if quantization != "none" and baseline_qps:
            for i, corpus_size in enumerate(corpus_sizes):
                if corpus_size in baseline_qps and baseline_qps[corpus_size] > 0:
                    current_qps = avg_qps_by_size[i]
                    if current_qps > 0:
                        speed_ratio = current_qps / baseline_qps[corpus_size]

                        if speed_ratio >= 1:
                            annotation = f"{speed_ratio:.1f}x"
                            color = "green"
                        else:
                            annotation = f"{1/speed_ratio:.1f}x"
                            color = "red"

                        ax.text(
                            corpus_size * 1.02,
                            current_qps,
                            annotation,
                            fontsize=8,
                            color=color,
                            fontweight="bold",
                            verticalalignment="center",
                        )

    ax.set_xlabel("Corpus Size")
    ax.set_ylabel("Queries per Second")
    ax.set_title("Performance vs Corpus Size by Quantization")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("quantization_performance_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()
