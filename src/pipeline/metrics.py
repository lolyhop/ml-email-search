import typing as tp


class MetricsCalculator:
    """Utility class for calculating metrics."""

    @staticmethod
    def calculate_recall_at_k(
        ground_truth_ids: tp.List[tp.Union[int, tp.List[int]]],
        predicted_ids: tp.List[tp.List[int]],
        k_values: tp.List[int],
    ) -> tp.Dict[int, float]:
        """Calculate recall@k for different k values."""
        recalls = {}

        for k in k_values:
            total_recall = 0.0

            for gt_ids, pred_ids in zip(ground_truth_ids, predicted_ids):
                if hasattr(gt_ids, "tolist"):
                    gt_ids = gt_ids.tolist()
                if hasattr(pred_ids, "tolist"):
                    pred_ids = pred_ids.tolist()

                gt_top_k = set(gt_ids[:k])
                pred_top_k = set(pred_ids[:k])

                recall = len(gt_top_k.intersection(pred_top_k)) / k
                total_recall += recall

            recalls[k] = total_recall / len(ground_truth_ids)

        return recalls


if __name__ == "__main__":
    ground_truth = [
        [1, 3, 5],  # Query 1 relevant docs
        [2, 4],  # Query 2 relevant docs
        [1, 2, 6, 8],  # Query 3 relevant docs
    ]

    predictions = [
        [1, 2, 3, 4, 5, 6],  # Query 1 predictions
        [4, 1, 2, 3, 5, 6],  # Query 2 predictions
        [1, 3, 2, 7, 6, 8],  # Query 3 predictions
    ]

    k_values = [1, 3, 5]

    recalls = MetricsCalculator.calculate_recall_at_k(
        ground_truth, predictions, k_values
    )

    print("Mean Recall:")
    print(list(recalls.values()))
