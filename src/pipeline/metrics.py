import typing as tp


class MetricsCalculator:
    """Utility class for calculating metrics."""

    @staticmethod
    def calculate_recall_at_k(
        ground_truth_ids: tp.List[int],
        predicted_ids: tp.List[int],
        k_values: tp.List[int],
    ) -> tp.Dict[int, float]:
        """Calculate recall@k for different k values."""
        recalls = {}

        for k in k_values:
            total_recall = 0.0

            for gt_ids, pred_ids in zip(ground_truth_ids, predicted_ids):
                gt_ids = gt_ids.tolist()
                pred_ids = pred_ids.tolist()

                gt_set = set(gt_ids)
                pred_set = set(pred_ids[:k])
                recall = len(gt_set.intersection(pred_set)) / len(gt_set)
                total_recall += recall

            recalls[k] = total_recall / len(ground_truth_ids)

        return recalls
