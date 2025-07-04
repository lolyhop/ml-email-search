import os

os.environ["OMP_NUM_THREADS"] = "1"

import time
import logging
import typing as tp
import json

from tqdm import tqdm

from src.retrievers.base import EmbedderConfig
from src.index.base import BaseIndex
from src.index.factory import IndexFactory
from src.index.config import IndexConfig
from src.index.indexes import BruteForceIndex
from src.pipeline.config import QualityPipelineConfig
from src.pipeline.metrics import MetricsCalculator
from src.pipeline.pipeline import Pipeline
from src.data_loader.data_loader import EmailsDataLoader

logger = logging.getLogger(__name__)
SAVE_REPORTS_PATH = "reports/quality-time"


class QualityPipeline(Pipeline):
    """Pipeline for running end-to-end quality experiment."""

    def __init__(self, config: QualityPipelineConfig) -> None:
        super().__init__(config)
        self.config: QualityPipelineConfig = config

    def build_index(self, index_config: IndexConfig) -> BaseIndex:
        """Build the configured index."""
        logger.info(f"Building {index_config.index_name} index...")

        start_time = time.time()

        index: BaseIndex = IndexFactory.create(index_config)
        index.build_from_corpus(self.corpus, self.embedder)

        build_time = time.time() - start_time
        logger.info(f"Index built in {build_time:.3f}s")
        return index

    def run_pipeline(self) -> tp.Any:
        """Run pipeline and return results."""
        self.setup()
        sota_index_config = IndexConfig(
            index_name="brute_force",
            dimension=self.config.embedder_config.head_size,
            metric="l2",
        )
        sota_index: BruteForceIndex = IndexFactory.create(sota_index_config)

        indexes_to_compare: tp.List[BaseIndex] = [
            IndexFactory.create(index_config)
            for index_config in self.config.indexes_to_compare
        ]

        # 1. Get SOTA results for retrieval using brute force index
        sota_index.build_from_corpus(self.corpus, self.embedder)
        query_embeddings = self.embedder.embed_query(self.config.queries)

        search_start = time.time()
        sota_results = sota_index.search(
            query_embeddings,
            min(max(self.config.recall_ranks), self.corpus.n_documents),
        )
        search_time_sota = time.time() - search_start
        comparison_results = {
            "brute_force": {
                "recall": {k: 1.0 for k in self.config.recall_ranks},
                "search_time": search_time_sota,
                "time_per_query": search_time_sota / len(self.config.queries),
            }
        }

        # 2. Get results for retrieval using other indexes
        for index in tqdm(indexes_to_compare, desc="Comparing indexes"):
            index.build_from_corpus(self.corpus, self.embedder)
            search_start = time.time()
            results = index.search(
                query_embeddings,
                min(max(self.config.recall_ranks), self.corpus.n_documents),
            )
            search_time = time.time() - search_start
            # 3. Calculate recall@k for each index
            recall_at_k = MetricsCalculator.calculate_recall_at_k(
                sota_results[1], results[1], self.config.recall_ranks
            )

            comparison_results[
                f"{index.config.index_name}-{index.config.quantization}"
            ] = {
                "recall": recall_at_k,
                "search_time": search_time,
                "time_per_query": search_time / len(self.config.queries),
            }

        with open("quality_results.json", "w") as file:
            file.write(json.dumps(comparison_results, ensure_ascii=False, indent=4))

        return comparison_results


if __name__ == "__main__":
    loader = EmailsDataLoader.load(
        "/Users/egor/Documents/code/ml-email-search/src/data_loader/emails.csv"
    )
    loader.preprocess(raw_email_col="message")
    documents: tp.List[tp.Tuple[int, str]] = loader.get_faiss_dataset()
    config = QualityPipelineConfig(
        documents=documents,
        queries=["What is the capital of France?"],
        indexes_to_compare=[
            IndexConfig(
                index_name="ivf", dimension=384, metric="l2", quantization="none"
            ),
            IndexConfig(
                index_name="ivf",
                dimension=384,
                metric="l2",
                quantization="pq",
                pq_m=8,
                pq_bits=8,
            ),
            IndexConfig(
                index_name="ivf",
                dimension=384,
                metric="l2",
                quantization="sq",
                sq_bits=8,
            ),
        ],
        recall_ranks=[1, 5, 10, 30, 50, 100, 200, 500],
        embedder_config=EmbedderConfig(
            model_name="dummy_model",
            model_id="dummy",
            head="universal",
            head_size=384,
        ),
    )
    pipeline = QualityPipeline(config)
    pipeline.run_pipeline()
