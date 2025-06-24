import os

os.environ["OMP_NUM_THREADS"] = "1"

import logging
import typing as tp
import time
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from src.retrievers.base import EmbedderConfig
from src.index.base import BaseIndex
from src.index.factory import IndexFactory
from src.index.config import IndexConfig
from src.index.entity import FaissEntity, FaissCorpus
from src.index.indexes import BruteForceIndex
from src.pipeline.config import QualityPipelineConfig
from src.pipeline.metrics import MetricsCalculator
from src.pipeline.pipeline import Pipeline

logger = logging.getLogger(__name__)


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
        embeddings = self.embedder.embed_query(self.config.queries)
        sota_results = sota_index.search(
            embeddings, min(max(self.config.recall_ranks), self.corpus.n_documents)
        )

        print(sota_results)
        # 2. Get results for retrieval using other indexes
        for index in indexes_to_compare:
            index.build_from_corpus(self.corpus, self.embedder)
            embeddings = self.embedder.embed_query(self.config.queries)
            results = index.search(embeddings, max(self.config.recall_ranks))

            # 3. Calculate recall@k for each index
            print(sota_results[1])
            print(results[1])
            recall_at_k = MetricsCalculator.calculate_recall_at_k(
                sota_results[1], results[1], self.config.recall_ranks
            )

            print(f"Recall@k for {index.config.index_name}: {recall_at_k}")


if __name__ == "__main__":
    config = QualityPipelineConfig(
        documents=[
            (1, "This is a test document"),
            (2, "This is another test document"),
            (3, "This is a third test document"),
        ],
        queries=["What is the capital of France?"],
        indexes_to_compare=[
            IndexConfig(index_name="hnsw", dimension=128, metric="l2"),
            IndexConfig(index_name="ivf", dimension=128, metric="l2"),
        ],
        recall_ranks=[1, 5, 10, 20, 30, 50, 100],
        embedder_config=EmbedderConfig(
            model_name="dummy_model", head="universal", head_size=128
        ),
    )
    pipeline = QualityPipeline(config)
    pipeline.run_pipeline()
