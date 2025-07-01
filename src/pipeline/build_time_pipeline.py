import os

os.environ["OMP_NUM_THREADS"] = "1"

import time
import json
import logging
import typing as tp

from tqdm import tqdm

from src.retrievers.base import EmbedderConfig
from src.retrievers.dummy import DummyEmbedder
from src.index.base import BaseIndex
from src.index.factory import IndexFactory
from src.index.config import IndexConfig
from src.pipeline.config import BuildTimePipelineConfig
from src.pipeline.pipeline import Pipeline
from src.data_loader.data_loader import EmailsDataLoader

logger = logging.getLogger(__name__)
SAVE_REPORTS_PATH = "reports/embedder-time"


class BuildTimePipeline(Pipeline):

    def __init__(self, config: BuildTimePipelineConfig) -> None:
        super().__init__(config)
        self.config: BuildTimePipelineConfig = config

    def build_index(self, index_config: IndexConfig) -> BaseIndex:
        """Build the configured index."""
        index: BaseIndex = IndexFactory.create(index_config)
        return index

    def run_pipeline(self) -> tp.Any:
        """Run pipeline and return results."""
        build_timings = {}

        self.corpus = self._create_corpus()

        for index_config in tqdm(self.config.indexes_to_compare):
            embedder = DummyEmbedder(self.config.embedder_config)
            embedder.config.head_size = index_config.dimension
            embedder.initialize()
            for corpus_size in self.config.slice_sizes:
                start_time = time.time()
                index = self.build_index(index_config)
                index.build_from_corpus(self.corpus[:corpus_size], embedder)
                build_time = time.time() - start_time
                build_timings[
                    f"{index_config.index_name}_{corpus_size}_{index_config.dimension}"
                ] = build_time

        with open("build_time_results.json", "w") as file:
            file.write(json.dumps(build_timings, ensure_ascii=False, indent=4))

        return build_timings


if __name__ == "__main__":
    loader = EmailsDataLoader.load(
        "/Users/egor/Documents/code/ml-email-search/src/data_loader/emails.csv"
    )
    loader.preprocess(raw_email_col="message")
    documents: tp.List[tp.Tuple[int, str]] = loader.get_faiss_dataset()

    config = BuildTimePipelineConfig(
        documents=documents,
        slice_sizes=[1000, 5000, 10000, 30000, 50000],
        indexes_to_compare=[
            IndexConfig(index_name="hnsw", dimension=64, metric="l2"),
            IndexConfig(index_name="hnsw", dimension=128, metric="l2"),
            IndexConfig(index_name="hnsw", dimension=256, metric="l2"),
            IndexConfig(index_name="hnsw", dimension=512, metric="l2"),
            IndexConfig(index_name="hnsw", dimension=768, metric="l2"),
        ],
        embedder_config=EmbedderConfig(
            model_name="dummy_model",
            model_id="dummy",
            head="universal",
            head_size=384,
        ),
    )
    pipeline = BuildTimePipeline(config)
    pipeline.run_pipeline()
