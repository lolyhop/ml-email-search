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
from src.pipeline.config import SearchTimePipelineConfig
from src.pipeline.pipeline import Pipeline
from src.data_loader.data_loader import EmailsDataLoader

logger = logging.getLogger(__name__)
SAVE_REPORTS_PATH = "reports/embedder-time"


class SearchTimePipeline(Pipeline):

    def __init__(self, config: SearchTimePipelineConfig) -> None:
        super().__init__(config)
        self.config: SearchTimePipelineConfig = config

    def build_index(self, index_config: IndexConfig) -> BaseIndex:
        """Build the configured index."""
        index: BaseIndex = IndexFactory.create(index_config)
        return index

    def run_pipeline(self) -> tp.Any:
        """Run pipeline and return results."""
        search_timings = {}


        self.corpus = self._create_corpus()

        for index_config in tqdm(self.config.indexes_to_compare):
            embedder = DummyEmbedder(self.config.embedder_config)
            embedder.initialize()
            embedder.config.head_size = index_config.dimension
            for corpus_size in self.config.slice_sizes:
                index = self.build_index(index_config)
                index.build_from_corpus(self.corpus[:corpus_size], embedder)
                queries = embedder.embed_query(self.config.queries)
                for k in self.config.k_list:
                    start_time = time.time()
                    index.search(queries, k)
                    search_time = time.time() - start_time
                    search_timings[
                        f"index:{index_config.index_name}_corpus-size:{corpus_size}_k:{k}_n-queries:{len(self.config.queries)}_dimension:{index_config.dimension}"
                    ] = search_time

        with open("search_time_results.json", "w") as file:
            file.write(json.dumps(search_timings, ensure_ascii=False, indent=4))

        return search_timings


if __name__ == "__main__":
    loader = EmailsDataLoader.load(
        "/Users/egor/Documents/code/ml-email-search/src/data_loader/emails.csv"
    )
    loader.preprocess(raw_email_col="message")
    documents: tp.List[tp.Tuple[int, str]] = loader.get_faiss_dataset()

    config = SearchTimePipelineConfig(
        documents=documents,
        slice_sizes=[50000],
        indexes_to_compare=[
            IndexConfig(index_name="brute_force", dimension=384, metric="l2"),
            IndexConfig(index_name="hnsw", dimension=384, metric="l2"),
            IndexConfig(index_name="ivf", dimension=384, metric="l2"),
            IndexConfig(index_name="lsh", dimension=384, metric="l2"),
            IndexConfig(index_name="pq", dimension=384, metric="l2"),
        ],
        embedder_config=EmbedderConfig(
            model_name="dummy_model",
            model_id="dummy",
            head="universal",
            head_size=384,
        ),
        queries=[
            "meeting",
            "project deadline update",
            "budget approval request for Q4",
            "team performance review and feedback session",
            "client communication regarding contract negotiations and timeline adjustments",
        ],
        k_list=[1, 50, 100, 200, 500, 1000],
    )
    pipeline = SearchTimePipeline(config)
    pipeline.run_pipeline()
