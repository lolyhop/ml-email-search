import os

os.environ["OMP_NUM_THREADS"] = "1"

import time
import json
import logging
import typing as tp

from tqdm import tqdm

from src.retrievers.base import EmbedderConfig
from src.index.base import BaseIndex
from src.index.factory import IndexFactory
from src.index.config import IndexConfig
from src.pipeline.config import QuantizationPipelineConfig
from src.pipeline.pipeline import Pipeline
from src.data_loader.data_loader import EmailsDataLoader
from src.retrievers.dummy import DummyEmbedder

logger = logging.getLogger(__name__)
SAVE_REPORTS_PATH = "reports/embedder-time"


class QuantizationPipeline(Pipeline):

    def __init__(self, config: QuantizationPipelineConfig) -> None:
        super().__init__(config)
        self.config: QuantizationPipelineConfig = config

    def build_index(self, index_config: IndexConfig) -> BaseIndex:
        """Build the configured index."""
        logger.info(f"Building {index_config.index_name} index...")

        start_time = time.time()

        index: BaseIndex = IndexFactory.create(index_config)

        build_time = time.time() - start_time
        logger.info(f"Index built in {build_time:.3f}s")
        return index

    def run_pipeline(self) -> tp.Any:
        """Run pipeline and return results."""
        search_timings = {}
        embedder = DummyEmbedder(self.config.embedder_config)
        embedder.initialize()
        self.embedder = embedder
        
        self.corpus = self._create_corpus()

        for index_config in tqdm(self.config.indexes_to_compare):
            for corpus_size in self.config.slice_sizes:
                index = self.build_index(index_config)
                index.build_from_corpus(self.corpus[:corpus_size], embedder)
                queries = embedder.embed_query(self.config.queries)
                for k in self.config.k_list:
                    start_time = time.time()
                    index.search(queries, k)
                    search_time = time.time() - start_time
                    search_timings[
                        f"index:{index_config.index_name}_quantization:{index_config.quantization}_corpus-size:{corpus_size}_k:{k}_n-queries:{len(self.config.queries)}"
                    ] = search_time

        with open("quantization_timings.json", "w") as file:
            file.write(json.dumps(search_timings, ensure_ascii=False, indent=4))

        return search_timings


if __name__ == "__main__":
    loader = EmailsDataLoader.load(
        "/Users/egor/Documents/code/ml-email-search/src/data_loader/emails.csv"
    )
    loader.preprocess(raw_email_col="message")
    documents: tp.List[tp.Tuple[int, str]] = loader.get_faiss_dataset()
    config = QuantizationPipelineConfig(
        documents=documents,
        slice_sizes=[1000, 5000, 10000, 30000, 50000],
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
    pipeline = QuantizationPipeline(config)
    pipeline.run_pipeline()
