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
from src.pipeline.config import EmbedderPipelineConfig
from src.pipeline.pipeline import Pipeline
from src.data_loader.data_loader import EmailsDataLoader
from src.retrievers.matryoshka import MatryoshkaEmbedder

logger = logging.getLogger(__name__)
SAVE_REPORTS_PATH = "reports/embedder-time"


class EmbedderPipeline(Pipeline):

    def __init__(self, config: EmbedderPipelineConfig) -> None:
        super().__init__(config)
        self.config: EmbedderPipelineConfig = config

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
        embedder_timings = {}
        for head_size in tqdm(
            self.config.dimensions_to_compare, desc="Comparing embedder dimensions"
        ):
            config = EmbedderConfig(
                model_name="matryoshka",
                model_id="tomaarsen/mpnet-base-nli-matryoshka",
                head_size=head_size,
                head="document",
                device="cpu",
            )
            print(f"Processing head size: {head_size}")
            embedder = MatryoshkaEmbedder(config)
            embedder.initialize()
            for slice_size in tqdm(
                self.config.slice_sizes, desc="Processing slice sizes"
            ):
                start_time = time.time()
                embedder.embed_document(
                    self.config.documents[:slice_size], batch_size=8
                )
                end_time = time.time()
                if head_size not in embedder_timings:
                    embedder_timings[head_size] = {}
                embedder_timings[head_size][slice_size] = end_time - start_time

        with open("output.json", "w") as file:
            file.write(
                json.dumps(embedder_timings, ensure_ascii=False, indent=4)
            )

        return embedder_timings


if __name__ == "__main__":
    loader = EmailsDataLoader.load(
        "~/ml-email-search/src/data_loader/emails.csv"
    )
    loader.preprocess(raw_email_col="message")
    documents: tp.List[tp.Tuple[int, str]] = loader.get_faiss_dataset()
    config = EmbedderPipelineConfig(
        documents=documents,
        dimensions_to_compare=[64, 128, 256, 512, 768],
        slice_sizes=[1000, 5000, 10000, 30000, 50000],
    )
    pipeline = EmbedderPipeline(config)
    pipeline.run_pipeline()
