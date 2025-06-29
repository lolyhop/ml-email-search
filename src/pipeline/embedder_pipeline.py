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
from src.retrievers.dense import DenseEmbedder

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
        for model_id in tqdm(
            self.config.model_ids_to_compare, desc="Comparing embedder dimensions"
        ):
            config = EmbedderConfig(
                model_name="dense",
                model_id=model_id,
                head="document",
                device="cuda",
            )
            print(f"Processing model: {model_id}")
            embedder = DenseEmbedder(config)
            embedder.initialize()
            for slice_size in tqdm(
                self.config.slice_sizes, desc="Processing slice sizes"
            ):
                start_time = time.time()
                embedder.embed_document(
                    self.config.documents[:slice_size], batch_size=8
                )
                end_time = time.time()
                if model_id not in embedder_timings:
                    embedder_timings[model_id] = {}
                embedder_timings[model_id][slice_size] = end_time - start_time

        with open("output.json", "w") as file:
            file.write(json.dumps(embedder_timings, ensure_ascii=False, indent=4))

        return embedder_timings


if __name__ == "__main__":
    loader = EmailsDataLoader.load("~/ml-email-search/src/data_loader/emails.csv")
    loader.preprocess(raw_email_col="message")
    documents: tp.List[tp.Tuple[int, str]] = loader.get_faiss_dataset()
    config = EmbedderPipelineConfig(
        documents=documents,
        model_ids_to_compare=[
            "intfloat/e5-small-v2",
            "intfloat/e5-base-v2",
            "intfloat/e5-large-v2",
        ],
        slice_sizes=[1000, 5000, 10000, 30000, 50000],
    )
    pipeline = EmbedderPipeline(config)
    pipeline.run_pipeline()
