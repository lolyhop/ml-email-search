import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import time
import json
import random
from tqdm import tqdm

from src.retrievers.bm25 import BM25Retriever
from src.pipeline.config import BuildTimePipelineConfig
from src.data_loader.data_loader import EmailsDataLoader

SAVE_QPS_PATH = "reports/bm25-qps"


class BM25QPSExperiment:
    def __init__(self, config: BuildTimePipelineConfig):
        self.config = config
        self.documents = config.documents

    def run(self, num_queries: int = 100) -> float:
        retriever = BM25Retriever()
        retriever.build_index(self.documents)

        queries = [doc[1] for doc in random.sample(self.documents, num_queries)]
        start_time = time.time()

        for query in tqdm(queries, desc="Evaluating QPS"):
            retriever.query(query, top_k=10)

        total_time = time.time() - start_time
        qps = num_queries / total_time

        with open(f"{SAVE_QPS_PATH}/qps.json", "w") as f:
            json.dump({"bm25_qps": qps}, f, indent=4)

        return qps


if __name__ == "__main__":
    loader = EmailsDataLoader.load(
        ...
    )
    loader.preprocess(raw_email_col="message")
    documents = loader.get_faiss_dataset()

    config = BuildTimePipelineConfig(
        documents=documents,
        slice_sizes=[],
        indexes_to_compare=[],
        embedder_config=None,
    )

    qps_exp = BM25QPSExperiment(config)
    qps = qps_exp.run(num_queries=100)
    print(f"BM25 QPS: {qps:.2f}")
