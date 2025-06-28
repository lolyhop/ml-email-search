import time
import typing as tp

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrievers.base import Embedder, EmbedderConfig


class MatryoshkaEmbedder(Embedder):
    def __init__(self, config: EmbedderConfig) -> None:
        super().__init__(config)
        self.model = SentenceTransformer(config.model_id, truncate_dim=config.head_size)

    def initialize(self) -> None:
        self.model.to(self.config.device)
        self.model.eval()
        self.is_initialized = True

    def embed_query(self, query: tp.List[str]) -> np.ndarray:
        return self.model.encode(query)

    def embed_document(self, document: tp.List[str]) -> np.ndarray:
        return self.model.encode(document)


if __name__ == "__main__":
    config = EmbedderConfig(
        model_name="matryoshka",
        model_id="tomaarsen/mpnet-base-nli-matryoshka",
        head_size=64,
        head="universal",
        device="cpu",
    )
    embedder = MatryoshkaEmbedder(config)
    embedder.initialize()
    queries = ["Hello, how are you?", "What is the capital of France?"]
    start_time = time.time()
    embeddings = embedder.embed_query(queries)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")
    print(embeddings.shape)
