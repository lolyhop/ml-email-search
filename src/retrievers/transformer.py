import time
import typing as tp

import numpy as np
from sentence_transformers import SentenceTransformer

from src.retrievers.base import Embedder, EmbedderConfig


class TransformerEmbedder(Embedder):
    def __init__(self, config: EmbedderConfig) -> None:
        super().__init__(config)
        self.model = SentenceTransformer(config.model_id)
        self.config = config

    def initialize(self) -> None:
        self.model.to(self.config.device)
        self.model.eval()
        self.is_initialized = True

    def embed_query(self, query: tp.List[str], **kwargs) -> np.ndarray:
        return self.model.encode(query, batch_size=kwargs.get("batch_size", 32), show_progress_bar=False)

    def embed_document(self, document: tp.List[str], **kwargs) -> np.ndarray:
        return self.model.encode(document, batch_size=kwargs.get("batch_size", 32), show_progress_bar=False)


if __name__ == "__main__":
    config = EmbedderConfig(
        model_name="transformer",
        model_id="sentence-transformers/all-MiniLM-L6-v2",  # or another 300M+ model
        head="universal",
        head_size=384,
        device="cpu"
    )
    embedder = TransformerEmbedder(config)
    embedder.initialize()
    
    queries = ["Hello, how are you?", "What is the capital of France?"]
    start = time.time()
    embeddings = embedder.embed_query(queries)
    print(f"Time: {time.time() - start:.4f}s")
    print("Shape:", embeddings.shape)
