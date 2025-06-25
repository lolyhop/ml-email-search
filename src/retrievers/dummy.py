import hashlib
import typing as tp

import torch
import numpy as np

from src.retrievers.base import Embedder, EmbedderConfig


class DummyEmbedder(Embedder):
    """Dummy embedder that generates random embeddings for testing."""

    def __init__(self, config: EmbedderConfig) -> None:
        super().__init__(config)
        self.base_seed: int = 42

    def _text_to_seed(self, text: str) -> int:
        """Convert text to a deterministic seed."""
        return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)

    def initialize(self) -> None:
        """Initialize the dummy embedder (no actual model loading)."""
        self._is_initialized = True

    def _generate_random_embeddings(self, texts: tp.List[str]) -> np.ndarray:
        """Generate deterministic 'random' embeddings for the given texts."""
        if not self._is_initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")

        embeddings = []
        for text in texts:
            text_seed = self.base_seed + self._text_to_seed(text)
            rng = np.random.RandomState(text_seed)

            embedding = rng.normal(0, 1, self.config.head_size).astype(np.float32)
            embeddings.append(embedding)

        embeddings = np.array(embeddings)

        if self.config.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / (norms + 1e-8)

        return embeddings

    def embed_query(self, query: tp.List[str]) -> tp.Union[np.ndarray, torch.Tensor]:
        """Generate random embeddings for queries."""
        assert (
            self.config.head == "query" or self.config.head == "universal"
        ), "Expected head to be 'query' for query embedding"
        return self._generate_random_embeddings(query)

    def embed_document(
        self, document: tp.List[str]
    ) -> tp.Union[np.ndarray, torch.Tensor]:
        """Generate random embeddings for documents."""
        assert (
            self.config.head == "doc" or self.config.head == "universal"
        ), "Expected head to be 'doc' for document embedding"
        return self._generate_random_embeddings(document)
