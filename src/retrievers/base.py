import typing as tp
from abc import ABC, abstractmethod

import numpy as np
import torch

from src.retrievers.config import EmbedderConfig


class Embedder(ABC):
    """Abstract base class for embedder implementations."""

    def __init__(self, config: EmbedderConfig) -> None:
        self.config: EmbedderConfig = config
        self._embedding_dim: tp.Optional[int] = None
        self._is_initialized: bool = False

    @abstractmethod
    def initialize(self) -> None:
        """Initialize the embedder model and required components."""
        pass

    @abstractmethod
    def embed_query(self, query: tp.List[str]) -> tp.Union[np.ndarray, torch.Tensor]:
        """Create embeddings for search queries."""
        raise NotImplementedError()

    @abstractmethod
    def embed_document(
        self, document: tp.List[str]
    ) -> tp.Union[np.ndarray, torch.Tensor]:
        """Create embeddings for documents/emails."""
        raise NotImplementedError()

    def __str__(self) -> str:
        """String representation of the embedder."""
        return f"{self.__class__.__name__}(model_name='{self.config.model_name}', dim={self.config.head_size})"

    def __repr__(self) -> str:
        """Detailed string representation of the embedder."""
        return f"{self.__class__.__name__}(config={self.config})"
