import typing as tp
import uuid
from abc import ABC, abstractmethod

import numpy as np
import faiss

from src.index.config import IndexConfig
from src.retrievers.base import Embedder
from src.index.entity import FaissEntity, FaissCorpus


class BaseIndex(ABC):

    def __init__(self, config: IndexConfig) -> None:
        self.config: IndexConfig = config
        self.index: faiss.Index = None
        self._is_trained: bool = False
        self.corpus: tp.Optional[FaissCorpus] = None

    @abstractmethod
    def build(self) -> None:
        """Build empty index structure."""
        pass

    @abstractmethod
    def add(self, embeddings: np.ndarray) -> None:
        """Add embeddings to existing index."""
        pass

    def build_from_corpus(self, corpus: FaissCorpus, embedder: Embedder) -> None:
        """Build and populate index from corpus data using embedder."""

        self.corpus: FaissCorpus = corpus

        embeddings: np.ndarray = embedder.embed_document(corpus.contents)

        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        embeddings = embeddings.astype(np.float32)

        self.build()
        self.add(embeddings)

    @abstractmethod
    def search(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        pass

    def search_with_ids(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> tp.Tuple[np.ndarray, tp.List[tp.List[tp.Any]]]:
        """Search and return document IDs instead of indices."""
        scores, indices = self.search(query_embeddings, k)
        
        if self.corpus is None:
            raise RuntimeError("Index was not built from corpus, cannot return document IDs")
        
        # Convert indices to document IDs
        doc_ids: tp.List[uuid.UUID] = []
        for query_indices in indices:
            query_doc_ids = [self.corpus.document_ids[idx] for idx in query_indices if idx < len(self.corpus.document_ids)]
            doc_ids.append(query_doc_ids)
        
        return scores, doc_ids

    def get_entities_by_indices(self, indices: tp.List[int]) -> tp.List[FaissEntity]:
        """Get original entities by FAISS indices."""
        if self.corpus is None:
            raise RuntimeError("Index was not built from corpus, cannot return entities")
        
        return [self.corpus.entities[i] for i in indices if 0 <= i < len(self.corpus.entities)]

    def _prepare_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        if self.config.metric == "cosine":
            faiss.normalize_L2(embeddings)
        return embeddings.astype(np.float32)
