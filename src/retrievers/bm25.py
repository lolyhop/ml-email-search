import typing as tp
import numpy as np
from rank_bm25 import BM25Okapi

from src.retrievers.base import EmbedderConfig  # dummy to match interface


class BM25Retriever:
    def __init__(self, config: tp.Optional[EmbedderConfig] = None) -> None:
        self.config = config
        self.corpus: tp.List[str] = []
        self.tokenized_corpus: tp.List[tp.List[str]] = []
        self.bm25: tp.Optional[BM25Okapi] = None

    def initialize(self) -> None:
        pass  # for interface parity

    def build_index(self, documents: tp.List[tp.Tuple[int, str]]) -> None:
        self.corpus = [text for _, text in documents]
        self.tokenized_corpus = [doc.lower().split() for doc in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

    def query(self, query_text: str, top_k: int = 10) -> tp.List[int]:
        if self.bm25 is None:
            raise RuntimeError("Index must be built before querying.")

        tokenized_query = query_text.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return top_indices.tolist()
