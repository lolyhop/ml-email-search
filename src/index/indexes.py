import typing as tp

import numpy as np
import faiss

from src.index.base import BaseIndex


class BruteForceIndex(BaseIndex):

    def build(self) -> None:
        if self.config.metric == "cosine":
            self.index = faiss.IndexFlatIP(self.config.dimension)
        else:
            self.index = faiss.IndexFlatL2(self.config.dimension)
        self._is_trained = True

    def add(self, embeddings: np.ndarray) -> None:
        embeddings = self._prepare_embeddings(embeddings)
        self.index.add(embeddings)

    def search(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        query_embeddings = self._prepare_embeddings(query_embeddings)
        return self.index.search(query_embeddings, k)


class HNSWIndex(BaseIndex):

    def build(self) -> None:
        if self.config.metric == "cosine":
            self.index = faiss.IndexHNSWFlat(
                self.config.dimension, self.config.hnsw_m, faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.index = faiss.IndexHNSWFlat(
                self.config.dimension, self.config.hnsw_m, faiss.METRIC_L2
            )

        self.index.hnsw.efConstruction = self.config.hnsw_ef_construction
        self.index.hnsw.efSearch = self.config.hnsw_ef_search
        self._is_trained = True

    def add(self, embeddings: np.ndarray) -> None:
        embeddings = self._prepare_embeddings(embeddings)
        self.index.add(embeddings)

    def search(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        query_embeddings = self._prepare_embeddings(query_embeddings)
        return self.index.search(query_embeddings, k)


class IVFIndex(BaseIndex):

    def build(self) -> None:
        if self.config.metric == "cosine":
            quantizer = faiss.IndexFlatIP(self.config.dimension)
            metric = faiss.METRIC_INNER_PRODUCT
        else:
            quantizer = faiss.IndexFlatL2(self.config.dimension)
            metric = faiss.METRIC_L2

        if self.config.quantization == "pq":
            self.index = faiss.IndexIVFPQ(
                quantizer,
                self.config.dimension,
                self.config.ivf_nlist,
                self.config.pq_m,
                self.config.pq_bits,
                metric,
            )
        elif self.config.quantization == "sq":
            self.index = faiss.IndexIVFScalarQuantizer(
                quantizer,
                self.config.dimension,
                self.config.ivf_nlist,
                faiss.ScalarQuantizer.QT_8bit,
                metric,
            )
        else:
            self.index = faiss.IndexIVFFlat(
                quantizer, self.config.dimension, self.config.ivf_nlist, metric
            )

        self.index.nprobe = self.config.ivf_nprobe

    def add(self, embeddings: np.ndarray) -> None:
        embeddings = self._prepare_embeddings(embeddings)

        if not self._is_trained:
            self.index.train(embeddings)
            self._is_trained = True

        self.index.add(embeddings)

    def search(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        query_embeddings = self._prepare_embeddings(query_embeddings)
        return self.index.search(query_embeddings, k)


class LSHIndex(BaseIndex):

    def build(self) -> None:
        self.index = faiss.IndexLSH(self.config.dimension, self.config.lsh_nbits)
        self._is_trained = True

    def add(self, embeddings: np.ndarray) -> None:
        embeddings = self._prepare_embeddings(embeddings)
        self.index.add(embeddings)

    def search(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        query_embeddings = self._prepare_embeddings(query_embeddings)
        return self.index.search(query_embeddings, k)


class PQIndex(BaseIndex):

    def build(self) -> None:
        if self.config.metric == "cosine":
            self.index = faiss.IndexPQ(
                self.config.dimension, self.config.pq_m, self.config.pq_bits,
                faiss.METRIC_INNER_PRODUCT
            )
        else:
            self.index = faiss.IndexPQ(
                self.config.dimension, self.config.pq_m, self.config.pq_bits,
                faiss.METRIC_L2
            )

    def add(self, embeddings: np.ndarray) -> None:
        embeddings = self._prepare_embeddings(embeddings)
        
        if not self._is_trained:
            self.index.train(embeddings)
            self._is_trained = True
            
        self.index.add(embeddings)

    def search(
        self, query_embeddings: np.ndarray, k: int = 10
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        query_embeddings = self._prepare_embeddings(query_embeddings)
        return self.index.search(query_embeddings, k)
