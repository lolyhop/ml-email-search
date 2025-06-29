import typing as tp

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

from src.retrievers.base import Embedder, EmbedderConfig


class DenseEmbedder(Embedder):

    def __init__(self, config: EmbedderConfig) -> None:
        super().__init__(config)
        self.model = None
        self.tokenizer = None

    def initialize(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_id)
        self.model = AutoModel.from_pretrained(self.config.model_id)
        self.model.to(self.config.device)
        self.model.eval()
        self._is_initialized = True

    def _pool_embeddings(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        if self.config.pooling_strategy == "cls":
            return embeddings[:, 0]
        elif self.config.pooling_strategy == "mean":
            masked_embeddings = embeddings * attention_mask.unsqueeze(-1)
            return masked_embeddings.sum(dim=1) / attention_mask.sum(
                dim=1, keepdim=True
            )
        elif self.config.pooling_strategy == "eos":
            seq_lengths = attention_mask.sum(dim=1) - 1
            return embeddings[range(embeddings.size(0)), seq_lengths]
        else:
            raise ValueError(
                f"Unknown pooling strategy: {self.config.pooling_strategy}"
            )

    def _embed_batch(self, texts: tp.List[str]) -> torch.Tensor:
        if not self._is_initialized:
            raise RuntimeError(f"{self.__class__.__name__} is not initialized")

        inputs = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.config.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            embeddings = self._pool_embeddings(
                outputs.last_hidden_state, inputs.attention_mask
            )

            if self.config.normalize:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=-1)

        return embeddings.cpu().float().numpy()

    def _embed_texts(self, texts: tp.List[str], batch_size: int = 32) -> np.ndarray:
        if not texts:
            return np.array([])

        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            batch_embeddings = self._embed_batch(batch)
            all_embeddings.append(batch_embeddings)

        return np.vstack(all_embeddings)

    def embed_query(
        self, query: tp.List[str], batch_size: int = 32
    ) -> tp.Union[np.ndarray, torch.Tensor]:
        assert (
            self.config.head == "query" or self.config.head == "universal"
        ), "Expected head to be 'query' for query embedding"
        return self._embed_texts(query, batch_size)

    def embed_document(
        self, document: tp.List[str], batch_size: int = 32
    ) -> tp.Union[np.ndarray, torch.Tensor]:
        assert (
            self.config.head == "doc" or self.config.head == "universal"
        ), "Expected head to be 'doc' for document embedding"
        return self._embed_texts(document, batch_size)
