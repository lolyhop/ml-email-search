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

        return embeddings

    def embed_query(self, query: tp.List[str]) -> tp.Union[np.ndarray, torch.Tensor]:
        assert (
            self.config.head == "query"
        ), "Expected head to be 'query' for query embedding"
        return self._embed_batch(query)

    def embed_document(
        self, document: tp.List[str]
    ) -> tp.Union[np.ndarray, torch.Tensor]:
        assert (
            self.config.head == "doc"
        ), "Expected head to be 'doc' for document embedding"
        return self._embed_batch(document)
