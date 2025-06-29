import typing as tp
from dataclasses import dataclass, field

import numpy as np

from src.retrievers.base import EmbedderConfig
from src.index.config import IndexConfig


@dataclass
class PipelineConfig:
    embedder_config: tp.Optional[EmbedderConfig] = None
    documents: tp.List[tp.Tuple[int, str]] = field(
        default_factory=list
    )  # (id, content)
    queries: tp.List[str] = field(default_factory=list)


@dataclass
class SearchResult:
    scores: np.ndarray
    document_ids: tp.List[int] = field(default_factory=list)
    search_time: tp.Optional[float] = None


@dataclass
class QualityPipelineConfig(PipelineConfig):
    indexes_to_compare: tp.List[IndexConfig] = field(default_factory=list)
    recall_ranks: tp.List[int] = field(default_factory=list)


@dataclass
class EmbedderPipelineConfig(PipelineConfig):
    dimensions_to_compare: tp.List[int] = field(default_factory=list)
    slice_sizes: tp.List[int] = field(default_factory=list)


@dataclass
class BuildTimePipelineConfig(PipelineConfig):
    indexes_to_compare: tp.List[IndexConfig] = field(default_factory=list)
    slice_sizes: tp.List[int] = field(default_factory=list)


@dataclass
class SearchTimePipelineConfig(PipelineConfig):
    indexes_to_compare: tp.List[IndexConfig] = field(default_factory=list)
    queries: tp.List[str] = field(default_factory=list)
    slice_sizes: tp.List[int] = field(default_factory=list)
    k_list: tp.List[int] = field(default_factory=list)
