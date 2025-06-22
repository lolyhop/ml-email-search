import typing as tp
import uuid
from dataclasses import dataclass, field

import numpy as np

from src.retrievers.base import EmbedderConfig
from src.index.config import IndexConfig
from src.index.entity import FaissEntity


@dataclass(kw_only=True)
class PipelineConfig:
    """Configuration for the search pipeline with single index and embedder."""

    embedder_config: EmbedderConfig
    index_type: str
    index_config: IndexConfig
    k: int = 10
    documents: tp.List[tp.Tuple[str, str]] = field(
        default_factory=list
    )  # (id, content)
    queries: tp.List[str] = field(default_factory=list)
    measure_time: bool = True


@dataclass
class SearchResult:
    """Results from a single search query."""

    query: str
    scores: np.ndarray
    document_ids: tp.List[uuid.UUID]
    documents: tp.List[FaissEntity]
    search_time: tp.Optional[float] = None


@dataclass
class PipelineResult:
    """Results from running the pipeline."""

    index_type: str
    build_time: float
    total_search_time: float
    avg_search_time: float
    results: tp.List[SearchResult]
