import typing as tp

from src.index.base import BaseIndex
from src.index.config import IndexConfig
from src.index.indexes import BruteForceIndex, HNSWIndex, IVFIndex, LSHIndex, PQIndex


class IndexFactory:
    """Factory for creating index instances."""

    _registry: tp.Dict[str, tp.Type[BaseIndex]] = {
        "brute_force": BruteForceIndex,
        "hnsw": HNSWIndex,
        "ivf": IVFIndex,
        "lsh": LSHIndex,
        "pq": PQIndex,
    }

    @classmethod
    def create(cls, config: IndexConfig) -> BaseIndex:
        """Create an index instance by type."""
        if config.index_name not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(
                f"Unknown index type '{config.index_name}'. Available: {available}"
            )

        return cls._registry[config.index_name](config)

    @classmethod
    def list_available(cls) -> tp.List[str]:
        """List all available index types."""
        return list(cls._registry.keys())
