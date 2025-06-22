import typing as tp

from src.index.config import IndexConfig
from src.index.base import BaseIndex
from src.index.indexes import BruteForceIndex, HNSWIndex, IVFIndex


class IndexFactory:
    _registry: tp.Dict[str, BaseIndex] = {
        "brute_force": BruteForceIndex,
        "hnsw": HNSWIndex,
        "ivf": IVFIndex,
    }

    @classmethod
    def create(cls, index_type: str, config: IndexConfig) -> BaseIndex:
        if index_type not in cls._registry:
            raise ValueError(
                f"Unknown index type '{index_type}'. Available index types: {cls._registry}"
            )

        index: BaseIndex = cls._registry[index_type](config)
        index.build()
        return index

    @classmethod
    def list_available(cls) -> tp.List[str]:
        return list(cls._registry.keys())
