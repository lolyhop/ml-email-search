import typing as tp

from src.index.base import BaseIndex
from src.index.config import IndexConfig
from src.index.indexes import BruteForceIndex, HNSWIndex, IVFIndex


class IndexFactory:
    """Factory for creating index instances."""
    
    _registry: tp.Dict[str, tp.Type[BaseIndex]] = {
        "brute_force": BruteForceIndex,
        "hnsw": HNSWIndex,
        "ivf": IVFIndex,
    }
    
    @classmethod
    def create(cls, index_type: str, config: IndexConfig) -> BaseIndex:
        """Create an index instance by type."""
        if index_type not in cls._registry:
            available = ", ".join(cls._registry.keys())
            raise ValueError(f"Unknown index type '{index_type}'. Available: {available}")
        
        return cls._registry[index_type](config)
    
    @classmethod
    def list_available(cls) -> tp.List[str]:
        """List all available index types."""
        return list(cls._registry.keys())
