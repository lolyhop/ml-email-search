import typing as tp
from dataclasses import dataclass


@dataclass(kw_only=True)
class EmbedderConfig:
    """Configuration class for embedder settings."""

    model_name: str
    head: tp.Literal["query", "doc"] = "query"
    model_id: tp.Optional[str] = None
    batch_size: int = 32
    max_length: int = 512
    head_size: int = 1024
    device: str = "cpu"
    normalize: bool = False
    pooling_strategy: tp.Literal["cls", "mean", "eos"] = "cls"
