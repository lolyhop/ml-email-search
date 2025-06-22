import typing as tp
from dataclasses import dataclass


@dataclass(kw_only=True)
class IndexConfig:
    dimension: int
    metric: tp.Literal["cosine", "l2"] = "cosine"
    quantization: tp.Literal["none", "pq", "sq"] = "none"

    # HNSW specific
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 100

    # IVF specific
    ivf_nlist: int = 100
    ivf_nprobe: int = 10

    # Quantization specific
    pq_m: int = 8
    pq_bits: int = 8
    sq_bits: int = 8
