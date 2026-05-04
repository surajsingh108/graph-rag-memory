from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Config:
    # Storage
    data_dir: Path = field(default_factory=lambda: Path("grag_data"))
    chroma_path: str = "grag_data/chroma"
    graph_path: str = "grag_data/knowledge_graph.gexf"

    # Models
    embed_model: str = "all-MiniLM-L6-v2"
    llm_model: str = "Qwen/Qwen2.5-1.5B-Instruct"

    # Chunking
    chunk_size: int = 1500
    chunk_overlap: int = 150

    # Retrieval
    top_k: int = 4
    derived_weight: float = 0.6
    min_relevance: float = 0.35
    max_facts: int = 15

    # Decay
    decay_every_n_queries: int = 10
    decay_per_sweep: float = 0.5
    access_increment: float = 1.0
    initial_score_source: float = 100.0
    initial_score_derived: float = 5.0
    source_decays: bool = True
    source_decay_factor: float = 0.1
    confirmation_floor: int = 5

    # Cull
    cull_threshold: float = 0.0
    compact_decay_multiplier: float = 2.0

    # Memory pressure
    soft_limit: int = 10000
    pressure_targets_derived_only: bool = True

    # Spreading activation
    spread_enabled: bool = False
    spread_factor: float = 0.3
    spread_max_hops: int = 2
    spread_min_contribution: float = 0.05

    # Dedup
    dedup_cosine_max: float = 0.95
    dedup_residual_min: float = 0.25
    dedup_neighbours: int = 5

    # Cache
    cache_ttl_seconds: int = 3600
    cache_hash_length: int = 12

    def __post_init__(self) -> None:
        Path(self.chroma_path).mkdir(parents=True, exist_ok=True)
        Path(self.graph_path).parent.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        import yaml

        with open(path) as f:
            data = yaml.safe_load(f) or {}

        decay = data.get("decay", {})
        cull = data.get("cull", {})
        pressure = data.get("memory_pressure", {})
        spread = data.get("spread", {})
        dedup = data.get("dedup", {})
        cache = data.get("cache", {})

        kwargs: dict[str, Any] = {}

        def maybe(d: dict, yaml_key: str, config_key: str) -> None:
            if yaml_key in d:
                kwargs[config_key] = d[yaml_key]

        maybe(decay, "decay_every_n_queries", "decay_every_n_queries")
        maybe(decay, "decay_per_sweep", "decay_per_sweep")
        maybe(decay, "access_increment", "access_increment")
        maybe(decay, "initial_score_source", "initial_score_source")
        maybe(decay, "initial_score_derived", "initial_score_derived")
        maybe(decay, "source_decays", "source_decays")
        maybe(decay, "source_decay_factor", "source_decay_factor")
        maybe(decay, "confirmation_floor", "confirmation_floor")
        maybe(cull, "cull_threshold", "cull_threshold")
        maybe(cull, "compact_decay_multiplier", "compact_decay_multiplier")
        maybe(pressure, "soft_limit", "soft_limit")
        maybe(pressure, "pressure_targets_derived_only", "pressure_targets_derived_only")
        maybe(spread, "enabled", "spread_enabled")
        maybe(spread, "factor", "spread_factor")
        maybe(spread, "max_hops", "spread_max_hops")
        maybe(spread, "min_contribution", "spread_min_contribution")
        maybe(dedup, "cosine_max", "dedup_cosine_max")
        maybe(dedup, "residual_min", "dedup_residual_min")
        maybe(dedup, "neighbours", "dedup_neighbours")
        maybe(cache, "ttl_seconds", "cache_ttl_seconds")
        maybe(cache, "hash_length", "cache_hash_length")

        return cls(**kwargs)
