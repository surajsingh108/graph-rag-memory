from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import chromadb
import numpy as np
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

_MODEL_CACHE: dict[str, SentenceTransformer] = {}


class _SentenceTransformerEF(EmbeddingFunction[Documents]):
    """sentence-transformers wrapper satisfying chromadb's EmbeddingFunction protocol."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        if model_name not in _MODEL_CACHE:
            _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        self._model = _MODEL_CACHE[model_name]

    def name(self) -> str:
        return f"sentence-transformers-{self._model_name}"

    def __call__(self, input: Documents) -> Embeddings:  # type: ignore[override]
        return self._model.encode(list(input), show_progress_bar=False).tolist()

    def embed_query(self, input: Documents) -> Embeddings:
        return self(input)

    @classmethod
    def build_from_config(cls, config: dict) -> "_SentenceTransformerEF":
        return cls(config["model_name"])

    def get_config(self) -> dict:
        return {"model_name": self._model_name}


def _safe_normalize(arr: np.ndarray) -> np.ndarray:
    """L2-normalize rows of a 2-D array or a 1-D vector; leaves zero-norm rows unchanged."""
    if arr.ndim == 1:
        n = np.linalg.norm(arr)
        return arr / n if n > 0 else arr
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


class Memory:
    """Two-tier ChromaDB store: 'sources' for ingested docs, 'derived' for LLM summaries."""

    def __init__(self, config) -> None:
        self._config = config
        self._derived_weight = config.derived_weight
        self._ef = _SentenceTransformerEF(config.embed_model)
        self._client = chromadb.PersistentClient(path=config.chroma_path)
        self._sources = self._client.get_or_create_collection(
            "sources", embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )
        self._derived = self._client.get_or_create_collection(
            "derived", embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )
        self._queries_since_sweep: int = 0
        self._warn_if_not_cosine()

    def _warn_if_not_cosine(self) -> None:
        for col, name in [(self._sources, "sources"), (self._derived, "derived")]:
            space = (col.metadata or {}).get("hnsw:space")
            if space != "cosine":
                logger.warning(
                    "Collection '%s' is using '%s' distance (not cosine). "
                    "Relevance scores will be lower than expected — run :reset to rebuild.",
                    name, space or "l2",
                )

    # ------------------------------------------------------------------
    # Write paths
    # ------------------------------------------------------------------

    def add_source(self, chunks: list[str], metadatas: list[dict]) -> None:
        """Add source chunks with score metadata. Idempotent: skips existing IDs."""
        ids = [f"{m['file_hash']}_{m['chunk_idx']}" for m in metadatas]
        if not ids:
            return
        existing = set(self._sources.get(ids=ids)["ids"])
        new = [(c, m, i) for c, m, i in zip(chunks, metadatas, ids) if i not in existing]
        if not new:
            logger.debug("All chunks already present — skipping add")
            return
        cs, ms, is_ = zip(*new)
        ms_scored = [
            {**m, "score": self._config.initial_score_source, "confirmations": 1}
            for m in ms
        ]
        self._sources.add(documents=list(cs), metadatas=ms_scored, ids=list(is_))
        logger.debug("Stored %d new source chunks", len(new))

    def add_derived(self, text: str, metadata: dict) -> None:
        """Store a derived summary, applying residual novelty dedup first."""
        doc_id = hashlib.md5(text.encode()).hexdigest()
        if self._derived.get(ids=[doc_id])["ids"]:
            return  # exact content-hash duplicate

        derived_count = self._derived.count()
        if derived_count > 0:
            k = min(self._config.dedup_neighbours, derived_count)
            try:
                results = self._derived.query(
                    query_texts=[text],
                    n_results=k,
                    include=["embeddings", "ids"],
                )
                neighbour_embs_raw = np.array(results["embeddings"][0], dtype=float)
                neighbour_ids: list[str] = results["ids"][0]

                new_emb_raw = np.array(self._ef([text])[0], dtype=float)
                new_emb = _safe_normalize(new_emb_raw)
                neighbour_embs = _safe_normalize(neighbour_embs_raw)

                from grag.dedup import residual_novelty
                max_cos, res_norm = residual_novelty(new_emb, neighbour_embs)

                if max_cos > self._config.dedup_cosine_max:
                    nearest_id = neighbour_ids[0]
                    self._bump_metadata(self._derived, nearest_id, self._config.access_increment, conf_delta=1)
                    logger.info(
                        "Dedup reject (cosine=%.3f > %.3f): bumped confirmations on %s",
                        max_cos, self._config.dedup_cosine_max, nearest_id,
                    )
                    return

                if res_norm < self._config.dedup_residual_min:
                    nearest_id = neighbour_ids[0]
                    self._bump_metadata(self._derived, nearest_id, self._config.access_increment, conf_delta=1)
                    logger.info(
                        "Dedup reject (residual=%.3f < %.3f): bumped confirmations on %s",
                        res_norm, self._config.dedup_residual_min, nearest_id,
                    )
                    return
            except Exception as exc:
                logger.debug("Dedup check failed (%s); inserting without filter", exc)

        full_meta = {
            **metadata,
            "score": self._config.initial_score_derived,
            "confirmations": 1,
        }
        self._derived.add(documents=[text], metadatas=[full_meta], ids=[doc_id])

    # ------------------------------------------------------------------
    # Read path
    # ------------------------------------------------------------------

    def query(self, text: str, k: int = 4) -> list[dict]:
        """Retrieve top-k chunks across both tiers; trigger decay sweep counter."""
        self._queries_since_sweep += 1
        if self._queries_since_sweep >= self._config.decay_every_n_queries:
            self._run_decay_sweep()
            self._queries_since_sweep = 0

        results: list[dict] = []
        for collection, tier in [(self._sources, "source"), (self._derived, "derived")]:
            count = collection.count()
            if count == 0:
                continue
            r = collection.query(query_texts=[text], n_results=min(k, count))
            for doc, meta, dist, doc_id in zip(
                r["documents"][0], r["metadatas"][0], r["distances"][0], r["ids"][0]
            ):
                score = max(0.0, 1.0 - dist)
                if tier == "derived":
                    score *= self._derived_weight
                results.append({
                    "text": doc,
                    "score": score,
                    "tier": tier,
                    "metadata": meta,
                    "id": doc_id,
                })
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def get_cached_answer(self, q_hash: str) -> tuple[str | None, str | None]:
        """Return (answer_text, doc_id) if a fresh cache entry exists, else (None, None)."""
        if self._config.cache_ttl_seconds == 0:
            return None, None
        try:
            results = self._derived.get(
                where={"question_hash": {"$eq": q_hash}},
                include=["documents", "metadatas"],
            )
        except Exception:
            return None, None

        if not results["ids"]:
            return None, None

        now = datetime.now(timezone.utc)
        for doc_id, doc, meta in zip(results["ids"], results["documents"], results["metadatas"]):
            meta = meta or {}
            created_at_str = meta.get("created_at")
            if not created_at_str:
                continue
            try:
                created_at = datetime.fromisoformat(created_at_str)
                if (now - created_at).total_seconds() <= self._config.cache_ttl_seconds:
                    return doc, doc_id
            except (ValueError, TypeError):
                continue
        return None, None

    # ------------------------------------------------------------------
    # Access tracking
    # ------------------------------------------------------------------

    def record_access(self, chunks: list[dict]) -> None:
        """Bump score for chunks that were included in the LLM context."""
        from grag.access import apply_access

        for chunk in chunks:
            cid = chunk.get("id")
            tier = chunk.get("tier", "derived")
            if not cid:
                continue
            collection = self._sources if tier == "source" else self._derived

            def _make_bump(col, cid_: str):
                def bump(inc: float) -> None:
                    self._bump_metadata(col, cid_, inc)
                return bump

            # TODO: wire neighbour_fn when spread.enabled is true
            # neighbour_fn = lambda: self._get_neighbour_bumps(collection, cid)
            apply_access(_make_bump(collection, cid), self._config.access_increment, self._config)

    # ------------------------------------------------------------------
    # Decay sweep
    # ------------------------------------------------------------------

    def _run_decay_sweep(self, decay_multiplier: float = 1.0) -> dict:
        """Reduce scores of all entries by decay amount; cull below-threshold entries."""
        config = self._config
        derived_count = self._derived.count()

        source_delete: list[str] = []
        derived_delete: list[str] = []

        for collection, tier, delete_list in [
            (self._sources, "source", source_delete),
            (self._derived, "derived", derived_delete),
        ]:
            if tier == "source" and not config.source_decays:
                continue

            all_data = collection.get(include=["metadatas"])
            if not all_data["ids"]:
                continue

            decay = config.decay_per_sweep * decay_multiplier
            if tier == "source":
                decay *= config.source_decay_factor
            if tier == "derived" or not config.pressure_targets_derived_only:
                if derived_count > config.soft_limit:
                    decay *= derived_count / config.soft_limit

            initial = (config.initial_score_source if tier == "source"
                       else config.initial_score_derived)

            update_ids: list[str] = []
            update_metas: list[dict] = []

            for doc_id, meta in zip(all_data["ids"], all_data["metadatas"]):
                meta = dict(meta or {})
                confirmations = int(meta.get("confirmations", 1))
                if confirmations >= config.confirmation_floor:
                    continue

                score = float(meta.get("score", initial))
                new_score = score - decay

                if new_score <= config.cull_threshold:
                    delete_list.append(doc_id)
                else:
                    update_ids.append(doc_id)
                    update_metas.append({**meta, "score": new_score})

            if update_ids:
                collection.update(ids=update_ids, metadatas=update_metas)

        total_deleting = len(source_delete) + len(derived_delete)
        if total_deleting > 100:
            self._write_snapshot(source_delete, derived_delete)

        if source_delete:
            self._sources.delete(ids=source_delete)
        if derived_delete:
            self._derived.delete(ids=derived_delete)

        logger.info(
            "Memory sweep: dropped %d derived, %d sources",
            len(derived_delete), len(source_delete),
        )
        return {"sources_deleted": len(source_delete), "derived_deleted": len(derived_delete)}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _bump_metadata(
        self, collection, doc_id: str, score_delta: float, conf_delta: int = 0
    ) -> None:
        """Read-modify-write score (and optionally confirmations) for one entry."""
        result = collection.get(ids=[doc_id], include=["metadatas"])
        if not result["ids"]:
            return
        meta = dict(result["metadatas"][0] or {})
        tier = "source" if collection is self._sources else "derived"
        initial = (self._config.initial_score_source if tier == "source"
                   else self._config.initial_score_derived)
        meta["score"] = float(meta.get("score", initial)) + score_delta
        if conf_delta:
            meta["confirmations"] = int(meta.get("confirmations", 1)) + conf_delta
        collection.update(ids=[doc_id], metadatas=[meta])

    def _write_snapshot(self, source_ids: list[str], derived_ids: list[str]) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(self._config.data_dir) / "backups" / f"sweep_{ts}"
        backup_dir.mkdir(parents=True, exist_ok=True)
        manifest = {"deleted_source_chunks": source_ids, "deleted_derived_chunks": derived_ids}
        (backup_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        backups_root = Path(self._config.data_dir) / "backups"
        old_sweeps = sorted(backups_root.glob("sweep_*"), key=lambda p: p.name)
        for old in old_sweeps[:-10]:
            shutil.rmtree(old, ignore_errors=True)

    def is_empty(self) -> bool:
        return self._sources.count() == 0

    def reset(self) -> tuple[bool, bool]:
        """Drop both collections and recreate them empty."""
        for name in ("sources", "derived"):
            try:
                self._client.delete_collection(name)
            except Exception:
                pass
        self._sources = self._client.get_or_create_collection(
            "sources", embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )
        self._derived = self._client.get_or_create_collection(
            "derived", embedding_function=self._ef, metadata={"hnsw:space": "cosine"}
        )
        self._queries_since_sweep = 0
        return True, True

    def stats(self) -> dict:
        """Return score distributions, sweep state, cache info, and memory pressure."""
        import statistics as _stats

        result: dict = {
            "queries_since_sweep": self._queries_since_sweep,
            "sweep_every": self._config.decay_every_n_queries,
            "sources": {},
            "derived": {},
            "cache": {"count": 0, "oldest_seconds": 0.0, "newest_seconds": 0.0},
            "pressure_pct": 0.0,
        }

        for collection, tier_key in [(self._sources, "sources"), (self._derived, "derived")]:
            all_data = collection.get(include=["metadatas"])
            count = len(all_data["ids"])
            result[tier_key]["count"] = count
            if count == 0:
                result[tier_key].update(score_min=0.0, score_median=0.0, score_max=0.0)
                if tier_key == "derived":
                    result[tier_key]["confirmations_dist"] = {1: 0, "2-4": 0, "5+": 0}
                continue

            initial = (self._config.initial_score_source if tier_key == "sources"
                       else self._config.initial_score_derived)
            scores = [float((m or {}).get("score", initial)) for m in all_data["metadatas"]]
            result[tier_key]["score_min"] = min(scores)
            result[tier_key]["score_median"] = _stats.median(scores)
            result[tier_key]["score_max"] = max(scores)

            if tier_key == "derived":
                dist: dict = {1: 0, "2-4": 0, "5+": 0}
                for m in all_data["metadatas"]:
                    c = int((m or {}).get("confirmations", 1))
                    if c == 1:
                        dist[1] += 1
                    elif c <= 4:
                        dist["2-4"] += 1
                    else:
                        dist["5+"] += 1
                result[tier_key]["confirmations_dist"] = dist

        derived_count = result["derived"]["count"]
        result["soft_limit"] = self._config.soft_limit
        result["pressure_pct"] = (derived_count / self._config.soft_limit) * 100

        # Cache entries = derived chunks that carry a question_hash
        now = datetime.now(timezone.utc)
        try:
            all_derived = self._derived.get(include=["metadatas"])
            ages: list[float] = []
            for meta in all_derived["metadatas"]:
                meta = meta or {}
                if "question_hash" not in meta:
                    continue
                created_str = meta.get("created_at")
                if created_str:
                    try:
                        age = (now - datetime.fromisoformat(created_str)).total_seconds()
                        ages.append(age)
                    except (ValueError, TypeError):
                        pass
            result["cache"]["count"] = len(ages)
            if ages:
                result["cache"]["oldest_seconds"] = max(ages)
                result["cache"]["newest_seconds"] = min(ages)
        except Exception:
            pass

        return result
