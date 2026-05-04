from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

import networkx as nx

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """NetworkX MultiDiGraph of (subject, relation, object) triples with provenance."""

    def __init__(self, config) -> None:
        self._config = config
        self._path = config.graph_path
        self._queries_since_sweep: int = 0

        if Path(self._path).exists():
            loaded = nx.read_gexf(self._path)
            self._g: nx.MultiDiGraph = (
                loaded if isinstance(loaded, nx.MultiDiGraph) else nx.MultiDiGraph(loaded)
            )
        else:
            self._g = nx.MultiDiGraph()

    def add_triple(self, s: str, r: str, o: str, **prov) -> None:
        """Add a triple; duplicate (s,r,o) increments confirmations instead of adding a new edge."""
        if not all(isinstance(v, str) and v.strip() for v in (s, r, o)):
            logger.debug("Skipping malformed triple s=%r r=%r o=%r", s, r, o)
            return
        s = s.lower().strip()
        o = o.lower().strip()
        r = r.lower().strip().replace(" ", "_")

        # Check for existing edge with same (s, relation, o)
        if self._g.has_node(s) and self._g.has_node(o) and self._g.has_edge(s, o):
            for key, data in self._g[s][o].items():
                if data.get("relation") == r:
                    data["confirmations"] = int(data.get("confirmations", 1)) + 1
                    data["last_seen"] = datetime.now(timezone.utc).isoformat()
                    initial = (self._config.initial_score_source
                               if data.get("tier") == "source"
                               else self._config.initial_score_derived)
                    data["score"] = float(data.get("score", initial)) + self._config.access_increment
                    # Tier promotion: source evidence always wins
                    new_tier = prov.get("tier", "derived")
                    if data.get("tier") == "derived" and new_tier == "source":
                        data["tier"] = "source"
                        data["source_id"] = prov.get("source_id", data.get("source_id"))
                    self._save()
                    logger.debug(
                        "Triple dedup: %s -[%s]-> %s (confirmations=%d)",
                        s, r, o, data["confirmations"],
                    )
                    return

        # New edge
        tier = prov.get("tier", "derived")
        initial_score = (self._config.initial_score_source
                         if tier == "source"
                         else self._config.initial_score_derived)
        self._g.add_node(s)
        self._g.add_node(o)
        self._g.add_edge(
            s, o,
            relation=r,
            created_at=datetime.now(timezone.utc).isoformat(),
            score=initial_score,
            confirmations=1,
            **prov,
        )
        self._save()

    def entities_in_text(self, text: str) -> list[str]:
        """Return known graph nodes that appear as substrings in text."""
        text_lower = text.lower()
        return [n for n in self._g.nodes() if n and n in text_lower]

    def expand(self, entities: list[str], hops: int = 1, max_facts: int = 0) -> list[dict]:
        """Walk the graph up to `hops` from seed entities.

        Returns list of dicts with 'fact' (display string) and 'edge_id' (u, v, key).
        Triggers the decay sweep counter. Source-tier facts before derived-tier.
        """
        self._queries_since_sweep += 1
        if self._queries_since_sweep >= self._config.decay_every_n_queries:
            self._run_decay_sweep()
            self._queries_since_sweep = 0

        visited: set[str] = set()
        seen: set[str] = set()
        frontier: list[str] = [
            e for e in entities if e in self._g and not (e in seen or seen.add(e))  # type: ignore[func-returns-value]
        ]
        source_results: list[dict] = []
        derived_results: list[dict] = []

        for _ in range(hops):
            next_seen: set[str] = set()
            next_frontier: list[str] = []
            for node in frontier:
                if node in visited:
                    continue
                visited.add(node)
                for src, dst, key, data in self._g.out_edges(node, keys=True, data=True):
                    tier = data.get("tier", "?")
                    relation = data.get("relation", "?")
                    src_id = data.get("source_id", "?")
                    fact = f"{src} --[{relation}]--> {dst}  (source: {src_id}, tier: {tier})"
                    entry = {"fact": fact, "edge_id": (src, dst, key)}
                    if tier == "source":
                        source_results.append(entry)
                    else:
                        derived_results.append(entry)
                    if dst not in visited and dst not in next_seen:
                        next_frontier.append(dst)
                        next_seen.add(dst)
            frontier = next_frontier

        all_results = source_results + derived_results
        return all_results[:max_facts] if max_facts > 0 else all_results

    def record_access(self, edges: list[tuple]) -> None:
        """Bump score for edges that were included in the LLM context."""
        from grag.access import apply_access

        changed = False
        for edge in edges:
            u, v, key = edge
            if not self._g.has_edge(u, v, key):
                continue

            def _make_bump(u_: str, v_: str, k_: int):
                def bump(inc: float) -> None:
                    data = self._g[u_][v_][k_]
                    initial = (self._config.initial_score_source
                               if data.get("tier") == "source"
                               else self._config.initial_score_derived)
                    data["score"] = float(data.get("score", initial)) + inc
                return bump

            # TODO: wire neighbour_fn when spread.enabled is true
            # neighbour_fn = lambda: self._get_neighbour_bumps(u, v, key)
            apply_access(_make_bump(u, v, key), self._config.access_increment, self._config)
            changed = True

        if changed:
            self._save()

    def _run_decay_sweep(self, decay_multiplier: float = 1.0) -> dict:
        """Subtract decay from all edge scores; remove entries at or below cull_threshold."""
        config = self._config
        logger.info("Graph sweep firing (multiplier=%.2f, edges=%d)", decay_multiplier, self._g.number_of_edges())

        derived_edge_count = sum(
            1 for _, _, data in self._g.edges(data=True)
            if data.get("tier", "derived") != "source"
        )

        to_remove: list[tuple] = []
        to_update: list[tuple] = []

        for u, v, key, data in list(self._g.edges(keys=True, data=True)):
            tier = data.get("tier", "derived")
            if tier == "source" and not config.source_decays:
                continue
            confirmations = int(data.get("confirmations", 1))
            if confirmations >= config.confirmation_floor:
                continue

            initial = config.initial_score_source if tier == "source" else config.initial_score_derived
            score = float(data.get("score", initial))

            decay = config.decay_per_sweep * decay_multiplier
            if tier == "source":
                decay *= config.source_decay_factor
            if tier != "source" or not config.pressure_targets_derived_only:
                if derived_edge_count > config.soft_limit:
                    decay *= derived_edge_count / config.soft_limit

            new_score = score - decay
            if new_score <= config.cull_threshold:
                to_remove.append((u, v, key))
            else:
                to_update.append((u, v, key, new_score))

        if len(to_remove) > 100:
            self._write_snapshot(to_remove)

        for u, v, key, new_score in to_update:
            if self._g.has_edge(u, v, key):
                self._g[u][v][key]["score"] = new_score

        for u, v, key in to_remove:
            if self._g.has_edge(u, v, key):
                self._g.remove_edge(u, v, key)

        orphan_count = 0
        for n in list(self._g.nodes()):
            if self._g.degree(n) == 0:
                self._g.remove_node(n)
                orphan_count += 1

        if to_remove or to_update:
            self._save()

        logger.info("Graph sweep: dropped %d edges, %d orphan nodes", len(to_remove), orphan_count)
        return {"edges_removed": len(to_remove), "orphans_removed": orphan_count}

    def _write_snapshot(self, edges_to_remove: list[tuple]) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        backup_dir = Path(self._config.data_dir) / "backups" / f"sweep_{ts}"
        backup_dir.mkdir(parents=True, exist_ok=True)

        src_gexf = Path(self._path)
        if src_gexf.exists():
            shutil.copy2(src_gexf, backup_dir / "knowledge_graph.gexf")

        manifest = {"deleted_edges": [[u, v, k] for u, v, k in edges_to_remove]}
        (backup_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        backups_root = Path(self._config.data_dir) / "backups"
        old_sweeps = sorted(backups_root.glob("sweep_*"), key=lambda p: p.name)
        for old in old_sweeps[:-10]:
            shutil.rmtree(old, ignore_errors=True)

    def reset(self) -> None:
        """Replace the in-memory graph with an empty one and overwrite the file on disk."""
        self._g = nx.MultiDiGraph()
        self._queries_since_sweep = 0
        self._save()

    def stats(self) -> dict:
        """Return node/edge counts split by tier and sweep state."""
        source_edges = sum(1 for _, _, d in self._g.edges(data=True) if d.get("tier") == "source")
        derived_edges = self._g.number_of_edges() - source_edges
        return {
            "nodes": self._g.number_of_nodes(),
            "edges": self._g.number_of_edges(),
            "source_edges": source_edges,
            "derived_edges": derived_edges,
            "queries_since_sweep": self._queries_since_sweep,
            "sweep_every": self._config.decay_every_n_queries,
        }

    def _save(self) -> None:
        Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(self._g, self._path)
