from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from grag.config import Config
from grag.extractor import extract_triples
from grag.graph import KnowledgeGraph
from grag.llm import LLM
from grag.memory import Memory

logger = logging.getLogger(__name__)


class RAG:
    """Orchestrates hybrid vector+graph retrieval with tiered memory."""

    def __init__(self, config: Optional[Config] = None) -> None:
        self.config = config or Config()
        self._llm: Optional[LLM] = None
        self._memory: Optional[Memory] = None
        self._graph: Optional[KnowledgeGraph] = None
        self._conversation_buffer: list[dict] = []

    @property
    def llm(self) -> LLM:
        if self._llm is None:
            self._llm = LLM(self.config.llm_model)
        return self._llm

    @property
    def memory(self) -> Memory:
        if self._memory is None:
            self._memory = Memory(self.config)
        return self._memory

    @property
    def graph(self) -> KnowledgeGraph:
        if self._graph is None:
            self._graph = KnowledgeGraph(self.config)
        return self._graph

    def query(self, question: str) -> dict:
        """Answer a question using hybrid retrieval, then feed the summary back into memory.

        Returns a dict with keys:
            answer        – full LLM response
            context_used  – list of chunk dicts from memory.query()
            facts_used    – list of fact strings from the knowledge graph
            cache_hit     – True if the answer was served from cache
        """
        q_hash = hashlib.md5(question.strip().lower().encode()).hexdigest()[
            : self.config.cache_hash_length
        ]

        # 1. Question-level cache check
        cached_answer, cached_id = self.memory.get_cached_answer(q_hash)
        if cached_answer is not None:
            logger.info("Cache hit for q_hash=%s", q_hash)
            self.memory.record_access([{"id": cached_id, "tier": "derived"}])
            return {
                "answer": cached_answer,
                "context_used": [],
                "facts_used": [],
                "cache_hit": True,
            }

        # 2. Vector retrieval — prepend last exchange for follow-up context
        if self._conversation_buffer:
            last = self._conversation_buffer[-1]
            retrieval_query = f"{last['question']} {last['answer'][:150]} {question}"
        else:
            retrieval_query = question
        chunks = self.memory.query(retrieval_query, k=self.config.top_k)

        # 3. Relevance gate: skip graph walk when retrieved context is off-topic
        top_score = chunks[0]["score"] if chunks else 0.0
        fact_entries: list[dict] = []
        if top_score < self.config.min_relevance:
            logger.info(
                "Top chunk score %.3f below min_relevance %.3f — skipping graph walk",
                top_score, self.config.min_relevance,
            )
        else:
            entity_scores: dict[str, float] = {}
            for chunk in chunks:
                for entity in self.graph.entities_in_text(chunk["text"]):
                    if entity not in entity_scores or chunk["score"] > entity_scores[entity]:
                        entity_scores[entity] = chunk["score"]
            ordered_entities = sorted(
                entity_scores, key=lambda e: entity_scores[e], reverse=True
            )
            fact_entries = self.graph.expand(
                ordered_entities, hops=1, max_facts=self.config.max_facts
            )

        facts_strings = [f["fact"] for f in fact_entries]
        used_edge_ids = [f["edge_id"] for f in fact_entries]

        # 4. Build prompt
        context_block = "\n\n".join(
            f"[{c['tier'].upper()}] {c['text']}" for c in chunks
        ) or "No passages retrieved."
        facts_block = "\n".join(facts_strings) or "No graph facts found."

        prompt = (
            "You are a knowledgeable assistant. Answer the question using only the "
            "retrieved passages and graph facts provided below.\n\n"
            "=== Retrieved passages ===\n"
            f"{context_block}\n\n"
            "=== Related facts from knowledge graph ===\n"
            f"{facts_block}\n\n"
            "Every factual claim in your answer must be supported by either a retrieved "
            "passage or a graph fact above. If the context does not contain enough to "
            "answer, say so explicitly. Do not introduce facts that are not in the context, "
            "even if you know them to be true from training.\n\n"
            f"Question: {question}\nAnswer:"
        )

        # 5. Generate answer
        answer = self.llm.answer(prompt)

        # 6. Record access for chunks and edges that went into the prompt
        self.memory.record_access(chunks)
        self.graph.record_access(used_edge_ids)

        # 7. Summarise → store in derived tier
        summary = self.llm.summarise(answer)
        self.memory.add_derived(
            summary,
            {
                "source": f"derived:q{q_hash}",
                "question": question,
                "question_hash": q_hash,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source_chunks": str([c["metadata"].get("file", "?") for c in chunks]),
            },
        )

        # 8. Extract triples from summary → store in graph as derived
        derived_triples = extract_triples(summary, self.llm)
        for t in derived_triples:
            self.graph.add_triple(
                t["subject"], t["relation"], t["object"],
                source_id=f"query:{question[:50]}",
                tier="derived",
                confidence=0.7,
            )

        # 9. Update conversation buffer (in-memory only, resets on restart)
        self._conversation_buffer.append({"question": question, "answer": answer[:300]})
        if len(self._conversation_buffer) > self.config.conversation_buffer_size:
            self._conversation_buffer.pop(0)

        return {
            "answer": answer,
            "context_used": chunks,
            "facts_used": facts_strings,
            "cache_hit": False,
        }

    def compact(self) -> dict:
        """Force an aggressive one-shot decay sweep across memory and graph."""
        multiplier = self.config.compact_decay_multiplier

        mem_stats_before = self.memory.stats()
        graph_stats_before = self.graph.stats()

        mem_result = self.memory._run_decay_sweep(decay_multiplier=multiplier)
        graph_result = self.graph._run_decay_sweep(decay_multiplier=multiplier)

        mem_stats_after = self.memory.stats()
        graph_stats_after = self.graph.stats()

        return {
            "memory": {
                "before": {
                    "sources": mem_stats_before["sources"]["count"],
                    "derived": mem_stats_before["derived"]["count"],
                },
                "after": {
                    "sources": mem_stats_after["sources"]["count"],
                    "derived": mem_stats_after["derived"]["count"],
                },
                "deleted": mem_result,
            },
            "graph": {
                "before": {
                    "nodes": graph_stats_before["nodes"],
                    "edges": graph_stats_before["edges"],
                },
                "after": {
                    "nodes": graph_stats_after["nodes"],
                    "edges": graph_stats_after["edges"],
                },
                "deleted": graph_result,
            },
            "score_dist": {
                "sources": {
                    k: mem_stats_after["sources"].get(k)
                    for k in ("score_min", "score_median", "score_max")
                },
                "derived": {
                    k: mem_stats_after["derived"].get(k)
                    for k in ("score_min", "score_median", "score_max")
                },
            },
        }

    def stats(self) -> dict:
        """Collect and return a full diagnostic snapshot."""
        return {
            "memory": self.memory.stats(),
            "graph": self.graph.stats(),
            "spread_enabled": self.config.spread_enabled,
        }

    def reset(self) -> dict:
        """Wipe all memory and graph state so the system starts fresh."""
        sources_cleared, derived_cleared = self.memory.reset()
        self.graph.reset()
        self._conversation_buffer.clear()
        ingest_log = Path("ingested_files.txt")
        ingest_log_cleared = False
        if ingest_log.exists():
            ingest_log.unlink()
            ingest_log_cleared = True
        return {
            "sources_cleared": sources_cleared,
            "derived_cleared": derived_cleared,
            "graph_cleared": True,
            "ingest_log_cleared": ingest_log_cleared,
        }
