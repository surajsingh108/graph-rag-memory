"""Command-line demo for graph-rag-memory.

Usage:
    python examples/demo.py

Commands:
    :graph   — print current knowledge graph stats
    :reset   — wipe all memory and the graph, then re-ingest the sample doc
    :compact — force an aggressive decay sweep and print before/after counts
    :stats   — print a full diagnostic snapshot
    :quit    — exit
"""
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("grag").setLevel(logging.INFO)

sys.path.insert(0, str(Path(__file__).parent.parent))

from grag import RAG, Config
from grag.ingest import ingest

EINSTEIN_DOC = Path(__file__).parent / "sample_docs" / "einstein.txt"
YAML_PATH = Path(__file__).parent.parent / "grag_params.yaml"


def _fmt_age(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def print_stats(s: dict) -> None:
    mem = s["memory"]
    g = s["graph"]

    print(
        f"Sweep counters: memory {mem['queries_since_sweep']}/{mem['sweep_every']}, "
        f"graph {g['queries_since_sweep']}/{g['sweep_every']}"
    )

    for tier_key, label in [("sources", "Sources"), ("derived", "Derived")]:
        t = mem[tier_key]
        count = t.get("count", 0)
        if count:
            print(
                f"{label}: {count} chunks, "
                f"score min/median/max: "
                f"{t.get('score_min', 0):.1f} / {t.get('score_median', 0):.1f} / {t.get('score_max', 0):.1f}"
            )
        else:
            print(f"{label}: 0 chunks")

    print(
        f"Graph:   {g['nodes']} nodes, {g['edges']} edges "
        f"({g['source_edges']} source-tier, {g['derived_edges']} derived-tier)"
    )

    dist = mem["derived"].get("confirmations_dist", {})
    if dist:
        print("Confirmations distribution (derived):")
        print(f"  1: {dist.get(1, 0)} entries")
        print(f"  2-4: {dist.get('2-4', 0)} entries")
        print(f"  5+: {dist.get('5+', 0)} entries (decay-exempt)")

    cache = mem["cache"]
    oldest = cache.get("oldest_seconds", 0.0)
    newest = cache.get("newest_seconds", 0.0)
    if cache["count"]:
        print(f"Cache: {cache['count']} entries, oldest {_fmt_age(oldest)}, newest {_fmt_age(newest)}")
    else:
        print("Cache: 0 entries")

    derived_count = mem["derived"].get("count", 0)
    soft_limit = mem.get("sweep_every", 10)  # fallback; pressure_pct is better
    print(
        f"Memory pressure: {derived_count} / {mem.get('soft_limit', '?')} derived chunks "
        f"({mem.get('pressure_pct', 0.0):.1f}% of soft limit)"
    )
    print(f"Spread activation: {'enabled' if s['spread_enabled'] else 'disabled'}")


def print_compact(result: dict) -> None:
    mem = result["memory"]
    g = result["graph"]
    print(
        f"Memory:  sources {mem['before']['sources']} → {mem['after']['sources']}, "
        f"derived {mem['before']['derived']} → {mem['after']['derived']}"
    )
    print(
        f"Graph:   nodes {g['before']['nodes']} → {g['after']['nodes']}, "
        f"edges {g['before']['edges']} → {g['after']['edges']}"
    )
    dist = result.get("score_dist", {})
    for tier in ("sources", "derived"):
        d = dist.get(tier, {})
        if d.get("score_min") is not None:
            print(
                f"  {tier} scores after: "
                f"min={d['score_min']:.1f} median={d['score_median']:.1f} max={d['score_max']:.1f}"
            )


def main() -> None:
    if YAML_PATH.exists():
        config = Config.from_yaml(YAML_PATH)
        logging.getLogger("grag").info("Loaded config from %s", YAML_PATH)
    else:
        config = Config()
        logging.getLogger("grag").info("Using default config (no grag_params.yaml found)")

    rag = RAG(config)

    if rag.memory.is_empty():
        print(f"Ingesting {EINSTEIN_DOC.name} into memory …")
        ingest(EINSTEIN_DOC, rag)
        print("Ingestion complete.\n")

    gs = rag.graph.stats()
    print(f"Knowledge graph ready: {gs['nodes']} nodes, {gs['edges']} edges")
    print("Ask a question, or type :graph / :reset / :compact / :stats / :quit.\n")

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not user_input:
            continue

        if user_input == ":quit":
            break

        if user_input == ":graph":
            s = rag.graph.stats()
            print(f"Nodes: {s['nodes']}  Edges: {s['edges']}  "
                  f"(source: {s['source_edges']}, derived: {s['derived_edges']})")
            continue

        if user_input == ":stats":
            print_stats(rag.stats())
            print()
            continue

        if user_input == ":compact":
            print("Running compact sweep …")
            result = rag.compact()
            print_compact(result)
            print()
            continue

        if user_input == ":reset":
            confirm = input(
                "Are you sure? This wipes all memory and the graph. (y/N): "
            ).strip().lower()
            if confirm in ("y", "yes"):
                result_reset = rag.reset()
                print(result_reset)
                print(f"\nRe-ingesting {EINSTEIN_DOC.name} …")
                ingest(EINSTEIN_DOC, rag)
                s = rag.graph.stats()
                print(f"Ready — {s['nodes']} nodes, {s['edges']} edges.\n")
            else:
                print("Aborted.")
            continue

        result = rag.query(user_input)

        if result.get("cache_hit"):
            print(f"\n[cached] {result['answer']}\n")
        else:
            print(f"\n{result['answer']}\n")

        if result["context_used"]:
            sources = [
                f"{c['metadata'].get('file') or c['metadata'].get('source', 'derived:unknown')} [{c['tier']}]"
                for c in result["context_used"]
            ]
            print(f"Based on: {', '.join(sources)}")

        if result["facts_used"]:
            print(f"Graph facts ({len(result['facts_used'])} used):")
            for fact in result["facts_used"][:5]:
                print(f"  {fact}")

        print()


if __name__ == "__main__":
    main()
