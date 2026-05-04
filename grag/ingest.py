from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from grag.extractor import extract_triples

if TYPE_CHECKING:
    from grag.rag import RAG

logger = logging.getLogger(__name__)


def ingest(path: str | Path, rag: RAG) -> None:
    """Ingest a .txt, .md, or .pdf file into memory and the knowledge graph.

    Idempotent: chunks are keyed by file hash + index, so re-ingesting the same file
    adds nothing new to the vector store.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    file_hash = _file_hash(path)
    text = _load_text(path)
    if not text.strip():
        logger.warning("No text extracted from %s", path)
        return

    chunks = _chunk(text, rag.config.chunk_size, rag.config.chunk_overlap)
    metadatas = [
        {"file": path.name, "file_hash": file_hash, "chunk_idx": i}
        for i in range(len(chunks))
    ]

    rag.memory.add_source(chunks, metadatas)
    logger.info("Stored %d chunks from %s", len(chunks), path.name)

    for i, chunk in enumerate(chunks):
        triples = extract_triples(chunk, rag.llm)
        for t in triples:
            rag.graph.add_triple(
                t["subject"], t["relation"], t["object"],
                source_id=f"{path.name}:{i}",
                tier="source",
                confidence=1.0,
            )
        print(f"  Chunk {i + 1}/{len(chunks)}: extracted {len(triples)} triples")


def _file_hash(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def _load_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8", errors="ignore")
    if ext == ".pdf":
        from pdfminer.high_level import extract_text
        return extract_text(str(path))
    raise ValueError(f"Unsupported file type: {ext}")


def _chunk(text: str, size: int, overlap: int) -> list[str]:
    step = max(1, size - overlap)
    return [text[i: i + size] for i in range(0, len(text), step)]
