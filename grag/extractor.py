from __future__ import annotations

import json
import logging
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from grag.llm import LLM

logger = logging.getLogger(__name__)

_TRIPLE_RE = re.compile(
    r'"subject"\s*:\s*"([^"]+)".*?"relation"\s*:\s*"([^"]+)".*?"object"\s*:\s*"([^"]+)"',
    re.DOTALL,
)


def extract_triples(text: str, llm: LLM) -> list[dict]:
    """Extract (subject, relation, object) triples from text using the LLM.

    Tries JSON parsing first; falls back to regex so malformed output still yields results.
    Values are stripped; any triple with a missing, None, non-string, or empty field is dropped.
    Capped at 10 triples per call.
    """
    raw = llm.extract_triples(text)
    candidates = _parse_json(raw) or _parse_regex(raw) or []
    triples = [_normalize(t) for t in candidates if _valid(t)]
    logger.debug("Extracted %d triples", len(triples))
    return triples[:10]


def _parse_json(raw: str) -> list[dict] | None:
    start = raw.find("[")
    end = raw.rfind("]") + 1
    if start == -1 or end == 0:
        return None
    try:
        parsed = json.loads(raw[start:end])
        if isinstance(parsed, list):
            return list(parsed)
    except (json.JSONDecodeError, ValueError):
        pass
    return None


def _parse_regex(raw: str) -> list[dict]:
    return [
        {"subject": m.group(1), "relation": m.group(2), "object": m.group(3)}
        for m in _TRIPLE_RE.finditer(raw)
    ]


def _valid(t: object) -> bool:
    """Return True only when all three fields are non-empty strings after stripping."""
    if not isinstance(t, dict):
        return False
    return all(
        isinstance(t.get(k), str) and bool(t[k].strip())
        for k in ("subject", "relation", "object")
    )


def _normalize(t: dict) -> dict:
    """Strip whitespace from all three fields."""
    return {k: t[k].strip() for k in ("subject", "relation", "object")}
