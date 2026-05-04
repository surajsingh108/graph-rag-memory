from __future__ import annotations

from typing import Callable, Optional


def apply_access(
    bump_fn: Callable[[float], None],
    increment: float,
    config,
    neighbour_fn: Optional[Callable[[], list]] = None,
    depth: int = 0,
) -> None:
    """
    Bump an entry's score by `increment`, then optionally spread to neighbours.

    bump_fn: callable that applies the score delta for this entry.
    neighbour_fn: returns a list of (bump_fn, neighbour_fn) pairs for neighbours.
      Provided by the caller; used only when spread is enabled.
    """
    bump_fn(increment)

    if not config.spread_enabled:
        return
    if depth >= config.spread_max_hops:
        return
    next_increment = increment * config.spread_factor
    if next_increment < config.spread_min_contribution:
        return
    if neighbour_fn is None:
        return
    for nbr_bump, nbr_neighbour_fn in neighbour_fn():
        apply_access(nbr_bump, next_increment, config, nbr_neighbour_fn, depth + 1)
