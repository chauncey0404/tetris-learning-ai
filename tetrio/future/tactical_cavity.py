from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from tetrio.future.search_cache import fast_landings
from tetrio.reachability import enumerate_tetrio_reachable_placements
from tetrio.ruleset import TETRIO_MOVEMENT
from tetris_ai.core.movement import clear_lines, lock_piece
from tetris_ai.core.spins.common import t_corner_counts
from tetris_ai.core.types import PieceState


def _game_profile():
    # V9.2+ has the explicit ranked Season 2 profile.  Keep a fail-soft
    # fallback for repositories that still expose only the older alias.
    import tetrio.ruleset as ruleset

    return getattr(
        ruleset,
        "TETRIO_TETRA_LEAGUE",
        getattr(ruleset, "TETRIO_MULTIPLAYER", None),
    )


@dataclass(frozen=True)
class TOpportunity:
    proxy_full: int = 0
    proxy_mini: int = 0
    exact_full: int = 0
    exact_mini: int = 0
    exact_max_lines: int = 0
    exact_checked: bool = False

    @property
    def proxy_total(self) -> int:
        return self.proxy_full + self.proxy_mini

    @property
    def exact_total(self) -> int:
        return self.exact_full + self.exact_mini


@dataclass(frozen=True)
class TSpinTarget:
    full: int = 0
    mini: int = 0
    lines: int = 0
    reachable: bool = False


def _proxy_t_geometry(
    board: np.ndarray,
    max_states: int,
    *,
    use_fast_cache: bool = True,
) -> TOpportunity:
    full = 0
    mini = 0
    for state in fast_landings(
        board,
        "T",
        max_states=max_states,
        use_cache=use_fast_cache,
    ):
        corners, front = t_corner_counts(board, state, TETRIO_MOVEMENT)
        if corners >= 3:
            if front >= 2:
                full += 1
            else:
                mini += 1
    return TOpportunity(proxy_full=full, proxy_mini=mini)


def scan_t_opportunities(
    board: np.ndarray,
    *,
    fast_max_states: int = 10_000,
    reference_max_states: int = 50_000,
    exact_if_proxy: bool = False,
    use_fast_cache: bool = True,
) -> TOpportunity:
    """Find reachable T-slot opportunities.

    Fast geometry + 3-corner counts are used as a cheap tactical proxy.
    Path-sensitive exact TETR.IO spin classification is only invoked when
    explicitly requested *and* a proxy opportunity exists.
    """
    proxy = _proxy_t_geometry(
        board,
        int(fast_max_states),
        use_fast_cache=use_fast_cache,
    )
    if not exact_if_proxy or proxy.proxy_total == 0:
        return proxy

    profile = _game_profile()
    if profile is None or getattr(profile, "spins", None) is None:
        return proxy

    by_geometry: dict[tuple, tuple[int, int]] = {}
    for placement in enumerate_tetrio_reachable_placements(
        board,
        "T",
        max_states=int(reference_max_states),
    ):
        locked = lock_piece(board, placement.landing_state, TETRIO_MOVEMENT)
        _, lines = clear_lines(locked, TETRIO_MOVEMENT)
        result = profile.spins.classify(
            board,
            placement,
            TETRIO_MOVEMENT,
            lines_cleared=int(lines),
        )
        kind = getattr(getattr(result, "kind", None), "value", "none")
        rank = 2 if kind == "full" else (1 if kind == "mini" else 0)
        key = placement.landing_state.geometry_key()
        old = by_geometry.get(key, (0, 0))
        if (rank, int(lines)) > old:
            by_geometry[key] = (rank, int(lines))

    exact_full = sum(rank == 2 for rank, _ in by_geometry.values())
    exact_mini = sum(rank == 1 for rank, _ in by_geometry.values())
    max_lines = max((lines for rank, lines in by_geometry.values() if rank), default=0)

    return TOpportunity(
        proxy_full=proxy.proxy_full,
        proxy_mini=proxy.proxy_mini,
        exact_full=int(exact_full),
        exact_mini=int(exact_mini),
        exact_max_lines=int(max_lines),
        exact_checked=True,
    )


def classify_target_t_spin(
    board_before: np.ndarray,
    target: PieceState,
    *,
    lines_cleared: int,
    reference_max_states: int = 50_000,
) -> TSpinTarget:
    """Path-sensitive classification for one T candidate geometry."""
    if target.piece != "T":
        return TSpinTarget()

    profile = _game_profile()
    if profile is None or getattr(profile, "spins", None) is None:
        return TSpinTarget(reachable=True)

    target_key = target.geometry_key()
    best_rank = 0
    best_lines = 0
    reachable = False

    for placement in enumerate_tetrio_reachable_placements(
        board_before,
        "T",
        max_states=int(reference_max_states),
    ):
        if placement.landing_state.geometry_key() != target_key:
            continue
        reachable = True
        result = profile.spins.classify(
            board_before,
            placement,
            TETRIO_MOVEMENT,
            lines_cleared=int(lines_cleared),
        )
        kind = getattr(getattr(result, "kind", None), "value", "none")
        rank = 2 if kind == "full" else (1 if kind == "mini" else 0)
        if rank > best_rank:
            best_rank = rank
            best_lines = int(lines_cleared)

    return TSpinTarget(
        full=int(best_rank == 2),
        mini=int(best_rank == 1),
        lines=int(best_lines if best_rank else 0),
        reachable=reachable,
    )
