"""Deterministic client membership schedule for the churn experiments, and the split of active clients by graph coverage."""

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class ChurnEvent:
    round_idx: int
    left: List[int] = field(default_factory=list)
    joined: List[int] = field(default_factory=list)


class MembershipSchedule:
    def __init__(self, num_clients: int, rounds: int, seed: int, enabled: bool,
                 initial_active=None, interval: int = 47, n_leave: int = 0, n_join: int = 0,
                 protect_rounds: int = 0):
        self.events: List[ChurnEvent] = []
        if not enabled:
            self._active: Dict[int, List[int]] = {r: list(range(num_clients)) for r in range(1, rounds + 1)}
            return
        rng = np.random.default_rng(seed)
        interval = max(int(interval), 1)
        n_init = min(initial_active if initial_active is not None else num_clients, num_clients)
        active = set(range(n_init))
        reserve = set(range(n_init, num_clients))
        departed: set = set()
        self._active = {}
        for r in range(1, rounds + 1):
            if r > protect_rounds and r % interval == 0 and (n_leave or n_join):
                left: List[int] = []
                if n_leave and len(active) - n_leave >= 2:
                    left = sorted(rng.choice(sorted(active), size=n_leave, replace=False).tolist())
                    active -= set(left)
                    departed |= set(left)
                joined: List[int] = []
                if n_join:
                    pool = sorted(reserve) or sorted(departed)
                    k = min(n_join, len(pool))
                    if k:
                        joined = sorted(rng.choice(pool, size=k, replace=False).tolist())
                        active |= set(joined)
                        reserve -= set(joined)
                        departed -= set(joined)
                if left or joined:
                    self.events.append(ChurnEvent(r, left, joined))
            self._active[r] = sorted(active)

    def active_at(self, r: int) -> List[int]:
        return list(self._active[r])

    def changed_at(self, r: int) -> bool:
        return any(e.round_idx == r for e in self.events)


def partition_recoverable(active: Sequence[int], graph_vertices: Sequence[int], sampled: Sequence[int]):
    active_set = set(int(a) for a in active)
    graph_set = set(int(v) for v in graph_vertices)
    sampled_set = set(int(s) for s in sampled)
    observed = sorted(sampled_set & graph_set)
    offgraph = sorted(sampled_set - graph_set)
    interpolated = sorted((graph_set & active_set) - sampled_set)
    unrepresented = sorted(active_set - sampled_set - graph_set)
    return observed, offgraph, interpolated, unrepresented
