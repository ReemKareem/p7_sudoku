"""CSP part: 81 variables, domain 1-9, all-different on rows/cols/boxes.

Backtracking (BT), Forward Checking (FC) and AC-3 are SEPARATE switches:
    solve(grid, use_fc=False, use_ac3=False)  -> plain BT
    solve(grid, use_fc=True,  use_ac3=False)  -> BT + FC
    solve(grid, use_fc=True,  use_ac3=True)   -> BT + FC + AC-3
Domains are 9-bit masks (bit v-1 set  <=>  value v still possible).
Counters: assignments, backtracks, AC-3 revisions/prunings, FC prunings.
"""
from __future__ import annotations
import time
from collections import deque
from dataclasses import dataclass, field, asdict


def _build():
    units = [[r * 9 + c for c in range(9)] for r in range(9)]
    units += [[r * 9 + c for r in range(9)] for c in range(9)]
    for br in range(3):
        for bc in range(3):
            units.append([(br * 3 + i) * 9 + bc * 3 + j for i in range(3) for j in range(3)])
    peers = [[] for _ in range(81)]
    for u in units:
        for a in u:
            for b in u:
                if a != b and b not in peers[a]:
                    peers[a].append(b)
    return units, peers


UNITS, PEERS = _build()
ALL = 511
POP = [bin(i).count("1") for i in range(512)]


def bit(v: int) -> int:
    return 1 << (v - 1)


def values(mask: int):
    return [v for v in range(1, 10) if mask & (1 << (v - 1))]


@dataclass
class CSPStats:
    assignments: int = 0
    backtracks: int = 0
    ac3_revisions: int = 0
    ac3_prunings: int = 0
    fc_prunings: int = 0
    wipeouts: int = 0
    time_s: float = 0.0

    def as_dict(self):
        return asdict(self)


@dataclass
class CSPResult:
    status: str                      # 'solved' | 'no_solution' | 'cap'
    solutions: list = field(default_factory=list)
    stats: CSPStats = field(default_factory=CSPStats)
    message: str = ""
    ac3_detected: bool = False       # conflict found by AC-3 before any search

    @property
    def solved(self):
        return self.status == "solved"

    @property
    def solution(self):
        return self.solutions[0] if self.solutions else None


def find_conflicts(grid):
    """Pairs (a, b), a<b, of equal non-zero cells sharing a row/column/box."""
    out = set()
    for u in UNITS:
        seen = {}
        for c in u:
            v = grid[c]
            if v:
                if v in seen:
                    for o in seen[v]:
                        out.add((min(o, c), max(o, c)))
                    seen[v].append(c)
                else:
                    seen[v] = [c]
    return sorted(out)


def is_valid_solution(grid) -> bool:
    return all(sorted(grid[c] for c in u) == list(range(1, 10)) for u in UNITS)


def ac3(dom, stats: CSPStats) -> bool:
    """Arc consistency for the '!=' constraints. Returns False if a domain empties."""
    q = deque((xi, xj) for xi in range(81) for xj in PEERS[xi])
    while q:
        xi, xj = q.popleft()
        stats.ac3_revisions += 1
        dj = dom[xj]
        # a value of xi has support in xj unless xj is forced to that same value
        if dj & (dj - 1) == 0 and dom[xi] & dj:
            dom[xi] &= ~dj
            stats.ac3_prunings += 1
            if dom[xi] == 0:
                return False
            for xk in PEERS[xi]:
                if xk != xj:
                    q.append((xk, xi))
    return True


class _Cap(Exception):
    pass


def solve(grid, use_fc=True, use_ac3=True, ordering="mrv", max_solutions=1, node_cap=None) -> CSPResult:
    """ordering: 'static' (row-major) or 'mrv' (smallest legal domain first)."""
    t0 = time.perf_counter()
    stats = CSPStats()
    res = CSPResult("no_solution", [], stats)
    g = list(grid)
    dom = [bit(v) if v else ALL for v in g]

    def finish(status, msg=""):
        res.status, res.message = status, msg
        stats.time_s = time.perf_counter() - t0
        return res

    # --- preprocessing -----------------------------------------------------
    if use_ac3:
        if not ac3(dom, stats):
            res.ac3_detected = True
            return finish("no_solution", "No solution: AC-3 emptied a domain (conflicting givens), no search needed.")
    else:
        if find_conflicts(g):
            return finish("no_solution", "No solution: two equal givens share a row/column/box.")
        if use_fc:  # forward-check the givens once
            for i in range(81):
                if g[i]:
                    for p in PEERS[i]:
                        if not g[p] and dom[p] & bit(g[i]):
                            dom[p] &= ~bit(g[i])
                            stats.fc_prunings += 1
                            if dom[p] == 0:
                                stats.wipeouts += 1
                                return finish("no_solution", "No solution: forward checking emptied a domain.")

    def legal(i, dom_):
        if use_fc:
            return dom_[i]
        m = dom_[i]
        for p in PEERS[i]:
            if g[p]:
                m &= ~bit(g[p])
        return m

    def pick(dom_):
        best, bm, bc = -1, 0, 10
        for i in range(81):
            if g[i]:
                continue
            m = legal(i, dom_)
            if ordering == "static":
                return i, m
            c = POP[m]
            if c < bc:
                best, bm, bc = i, m, c
                if c <= 1:
                    break
        return best, bm

    def rec(dom_) -> bool:          # True = enough solutions found, stop
        i, m = pick(dom_)
        if i == -1:
            res.solutions.append(list(g))
            return len(res.solutions) >= max_solutions
        for v in values(m):
            stats.assignments += 1
            if node_cap and stats.assignments > node_cap:
                raise _Cap
            g[i] = v
            ok, nd = True, dom_
            if use_fc:
                nd = dom_[:]
                nd[i] = bit(v)
                b = bit(v)
                for p in PEERS[i]:
                    if not g[p] and nd[p] & b:
                        nd[p] &= ~b
                        stats.fc_prunings += 1
                        if nd[p] == 0:
                            stats.wipeouts += 1
                            ok = False
                            break
            if ok and rec(nd):
                g[i] = 0
                return True
            g[i] = 0
            stats.backtracks += 1
        return False

    try:
        rec(dom)
    except _Cap:
        return finish("cap", f"Stopped: assignment cap {node_cap} reached.")
    if res.solutions:
        return finish("solved", "Solved.")
    return finish("no_solution", "No solution: search space exhausted.")
