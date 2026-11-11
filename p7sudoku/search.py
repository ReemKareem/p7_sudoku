"""Search part: find the cheapest set of digit corrections that makes the puzzle valid.

State  = tuple of (cell, new_digit), sorted by cell (each cell changed at most once).
Action = change one NOT-yet-changed uncertain cell to one of its top alternatives.
Cost   = max(0, log p(read) - log p(new))   (never negative); path cost is the SUM over cells,
         so it does not depend on the order of actions.
Goal   = the corrected puzzle is solvable by the CSP (optionally: exactly one solution).
Algorithms: UCS (optimal), Greedy best-first (h = #conflicts), A* (admissible h below).
"""
from __future__ import annotations
import heapq, itertools, math, time
from dataclasses import dataclass, field
from .csp import find_conflicts, solve

EPS = 1e-12
INF = float("inf")


@dataclass
class Correction:
    cell: int
    read_digit: int
    new_digit: int
    cost: float

    def as_row(self):
        return dict(cell=self.cell, row=self.cell // 9 + 1, col=self.cell % 9 + 1,
                    read=self.read_digit, new=self.new_digit, cost=round(self.cost, 4))


@dataclass
class SearchStats:
    expanded: int = 0
    generated: int = 0
    goal_tests: int = 0
    csp_calls: int = 0
    quick_rejects: int = 0
    max_frontier: int = 0
    time_s: float = 0.0


@dataclass
class SearchResult:
    algorithm: str
    found: bool
    corrections: list
    cost: float
    stats: SearchStats
    message: str = ""
    trace: list = field(default_factory=list)


class GoalChecker:
    """Goal test = CSP can solve the grid (cached). Duplicate-digit conflicts are rejected
    immediately (this is exactly what AC-3 would detect, without paying for a full run)."""

    def __init__(self, use_fc=True, use_ac3=True, unique=False, ordering="mrv"):
        self.use_fc, self.use_ac3, self.unique, self.ordering = use_fc, use_ac3, unique, ordering
        self.csp_calls = 0
        self.quick_rejects = 0
        self.cache = {}

    def __call__(self, grid) -> bool:
        key = tuple(grid)
        if key in self.cache:
            return self.cache[key]
        if find_conflicts(grid):
            self.quick_rejects += 1
            ok = False
        else:
            self.csp_calls += 1
            r = solve(grid, self.use_fc, self.use_ac3, self.ordering, 2 if self.unique else 1)
            n = len(r.solutions)
            ok = (n == 1) if self.unique else (n >= 1)
        self.cache[key] = ok
        return ok


class CorrectionProblem:
    def __init__(self, read_grid, probs, goal_fn, max_corr=3, n_uncertain=12, top_alts=3, max_cost=None):
        self.read = list(read_grid)
        self.max_cost = max_cost      # optional: refuse corrections that are too unlikely in total
        self.goal_fn = goal_fn
        self.max_corr = max_corr
        givens = [i for i in range(81) if self.read[i]]
        order = sorted(givens, key=lambda i: (float(probs[i][self.read[i] - 1]), i))
        self.cands = {}      # cell -> [(digit, cost)] ascending cost
        for c in order[:n_uncertain]:
            p, r = probs[c], self.read[c]
            alts = [d for d in sorted(range(9), key=lambda d: (-float(p[d]), d)) if d + 1 != r][:top_alts]
            lst = [(d + 1, max(0.0, math.log(max(float(p[r - 1]), EPS)) - math.log(max(float(p[d]), EPS))))
                   for d in alts]
            self.cands[c] = sorted(lst, key=lambda t: t[1])
        self.min_cost = {c: l[0][1] for c, l in self.cands.items()}

    # --- helpers ------------------------------------------------------------
    def grid_of(self, state):
        g = list(self.read)
        for c, d in state:
            g[c] = d
        return g

    def cost_of(self, state):
        return sum(next(k for dd, k in self.cands[c] if dd == d) for c, d in state)

    def corrections_of(self, state):
        return [Correction(c, self.read[c], d, next(k for dd, k in self.cands[c] if dd == d))
                for c, d in state]

    # --- heuristics ---------------------------------------------------------
    def h_greedy(self, state):
        return len(find_conflicts(self.grid_of(state)))

    def h_astar(self, state):
        """Admissible: every duplicate-digit conflict needs one of its two cells changed.
        Take cell-disjoint conflicts; each needs a DIFFERENT changed cell costing >= min cost.
        Returns INF if the state cannot be repaired within the correction budget."""
        conflicts = find_conflicts(self.grid_of(state))
        if not conflicts:
            return 0.0
        changed = {c for c, _ in state}
        opts = []
        for a, b in conflicts:
            costs = [self.min_cost[x] for x in (a, b) if x in self.cands and x not in changed]
            if not costs:
                return INF
            opts.append((min(costs), a, b))
        opts.sort()
        used, h, n = set(), 0.0, 0
        for c, a, b in opts:
            if a not in used and b not in used:
                used.update((a, b))
                h += c
                n += 1
        if n > self.max_corr - len(state):
            return INF
        return h


def search(problem: CorrectionProblem, algorithm="ucs", record_trace=False, max_expansions=None) -> SearchResult:
    t0 = time.perf_counter()
    st = SearchStats()
    cnt = itertools.count()
    goal = problem.goal_fn
    csp0, qr0 = getattr(goal, "csp_calls", 0), getattr(goal, "quick_rejects", 0)
    trace = []

    def prio(state, g):
        if algorithm == "ucs":
            return (g, 0.0)
        if algorithm == "astar":
            h = problem.h_astar(state)
            return None if h == INF else (g + h, g)
        if algorithm == "greedy":
            return (float(problem.h_greedy(state)), g)
        raise ValueError(algorithm)

    start = ()
    p0 = prio(start, 0.0) or (0.0, 0.0)
    frontier = [(*p0, next(cnt), 0.0, start)]
    seen = {start}
    st.generated = 1
    result = None
    while frontier:
        st.max_frontier = max(st.max_frontier, len(frontier))
        f1, f2, _, g, state = heapq.heappop(frontier)
        st.goal_tests += 1
        if goal(problem.grid_of(state)):
            if record_trace:
                trace.append(("GOAL", state, round(g, 4)))
            result = SearchResult(algorithm, True, problem.corrections_of(state), g, st,
                                  "Solvable puzzle found.")
            break
        st.expanded += 1
        if record_trace:
            trace.append(("expand", state, round(g, 4)))
        if max_expansions and st.expanded >= max_expansions:
            break
        if len(state) >= problem.max_corr:
            continue
        done = {c for c, _ in state}
        for c, lst in problem.cands.items():
            if c in done:
                continue
            for d, k in lst:
                ns = tuple(sorted(state + ((c, d),)))
                if ns in seen:
                    continue
                if problem.max_cost is not None and g + k > problem.max_cost:
                    continue
                seen.add(ns)
                st.generated += 1
                p = prio(ns, g + k)
                if p is None:      # A*: provably cannot be repaired within budget
                    continue
                heapq.heappush(frontier, (*p, next(cnt), g + k, ns))
    if result is None:
        result = SearchResult(algorithm, False, [], INF, st,
                              "No correction set within the budget makes the puzzle solvable.")
    st.time_s = time.perf_counter() - t0
    st.csp_calls = getattr(goal, "csp_calls", 0) - csp0
    st.quick_rejects = getattr(goal, "quick_rejects", 0) - qr0
    result.trace = trace
    return result


def brute_force(problem: CorrectionProblem):
    """Reference answer: enumerate EVERY correction set (<= max_corr cells) and keep the cheapest goal."""
    cells = list(problem.cands)
    best, best_state = INF, None
    for k in range(problem.max_corr + 1):
        for subset in itertools.combinations(cells, k):
            for digs in itertools.product(*[problem.cands[c] for c in subset]):
                state = tuple(sorted((c, d) for c, (d, _) in zip(subset, digs)))
                cost = sum(kk for _, kk in digs)
                if cost < best and problem.goal_fn(problem.grid_of(state)):
                    best, best_state = cost, state
    return best, best_state
