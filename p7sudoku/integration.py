"""NN reads grid -> AC-3 detects conflict -> Search tries cheapest corrections -> CSP tests each one."""
from __future__ import annotations
import random
import numpy as np
from .csp import ac3, bit, ALL, CSPStats, find_conflicts, solve
from .search import CorrectionProblem, GoalChecker, search, Correction


def read_cells(cells, model, T=1.0):
    """cells (81,28,28) -> read_grid, probs. Blank cells found by brightness rule, digits by the NN."""
    from .generator import is_blank
    probs = np.zeros((81, 9), np.float32)
    read = [0] * 81
    ink = [i for i in range(81) if not is_blank(cells[i])]
    if ink:
        p = model.proba(cells[ink].reshape(len(ink), 784), T)
        for j, i in enumerate(ink):
            probs[i] = p[j]
            read[i] = int(p[j].argmax()) + 1
    return read, probs


def ac3_conflict_report(read_grid):
    dom = [bit(v) if v else ALL for v in read_grid]
    stats = CSPStats()
    ok = ac3(dom, stats)
    return dict(ac3_conflict=not ok, conflict_pairs=find_conflicts(read_grid), ac3_stats=stats)


def least_confident(read_grid, probs, k=5):
    cells = sorted((i for i in range(81) if read_grid[i]), key=lambda i: probs[i][read_grid[i] - 1])[:k]
    return [dict(cell=i, row=i // 9 + 1, col=i % 9 + 1, read=read_grid[i],
                 confidence=round(float(probs[i][read_grid[i] - 1]), 3)) for i in cells]


def solve_with_correction(read_grid, probs, algorithm="ucs", max_corr=3, n_uncertain=12, top_alts=3,
                          use_fc=True, use_ac3=True, unique=False, record_trace=False, max_cost=None):
    rep = ac3_conflict_report(read_grid)
    goal = GoalChecker(use_fc, use_ac3, unique)
    prob = CorrectionProblem(read_grid, probs, goal, max_corr, n_uncertain, top_alts, max_cost)
    sr = search(prob, algorithm, record_trace)
    out = dict(algorithm=algorithm, ac3_conflict=rep["ac3_conflict"], conflict_pairs=rep["conflict_pairs"],
               search=sr, search_stats=sr.stats, corrections=sr.corrections, cost=sr.cost,
               solution=None, least_confident=[], trace=sr.trace)
    if sr.found:
        grid = prob.grid_of(tuple((c.cell, c.new_digit) for c in sr.corrections))
        res = solve(grid, use_fc, use_ac3, "mrv", 1)
        out.update(status="solved", solution=res.solution, corrected_grid=grid, csp_stats=res.stats,
                   message=f"Solved with {len(sr.corrections)} correction(s), total cost {sr.cost:.3f}.")
    else:
        out.update(status="unreadable", least_confident=least_confident(read_grid, probs),
                   message="UNREADABLE: no solvable puzzle within the correction budget. "
                           "Least confident cells are listed.")
    return out


def no_correction(read_grid, use_fc=True, use_ac3=True):
    r = solve(read_grid, use_fc, use_ac3)
    return dict(status="solved" if r.solved else "unreadable", solution=r.solution, corrections=[],
                message=r.message)


def random_correction(read_grid, probs, rng: random.Random, max_corr=3, budget=200, use_fc=True, use_ac3=True):
    """Baseline: random cells -> random digits, `budget` random tries, first solvable puzzle wins."""
    goal = GoalChecker(use_fc, use_ac3)
    givens = [i for i in range(81) if read_grid[i]]
    if goal(read_grid):
        return dict(status="solved", solution=solve(read_grid).solution, corrections=[], tries=0)
    for t in range(1, budget + 1):
        k = rng.randint(1, max_corr)
        g = list(read_grid)
        for c in rng.sample(givens, k):
            g[c] = rng.choice([d for d in range(1, 10) if d != read_grid[c]])
        if goal(g):
            return dict(status="solved", solution=solve(g).solution, corrections=[], tries=t)
    return dict(status="unreadable", solution=None, corrections=[], tries=budget)
