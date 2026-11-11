import random
import numpy as np
from p7sudoku.csp import PEERS

PUZZLE = [int(c) for c in "530070000600195000098000060800060003400803001700020006060000280000419005000080079"]
SOLUTION = [int(c) for c in "534678912672195348198342567859761423426853791713924856961537284287419635345286179"]


def conflicting_wrong_digit(grid, cell):
    """A wrong digit for `cell` that duplicates a given digit in one of its peers (so AC-3 sees a conflict)."""
    return next(d for d in sorted({grid[p] for p in PEERS[cell] if grid[p]}) if d != grid[cell])


def synth(grid, misread=None, conf=0.92, true_rank=2):
    """Probability matrix for a grid. misread = {cell: wrong_digit}: the network reads wrong_digit (p=.52)
    while the true digit gets p=.45 (rank 2) - the example from the project brief.
    true_rank=5 pushes the true digit down to the 5th place (cannot be recovered with top-3 alternatives)."""
    misread = misread or {}
    probs = np.zeros((81, 9), np.float32)
    read = list(grid)
    for c in range(81):
        if not grid[c]:
            continue
        p = np.full(9, (1 - conf) / 8, np.float32); p[grid[c] - 1] = conf
        if c in misread:
            w = misread[c]
            if true_rank == 2:
                p = np.full(9, 0.03 / 7, np.float32); p[w - 1] = 0.52; p[grid[c] - 1] = 0.45
            else:
                others = [d for d in range(9) if d not in (w - 1, grid[c] - 1)]
                p = np.zeros(9, np.float32); p[w - 1] = 0.6
                order = others[:3]
                for q, d in zip((0.2, 0.1, 0.07), order): p[d] = q
                for d in others[3:]: p[d] = 0.03 / len(others[3:])
                p[grid[c] - 1] = 0.001
                p /= p.sum()
            read[c] = w
        probs[c] = p
    return read, probs
