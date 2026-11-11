"""Seeded generator: full grid (randomised backtracking) -> unique puzzle -> mosaic of MNIST images
-> controlled injection of 'hard' (misread) digits.  All randomness comes from random.Random(seed)."""
from __future__ import annotations
import random
from dataclasses import dataclass
import numpy as np
from .csp import PEERS, solve


def random_full_grid(rng: random.Random):
    g = [0] * 81

    def rec(i):
        if i == 81:
            return True
        digits = list(range(1, 10)); rng.shuffle(digits)
        for v in digits:
            if all(g[p] != v for p in PEERS[i] if p < i):
                g[i] = v
                if rec(i + 1):
                    return True
                g[i] = 0
        return False

    rec(0)
    return g


def make_puzzle(rng: random.Random, n_givens: int, max_restarts=60):
    """Remove cells one by one while the solution stays unique. Returns (puzzle, solution)."""
    for _ in range(max_restarts):
        sol = random_full_grid(rng)
        grid, n = sol[:], 81
        cells = list(range(81)); rng.shuffle(cells)
        for c in cells:
            if n <= n_givens:
                break
            v = grid[c]; grid[c] = 0
            if len(solve(grid, True, True, "mrv", max_solutions=2).solutions) == 1:
                n -= 1
            else:
                grid[c] = v
        if n == n_givens:
            return grid, sol
    raise RuntimeError(f"could not build a unique puzzle with {n_givens} givens")


class ImagePool:
    """Index pools over a labelled image set whose class probabilities are known.
    correct[d]: images of true digit d that the NN reads correctly.
    hard[d]   : images of digit d the NN MISREADS (true class still in its top-3) - the documented
                'hard digit' rule. If fewer than `min_hard` exist, the lowest-p(true) images are added."""

    def __init__(self, labels_1to9, probs, min_hard=5):
        pred = probs.argmax(1) + 1
        self.correct, self.hard = {}, {}
        for d in range(1, 10):
            idx = np.where(labels_1to9 == d)[0]
            ok = idx[pred[idx] == d]
            bad = [i for i in idx if pred[i] != d and (probs[i] > probs[i][d - 1]).sum() <= 2]
            if len(bad) < min_hard:
                extra = sorted((i for i in idx if i not in bad), key=lambda i: probs[i][d - 1])
                bad = bad + extra[:min_hard - len(bad)]
            self.correct[d] = list(ok)
            self.hard[d] = list(bad)


@dataclass
class Instance:
    puzzle: list          # true givens (0 = blank)
    solution: list
    img_idx: list         # test-image index per cell, -1 for blank
    read_grid: list       # what the NN read (0 = blank)
    probs: np.ndarray     # (81, 9) NN probabilities (zeros for blanks)
    mis_cells: list       # cells that really are misread
    n_injected: int


def make_instance(rng: random.Random, pool: ImagePool, test_probs, puzzle, solution, k_mis) -> Instance:
    givens = [i for i in range(81) if puzzle[i]]
    inj = set(rng.sample(givens, k_mis))
    img = [-1] * 81
    probs = np.zeros((81, 9), np.float32)
    read = [0] * 81
    for c in givens:
        d = solution[c]
        img[c] = rng.choice(pool.hard[d] if c in inj else pool.correct[d])
        probs[c] = test_probs[img[c]]
        read[c] = int(probs[c].argmax()) + 1
    mis = [c for c in givens if read[c] != puzzle[c]]
    return Instance(list(puzzle), list(solution), img, read, probs, mis, k_mis)


# ---- mosaic helpers -------------------------------------------------------------------------
def build_mosaic(x_test, img_idx, rng=None, noise=0.02):
    """81 cells -> (81,28,28) images; blank cells get faint noise (so blank detection is a real rule)."""
    nrng = np.random.default_rng(0 if rng is None else rng.randrange(1 << 30))
    cells = np.zeros((81, 28, 28), np.float32)
    for c, i in enumerate(img_idx):
        cells[c] = x_test[i].reshape(28, 28) if i >= 0 else np.clip(nrng.normal(0, noise, (28, 28)), 0, 1)
    return cells


def is_blank(cell_img, ink_pixels=10):
    """Brightness rule (NOT the NN): a cell is blank if fewer than 10 pixels are brighter than 0.5."""
    return int((cell_img > 0.5).sum()) < ink_pixels


def mosaic_image(cells):
    rows = [np.concatenate(list(cells[r * 9:(r + 1) * 9]), axis=1) for r in range(9)]
    return np.concatenate(rows, axis=0)
