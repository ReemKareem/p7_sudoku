# P7 - Handwritten Sudoku Reader and Solver (NN + Search + CSP)

A neural network reads a 9x9 mosaic of handwritten (MNIST) digits. A misread digit can break the puzzle.
**AC-3** detects the conflict, **Search** (UCS / Greedy / A*) finds the cheapest set of digit corrections
(cost = how much less likely the new digit is), and the **CSP** (BT / FC / AC-3, separate switches) tests each
corrected puzzle - the first solvable one wins. If none exists: `UNREADABLE` + the least confident cells.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
## Run (exact commands)
```bash
python make_data.py                    # downloads MNIST (or put the 4 .gz files in data/raw/), seed 0
python train_nn.py --epochs 12         # NN + logistic regression + temperature scaling -> results/classifier.*
python -m unittest discover -s tests -t . -v      # 33 tests (also runs under pytest)
python run_demo.py --seed 0            # console walk-through + results/demo.png
python run_experiments.py --unique-e1  # 3 experiments (+ E1 again with the exactly-one-solution goal), 5 seeds x 20 puzzles (=100 per level) -> results/e*.{csv,md,png}
                                       # + results/correction_log.csv (cell, read digit, new digit, cost per puzzle)
```
MNIST is downloaded from a Google Storage mirror (the files match the official md5 checksums) and kept
locally in `data/raw/` (keep that folder - the brief asks for a local copy).

No internet? `python make_data.py --source digits` builds a tiny **dev-only** dataset (sklearn 8x8 digits).
Never report numbers from it - it is only for checking that the code runs.

## Team seed
Every team has its own generator seed: edit `TEAM_SEED` in `p7sudoku/config.py` (or pass `--team-seed N`)
and re-run the demo and the experiments. The seed used is printed in every results table.

## Layout
| Module | Content |
|---|---|
| `p7sudoku/data.py`, `make_data.py` | MNIST download/load, 50k/10k train/val split (seed) **then** digit 0 dropped -> 45,072 train / 9,005 val / 9,020 test |
| `p7sudoku/nn.py`, `train_nn.py` | numpy MLP 784-128-64-9, logistic-regression baseline, temperature scaling, ECE, confusion matrix |
| `p7sudoku/csp.py` | Sudoku CSP: `solve(grid, use_fc, use_ac3, ordering)`; counters: assignments, backtracks, AC-3 revisions/prunings, FC prunings |
| `p7sudoku/search.py` | UCS, Greedy, A* over correction sets; counters: expanded, generated, goal tests, CSP calls; brute-force reference |
| `p7sudoku/generator.py` | seeded puzzle generator (unique solution), mosaic builder, brightness blank-cell rule, hard-digit injection |
| `p7sudoku/integration.py` | NN read -> AC-3 -> Search -> CSP; baselines (no correction, random correction) |
| `p7sudoku/experiments.py`, `run_experiments.py` | Experiments 1-3, tables + plots |
| `tests/` | 5 normal + 3 edge + 2 impossible system tests, CSP/Search/NN/generator tests, search vs brute force |

## Design decisions (put these in the report)
* **State** = set of (cell, new digit); each cell changed at most once; max 3 corrections; cost per cell
  `max(0, ln p(read) - ln p(new))`, so path cost does not depend on the order of actions.
* **Candidates**: the 12 least confident given cells, top-3 alternative digits each (parameters `n_uncertain`, `top_alts`).
* **A\* heuristic** (admissible): each duplicate-digit conflict needs one of its two cells changed; take
  cell-disjoint conflicts, each costs at least the cheaper cell's cheapest alternative. States that need more
  corrections than the budget are pruned. UCS and A* return the same optimal cost (tested against brute force).
* **Greedy** orders by number of remaining conflicts (not optimal - the demo shows it can pay a much higher cost).
* **Goal test** = CSP solvable (optionally `unique=True`: exactly one solution). Equal digits in a unit are rejected
  immediately (this is what AC-3 would detect) so the full CSP only runs on conflict-free candidates.
* **Hard digits** (injection rule): for k injected misreads, the cell shows a test image of the right digit that the
  calibrated NN misreads while the true class is still in its top-3; all other cells show correctly read images.
  So "level k" really means k misreads. Blank cells are found by a brightness rule (fewer than 10 pixels > 0.5), not by the NN.
* **BT vs FC vs AC-3** are independent switches. Experiment 2 uses static (row-major) order so the effect of each part
  is visible; MRV ordering is available (`ordering="mrv"`) and used in the generator/search.

## Reading the result tables
* E1 reports the mean over seeds **and** `nodes_expanded_median` over all puzzles: a few hard puzzles dominate the mean (large std).
  At level 4 the median (6571) is the whole search space: no goal exists inside the budget, so every state is expanded.
* E1 also has `solved_but_wrong` (solvable puzzle, but not the true solution). `--unique-e1` repeats E1 with the exactly-one-solution goal
  (`results/e1_search_unique.*`): level 1: 0.86 -> 0.96, level 2: 0.64 -> 0.92, level 3: 0.43 -> 0.75 exact recovery (UCS and A*).
* E2 stops a run at 200,000 assignments; stopped runs count as unsolved, so BT at 25 givens is a lower bound. AC-3 cuts assignments
  but costs propagation time, so on easy puzzles (35 givens) BT+FC can be faster in seconds.
* The injected misreads are an experimental device (hard test digits chosen with the calibrated NN); they are not the natural error rate (~2.3%).

## Known limits (honest list)
1. "Solvable" is a weak goal: a wrong repair can also give a solvable (even unique) puzzle. Remedies in the code:
   `unique=True` and `max_cost` (refuse expensive corrections -> `UNREADABLE`). Experiment 3 reports `solved_but_wrong`.
2. Misreads where the true digit is outside the top-3 alternatives, or the cell is outside the 12 least confident, cannot be fixed.
3. With max 3 corrections, level-4 puzzles cannot be recovered completely (by design of the brief).
4. Mosaics are built from MNIST, not photographed Sudoku. Puzzles and injected hard digits are generated.
