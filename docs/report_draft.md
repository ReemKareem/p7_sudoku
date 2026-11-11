# P7 - Handwritten Sudoku Reader and Solver (NN + Search + CSP)
Team seed 2024. DRAFT: every number below comes from `results/`; re-check them after you re-run, add names, and trim to 12 pages.

## 1. Problem
A neural network reads a 9x9 mosaic of handwritten digits. A misread digit can create two equal digits in one row, column or box, so the puzzle breaks.
The system finds the most likely set of corrections and solves the puzzle. Users: puzzle apps, education.
**The mosaics are built from MNIST digits; they are not photographs of real Sudoku.**

## 2. Data
See `docs/data_dictionary.md`. MNIST (real): the 60,000 training images are split with seed 0 into train and validation, then digit 0 is dropped
-> 45,072 train / 9,005 validation / 9,020 test (the 50,000/10,000 of the brief are the sizes before dropping 0). The test set is not used for training or tuning.
Generated: Sudoku grids (randomised backtracking), puzzles (cells removed while the solution stays unique; 25/30/35 givens), mosaics (test images of the right digit),
blank cells (rule: fewer than 10 pixels > 0.5, not the NN).
**Hard-digit injection:** for k misreads, k given cells show a test image that the calibrated NN misreads while the true digit is still in its top 3; all other cells show correctly read images.
So level k really means k misreads. This is an experimental device: the natural error rate of the NN is about 2.3%.

## 3. Neural network
NumPy MLP 784-128-64-9 (digits 1-9), compared with logistic regression. Temperature scaling (T = 1.387, fitted on validation).

| Model | Test acc. | NLL | ECE |
|---|---|---|---|
| Logistic regression | 0.9218 | 0.2727 | 0.0203 |
| NN (uncalibrated) | 0.9772 | 0.0756 | 0.0072 |
| NN + temperature scaling | 0.9772 | 0.0718 | 0.0036 |

Figure: `results/classifier.png` (confusion matrix, reliability diagram). Low-confidence bins have few samples and are noisy.
The calibrated probabilities are the correction costs, so calibration matters here.

## 4. Search
State = set of (cell, new digit), each cell changed once, at most 3 corrections. Action = change one more uncertain cell (12 least confident givens, top-3 alternatives).
Cost of one change = max(0, ln p(read) - ln p(new)); path cost = sum, so it does not depend on action order. Goal = the CSP solves the corrected puzzle (optional: exactly one solution).
UCS is optimal. Greedy uses h = number of remaining conflicts (not optimal). A* uses an admissible h: take cell-disjoint duplicate conflicts; each needs at least one changed cell that costs at least the cheapest alternative of that cell; INF if the budget cannot repair them.
Tests compare UCS and A* with brute force on small cases (`tests/test_search.py`).

Experiment 1 (100 puzzles per cell, 30 givens, mean ± std over 5 seeds; `results/e1_search.md`):

| misreads | UCS nodes (mean / median) | A* nodes (mean / median) | exact recovery UCS = A* | Greedy exact | cost UCS = A* | Greedy cost |
|---|---|---|---|---|---|---|
| 1 | 133.9 / 1 | 1.1 / 1 | 0.86 | 0.86 | 1.22 | 1.22 |
| 2 | 80.6 / 6 | 4.8 / 3 | 0.64 | 0.58 | 2.50 | 2.74 |
| 3 | 181.5 / 18 | 7.5 / 5 | 0.43 | 0.40 | 3.32 | 3.50 |
| 4 | 3673 / 6571 | 33.4 / 12.5 | 0.00 | 0.00 | 5.49 | 5.70 |

Reading: A* expands far fewer nodes than UCS for the same optimal cost; Greedy is cheap in nodes but pays a higher cost. The mean is dominated by a few hard puzzles (large std), so the median is also given.
At level 4 the budget (3) is too small by design; the median 6571 is the whole search space.

## 5. CSP
81 variables, domains 1-9, givens fixed, all-different on rows, columns and boxes. BT, FC and AC-3 are independent switches; counters: assignments, backtracks, AC-3 revisions/prunings, FC prunings.
Experiment 2 (static order, cap 200,000 assignments; `results/e2_csp.md`):

| givens | BT assignments | BT+FC | BT+FC+AC-3 |
|---|---|---|---|
| 25 | 41287 (solved 0.94) | 10925 | 5752 |
| 30 | 5523 | 1193 | 645 |
| 35 | 499 | 187 | 72 |

FC removes roughly 4-5x of the work and AC-3 about 2x more. AC-3 has its own cost: at 35 givens BT+FC is faster in seconds although AC-3 makes fewer assignments.
BT at 25 givens is stopped by the cap in 6% of the runs, so its mean is a lower bound.

## 6. Integration
NN reads 81 cells -> AC-3 detects the conflict -> Search tries the cheapest corrections -> the CSP tests each corrected puzzle; the first solvable one wins.
If none exists: `UNREADABLE` + the least confident cells. Correction log: `results/correction_log.csv` (cell, read digit, new digit, cost). Demo: `python run_demo.py` -> `results/demo.png`.
Experiment 3 (`results/e3_correction.md`), exact recovery for 1 / 2 / 3 misreads: NN-guided 0.84 / 0.66 / 0.49; NN-guided with unique goal 0.94 / 0.88 / 0.86; no correction 0 / 0 / 0; random correction 0.16 / 0 / 0.
E1 with the unique goal (`results/e1_search_unique.md`): UCS and A* reach 0.96 / 0.92 / 0.75.

## 7. Limits
1. "Solvable" is a weak goal: a wrong repair can also give a solvable puzzle (`solved_but_wrong` 0.12 / 0.35 / 0.55 for 1-3 misreads). Remedies: unique goal, `max_cost`; neither is perfect.
2. The true digit must be in the top-3 alternatives and the cell among the 12 least confident.
3. At most 3 corrections: level 4 cannot be recovered.
4. Mosaics come from MNIST, not photos; injected misreads are artificial.
5. Calibration is measured on MNIST only; Python speed limits the size of the experiments.

## 8. Conclusions
Calibrated probabilities turn misreads into a cost, A* finds the optimal correction with much less search than UCS, and AC-3/FC make the CSP check cheap. The unique-solution goal is the main protection against wrong repairs.

## Appendix
How to run: README. Git log: `git log --format='%h %an %ad %s' --date=short`. Contribution statements: `docs/contributions_template.md`.
