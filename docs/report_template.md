# Report skeleton (max 12 pages) - fill in with YOUR results from `results/`

1. **Problem** (1 p): handwritten Sudoku, NN misreads break puzzles, goal = most likely corrected puzzle. Real users: puzzle apps, education. Say clearly: mosaics are built from MNIST, not photographed Sudoku.
2. **Data** (1 p): data dictionary (real vs generated), split, hard-digit injection rule, blank rule.
3. **Neural network** (1.5 p): architecture, training, `results/classifier.md` (NN vs logistic regression: accuracy, NLL, ECE before/after temperature scaling), confusion matrix, reliability diagram.
4. **Search** (2 p): state/action/cost/goal, UCS vs Greedy vs A*, the admissible heuristic and why it is admissible, brute-force check, Experiment 1 table + plot.
5. **CSP** (2 p): variables/domains/constraints, BT, FC, AC-3 as separate switches, counters, Experiment 2 table + plot.
6. **Integration** (1.5 p): NN -> AC-3 -> Search -> CSP flow, correction log example, "unreadable" behaviour, Experiment 3 (NN-guided vs none vs random, unique goal).
7. **Limits** (1.5 p): weak "solvable" goal (wrong-but-solvable repairs, `max_cost`, `unique`), top-3/12-cell candidate limit, 3-correction budget, MNIST mosaics, calibration quality, python speed.
8. **Conclusions** (0.5 p).
9. Appendix: how to run, git log, individual contribution statements (max 200 words each).

## 10-minute talk + 5-minute live demo
Talk: problem (1) - NN + calibration (2) - search (2) - CSP (2) - integration + results (2) - limits (1).
Live demo (`python run_demo.py`): (a) search trace, (b) CSP pruning table BT / BT+FC / BT+AC-3 / BT+FC+AC-3,
(c) correction ON vs OFF ("UNREADABLE" vs solved), (d) the figure `results/demo.png`.
