import random, unittest
from p7sudoku.csp import solve, find_conflicts, is_valid_solution
from p7sudoku.generator import make_puzzle
from tests.helpers import PUZZLE, SOLUTION

CFG = [(False, False), (True, False), (False, True), (True, True)]


class CSPTests(unittest.TestCase):
    def test_all_configs_find_known_solution(self):
        for fc, ac in CFG:
            r = solve(PUZZLE, fc, ac)
            self.assertTrue(r.solved); self.assertEqual(r.solution, SOLUTION)

    def test_counters_fc_and_ac3_prune(self):
        p, _ = make_puzzle(random.Random(3), 28)
        bt, fc, full = (solve(p, False, False, "static"), solve(p, True, False, "static"), solve(p, True, True, "static"))
        self.assertGreater(bt.stats.assignments, fc.stats.assignments)
        self.assertGreaterEqual(fc.stats.assignments, full.stats.assignments)
        self.assertGreater(fc.stats.fc_prunings, 0); self.assertGreater(full.stats.ac3_prunings, 0)
        self.assertEqual(bt.stats.fc_prunings, 0); self.assertEqual(bt.stats.ac3_prunings, 0)

    def test_generated_puzzles_have_unique_solution(self):
        rng = random.Random(0)
        for n in (25, 30, 35):
            p, s = make_puzzle(rng, n)
            self.assertEqual(sum(1 for v in p if v), n)
            r = solve(p, True, True, "mrv", max_solutions=2)
            self.assertEqual(len(r.solutions), 1); self.assertEqual(r.solution, s)

    def test_ordering_static_and_mrv_agree(self):
        self.assertEqual(solve(PUZZLE, True, True, "static").solution, solve(PUZZLE, True, True, "mrv").solution)

    def test_node_cap_reports_cap(self):
        p, _ = make_puzzle(random.Random(5), 25)
        r = solve(p, False, False, "static", node_cap=5)
        self.assertEqual(r.status, "cap")


class CSPEdgeTests(unittest.TestCase):
    def test_edge_full_valid_grid(self):
        r = solve(SOLUTION); self.assertTrue(r.solved); self.assertEqual(r.solution, SOLUTION)

    def test_edge_empty_grid_has_many_solutions(self):
        r = solve([0] * 81, max_solutions=2)
        self.assertEqual(len(r.solutions), 2); self.assertTrue(is_valid_solution(r.solutions[0]))

    def test_edge_single_blank(self):
        g = list(SOLUTION); g[40] = 0
        self.assertEqual(solve(g, False, False).solution, SOLUTION)


class CSPImpossibleTests(unittest.TestCase):
    def test_impossible_duplicate_givens_ac3_detects_without_search(self):
        g = list(PUZZLE); g[1] = 5                       # two 5s in row 1
        r = solve(g, True, True)
        self.assertEqual(r.status, "no_solution"); self.assertTrue(r.ac3_detected)
        self.assertEqual(r.stats.assignments, 0); self.assertIn("No solution", r.message)
        for fc, ac in CFG:
            self.assertEqual(solve(g, fc, ac).status, "no_solution")
        self.assertTrue(find_conflicts(g))

    def test_impossible_cell_without_candidates(self):
        g = [0] * 81
        g[0:8] = [1, 2, 3, 4, 5, 6, 7, 8]; g[17] = 9     # cell 8 sees 1..8 in its row and 9 in its box
        for fc, ac in CFG:
            r = solve(g, fc, ac)
            self.assertEqual(r.status, "no_solution"); self.assertIn("No solution", r.message)


if __name__ == "__main__":
    unittest.main()
