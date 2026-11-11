import math, random, unittest
import numpy as np
from p7sudoku.search import CorrectionProblem, GoalChecker, brute_force, search
from tests.helpers import PUZZLE, SOLUTION, conflicting_wrong_digit, synth


def make_problem(mis, max_corr=3, n_unc=8, alg_goal=None):
    wrong = {c: conflicting_wrong_digit(PUZZLE, c) for c in mis}
    read, probs = synth(PUZZLE, wrong)
    return CorrectionProblem(read, probs, alg_goal or GoalChecker(), max_corr, n_unc), read


class SearchTests(unittest.TestCase):
    def test_known_answer_from_project_brief(self):
        """NN reads 5 as w with p=.52 vs .45 for the true digit -> cost = ln(.52/.45) = 0.1445."""
        prob, read = make_problem([0])
        r = search(prob, "ucs")
        self.assertTrue(r.found); self.assertEqual(len(r.corrections), 1)
        c = r.corrections[0]
        self.assertEqual((c.cell, c.new_digit), (0, PUZZLE[0]))
        self.assertAlmostEqual(c.cost, math.log(0.52 / 0.45), places=4)

    def test_costs_never_negative(self):
        prob, _ = make_problem([0, 2])
        for lst in prob.cands.values():
            for _, k in lst:
                self.assertGreaterEqual(k, 0.0)

    def test_ucs_and_astar_match_brute_force(self):
        rng = random.Random(1)
        for _ in range(6):
            cells = rng.sample([i for i in range(81) if PUZZLE[i]], rng.choice([1, 2]))
            prob, _ = make_problem(cells, max_corr=2, n_unc=7)
            bf, _ = brute_force(prob)
            u, a = search(prob, "ucs"), search(prob, "astar")
            self.assertAlmostEqual(u.cost, bf, places=9)
            self.assertAlmostEqual(a.cost, bf, places=9)      # admissible heuristic keeps optimality

    def test_greedy_finds_a_solution_not_worse_than_infinity(self):
        prob, _ = make_problem([0, 2])
        g = search(prob, "greedy"); self.assertTrue(g.found)
        self.assertGreaterEqual(g.cost, search(prob, "ucs").cost - 1e-12)

    def test_counters(self):
        prob, _ = make_problem([0, 2])
        r = search(prob, "ucs")
        s = r.stats
        self.assertGreaterEqual(s.generated, s.expanded); self.assertGreaterEqual(s.goal_tests, 1)
        self.assertGreaterEqual(s.quick_rejects + s.csp_calls, 1)
        a = search(make_problem([0, 2])[0], "astar").stats
        self.assertLessEqual(a.expanded, s.expanded)           # A* never expands more than UCS here

    def test_budget_respected(self):
        prob, _ = make_problem([0, 20, 49, 70], max_corr=2)
        r = search(prob, "ucs")
        self.assertFalse(r.found)


if __name__ == "__main__":
    unittest.main()
