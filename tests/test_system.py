"""End-to-end tests of NN-reading -> AC-3 -> Search -> CSP with synthetic NN outputs.
NormalCases (5), EdgeCases (3), ImpossibleCases (2)."""
import unittest
from p7sudoku.csp import find_conflicts
from p7sudoku.integration import solve_with_correction
from tests.helpers import PUZZLE, SOLUTION, conflicting_wrong_digit, synth


def case(cells, **kw):
    wrong = {c: conflicting_wrong_digit(PUZZLE, c) for c in cells}
    read, probs = synth(PUZZLE, wrong, **{k: v for k, v in kw.items() if k == "true_rank"})
    return read, probs


class NormalCases(unittest.TestCase):
    def test_1_clean_reading_needs_no_correction(self):
        read, probs = synth(PUZZLE)
        o = solve_with_correction(read, probs)
        self.assertEqual(o["status"], "solved"); self.assertEqual(o["corrections"], []); self.assertEqual(o["solution"], SOLUTION)

    def test_2_one_misread_is_corrected(self):
        read, probs = case([0])
        self.assertTrue(find_conflicts(read))
        o = solve_with_correction(read, probs)
        self.assertEqual(o["status"], "solved"); self.assertEqual(o["solution"], SOLUTION)
        self.assertEqual([c.cell for c in o["corrections"]], [0]); self.assertTrue(o["ac3_conflict"])

    def test_3_two_misreads_are_corrected(self):
        read, probs = case([0, 20])
        o = solve_with_correction(read, probs)
        self.assertEqual(o["solution"], SOLUTION); self.assertEqual(len(o["corrections"]), 2)

    def test_4_all_algorithms_recover_the_solution(self):
        read, probs = case([0, 20])
        costs = {}
        for al in ("ucs", "greedy", "astar"):
            o = solve_with_correction(read, probs, al)
            self.assertEqual(o["status"], "solved"); costs[al] = o["cost"]
        self.assertAlmostEqual(costs["ucs"], costs["astar"], places=9)

    def test_5_unique_goal_and_switching_csp_parts(self):
        read, probs = case([0])
        for fc, ac in [(False, False), (True, False), (True, True)]:
            o = solve_with_correction(read, probs, use_fc=fc, use_ac3=ac, unique=True)
            self.assertEqual(o["solution"], SOLUTION)


class EdgeCases(unittest.TestCase):
    def test_1_full_grid_input(self):
        read, probs = synth(SOLUTION)
        o = solve_with_correction(read, probs)
        self.assertEqual(o["solution"], SOLUTION); self.assertEqual(o["corrections"], [])

    def test_2_all_blank_grid_is_solvable(self):
        o = solve_with_correction([0] * 81, [[0] * 9] * 81)
        self.assertEqual(o["status"], "solved"); self.assertEqual(o["corrections"], [])

    def test_3_budget_exactly_equals_number_of_misreads(self):
        read, probs = case([0, 20, 49])
        o = solve_with_correction(read, probs, max_corr=3, n_uncertain=10)
        self.assertEqual(o["solution"], SOLUTION); self.assertEqual(len(o["corrections"]), 3)


class ImpossibleCases(unittest.TestCase):
    def test_1_more_misreads_than_budget_is_unreadable(self):
        read, probs = case([0, 20, 49, 70])
        o = solve_with_correction(read, probs, max_corr=2)
        self.assertEqual(o["status"], "unreadable"); self.assertIn("UNREADABLE", o["message"])
        self.assertEqual(len(o["least_confident"]), 5); self.assertIsNone(o["solution"])

    def test_2_true_digit_outside_top3_alternatives_is_unreadable(self):
        read, probs = case([0], true_rank=5)
        o = solve_with_correction(read, probs)               # without a cost limit a WRONG repair is accepted...
        self.assertEqual(o["status"], "solved"); self.assertNotEqual(o["solution"], SOLUTION)
        o = solve_with_correction(read, probs, max_cost=3.0)  # ...a cost limit turns it into an honest 'unreadable'
        self.assertEqual(o["status"], "unreadable"); self.assertIn("UNREADABLE", o["message"])


if __name__ == "__main__":
    unittest.main()
