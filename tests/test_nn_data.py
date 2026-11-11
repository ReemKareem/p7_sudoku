import os, random, tempfile, unittest
import numpy as np
from p7sudoku.csp import solve
from p7sudoku.generator import build_mosaic, is_blank, make_puzzle, ImagePool, make_instance
from p7sudoku.nn import MLP, softmax, ece, fit_temperature, nll, accuracy, confusion_matrix


class NNTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(0)
        self.y = rng.integers(0, 9, 900)
        self.x = (np.eye(9)[self.y] @ rng.normal(0, 1, (9, 784)) + rng.normal(0, 1.2, (900, 784))).astype(np.float32)

    def test_trains_and_beats_chance(self):
        m = MLP([784, 32, 9]); m.fit(self.x[:700], self.y[:700], self.x[700:], self.y[700:], epochs=5, verbose=False)
        self.assertGreater(accuracy(m.logits(self.x[700:]), self.y[700:]), 0.7)

    def test_probabilities_sum_to_one_and_save_load(self):
        m = MLP([784, 16, 9]); p = m.proba(self.x[:5]); np.testing.assert_allclose(p.sum(1), 1, atol=1e-5)
        path = os.path.join(tempfile.gettempdir(), "_m.npz")
        m.save(path); np.testing.assert_allclose(MLP.load(path).proba(self.x[:5]), p, atol=1e-6)

    def test_temperature_does_not_hurt_nll_and_ece_in_range(self):
        rng = np.random.default_rng(1)
        y = rng.integers(0, 9, 600); lg = np.eye(9)[y] * 3 + rng.normal(0, 2, (600, 9)); lg *= 3   # overconfident
        T = fit_temperature(lg, y)
        self.assertLessEqual(nll(softmax(lg, T), y), nll(softmax(lg), y) + 1e-9)
        self.assertTrue(0 <= ece(softmax(lg, T), y) <= 1)

    def test_confusion_matrix_counts(self):
        cm = confusion_matrix(np.array([0, 1, 1]), np.array([0, 1, 0]), 2)
        self.assertEqual(cm.tolist(), [[1, 1], [0, 1]])


class GeneratorTests(unittest.TestCase):
    def test_seed_reproducible(self):
        a = make_puzzle(random.Random(7), 30); b = make_puzzle(random.Random(7), 30)
        self.assertEqual(a, b)

    def test_blank_rule_and_mosaic(self):
        imgs = np.zeros((3, 784), np.float32); imgs[1, 300:340] = 1.0
        cells = build_mosaic(imgs, [0, 1, -1] + [-1] * 78, random.Random(0))
        self.assertTrue(is_blank(cells[0])); self.assertFalse(is_blank(cells[1])); self.assertTrue(is_blank(cells[2]))

    def test_instance_injection_counts(self):
        rng = np.random.default_rng(0)
        labels = np.tile(np.arange(1, 10), 40)
        probs = np.full((len(labels), 9), 0.02, np.float32)
        for i, l in enumerate(labels):
            if i % 5 == 0:
                probs[i, (l % 9)] = 0.8        # misread as the next digit
            else:
                probs[i, l - 1] = 0.8
        pool = ImagePool(labels, probs / probs.sum(1, keepdims=True))
        p, s = make_puzzle(random.Random(2), 30)
        ins = make_instance(random.Random(2), pool, probs / probs.sum(1, keepdims=True), p, s, 2)
        self.assertEqual(len(ins.mis_cells), 2)
        self.assertEqual(make_instance(random.Random(2), pool, probs, p, s, 0).mis_cells, [])


if __name__ == "__main__":
    unittest.main()
