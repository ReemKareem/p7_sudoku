"""Small numpy neural network (MLP) + logistic-regression baseline + calibration metrics.
MLP([784,128,64,9]) is the NN; MLP([784,9]) (no hidden layer) is multinomial logistic regression.
"""
from __future__ import annotations
import numpy as np


def softmax(z, T=1.0):
    z = z / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MLP:
    def __init__(self, sizes, seed=0):
        rng = np.random.default_rng(seed)
        self.sizes = list(sizes)
        self.W = [(rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(np.float32)
                  for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b, np.float32) for b in sizes[1:]]

    def _forward(self, x):
        acts = [x]
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = acts[-1] @ W + b
            acts.append(np.maximum(z, 0) if i < len(self.W) - 1 else z)
        return acts

    def logits(self, x, batch=4096):
        return np.concatenate([self._forward(x[i:i + batch])[-1] for i in range(0, len(x), batch)])

    def proba(self, x, T=1.0):
        return softmax(self.logits(x), T)

    def fit(self, x, y, xv, yv, epochs=10, lr=1e-3, batch=128, l2=1e-5, seed=0, verbose=True):
        rng = np.random.default_rng(seed)
        params = self.W + self.b
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        t, best, best_params = 0, -1, None
        L = len(self.W)
        for ep in range(epochs):
            perm = rng.permutation(len(x))
            for s in range(0, len(x), batch):
                idx = perm[s:s + batch]
                acts = self._forward(x[idx])
                p = softmax(acts[-1])
                d = p
                d[np.arange(len(idx)), y[idx]] -= 1
                d /= len(idx)
                gW, gb = [None] * L, [None] * L
                for i in reversed(range(L)):
                    gW[i] = acts[i].T @ d + l2 * self.W[i]
                    gb[i] = d.sum(0)
                    if i > 0:
                        d = (d @ self.W[i].T) * (acts[i] > 0)
                grads = gW + gb
                t += 1
                for k, (p_, g_) in enumerate(zip(params, grads)):
                    m[k] = 0.9 * m[k] + 0.1 * g_
                    v[k] = 0.999 * v[k] + 0.001 * g_ * g_
                    p_ -= lr * (m[k] / (1 - 0.9 ** t)) / (np.sqrt(v[k] / (1 - 0.999 ** t)) + 1e-8)
            acc = accuracy(self.logits(xv), yv)
            if verbose:
                print(f"  epoch {ep + 1:2d}/{epochs}  val acc {acc:.4f}")
            if acc > best:
                best, best_params = acc, [p_.copy() for p_ in params]
        for p_, bp in zip(params, best_params):
            p_[...] = bp
        return best

    def save(self, path):
        np.savez(path, sizes=np.array(self.sizes), **{f"W{i}": w for i, w in enumerate(self.W)},
                 **{f"b{i}": b for i, b in enumerate(self.b)})

    @classmethod
    def load(cls, path):
        d = np.load(path)
        m = cls(list(d["sizes"]))
        m.W = [d[f"W{i}"] for i in range(len(m.W))]
        m.b = [d[f"b{i}"] for i in range(len(m.b))]
        return m


# ---------------------------------------------------------------- metrics
def accuracy(logits, y):
    return float((logits.argmax(1) == y).mean())


def nll(probs, y):
    return float(-np.log(np.clip(probs[np.arange(len(y)), y], 1e-12, 1)).mean())


def ece(probs, y, bins=15):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            total += mask.mean() * abs((pred[mask] == y[mask]).mean() - conf[mask].mean())
    return float(total)


def reliability(probs, y, bins=10):
    conf, pred = probs.max(1), probs.argmax(1)
    edges = np.linspace(0, 1, bins + 1)
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            out.append((float(conf[mask].mean()), float((pred[mask] == y[mask]).mean()), int(mask.sum())))
    return out


def confusion_matrix(pred, y, k=9):
    cm = np.zeros((k, k), int)
    for a, b in zip(y, pred):
        cm[a, b] += 1
    return cm


def fit_temperature(val_logits, y):
    """Temperature scaling: pick T minimising validation NLL (coarse-to-fine grid search)."""
    grid = np.linspace(0.3, 5.0, 236)
    for _ in range(2):
        best = min(grid, key=lambda T: nll(softmax(val_logits, T), y))
        step = (grid[1] - grid[0])
        grid = np.linspace(max(0.05, best - step), best + step, 41)
    return float(min(grid, key=lambda T: nll(softmax(val_logits, T), y)))
