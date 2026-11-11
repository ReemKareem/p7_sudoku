"""MNIST loading (local copy / download) + processed split. Digit 0 is dropped (labels 1-9 -> 0-8)."""
from __future__ import annotations
import gzip, hashlib, os, urllib.request
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed", "mnist19.npz")
MIRRORS = ["https://storage.googleapis.com/cvdf-datasets/mnist/",
           "https://ossci-datasets.s3.amazonaws.com/mnist/",
           "http://yann.lecun.com/exdb/mnist/"]
FILES = {"train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
         "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
         "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
         "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c"}


def _download(name):
    path = os.path.join(RAW, name)
    if not os.path.exists(path):
        os.makedirs(RAW, exist_ok=True)
        last = None
        for m in MIRRORS:
            try:
                print(f"downloading {m}{name}")
                urllib.request.urlretrieve(m + name, path)
                break
            except Exception as e:  # try next mirror
                last = e
        else:
            raise RuntimeError(f"Could not download {name}. Put the 4 MNIST .gz files in {RAW} manually. ({last})")
    md5 = hashlib.md5(open(path, "rb").read()).hexdigest()
    if md5 != FILES[name]:
        print(f"WARNING: md5 of {name} is {md5}, expected {FILES[name]}")
    return path


def _idx(path):
    with gzip.open(path, "rb") as f:
        raw = f.read()
    magic = int.from_bytes(raw[:4], "big")
    n = int.from_bytes(raw[4:8], "big")
    if magic == 2051:
        r, c = int.from_bytes(raw[8:12], "big"), int.from_bytes(raw[12:16], "big")
        return np.frombuffer(raw, np.uint8, offset=16).reshape(n, r * c)
    return np.frombuffer(raw, np.uint8, offset=8)


def load_mnist_raw():
    xs = _idx(_download("train-images-idx3-ubyte.gz")); ys = _idx(_download("train-labels-idx1-ubyte.gz"))
    xt = _idx(_download("t10k-images-idx3-ubyte.gz")); yt = _idx(_download("t10k-labels-idx1-ubyte.gz"))
    return xs, ys, xt, yt


def load_digits_dev():
    """DEV ONLY (no internet): sklearn's 8x8 digits up-sampled to 28x28. NOT MNIST."""
    from sklearn.datasets import load_digits
    from scipy.ndimage import zoom
    d = load_digits()
    x = np.stack([zoom(im, 28 / 8, order=1) for im in d.images])
    x = np.clip(x / 16.0 * 255, 0, 255).astype(np.uint8).reshape(len(x), 784)
    rng = np.random.default_rng(0)
    p = rng.permutation(len(x))
    cut = int(0.85 * len(x))
    return x[p[:cut]], d.target[p[:cut]], x[p[cut:]], d.target[p[cut:]]


def make_processed(source="mnist", seed=0, n_val=10000):
    """Split the TRAINING set into train/val (fixed seed) first, then drop digit 0. Test set untouched."""
    xs, ys, xt, yt = load_mnist_raw() if source == "mnist" else load_digits_dev()
    if source != "mnist":
        n_val = int(0.15 * len(xs))
    rng = np.random.default_rng(seed)
    p = rng.permutation(len(xs))
    val_i, tr_i = p[:n_val], p[n_val:]

    def pick(x, y, idx=None):
        if idx is not None:
            x, y = x[idx], y[idx]
        keep = y != 0
        return x[keep], (y[keep] - 1).astype(np.int64)

    x_tr, y_tr = pick(xs, ys, tr_i); x_va, y_va = pick(xs, ys, val_i); x_te, y_te = pick(xt, yt)
    os.makedirs(os.path.dirname(PROCESSED), exist_ok=True)
    np.savez_compressed(PROCESSED, x_train=x_tr, y_train=y_tr, x_val=x_va, y_val=y_va,
                        x_test=x_te, y_test=y_te, source=np.array(source))
    return dict(source=source, train=len(x_tr), val=len(x_va), test=len(x_te))


def load_processed():
    d = np.load(PROCESSED)
    f = lambda a: a.astype(np.float32) / 255.0
    return (f(d["x_train"]), d["y_train"], f(d["x_val"]), d["y_val"], f(d["x_test"]), d["y_test"],
            str(d["source"]))
