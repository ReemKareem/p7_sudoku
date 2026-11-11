"""Reproducible data preparation (fixed seed).
    python make_data.py                 # real MNIST (downloads / uses data/raw/*.gz)
    python make_data.py --source digits # DEV ONLY fallback (sklearn 8x8 digits, no internet)
"""
import argparse, json
from p7sudoku.data import make_processed

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="mnist", choices=["mnist", "digits"])
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    info = make_processed(a.source, a.seed)
    print(json.dumps(info))
    if a.source != "mnist":
        print("WARNING: dev fallback data, NOT MNIST. Do not report these numbers.")
