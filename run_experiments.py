"""python run_experiments.py [--seeds 5] [--per-seed 20] [--only e1 e2 e3] [--team-seed N]
5 seeds x 20 puzzles = 100 puzzles per misread level (required). Also writes results/correction_log.csv
(cell, read digit, new digit, cost for every UCS-solved puzzle)."""
import argparse, time
from p7sudoku import experiments as X
from p7sudoku.config import TEAM_SEED

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, default=5)
ap.add_argument("--per-seed", type=int, default=20)
ap.add_argument("--only", nargs="*", default=["e1", "e2", "e3"])
ap.add_argument("--unique-e1", action="store_true", help="also run E1 with the exactly-one-solution goal")
ap.add_argument("--team-seed", type=int, default=TEAM_SEED)
a = ap.parse_args()
t0 = time.time()
ctx = X.load_context()
print("data source:", ctx["source"], "| temperature:", round(ctx["T"], 3), "| team seed:", a.team_seed)
if "e1" in a.only: X.run_e1(ctx, a.seeds, a.per_seed, team_seed=a.team_seed)
if "e1" in a.only and a.unique_e1: X.run_e1(ctx, a.seeds, a.per_seed, team_seed=a.team_seed, unique=True)
if "e2" in a.only: X.run_e2(a.seeds, a.per_seed, team_seed=a.team_seed)
if "e3" in a.only: X.run_e3(ctx, a.seeds, a.per_seed, team_seed=a.team_seed)
print(f"done in {time.time()-t0:.0f}s -> results/")
