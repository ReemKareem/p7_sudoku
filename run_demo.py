"""python run_demo.py [--seed 0]   ->  console walk-through + results/demo.png
Shows: input mosaic, NN reading, AC-3 conflict, search trace, correction log, CSP pruning with/without FC/AC-3,
system with NN-correction ON vs OFF, and plots of how the three parts cooperate."""
import argparse, os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from p7sudoku.csp import solve, find_conflicts
from p7sudoku.experiments import load_context, RES
from p7sudoku.generator import build_mosaic, make_instance, make_puzzle, mosaic_image
from p7sudoku.integration import read_cells, solve_with_correction, no_correction, random_correction
from p7sudoku.config import TEAM_SEED

ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--misreads", type=int, default=2)
a = ap.parse_args()
ctx = load_context()
rng = random.Random(TEAM_SEED * 100000 + a.seed)


def show(g, mark=()):
    out = []
    for r in range(9):
        row = ""
        for c in range(9):
            v = g[r * 9 + c]
            ch = str(v) if v else "."
            row += (f"[{ch}]" if r * 9 + c in mark else f" {ch} ") + ("|" if c in (2, 5) else "")
        out.append(row)
        if r in (2, 5):
            out.append("-" * 33)
    return "\n".join(out)


# ---- pick a demo instance where misreads really break the puzzle ----------------------
for attempt in range(500):
    puzzle, sol = make_puzzle(rng, 30)
    ins = make_instance(rng, ctx["pool"], ctx["test_probs"], puzzle, sol, a.misreads)
    if len(ins.mis_cells) == a.misreads and find_conflicts(ins.read_grid):
        o = solve_with_correction(ins.read_grid, ins.probs, "ucs", record_trace=True)
        if o["status"] == "solved" and o["solution"] == sol:
            break
else:
    raise SystemExit("no suitable demo instance found; try another --seed")

cells = build_mosaic(ctx["x_test"], ins.img_idx, rng)
read, probs = read_cells(cells, ctx["model"], ctx["T"])          # blank rule + NN, from pixels
assert read == ins.read_grid

print("=" * 70, "\nINPUT: 9x9 mosaic of handwritten digits (saved in results/demo.png)")
print(f"data source: {ctx['source']}   (true misread cells: {ins.mis_cells})")
print("\n--- true puzzle ---\n" + show(ins.puzzle))
print("\n--- NN reading ([x] = misread) ---\n" + show(read, ins.mis_cells))
print(f"\nAC-3 on the NN reading: conflict={o['ac3_conflict']}  duplicate pairs={o['conflict_pairs'][:6]}")
r0 = no_correction(read)
print("CSP without correction ->", r0["message"])

print("\n--- UCS search trace (first 8 steps) ---")
for k, (kind, state, g) in enumerate(o["trace"][:8]):
    print(f"{k:2d} {kind:6s} g={g:<7} corrections={[(c//9+1, c%9+1, d) for c, d in state]}")
print("   ...  total:", o["search_stats"])
print("\n--- Correction log ---")
print(f"{'cell(r,c)':>10} {'read':>5} {'new':>4} {'cost':>7}")
for c in o["corrections"]:
    print(f"{f'({c.cell//9+1},{c.cell%9+1})':>10} {c.read_digit:>5} {c.new_digit:>4} {c.cost:>7.3f}")
print("\n--- Solution ---\n" + show(o["solution"], [c.cell for c in o["corrections"]]))
print("\nexact solution recovered:", o["solution"] == ins.solution)

print("\n--- CSP pruning on the corrected puzzle (static order, counters) ---")
cfg = [("BT", False, False), ("BT+FC", True, False), ("BT+AC-3", False, True), ("BT+FC+AC-3", True, True)]
csp_rows = []
for name, fc, ac in cfg:
    r = solve(o["corrected_grid"], fc, ac, "static")
    csp_rows.append((name, r.stats.assignments, r.stats.backtracks))
    print(f"{name:11s} assignments={r.stats.assignments:6d} backtracks={r.stats.backtracks:6d} "
          f"fc_prunings={r.stats.fc_prunings:5d} ac3_prunings={r.stats.ac3_prunings:4d}")

print("\n--- Search algorithms on this instance ---")
alg_rows = []
for al in ("ucs", "greedy", "astar"):
    oo = solve_with_correction(ins.read_grid, ins.probs, al)
    s = oo["search_stats"]; alg_rows.append((al.upper(), s.expanded, oo["cost"]))
    print(f"{al.upper():7s} expanded={s.expanded:4d} generated={s.generated:5d} cost={oo['cost']:.3f}")

print("\n--- NN-correction ON vs OFF ---")
rr = random_correction(read, probs, random.Random(a.seed), 3, 200)
print("OFF   (no correction)    :", r0["status"].upper(), "|", r0["message"])
print("RANDOM (200 random tries):", rr["status"].upper(), "| exact:", rr["solution"] == ins.solution if rr["solution"] else False)
print("ON    (NN-guided UCS)    :", o["status"].upper(), "| exact:", o["solution"] == ins.solution, "|", o["message"])

# ---- figure ---------------------------------------------------------------------------
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(2, 3, 1); ax.imshow(mosaic_image(cells), cmap="gray_r"); ax.axis("off")
ax.set_title("1. Input mosaic (red = digit the NN misreads)")
for c in ins.mis_cells:
    ax.add_patch(Rectangle(((c % 9) * 28, (c // 9) * 28), 28, 28, fill=False, ec="red", lw=2))
for k in range(10):
    ax.axhline(k * 28 - .5, color="b", lw=1.2 if k % 3 == 0 else .3); ax.axvline(k * 28 - .5, color="b", lw=1.2 if k % 3 == 0 else .3)


def draw_grid(ax, g, title, red=(), green=(), conflicts=()):
    ax.set_xlim(0, 9); ax.set_ylim(9, 0); ax.set_aspect("equal"); ax.set_title(title)
    for k in range(10):
        ax.plot([0, 9], [k, k], "k", lw=2 if k % 3 == 0 else .5); ax.plot([k, k], [0, 9], "k", lw=2 if k % 3 == 0 else .5)
    bad = {x for p in conflicts for x in p}
    for c in range(81):
        col = "#f4a6a6" if c in bad else ("#a8e6a1" if c in green else None)
        if col: ax.add_patch(Rectangle((c % 9, c // 9), 1, 1, color=col))
        if g[c]: ax.text(c % 9 + .5, c // 9 + .55, g[c], ha="center", va="center", fontsize=13,
                         color="red" if c in red else "black")
    ax.axis("off")


draw_grid(fig.add_subplot(2, 3, 2), read, "2. NN reading + AC-3 conflicts (pink)", red=ins.mis_cells, conflicts=o["conflict_pairs"])
draw_grid(fig.add_subplot(2, 3, 3), o["solution"], "3. After Search corrections + CSP solve (green = corrected)",
          green=[c.cell for c in o["corrections"]])
ax = fig.add_subplot(2, 3, 4); ax.bar([r[0] for r in alg_rows], [r[1] for r in alg_rows], color=["#4c72b0", "#dd8452", "#55a868"])
ax.set_title("4. Search: nodes expanded (same instance)"); ax.set_ylabel("expanded")
ax = fig.add_subplot(2, 3, 5); ax.bar([r[0] for r in csp_rows], [max(r[1], 1) for r in csp_rows], color="#8172b3")
ax.set_yscale("log"); ax.set_title("5. CSP: assignments, with/without FC and AC-3"); ax.tick_params(axis="x", labelsize=8)
ax = fig.add_subplot(2, 3, 6); ax.axis("off")
ax.text(0, .95, "Cooperation", fontsize=13, weight="bold", va="top")
ax.text(0, .85, f"NN reads 81 cells -> {len(ins.mis_cells)} misread\nAC-3 finds {len(o['conflict_pairs'])} duplicate pair(s)\n"
                f"Search (UCS) expands {o['search_stats'].expanded} nodes,\n  CSP checks {o['search_stats'].csp_calls} candidate puzzles\n"
                f"Corrections: {[(c.cell//9+1, c.cell%9+1, c.read_digit, c.new_digit) for c in o['corrections']]}\n"
                f"Total cost {o['cost']:.3f}\nExact solution recovered: {o['solution']==ins.solution}\n\n"
                f"Correction OFF   : {r0['status'].upper()}\n"
                f"Random correction: {rr['status'].upper()} (exact: {bool(rr['solution']) and rr['solution']==ins.solution})", va="top", fontsize=10, family="monospace")
plt.tight_layout(); os.makedirs(RES, exist_ok=True); plt.savefig(os.path.join(RES, "demo.png"), dpi=120)
print("\nsaved results/demo.png")
