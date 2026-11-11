"""The three experiments (5 seeds, mean +- std). Each returns rows and writes CSV / Markdown / PNG."""
from __future__ import annotations
import csv, json, os, random, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .csp import solve
from .data import load_processed, ROOT
from .generator import ImagePool, make_instance, make_puzzle
from .integration import no_correction, random_correction, solve_with_correction
from .nn import MLP, softmax
from .config import TEAM_SEED

RES = os.path.join(ROOT, "results")


def load_context():
    xtr, ytr, xva, yva, xte, yte, source = load_processed()
    model = MLP.load(os.path.join(ROOT, "models", "nn.npz"))
    T = json.load(open(os.path.join(ROOT, "models", "calibration.json")))["T_nn"]
    test_probs = softmax(model.logits(xte), T).astype(np.float32)
    pool = ImagePool(yte + 1, test_probs)
    return dict(model=model, T=T, x_test=xte, y_test=yte, test_probs=test_probs, pool=pool, source=source)


def mean_std(vals):
    a = np.array(vals, float)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def write_csv(name, rows):
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)


def write_md(name, rows, title, note=""):
    with open(os.path.join(RES, name), "w", encoding="utf-8") as f:
        f.write(f"### {title}\n\n{note}\n\n| " + " | ".join(rows[0]) + " |\n|" + "---|" * len(rows[0]) + "\n")
        for r in rows:
            f.write("| " + " | ".join(str(v) for v in r.values()) + " |\n")


def fmt(vals, d=2):
    m, s = mean_std(vals)
    return f"{m:.{d}f} ± {s:.{d}f}"


# --------------------------------------------------------------------------- E1
def run_e1(ctx, seeds=5, per_seed=20, n_givens=30, levels=(0, 1, 2, 3, 4), algos=("ucs", "greedy", "astar"),
           max_corr=3, n_uncertain=12, team_seed=TEAM_SEED, unique=False):
    s0 = team_seed * 100000
    tag = "_unique" if unique else ""
    pooled = {}   # (algo, level) -> expanded nodes of EVERY puzzle (for the median)
    log = []   # correction log of every UCS-solved puzzle
    acc = {}   # (algo, level) -> metric -> list over seeds
    for seed in range(seeds):
        rng = random.Random(s0 + 1000 + seed)
        puzzles = [make_puzzle(rng, n_givens) for _ in range(per_seed)]
        for lv in levels:
            irng = random.Random(s0 + seed * 100 + lv)
            insts = [make_instance(irng, ctx["pool"], ctx["test_probs"], p, s, lv) for p, s in puzzles]
            for al in algos:
                per = dict(expanded=[], generated=[], csp_calls=[], exact=[], found=[], cost=[], time=[])
                for pi, ins in enumerate(insts):
                    o = solve_with_correction(ins.read_grid, ins.probs, al, max_corr, n_uncertain, unique=unique)
                    if al == "ucs":
                        exact_ = o["status"] == "solved" and o["solution"] == ins.solution
                        base = dict(team_seed=team_seed, seed=seed, misreads=lv, puzzle=pi, status=o["status"],
                                    exact_solution=exact_, true_misread_cells=" ".join(map(str, ins.mis_cells)))
                        if o["corrections"]:
                            for c in o["corrections"]:
                                log.append(dict(base, cell=c.cell, row=c.cell // 9 + 1, col=c.cell % 9 + 1,
                                                read_digit=c.read_digit, new_digit=c.new_digit, cost=round(c.cost, 4),
                                                new_digit_is_true=(c.new_digit == ins.puzzle[c.cell])))
                        else:
                            log.append(dict(base, cell="", row="", col="", read_digit="", new_digit="", cost=0,
                                            new_digit_is_true=""))
                    s = o["search_stats"]
                    per["expanded"].append(s.expanded); per["generated"].append(s.generated)
                    per["csp_calls"].append(s.csp_calls); per["time"].append(s.time_s)
                    per["found"].append(o["status"] == "solved")
                    per["exact"].append(o["status"] == "solved" and o["solution"] == ins.solution)
                    if o["status"] == "solved":
                        per["cost"].append(o["cost"])
                pooled.setdefault((al, lv), []).extend(per["expanded"])
                d = acc.setdefault((al, lv), {k: [] for k in per})
                for k, v in per.items():
                    d[k].append(float(np.mean(v)) if v else float("nan"))
        print(f"E1 seed {seed} done")
    rows = []
    for (al, lv), d in acc.items():
        rows.append(dict(algorithm=al, misreads=lv, exact_recovery=fmt(d["exact"], 3), solvable_found=fmt(d["found"], 3),
                         solved_but_wrong=fmt([f - e for f, e in zip(d["found"], d["exact"])], 3),
                         nodes_expanded=fmt(d["expanded"], 1), nodes_expanded_median=float(np.median(pooled[(al, lv)])), nodes_generated=fmt(d["generated"], 1),
                         csp_calls=fmt(d["csp_calls"], 1), mean_cost=fmt([c for c in d["cost"] if c == c], 3),
                         time_s=fmt(d["time"], 3)))
    if not unique:
        write_csv("correction_log.csv", log)
    write_csv(f"e1_search{tag}.csv", rows)
    write_md(f"e1_search{tag}.md", rows, "Experiment 1: UCS vs Greedy vs A*" + (" (unique-solution goal)" if unique else ""),
             f"{seeds} seeds x {per_seed} puzzles = {seeds*per_seed} puzzles per cell; {n_givens} givens; "
             f"max {max_corr} corrections; {n_uncertain} uncertain cells; team seed {team_seed}; mean ± std over seeds. "
             f"Level 4 cannot be recovered fully with a budget of {max_corr} corrections. "
             "solved_but_wrong = solvable puzzle found but not the true solution. "
             "nodes_expanded_median is pooled over all puzzles (the mean is dominated by a few hard puzzles).")
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for al in algos:
        for k, a_, key in ((0, ax[0], "exact"), (1, ax[1], "expanded")):
            m = [mean_std(acc[(al, lv)][key]) for lv in levels]
            a_.errorbar(levels, [x[0] for x in m], [x[1] for x in m], marker="o", capsize=3, label=al.upper())
    ax[0].set_title("Exact-solution recovery rate"); ax[1].set_title("Search nodes expanded"); ax[1].set_yscale("log")
    for a_ in ax: a_.set_xlabel("injected misreads"); a_.legend(); a_.set_xticks(levels)
    plt.tight_layout(); plt.savefig(os.path.join(RES, f"e1_search{tag}.png"), dpi=130); plt.close()
    return rows


# --------------------------------------------------------------------------- E2
CONFIGS = [("BT", dict(use_fc=False, use_ac3=False)), ("BT+FC", dict(use_fc=True, use_ac3=False)),
           ("BT+FC+AC-3", dict(use_fc=True, use_ac3=True))]


def run_e2(seeds=5, per_seed=20, givens=(25, 30, 35), ordering="static", node_cap=200_000, team_seed=TEAM_SEED):
    s0 = team_seed * 100000
    acc = {}
    for seed in range(seeds):
        for n in givens:
            rng = random.Random(s0 + 5000 + seed * 10 + n)
            puzzles = [make_puzzle(rng, n) for _ in range(per_seed)]
            for name, kw in CONFIGS:
                per = dict(assignments=[], backtracks=[], time=[], solved=[])
                for p, s in puzzles:
                    r = solve(p, ordering=ordering, node_cap=node_cap, **kw)
                    per["assignments"].append(r.stats.assignments); per["backtracks"].append(r.stats.backtracks)
                    per["time"].append(r.stats.time_s); per["solved"].append(r.solved and r.solution == s)
                d = acc.setdefault((name, n), {k: [] for k in per})
                for k, v in per.items():
                    d[k].append(float(np.mean(v)))
        print(f"E2 seed {seed} done")
    rows = [dict(config=nm, givens=n, assignments=fmt(d["assignments"], 0), backtracks=fmt(d["backtracks"], 0),
                 time_s=fmt(d["time"], 4), solved_rate=fmt(d["solved"], 2)) for (nm, n), d in acc.items()]
    write_csv("e2_csp.csv", rows)
    write_md("e2_csp.md", rows, "Experiment 2: BT vs BT+FC vs BT+FC+AC-3",
             f"{seeds} seeds x {per_seed} puzzles per cell, variable order = {ordering}, assignment cap {node_cap}, team seed {team_seed}. "
             "Assignments/backtracks are per puzzle (mean ± std over seeds). Runs stopped by the cap count as NOT solved (solved_rate < 1) "
             "and their assignments are truncated at the cap, so the BT mean at 25 givens is a lower bound. "
             "AC-3 lowers assignments but adds its own propagation cost, so wall-clock time can be higher for easy puzzles (35 givens).")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    w = 0.25
    for k, (nm, _) in enumerate(CONFIGS):
        for a_, key in ((ax[0], "assignments"), (ax[1], "backtracks")):
            m = [mean_std(acc[(nm, n)][key]) for n in givens]
            a_.bar(np.arange(len(givens)) + k * w, [x[0] for x in m], w, yerr=[x[1] for x in m], capsize=2, label=nm)
    for a_, t in zip(ax, ("assignments", "backtracks")):
        a_.set_xticks(np.arange(len(givens)) + w, [str(n) for n in givens]); a_.set_xlabel("givens")
        a_.set_yscale("log"); a_.set_title(t); a_.legend()
    plt.tight_layout(); plt.savefig(os.path.join(RES, "e2_csp.png"), dpi=130); plt.close()
    return rows


# --------------------------------------------------------------------------- E3
METHODS = ("NN-guided (UCS)", "NN-guided (UCS, unique goal)", "No correction", "Random correction")

def run_e3(ctx, seeds=5, per_seed=20, n_givens=30, levels=(0, 1, 2, 3), max_corr=3, n_uncertain=12, budget=200, team_seed=TEAM_SEED):
    s0 = team_seed * 100000
    acc = {}
    for seed in range(seeds):
        rng = random.Random(s0 + 9000 + seed)
        puzzles = [make_puzzle(rng, n_givens) for _ in range(per_seed)]
        for lv in levels:
            irng = random.Random(s0 + seed * 100 + lv + 7)
            insts = [make_instance(irng, ctx["pool"], ctx["test_probs"], p, s, lv) for p, s in puzzles]
            rrng = random.Random(s0 + seed)
            res = {m: [] for m in METHODS}
            for ins in insts:
                o = solve_with_correction(ins.read_grid, ins.probs, "ucs", max_corr, n_uncertain)
                res["NN-guided (UCS)"].append((o["status"], o["solution"]))
                o = solve_with_correction(ins.read_grid, ins.probs, "ucs", max_corr, n_uncertain, unique=True)
                res["NN-guided (UCS, unique goal)"].append((o["status"], o["solution"]))
                o = no_correction(ins.read_grid); res["No correction"].append((o["status"], o["solution"]))
                o = random_correction(ins.read_grid, ins.probs, rrng, max_corr, budget)
                res["Random correction"].append((o["status"], o["solution"]))
            for m, lst in res.items():
                d = acc.setdefault((m, lv), dict(exact=[], solved_wrong=[], unreadable=[]))
                d["exact"].append(np.mean([st == "solved" and sol == ins.solution for (st, sol), ins in zip(lst, insts)]))
                d["solved_wrong"].append(np.mean([st == "solved" and sol != ins.solution for (st, sol), ins in zip(lst, insts)]))
                d["unreadable"].append(np.mean([st != "solved" for st, _ in lst]))
        print(f"E3 seed {seed} done")
    rows = [dict(method=m, misreads=lv, exact_recovery=fmt(d["exact"], 3), solved_but_wrong=fmt(d["solved_wrong"], 3),
                 unreadable=fmt(d["unreadable"], 3)) for (m, lv), d in acc.items()]
    write_csv("e3_correction.csv", rows)
    write_md("e3_correction.md", rows, "Experiment 3: NN-guided vs no vs random correction",
             f"{seeds} seeds x {per_seed} puzzles per cell, {n_givens} givens, random baseline: {budget} tries, "
             f"max {max_corr} changed cells, team seed {team_seed}.")
    fig, ax = plt.subplots(figsize=(8, 4)); w = 0.2
    for k, m in enumerate(METHODS):
        mm = [mean_std(acc[(m, lv)]["exact"]) for lv in levels]
        ax.bar(np.arange(len(levels)) + k * w, [x[0] for x in mm], w, yerr=[x[1] for x in mm], capsize=3, label=m)
    ax.set_xticks(np.arange(len(levels)) + w, [str(l) for l in levels]); ax.set_xlabel("injected misreads")
    ax.set_ylabel("exact-solution recovery"); ax.legend(); ax.set_title("Experiment 3")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "e3_correction.png"), dpi=130); plt.close()
    return rows
