"""Train NN (784-128-64-9) and logistic-regression baseline, fit temperature on validation,
write results/classifier.md + confusion matrix + reliability diagram."""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from p7sudoku.data import load_processed, ROOT
from p7sudoku.nn import MLP, accuracy, nll, ece, softmax, fit_temperature, confusion_matrix, reliability

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

xtr, ytr, xva, yva, xte, yte, source = load_processed()
print(f"data source: {source}  train {len(xtr)}  val {len(xva)}  test {len(xte)}")
os.makedirs(os.path.join(ROOT, "models"), exist_ok=True)
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

print("training NN"); nn = MLP([784, 128, 64, 9], a.seed); nn.fit(xtr, ytr, xva, yva, a.epochs, seed=a.seed)
print("training logistic regression"); lr = MLP([784, 9], a.seed); lr.fit(xtr, ytr, xva, yva, a.epochs, seed=a.seed)
nn.save(os.path.join(ROOT, "models", "nn.npz")); lr.save(os.path.join(ROOT, "models", "logreg.npz"))

rows, figs = [], {}
T_nn = fit_temperature(nn.logits(xva), yva)
json.dump({"T_nn": T_nn, "source": source}, open(os.path.join(ROOT, "models", "calibration.json"), "w"))
for name, m, T in [("Logistic regression", lr, 1.0), ("NN (uncalibrated)", nn, 1.0), ("NN + temperature scaling", nn, T_nn)]:
    lg = m.logits(xte); p = softmax(lg, T)
    rows.append((name, accuracy(lg, yte), nll(p, yte), ece(p, yte), T))
    figs[name] = p

with open(os.path.join(ROOT, "results", "classifier.md"), "w") as f:
    f.write(f"Data source: **{source}** (test set, digits 1-9 only, n={len(yte)})\n\n")
    f.write("| Model | Test accuracy | NLL | ECE (15 bins) | Temperature |\n|---|---|---|---|---|\n")
    for r in rows:
        f.write(f"| {r[0]} | {r[1]:.4f} | {r[2]:.4f} | {r[3]:.4f} | {r[4]:.3f} |\n")
print(open(os.path.join(ROOT, "results", "classifier.md")).read())

p = figs["NN + temperature scaling"]
cm = confusion_matrix(p.argmax(1), yte)
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].imshow(cm, cmap="Blues"); ax[0].set_title("Confusion matrix (NN, test)")
ax[0].set_xticks(range(9), range(1, 10)); ax[0].set_yticks(range(9), range(1, 10))
ax[0].set_xlabel("predicted"); ax[0].set_ylabel("true")
for i in range(9):
    for j in range(9):
        ax[0].text(j, i, cm[i, j], ha="center", va="center", fontsize=7, color="black" if cm[i, j] < cm.max() / 2 else "white")
ax[1].plot([0, 1], [0, 1], "k--", label="perfect")
for name in ("Logistic regression", "NN (uncalibrated)", "NN + temperature scaling"):
    r = reliability(figs[name], yte)
    ax[1].plot([x[0] for x in r], [x[1] for x in r], "o-", label=name)
ax[1].set_xlabel("confidence"); ax[1].set_ylabel("accuracy"); ax[1].set_title("Reliability diagram"); ax[1].legend()
plt.tight_layout(); plt.savefig(os.path.join(ROOT, "results", "classifier.png"), dpi=130)
