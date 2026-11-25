"""
Makes the 4 plots we used in the report, using the results saved by
experiments.py. Just loss/accuracy curves, the confusion matrix, and a bar
chart comparing architectures.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

with open("data/final_report.json") as f:
    report = json.load(f)
with open("data/hparam_results.json") as f:
    hparams = json.load(f)

plt.rcParams.update({
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888888",
    "axes.labelcolor": "#333333",
    "xtick.color": "#555555",
    "ytick.color": "#555555",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

BLUE = "#2563EB"
ORANGE = "#F59E0B"
GRAY = "#9CA3AF"

# --- Figure 1: loss curves for the final model (train vs val) ---
# also marks where early stopping kicked in
hist = report["history"]
epochs = np.arange(len(hist["train_loss"]))
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(epochs, hist["train_loss"], color=BLUE, lw=2, label="Training loss")
ax.plot(epochs, hist["val_loss"], color=ORANGE, lw=2, label="Validation loss")
ax.axvline(hist["stopped_epoch"], color=GRAY, lw=1.5, ls="--", label="Early stop")
ax.set_xlabel("Epoch")
ax.set_ylabel("Binary cross-entropy loss")
ax.set_title("Training vs. validation loss")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("plots/loss_curve.png", dpi=180)
plt.close(fig)

# --- Figure 2: same thing but for accuracy instead of loss ---
fig, ax = plt.subplots(figsize=(7, 4.3))
ax.plot(epochs, hist["train_acc"], color=BLUE, lw=2, label="Training accuracy")
ax.plot(epochs, hist["val_acc"], color=ORANGE, lw=2, label="Validation accuracy")
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")
ax.set_ylim(0.5, 1.02)
ax.set_title("Training vs. validation accuracy")
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig("plots/accuracy_curve.png", dpi=180)
plt.close(fig)

# --- Figure 3: confusion matrix on the test set ---
cm = report["test_confusion_matrix"]
matrix = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
fig, ax = plt.subplots(figsize=(4.6, 4.2))
im = ax.imshow(matrix, cmap="Blues", vmin=0)
labels = [["TN", "FP"], ["FN", "TP"]]
for i in range(2):
    for j in range(2):
        val = matrix[i, j]
        color = "white" if val > matrix.max() / 2 else "#1f2937"
        ax.text(j, i, f"{labels[i][j]}\n{val}", ha="center", va="center",
                color=color, fontsize=13, fontweight="bold")
ax.set_xticks([0, 1]); ax.set_xticklabels(["Predicted notckd", "Predicted ckd"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual notckd", "Actual ckd"])
ax.set_title("Confusion matrix (test set, n=80)")
fig.tight_layout()
fig.savefig("plots/confusion_matrix.png", dpi=180)
plt.close(fig)

# --- Figure 4: compare architectures ---
# for each architecture we tried, keep only its best result (best lr/l2/
# optimizer combo) so we can see which architecture worked best on its own
by_arch = {}
for r in hparams:
    a = r["arch"]
    if a not in by_arch or r["val_acc_mean"] > by_arch[a]["val_acc_mean"]:
        by_arch[a] = r
archs = sorted(by_arch.keys(), key=lambda a: len(a.split("-")))
accs = [by_arch[a]["val_acc_mean"] for a in archs]

fig, ax = plt.subplots(figsize=(7, 4.3))
bars = ax.bar(archs, accs, color=BLUE, width=0.55)
for b, a in zip(bars, accs):
    ax.text(b.get_x() + b.get_width() / 2, a + 0.005, f"{a:.3f}",
            ha="center", va="bottom", fontsize=11, color="#333333")
ax.set_ylim(0.9, 1.01)
ax.set_xlabel("Architecture (layer sizes)")
ax.set_ylabel("Best validation accuracy")
ax.set_title("Validation accuracy by network architecture\n(best learning rate / L2 / optimizer per architecture)")
fig.tight_layout()
fig.savefig("plots/architecture_comparison.png", dpi=180)
plt.close(fig)

print("Plots written to plots/")
