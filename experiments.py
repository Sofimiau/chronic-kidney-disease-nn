"""
This is where we actually try out different settings for our network and
see which one works best, then report the final result.

What we do, step by step:
1. Try a bunch of combinations of architecture / learning rate / L2 /
   optimizer. For each combination we train 3 times with different random
   seeds and average the results, since a single run can get lucky or
   unlucky depending on how the weights start.
2. Take the combination that did best on validation, and train it again
   with 10 different seeds, keeping whichever run got the lowest
   validation loss. This is our "final" model.
3. Only at the very end do we check it on the test set, a single time.
   We never look at the test set while choosing the architecture or
   hyperparameters -- otherwise we'd basically be tuning to the test set
   and our reported accuracy wouldn't mean much for new patients.
"""
import json
import numpy as np
from mlp import MLP

X_train = np.load("data/X_train.npy"); y_train = np.load("data/y_train.npy")
X_val = np.load("data/X_val.npy"); y_val = np.load("data/y_val.npy")
X_test = np.load("data/X_test.npy"); y_test = np.load("data/y_test.npy")

# all the combinations we want to try. 24 is the number of input features.
# 4 architectures x 3 learning rates x 3 L2 values x 2 optimizers = 72 runs
configs = []
for arch in [[24, 1], [24, 8, 1], [24, 16, 8, 1], [24, 32, 16, 1]]:
    for lr in [0.1, 0.01, 0.001]:
        for l2 in [0.0, 0.001, 0.01]:
            for opt in ["adam", "momentum"]:
                configs.append(dict(arch=arch, lr=lr, l2=l2, optimizer=opt))

print(f"Total configs: {len(configs)}")

# --- try every combination, 3 seeds each ---
results = []
for i, cfg in enumerate(configs):
    accs, losses, histories = [], [], []
    for seed in [0, 1, 2]:  # 3 seeds per config for stability
        model = MLP(cfg["arch"], optimizer=cfg["optimizer"], lr=cfg["lr"],
                     l2=cfg["l2"], seed=seed)
        hist = model.fit(X_train, y_train, X_val, y_val, epochs=400,
                          batch_size=16, patience=40, verbose=False, seed=seed)
        val_proba = model.predict_proba(X_val)
        val_acc = float(np.mean((val_proba >= 0.5) == y_val))
        val_loss = model.compute_loss(val_proba, y_val)
        accs.append(val_acc)
        losses.append(val_loss)
    results.append({
        **cfg,
        "arch": "-".join(map(str, cfg["arch"])),
        "val_acc_mean": float(np.mean(accs)),
        "val_acc_std": float(np.std(accs)),
        "val_loss_mean": float(np.mean(losses)),
    })
    if i % 20 == 0:
        print(f"[{i+1}/{len(configs)}] {cfg} -> val_acc={np.mean(accs):.3f}")

# sort by accuracy first, and if two configs tie, prefer the one with
# lower loss (with only 80 val examples, accuracy ties happen a lot)
results.sort(key=lambda r: (-r["val_acc_mean"], r["val_loss_mean"]))
with open("data/hparam_results.json", "w") as f:
    json.dump(results, f, indent=2)

print("\nTop 5 configs by validation accuracy:")
for r in results[:5]:
    print(r)

best = results[0]
print("\n=== BEST CONFIG ===")
print(best)

# --- retrain the winning config a few more times to pick the best seed ---
# since the starting weights are random, the exact same config can give
# slightly different results depending on the seed. we train it 10 times
# and keep the run that had the lowest val loss, instead of just trusting
# whatever seed we happened to use first.
arch = [int(x) for x in best["arch"].split("-")]
best_model, best_hist, best_val_loss = None, None, np.inf
for seed in range(10):
    model = MLP(arch, optimizer=best["optimizer"], lr=best["lr"], l2=best["l2"], seed=seed)
    hist = model.fit(X_train, y_train, X_val, y_val, epochs=500, batch_size=16,
                      patience=50, verbose=False, seed=seed)
    if hist["best_val_loss"] < best_val_loss:
        best_val_loss = hist["best_val_loss"]
        best_model, best_hist = model, hist
        best_seed = seed

print(f"Chosen seed={best_seed}, val_loss={best_val_loss:.4f}")

# --- final check on the test set (only done once, here) ---
test_proba = best_model.predict_proba(X_test)
test_pred = (test_proba >= 0.5).astype(int)
test_acc = float(np.mean(test_pred == y_test))

# counts for the confusion matrix / precision / recall / f1
tp = int(np.sum((test_pred == 1) & (y_test == 1)))
tn = int(np.sum((test_pred == 0) & (y_test == 0)))
fp = int(np.sum((test_pred == 1) & (y_test == 0)))
fn = int(np.sum((test_pred == 0) & (y_test == 1)))
precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

final_report = {
    "best_config": best,
    "chosen_seed": best_seed,
    "test_accuracy": test_acc,
    "test_confusion_matrix": {"tp": tp, "tn": tn, "fp": fp, "fn": fn},
    "test_precision": precision,
    "test_recall": recall,
    "test_f1": f1,
    "history": {k: (v if not isinstance(v, list) else [float(x) for x in v])
                for k, v in best_hist.items()},
}
with open("data/final_report.json", "w") as f:
    json.dump(final_report, f, indent=2)

print("\n=== FINAL TEST RESULTS ===")
print(f"Accuracy:  {test_acc:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"Confusion matrix: TP={tp} TN={tn} FP={fp} FN={fn}")

# save the trained weights too, in case we want to reload the model later
# without rerunning the whole search again
import pickle
with open("data/best_model.pkl", "wb") as f:
    pickle.dump({"W": best_model.W, "b": best_model.b, "arch": arch}, f)
