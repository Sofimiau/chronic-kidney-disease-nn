"""
Cleans up the raw CKD dataset and turns it into train/val/test arrays
ready for our neural network.

Important thing we were careful about: we split the data into
train/val/test FIRST, before doing any cleaning that involves computing a
statistic (like the median or the mean). Then we calculate the medians/
means etc. using ONLY the training set, and reuse those same numbers for
val and test. If we calculated them using all the data first and split
after, the val/test sets would have "leaked" a bit of information into the
training process (via those averages), and our accuracy would look better
than it actually is on truly unseen patients.
"""
import numpy as np
import pandas as pd
import json

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

df = pd.read_csv("data/kidney_disease.csv")
df = df.drop(columns=["id"])  # just a row number, not useful for prediction

NUMERIC_COLS = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
                 "hemo", "pcv", "wc", "rc"]
CATEGORICAL_COLS = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]
TARGET_COL = "classification"

# the raw csv has some messy text: extra spaces, tab characters, etc, so we
# clean that up first or it would mess up the imputation step later
for c in CATEGORICAL_COLS + [TARGET_COL] + ["pcv", "wc", "rc"]:
    df[c] = df[c].astype(str).str.strip().str.replace("\t", "", regex=False)
    df[c] = df[c].replace({"nan": np.nan, "": np.nan})

# pcv/wc/rc are supposed to be numbers but come in as text in the raw file
for c in ["pcv", "wc", "rc"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
for c in NUMERIC_COLS:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# turn the label into 0/1 so we can use it with our model
df[TARGET_COL] = df[TARGET_COL].map({"ckd": 1, "notckd": 0})
assert df[TARGET_COL].isna().sum() == 0, "some label didn't map to ckd/notckd"

# --- split into train (60%) / val (20%) / test (20%) ---
# we do this before any imputation/scaling, see note at the top of the file
y_all = df[TARGET_COL].values
idx = np.arange(len(df))


def stratified_split(idx, y, frac_train=0.6, frac_val=0.2, seed=RNG_SEED):
    """Splits the data but keeps the same proportion of ckd/notckd in each
    split. We do this instead of a totally random split because our
    dataset is a bit imbalanced (more ckd cases than notckd), and with
    only 400 rows a plain random split could easily give us a val or test
    set with way more of one class than the other just by bad luck.
    """
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for cls in np.unique(y):
        cls_idx = idx[y == cls]
        rng.shuffle(cls_idx)
        n = len(cls_idx)
        n_train = int(round(n * frac_train))
        n_val = int(round(n * frac_val))
        train_idx.extend(cls_idx[:n_train])
        val_idx.extend(cls_idx[n_train:n_train + n_val])
        test_idx.extend(cls_idx[n_train + n_val:])
    rng.shuffle(train_idx); rng.shuffle(val_idx); rng.shuffle(test_idx)
    return np.array(train_idx), np.array(val_idx), np.array(test_idx)


train_idx, val_idx, test_idx = stratified_split(idx, y_all)
print("split sizes:", len(train_idx), len(val_idx), len(test_idx))

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)
df_test = df.iloc[test_idx].reset_index(drop=True)

# --- fill in missing values ---
# numeric columns: use the median (less sensitive to outliers than the mean)
# categorical columns: use the most common value (mode)
# we calculate these using df_train only, then apply the same numbers to
# val/test (again, to avoid leakage)
num_medians = df_train[NUMERIC_COLS].median()
cat_modes = df_train[CATEGORICAL_COLS].mode().iloc[0]

for part in (df_train, df_val, df_test):
    part[NUMERIC_COLS] = part[NUMERIC_COLS].fillna(num_medians)
    part[CATEGORICAL_COLS] = part[CATEGORICAL_COLS].fillna(cat_modes)

# --- encode the categorical columns as 0/1 ---
# all of them only have two possible values (e.g. "normal"/"abnormal"), so
# we don't need one-hot encoding, a simple 0/1 mapping is enough
binary_maps = {}
for c in CATEGORICAL_COLS:
    cats = sorted(df_train[c].unique())
    assert len(cats) == 2, f"{c} has {cats}"
    binary_maps[c] = {cats[0]: 0, cats[1]: 1}

for part in (df_train, df_val, df_test):
    for c in CATEGORICAL_COLS:
        part[c] = part[c].map(binary_maps[c])

FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

# --- standardize the numeric columns (mean 0, std 1) ---
# the raw numbers are on very different scales (e.g. white cell count is in
# the thousands while specific gravity is around 1.0-1.03). if we don't
# scale them, the network would basically learn mostly from whichever
# column happens to have the biggest numbers, which isn't what we want.
mu = df_train[NUMERIC_COLS].mean()
sigma = df_train[NUMERIC_COLS].std().replace(0, 1.0)  # avoid dividing by 0

for part in (df_train, df_val, df_test):
    part[NUMERIC_COLS] = (part[NUMERIC_COLS] - mu) / sigma

# --- turn everything into numpy arrays and save ---
X_train = df_train[FEATURE_COLS].values.astype(np.float64)
X_val = df_val[FEATURE_COLS].values.astype(np.float64)
X_test = df_test[FEATURE_COLS].values.astype(np.float64)
y_train = df_train[TARGET_COL].values.astype(np.float64).reshape(-1, 1)
y_val = df_val[TARGET_COL].values.astype(np.float64).reshape(-1, 1)
y_test = df_test[TARGET_COL].values.astype(np.float64).reshape(-1, 1)

np.save("data/X_train.npy", X_train)
np.save("data/X_val.npy", X_val)
np.save("data/X_test.npy", X_test)
np.save("data/y_train.npy", y_train)
np.save("data/y_val.npy", y_val)
np.save("data/y_test.npy", y_test)

# save some info about the splits so we can reference it in the report
# without having to recompute it
with open("data/feature_info.json", "w") as f:
    json.dump({
        "feature_cols": FEATURE_COLS,
        "n_features": len(FEATURE_COLS),
        "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
        "train_class_balance": {"ckd": int(y_train.sum()), "notckd": int(len(y_train)-y_train.sum())},
        "val_class_balance": {"ckd": int(y_val.sum()), "notckd": int(len(y_val)-y_val.sum())},
        "test_class_balance": {"ckd": int(y_test.sum()), "notckd": int(len(y_test)-y_test.sum())},
    }, f, indent=2)

print("Done. Shapes:", X_train.shape, X_val.shape, X_test.shape)
print(open("data/feature_info.json").read())
