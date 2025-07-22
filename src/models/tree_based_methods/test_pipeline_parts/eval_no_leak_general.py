#!/usr/bin/env python
"""
usage:
    python models/tree_based_methods/eval_no_leak_general.py data/wine_quality
    python models/tree_based_methods/eval_no_leak_general.py data/brazilian_houses
"""

import sys, glob, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score

if len(sys.argv) != 2:
    print(__doc__); sys.exit(1)

# ----------------------------------------------------------------- paths
DATA_DIR  = Path(sys.argv[1]).expanduser()          # e.g. data/wine_quality
ds_name   = DATA_DIR.name                           # "wine_quality"
BASE_DIR  = Path(__file__).resolve().parent         # folder of this script
MODEL_DIR = BASE_DIR / "models" / ds_name           # e.g. …/models/wine_quality

if not MODEL_DIR.exists():
    sys.exit(f"❌ no models found in {MODEL_DIR}")

scores = []

# ----------------------------------------------------------------- loop splits
for split in sorted(p.name for p in DATA_DIR.iterdir() if p.is_dir()
                    and p.name.isdigit()):
    split_id = int(split)
    # find artefact file (lightgbm_… or catboost_…)
    pattern = str(MODEL_DIR / f"*_split{split_id}.pkl")
    hits = glob.glob(pattern)
    if not hits:
        print(f"⚠️  no model for split {split} – skipping")
        continue

    art = joblib.load(hits[0])

    # load test matrices
    X_test = pd.read_parquet(DATA_DIR / split / "X_test.parquet")[art["columns"]]
    y_test = pd.read_parquet(DATA_DIR / split / "y_test.parquet").squeeze("columns")

    for c in art["categorical_cols"]:
        X_test[c] = X_test[c].astype("category")

    if art["algo"] == "lightgbm":
        y_pred = art["model"].predict(
            X_test, num_iteration=art["model"].best_iteration)
    else:  # catboost
        y_pred = art["model"].predict(X_test.values)

    r2 = r2_score(y_test, y_pred)
    scores.append(r2)
    print(f"split {split}:  R² = {r2:.4f}")

# ----------------------------------------------------------------- summary
if scores:
    print("\nMean R² =", round(np.mean(scores), 4))
else:
    print("\nNo splits evaluated.")
