#!/usr/bin/env python
"""
Evaluate the one-shot final_model.pkl against all public test splits.

usage:
    python eval_final_model.py  \
        /work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template/data/bike_sharing_demand
"""

import sys, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score

if len(sys.argv) != 2:
    print(__doc__); sys.exit(1)

# ---------------------------------------------------------------- paths
DATA_DIR = Path(sys.argv[1]).expanduser()          # dataset folder with 1/ … 10/
ds_name  = DATA_DIR.name

MODEL_PKL = (Path(__file__).resolve().parent /
             "models" / ds_name / "final_model.pkl")

if not MODEL_PKL.exists():
    sys.exit(f"❌ cannot find {MODEL_PKL}")

art = joblib.load(MODEL_PKL)
print(f"Loaded final model for {ds_name}  →  algo={art['algo']}")

# ---------------------------------------------------------------- eval
scores = []
for sp in sorted(p for p in DATA_DIR.iterdir() if p.is_dir() and p.name.isdigit(),
                 key=lambda p: int(p.name)):
    split_id = sp.name
    X = pd.read_parquet(sp / "X_test.parquet")[art["columns"]]
    y = pd.read_parquet(sp / "y_test.parquet").squeeze("columns")

    for c in art["categorical_cols"]:
        X[c] = X[c].astype("category")

    if art["algo"] == "lightgbm":
        y_hat = art["model"].predict(X, num_iteration=art["model"].best_iteration)
    else:  # catboost
        y_hat = art["model"].predict(X.values)

    r2 = r2_score(y, y_hat)
    scores.append(r2)
    print(f"split {split_id}:  R² = {r2:.4f}")

print("\nMean R² =", round(np.mean(scores), 4))
