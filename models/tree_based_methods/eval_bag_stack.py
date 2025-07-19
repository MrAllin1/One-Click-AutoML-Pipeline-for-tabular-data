#!/usr/bin/env python
"""
Evaluate bagged average and (optional) stacked Ridge ensemble.

usage:
    python models/tree_based_methods/eval_bag_stack.py data/wine_quality
    python eval_bag_stack.py  data/brazilian_houses
"""

import sys, glob, joblib, numpy as np, pandas as pd
from pathlib import Path
from sklearn.metrics import r2_score

# --------------------------------------------------------------------------- #
def load_artefacts(model_dir: Path):
    """Return list of artefact dicts ordered by split index."""
    arts = []
    for i in range(1, 11):
        hits = glob.glob(str(model_dir / f"*split{i}.pkl"))
        if not hits:
            raise FileNotFoundError(f"no model for split {i} in {model_dir}")
        arts.append(joblib.load(hits[0]))
    return arts

# --------------------------------------------------------------------------- #
def main():
    if len(sys.argv) != 2:
        print(__doc__); sys.exit(1)

    root = Path(sys.argv[1]).expanduser()          # e.g. data/wine_quality
    ds_name = root.name
    model_dir = Path("models/tree_based_methods/models") / ds_name

    arts = load_artefacts(model_dir)
    meta_path = model_dir / "meta_ridge.pkl"
    meta = joblib.load(meta_path) if meta_path.exists() else None
    if meta is None:
        print("⚠︎ no meta_ridge.pkl found — stack scores will be skipped\n")

    bag_scores, stack_scores = [], []

    for split in range(1, 11):
        X = pd.read_parquet(root / str(split) / "X_test.parquet")
        y = pd.read_parquet(root / str(split) / "y_test.parquet").squeeze("columns")

        base_preds = []
        for art in arts:
            X_ = X[art["columns"]].copy()
            for col in art["categorical_cols"]:
                X_[col] = X_[col].astype("category")
            if art["algo"] == "lightgbm":
                y_hat = art["model"].predict(
                    X_, num_iteration=art["model"].best_iteration)
            else:  # catboost
                y_hat = art["model"].predict(X_.values)
            base_preds.append(y_hat)

        # bagging
        y_bag = np.mean(base_preds, axis=0)
        bag_scores.append(r2_score(y, y_bag))

        # stacking (if meta-model exists)
        if meta is not None:
            y_stack = meta.predict(np.column_stack(base_preds))
            stack_scores.append(r2_score(y, y_stack))
            print(f"split {split}:  bag R² = {bag_scores[-1]:.4f}   stack R² = {stack_scores[-1]:.4f}")
        else:
            print(f"split {split}:  bag R² = {bag_scores[-1]:.4f}")

    print("\nMean bag  R² =", round(np.mean(bag_scores), 4))
    if meta is not None:
        print("Mean stack R² =", round(np.mean(stack_scores), 4))

# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
