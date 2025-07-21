# src/models/tabnet/tabnet_ensembling.py

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from pytorch_tabnet.tab_model import TabNetRegressor
from sklearn.metrics import r2_score
from data import get_available_folds, load_fold
from data.preprocess import preprocess_features

#-----------------------------
# Utility: Function to convert data to float32 numpy arrays
#-----------------------------
def to_numpy_float32(x):
    if isinstance(x, pd.DataFrame):
        return x.astype(np.float32).values
    elif isinstance(x, pd.Series):
        return x.astype(np.float32).values
    else:
        return np.array(x).astype(np.float32)

#-----------------------------
# Utility: load model from folder
#-----------------------------
def load_tabnet_model(model_path):
    model = TabNetRegressor()
    model.load_model(model_path)
    return model

#-----------------------------
# Utility: Function to ensemble predictions from multiple models
#-----------------------------
def ensemble_predictions(models, X):
    preds = [model.predict(X).ravel() for model in models]
    return np.mean(preds, axis=0)

#-----------------------------
# Main function to run ensembling across all folds
#-----------------------------  
def main(dataset: str, model_dir: Path):
    folds = get_available_folds(dataset)
    r2_scores = []

    for fold in folds:
        print(f"\n[+] Evaluating Fold {fold} with Ensemble")
        # load fold data
        X_tr, X_te, y_tr, y_te = load_fold(dataset, fold)
        X_tr, X_te = preprocess_features(X_tr, X_te)

        X_te_np = to_numpy_float32(X_te)
        y_te_np = to_numpy_float32(y_te).reshape(-1, 1)

        # Load all models except current fold
        model_paths = [model_dir / f"tabnet_fold{f}.zip" for f in folds if f != fold]
        models = [load_tabnet_model(str(path)) for path in model_paths]

        # Ensemble Predictions
        preds = ensemble_predictions(models, X_te_np)

        r2 = r2_score(y_te_np.ravel(), preds)
        print(f"[+] Ensemble R2 for Fold {fold}: {r2:.4f}")
        r2_scores.append(r2)
    # Overall R2 score across all folds
    print(f"\n[+] Mean Ensemble R2 across folds: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

#-----------------------------
# CLI
#-----------------------------  
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-m", "--model-dir", type=Path, default=Path("models/trial_tabnet"))
    args = parser.parse_args()

    main(args.dataset, args.model_dir)