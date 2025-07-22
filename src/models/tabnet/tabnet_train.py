# src/models/tabnet/tabnet_train.py

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from sklearn.metrics import r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
from optuna import create_study
from optuna.samplers import TPESampler
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
# Function to train TabNet with given hyperparameters
#-----------------------------
def train_tabnet_with_params(X_tr, y_tr, X_val, y_val, params):
    model = TabNetRegressor(
        n_d=params['n_d'],
        n_a=params['n_a'],
        n_steps=params['n_steps'],
        gamma=params['gamma'],
        lambda_sparse=params['lambda_sparse'],
        optimizer_params=dict(lr=params['lr']),
        seed=42,
        verbose=0,
    )

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        eval_metric=['rmse'],
        patience=15,
        max_epochs=100,
        batch_size=512,
        virtual_batch_size=128,
    )

    return model

#-----------------------------
# Optuna objective function for hyperparameter tuning
#-----------------------------
def objective(trial, X_tr, y_tr, X_val, y_val):
    params = {
        'n_d': trial.suggest_int('n_d', 8, 64),
        'n_a': trial.suggest_int('n_a', 8, 64),
        'n_steps': trial.suggest_int('n_steps', 3, 10),
        'gamma': trial.suggest_float('gamma', 1.0, 2.0),
        'lambda_sparse': trial.suggest_float('lambda_sparse', 1e-5, 1e-3, log=True),
        'lr': trial.suggest_float('lr', 1e-3, 1e-1, log=True)
    }

    model = train_tabnet_with_params(X_tr, y_tr, X_val, y_val, params)

    preds = model.predict(X_val).ravel()
    score = r2_score(y_val.ravel(), preds)

    return -score  # because Optuna minimizes

# -----------------------------
# Main function: fold-wise training, tuning, and model saving
# -----------------------------

def main(dataset: str, out_dir: Path):
    folds = get_available_folds(dataset)
    scores = []

    out_dir.mkdir(parents=True, exist_ok=True)

    for fold in folds:
        print(f"\n[+] Fold {fold}:")

        # --- Load and preprocess data ---
        X_tr, X_te, y_tr, y_te = load_fold(dataset, fold)

        X_tr, X_te = preprocess_features(X_tr, X_te)

        # --- Convert data to float32 numpy arrays ---
        X_tr_np = to_numpy_float32(X_tr)
        y_tr_np = to_numpy_float32(y_tr).reshape(-1, 1)

        X_te_np = to_numpy_float32(X_te)
        y_te_np = to_numpy_float32(y_te).reshape(-1, 1)

        # --- Hyperparameter tuning with Optuna ---
        study = create_study(direction='minimize', sampler=TPESampler(seed=42))
        study.optimize(lambda trial: objective(trial, X_tr_np, y_tr_np, X_te_np, y_te_np), n_trials=20)

        best_params = study.best_params
        print(f"[+] Best TabNet Params for fold {fold}: {best_params}")

        # --- Train final model with best params ---
        final_model = train_tabnet_with_params(X_tr_np, y_tr_np, X_te_np, y_te_np, best_params)

        # --- Evaluate final model ---
        preds = final_model.predict(X_te_np).ravel()
        r2 = r2_score(y_te_np.ravel(), preds)
        print(f"[+] Final TabNet R2 for Fold {fold}: {r2:.4f}")

        scores.append(r2)

        final_model.save_model(str(out_dir / f"tabnet_fold{fold}"))
    
    print(f"\n[+] Mean R2 across folds: {np.mean(scores):.4f} ± {np.std(scores):.4f}")

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", required=True)
    parser.add_argument("-o", "--out-dir", type=Path, default=Path("models/tabnet_ensemble"))
    args = parser.parse_args()

    main(args.dataset, args.out_dir)