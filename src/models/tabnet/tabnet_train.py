import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from pytorch_tabnet.tab_model import TabNetRegressor
import optuna
from sklearn.preprocessing import StandardScaler
import joblib
from . import config

def to_numpy_float32(x):
    if isinstance(x, (pd.DataFrame, pd.Series)):
        return x.astype(np.float32).values
    return np.array(x).astype(np.float32)

def objective(trial, X_train, y_train, X_val, y_val):
    params = {
        'n_d': trial.suggest_int('n_d', *config.n_d),
        'n_a': trial.suggest_int('n_a', *config.n_a),
        'n_steps': trial.suggest_int('n_steps', *config.n_steps),
        'gamma': trial.suggest_float('gamma', *config.gamma),
        'lambda_sparse': trial.suggest_float(
            'lambda_sparse',
            *config.lambda_sparse['range'],
            log=config.lambda_sparse['log_scale']
        ),
        'lr': trial.suggest_float(
            'lr',
            *config.lr['range'],
            log=config.lr['log_scale']
        ),
        'weight_decay': trial.suggest_float(
            'weight_decay',
            *config.weight_decay['range'],
            log=config.weight_decay['log_scale']
        ),
    }

    clf = TabNetRegressor(
        n_d=params['n_d'],
        n_a=params['n_a'],
        n_steps=params['n_steps'],
        gamma=params['gamma'],
        lambda_sparse=params['lambda_sparse'],
        optimizer_params=dict(lr=params['lr'], weight_decay=params['weight_decay']),
        mask_type=config.mask_type,
        verbose=0,
    )

    clf.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric=['rmse'],
        max_epochs=config.initial_max_epochs,
        patience=config.initial_patience,
        batch_size=config.batch_size,
        virtual_batch_size=config.virtual_batch_size,
        num_workers=config.num_workers,
        drop_last=config.drop_last,     
    )

    preds = clf.predict(X_val).ravel()
    return r2_score(y_val.ravel(), preds)

def train_fold_with_optuna(X_tr, y_tr, X_val, y_val, fold_idx, output_dir, n_trials=config.n_trials):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Fold {fold_idx} started")

    X_train_np = to_numpy_float32(X_tr)
    X_val_np   = to_numpy_float32(X_val)

    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_tr.values.reshape(-1, 1))
    y_val_scaled   = scaler.transform(y_val.values.reshape(-1, 1))
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, output_dir / f"scaler_fold{fold_idx}.pkl")

    study = optuna.create_study(direction=config.direction)
    study.optimize(
        lambda trial: objective(trial, X_train_np, y_train_scaled, X_val_np, y_val_scaled),
        n_trials=n_trials
    )

    print(f"Best params for fold {fold_idx}: {study.best_params}")
    print(f"Best R² for fold {fold_idx}: {study.best_value:.4f}")

    best = study.best_params
    clf = TabNetRegressor(
        n_d=best['n_d'],
        n_a=best['n_a'],
        n_steps=best['n_steps'],
        gamma=best['gamma'],
        lambda_sparse=best['lambda_sparse'],
        optimizer_params=dict(lr=best['lr'], weight_decay=best['weight_decay']),
        mask_type=config.mask_type,
        verbose=0,
    )
    clf.fit(
        X_train_np, y_train_scaled,
        eval_set=[(X_val_np, y_val_scaled)],
        eval_metric=['rmse'],
        max_epochs=config.final_max_epochs,
        patience=config.final_patience,
        batch_size=config.batch_size,
        virtual_batch_size=config.virtual_batch_size,
        num_workers=config.num_workers,
        drop_last=config.drop_last,
    )

    val_preds_scaled = clf.predict(X_val_np).ravel()
    val_preds = scaler.inverse_transform(val_preds_scaled.reshape(-1, 1)).ravel()
    y_val_orig = scaler.inverse_transform(y_val_scaled).ravel()
    val_r2 = r2_score(y_val_orig, val_preds)
    print(f"Fold {fold_idx} final validation R²: {val_r2:.4f}")

    clf.save_model(str(output_dir / f"tabnet_fold{fold_idx}"))
    print(f"Saved fold {fold_idx} model to {output_dir / f'tabnet_fold{fold_idx}'}")

    return val_r2, study.best_params

def main(dataset_dir, output_dir, n_splits=config.n_splits, n_trials=config.n_trials):
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)

    X = pd.read_parquet(dataset_dir / "X_train.parquet")
    y = pd.read_parquet(dataset_dir / "y_train.parquet")

    kf = KFold(n_splits=n_splits, shuffle=config.shuffle, random_state=config.random_state)
    r2_scores = []

    for fold_idx, (train_index, val_index) in enumerate(kf.split(X), start=1):
        X_tr, X_val = X.iloc[train_index], X.iloc[val_index]
        y_tr, y_val = y.iloc[train_index], y.iloc[val_index]
        val_r2, best = train_fold_with_optuna(X_tr, y_tr, X_val, y_val, fold_idx, output_dir, n_trials)
        r2_scores.append(val_r2)

    print(f"\nMean R² across {n_splits} folds: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--n_splits", type=int, default=config.n_splits)
    parser.add_argument("--n_trials", type=int, default=config.n_trials)
    args = parser.parse_args()
    main(args.dataset_dir, args.output_dir, args.n_splits, args.n_trials)
