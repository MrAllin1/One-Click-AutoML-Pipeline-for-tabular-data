#!/usr/bin/env python3
import argparse
import subprocess
import os
import numpy as np
from typing import Tuple
from .tabnet_train import main as _train_main, train_fold_with_optuna
from .tabnet_ensembling import main as _ensemble_main
from data.load import load_only_train_for_dataset
from pathlib import Path
from sklearn.model_selection import KFold
from torch.utils.tensorboard import SummaryWriter
from . import config
from datetime import datetime

def tabnet_model(
    dataset_path: str,
    output_dir: str,
    n_splits: int = config.cv_folds,
    n_trials: int = config.cv_trials,
    seed: int = config.random_state
) -> Tuple[str, float]:
    """
    Train TabNet via K-Fold + Optuna on merged splits, reporting progress via prints
    and logging metrics to TensorBoard in the directory specified by config.tb_log_subdir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory '{output_dir}' ready.")

    # build TB log path from config
    ds_name = Path(dataset_path).name
    tb_dir = os.path.join(
        output_dir,
        config.tb_log_subdir,
        ds_name,
        "tabnet",
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(tb_dir, exist_ok=True) 
    writer = SummaryWriter(log_dir=str(tb_dir))

    ds = Path(dataset_path).name
    X_all, y_all = load_only_train_for_dataset(ds)
    print(f"[INFO] Loaded merged train data: {X_all.shape[0]} rows")

    kf = KFold(n_splits=n_splits, shuffle=config.shuffle, random_state=seed)
    r2_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_all), start=1):
        print(f"[INFO] Fold {fold_idx}/{n_splits} starting...")
        X_tr, X_val = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_val = y_all.iloc[train_idx], y_all.iloc[val_idx]

        fold_dir = Path(output_dir) / f"fold{fold_idx}"
        os.makedirs(fold_dir, exist_ok=True)

        val_r2, best_params = train_fold_with_optuna(
            X_tr, y_tr,
            X_val, y_val,
            fold_idx=fold_idx,
            output_dir=fold_dir,
            n_trials=n_trials,
            tb_writer=writer
        )
        r2_scores.append(val_r2)
        print(f"[INFO] Fold {fold_idx} done. R²={val_r2:.4f}")

    mean_r2 = float(np.mean(r2_scores))
    std_r2  = float(np.std(r2_scores))
    print(f"[INFO] TabNet CV complete — mean R² = {mean_r2:.4f} ± {std_r2:.4f}")

    # log overall summary
    writer.add_scalar("TabNet/mean_r2", mean_r2, 0)
    writer.add_scalar("TabNet/std_r2",  std_r2,  0)
    writer.close()

    return output_dir, mean_r2

def run_training(dataset_path: str, output_dir: str):
    n_splits = config.cv_folds
    n_trials = config.cv_trials
    print(f"\n[+] Starting TabNet {n_splits}-Fold CV Training...")
    train_cmd = [
        "python", "src/models/tabnet/tabnet_train.py",
        "--dataset_dir", dataset_path,
        "--output_dir", output_dir,
        "--n_splits", str(n_splits),
        "--n_trials", str(n_trials)
    ]
    subprocess.run(train_cmd, check=True)
    print(f"[+] Training completed. Fold models saved to {output_dir}")

def run_prediction(dataset_path: str, model_dir: str):
    output_file = config.ensemble_predictions
    print(f"\n[+] Starting TabNet Ensemble Prediction on Test Set...")
    predict_cmd = [
        "python", "src/models/tabnet/tabnet_ensembling.py",
        "--dataset_dir", dataset_path,
        "--model_dir", model_dir,
        "--output_file", output_file
    ]
    subprocess.run(predict_cmd, check=True)
    print(f"[+] Prediction completed. Predictions saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline: Train TabNet with KFold + Ensemble Predict")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--output_dir",   type=str, default=config.model_dir, help="Directory to save fold models")
    args = parser.parse_args()

    run_training(args.dataset_path, args.output_dir)
    run_prediction(args.dataset_path, args.output_dir)
