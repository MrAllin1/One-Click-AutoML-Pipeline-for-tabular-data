import argparse
import subprocess
import re
import sys
import io
from typing import Tuple
from .tabnet_train import main as _train_main
from .tabnet_ensembling import main as _ensemble_main

# New imports for final pipeline
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import r2_score
import joblib
from .tabnet_ensembling import load_model, to_numpy_float32

#-----------------------------
#  Function to run TabNet training via subprocess (unchanged)
#-----------------------------
def run_training(dataset_path: str, output_dir: str):
    n_splits = 10
    n_trials = 20
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

#-----------------------------
#  Function to run TabNet ensembling and prediction
#-----------------------------  
def run_prediction(dataset_path: str, model_dir: str):
    output_file = "models/exam_output/y_pred.csv"  # fixed prediction output path
    print(f"\n[+] Starting TabNet Ensemble Prediction on Test Set...")
    predict_cmd = [
        "python", "src/models/tabnet/tabnet_ensembling.py",
        "--dataset_dir", dataset_path,
        "--model_dir", model_dir,
        "--output_file", output_file
    ]
    subprocess.run(predict_cmd, check=True)
    print(f"[+] Prediction completed. Predictions saved to {output_file}")

#-----------------------------
#  High-level API for external scripts
#-----------------------------
def tabnet_final_pipeline(
    dataset_path: str,
    output_dir: str,
    n_splits: int = 10,
    n_trials: int = 20
) -> Tuple[Path, float]:
    """
    Train TabNet with KFold CV + Optuna on the train set,
    ensemble the fold models on the test set,
    compute R², pickle the ensemble,
    and return (model_path, r2).
    """
    # Prepare directory for TabNet artifacts
    outdir = Path(output_dir) / "tabnet"
    outdir.mkdir(parents=True, exist_ok=True)

    # 1) run CV+Optuna training
    _train_main(dataset_path, str(outdir), n_splits, n_trials)

    # 2) ensemble predictions into y_pred.csv
    preds_csv = outdir / "y_pred.csv"
    _ensemble_main(Path(dataset_path), outdir, preds_csv)

    # 3) compute R² on test set
    df_pred = pd.read_csv(preds_csv)
    y_pred = df_pred["prediction"].values
    y_true = pd.read_parquet(Path(dataset_path) / "y_test.parquet").values.ravel()
    test_r2 = r2_score(y_true, y_pred)

    # 4) load individual fold models and build average ensemble
    model_files = sorted(outdir.glob("tabnet_fold*.zip"))
    models = [load_model(p) for p in model_files]

    class TabNetEnsemble:
        def __init__(self, models):
            self.models = models
        def predict(self, X):
            X_np = to_numpy_float32(X)
            preds = np.vstack([m.predict(X_np).ravel() for m in self.models])
            return preds.mean(axis=0)

    ensemble = TabNetEnsemble(models)

    # 5) pickle the ensemble for downstream use
    model_pkl = outdir / "final_tabnet.pkl"
    joblib.dump(ensemble, model_pkl)

    return model_pkl, test_r2

#-----------------------------
#  CLI
#-----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pipeline: Train TabNet with KFold + Ensemble Predict")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to dataset folder")
    parser.add_argument("--output_dir", type=str, default="models/exam_output", help="Directory to save fold models")

    args = parser.parse_args()

    run_training(args.dataset_path, args.output_dir)
    run_prediction(args.dataset_path, args.output_dir)
