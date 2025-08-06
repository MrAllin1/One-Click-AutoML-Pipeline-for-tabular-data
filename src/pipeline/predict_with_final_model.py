#!/usr/bin/env python3
"""
Predict with the final_model.pkl ensemble, handling:
  - Tree-based feature engineering (as in training)
  - TabNet fold-wise prediction + inverse-scaling
  - TabPFN direct prediction (no special FE)

Prints every 100th sample’s per-model vs. ensemble output,
then writes all ensemble predictions to CSV.
"""
import argparse
import logging
import pickle                              # <— needed for unpickling
import sys
from pathlib import Path
from tabpfn import TabPFNRegressor 
import numpy as np
import pandas as pd
import joblib
from catboost.core import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

from data import get_test_data
from data.load import load_only_train_for_dataset
from models.tree_based_methods.auto_ml_pipeline_project.auto_ml_pipeline.feature_engineering import engineer_features

# ── LOGGER SETUP ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


# ── MUST match the training‐time definition exactly ─────────────────────────────
class WeightedEnsemble:
    """
    Wraps models supporting .predict(X) to return a weighted average of predictions.
    """
    def __init__(self, models, weights):
        self.models = models
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()

    def predict(self, X):
        # We won't actually use this—see below for per‐model logic.
        preds = [m.predict(X) for m in self.models]
        return np.tensordot(self.weights, preds, axes=[0, 0])


def load_tabnet_fold_preds(tabnet_root: Path, X_np: np.ndarray) -> np.ndarray:
    """
    For each fold directory under tabnet_root:
      1) load its StandardScaler,
      2) load its TabNet zip,
      3) predict on X_np,
      4) inverse‐transform,
    then return the mean across folds.
    """
    preds = []
    for fold_dir in sorted(tabnet_root.iterdir()):
        if not fold_dir.is_dir() or not fold_dir.name.startswith("fold"):
            continue
        scaler = joblib.load(fold_dir / f"scaler_{fold_dir.name}.pkl")
        zip_files = list(fold_dir.glob("*.zip"))
        model_file = zip_files[0]
        tb = TabNetRegressor()
        tb.load_model(str(model_file))
        raw = tb.predict(X_np).ravel()
        preds.append(scaler.inverse_transform(raw.reshape(-1, 1)).ravel())
    return np.mean(preds, axis=0)


def main():
    p = argparse.ArgumentParser(description="Predict with final ensemble")
    p.add_argument("-d", "--dataset",   required=True, help="Path to dataset root")
    p.add_argument("-m", "--model-file",required=True, help="Path to final_model.pkl")
    p.add_argument("-f", "--fold",      type=int, default=None, help="Fold number for get_test_data")
    p.add_argument("-o", "--output",    required=True, help="CSV file to save ensemble preds")
    args = p.parse_args()

    # 1) load test set
    logger.info(f"Loading X_test (fold {args.fold}) from {args.dataset}")
    X_test = get_test_data(str(args.dataset), fold=args.fold)

    # 2) load training set (to drive FE)
    ds_name = Path(args.dataset).name
    X_train_full, _ = load_only_train_for_dataset(ds_name)
    logger.info(f"Loaded full training data ({X_train_full.shape[0]} rows, {X_train_full.shape[1]} cols)")

    # 3) engineer features exactly as in training
    X_tr_fe, X_te_fe = engineer_features(X_train_full, X_test)

    # 4) convert all categories → integer codes, then float32
    for c in X_te_fe.select_dtypes(include="category"):
        X_te_fe[c] = X_te_fe[c].cat.codes
    X_te_fe = X_te_fe.astype(np.float32)

    # 5) unpickle ensemble (uses the inline WeightedEnsemble above)
    logger.info(f"Loading ensemble from {args.model_file}")
    with open(args.model_file, "rb") as f:
        ensemble: WeightedEnsemble = pickle.load(f)

    # 6) assign human-readable names
    names = []
    for m in ensemble.models:
        if isinstance(m, TabNetRegressor):
            names.append("TabNet")
        elif isinstance(m, TabPFNRegressor):
            names.append("TabPFN")
        else:
            names.append("Tree")

    # 7) per-model predictions
    #    - TabPFN on raw X_test
    pfn_preds  = np.asarray(ensemble.models[0].predict(X_test)).ravel()
    #    - CatBoost on engineered X_te_fe
    tree_preds = np.asarray(ensemble.models[1].predict(X_te_fe)).ravel()
    #    - TabNet folds on the same engineered features
    X_np = X_te_fe.values.astype(np.float32)
    tabnet_root = Path(args.model_file).parent / "tabnet"
    tabnet_preds = load_tabnet_fold_preds(tabnet_root, X_np)

    # 8) blend
    mat = np.vstack([pfn_preds, tree_preds, tabnet_preds])      # shape = (3, n)
    ens = ensemble.weights.dot(mat)                            # shape = (n,)

    # 9) print every 100th
    for i in range(0, len(ens), 100):
        parts = [f"{names[j]}={mat[j,i]:.6f}" for j in range(3)]
        parts.append(f"Ensemble={ens[i]:.6f}")
        print(f"Sample {i}: " + ", ".join(parts))

    # 10) save full results
    logger.info(f"Saving predictions to {args.output}")
    pd.DataFrame({"y_pred": ens}).to_csv(args.output, index=False)
    print(f"Saved predictions to: {args.output}", file=sys.stdout)


if __name__ == "__main__":
    main()
