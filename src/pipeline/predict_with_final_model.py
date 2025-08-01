#!/usr/bin/env python3
"""
Script to load a weighted ensemble model and generate predictions on X_test via get_test_data,
applying feature engineering only for the tree-based and TabNet submodels.
Emits an INFO log every 100th sample showing each model's prediction and the final ensemble.
"""
import argparse
import pickle
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from data import get_test_data
# feature engineering for tree-based and TabNet
from models.tree_based_methods.auto_ml_pipeline_project.auto_ml_pipeline.feature_engineering import engineer_features
from catboost.core import CatBoostRegressor
from pytorch_tabnet.tab_model import TabNetRegressor

# ── LOGGER SETUP ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    stream=sys.stderr
)
logger = logging.getLogger(__name__)


class WeightedEnsemble:
    """
    Unpickle-compatible ensemble that applies:
      – raw DataFrame inputs to PFN-based models,
      – engineered numpy inputs to CatBoost and TabNet,
    and logs every 100th prediction.
    """
    def __init__(self, models, weights):
        self.models = models
        # human-readable names in parallel with models[]
        self.model_names = []
        for m in models:
            if isinstance(m, CatBoostRegressor):
                self.model_names.append("CatBoost")
            elif isinstance(m, TabNetRegressor):
                self.model_names.append("TabNet")
            else:
                self.model_names.append("TabPFN")
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # engineer features once (for CatBoost & TabNet)
        X_fe, _ = engineer_features(X, X.copy())
        n_samples = X.shape[0]

        # collect flattened preds
        pred_list = []
        for m in self.models:
            if isinstance(m, CatBoostRegressor) or isinstance(m, TabNetRegressor):
                inp = X_fe.values
            else:
                inp = X  # PFN-based expects DataFrame with names
            raw = m.predict(inp)
            arr = np.asarray(raw).ravel()              # ensure 1-D
            if arr.shape[0] != n_samples:
                raise ValueError(f"Model {type(m).__name__} returned {arr.shape[0]} preds but expected {n_samples}")
            pred_list.append(arr)

        # stack into (num_models, n_samples)
        pred_matrix = np.vstack(pred_list)             # shape = (3, n_samples)

        # weighted blend → length‐n array
        ensemble_preds = self.weights.dot(pred_matrix) # shape = (n_samples,)

        # log chain-of-thought every 100 samples
        for idx in range(0, n_samples, 100):
            parts = [f"{name}={pred_matrix[i, idx]:.4f}"
                     for i, name in enumerate(self.model_names)]
            parts.append(f"Ensemble={ensemble_preds[idx]:.4f}")
            logger.info(f"Sample {idx}: " + ", ".join(parts))

        return ensemble_preds


def load_ensemble_model(path: Path) -> WeightedEnsemble:
    """
    Load the ensemble pickle and restore model_names attribute for logging.
    """
    with open(path, 'rb') as f:
        ensemble = pickle.load(f)

    names = []
    for m in ensemble.models:
        if isinstance(m, CatBoostRegressor):
            names.append("CatBoost")
        elif isinstance(m, TabNetRegressor):
            names.append("TabNet")
        else:
            names.append("TabPFN")
    ensemble.model_names = names

    return ensemble



def save_predictions(y_pred: np.ndarray, out_path: Path):
    """Save predictions DataFrame to CSV."""
    pd.DataFrame({'y_pred': y_pred}).to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction using a weighted ensemble pickle"
    )
    parser.add_argument(
        "-d", "--dataset", required=True,
        help="Path to dataset root (contains fold subfolders)"
    )
    parser.add_argument(
        "-m", "--model-file", required=True,
        help="Path to the final_model.pkl file"
    )
    parser.add_argument(
        "-f", "--fold", type=int, default=None,
        help="Fold number to use (overrides any simulated fold)"
    )
    parser.add_argument(
        "-o", "--output", required=True,
        help="CSV file to save predictions"
    )
    args = parser.parse_args()

    logger.info(f"Loading X_test for fold {args.fold} of dataset: {args.dataset}")
    X_test = get_test_data(str(args.dataset), fold=args.fold)

    logger.info(f"Loading ensemble model from: {args.model_file}")
    ensemble = load_ensemble_model(Path(args.model_file))

    logger.info("Computing predictions on X_test...")
    y_pred = ensemble.predict(X_test)

    logger.info(f"Saving predictions to: {args.output}")
    save_predictions(y_pred, Path(args.output))
    # still print to stdout so Slurm notices success
    print(f"Saved predictions to: {args.output}", file=sys.stdout)


if __name__ == '__main__':
    main()
