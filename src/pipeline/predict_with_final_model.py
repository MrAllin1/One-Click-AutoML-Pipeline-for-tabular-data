#!/usr/bin/env python3
"""
Script to load a weighted ensemble model and generate predictions on X_test via get_test_data,
applying feature engineering only for the tree-based submodel.
Pass raw DataFrame to PFN to preserve feature names, and numpy array to CatBoost.

Usage:
    python predict_with_ensemble.py \
        --dataset /path/to/data_root \
        --model-file /path/to/final_model.pkl \
        [--fold 0] \
        --output /path/to/y_pred.csv

Supported output formats: .csv
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from data import get_test_data
# feature engineering for tree-based model
from models.tree_based_methods.auto_ml_pipeline_project.auto_ml_pipeline.feature_engineering import engineer_features
from catboost.core import CatBoostRegressor

class WeightedEnsemble:
    """
    Unpickle-compatible ensemble that applies raw DataFrame inputs to PFN-derived models
    (preserving feature names) and engineered numpy inputs to CatBoost.
    """
    def __init__(self, models, weights):
        self.models = models
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        X: raw DataFrame of test features.
        Returns weighted blend of submodel predictions.
        """
        # generate engineered features DataFrame for tree-based
        X_fe, _ = engineer_features(X, X.copy())
        preds = []
        for m in self.models:
            # CatBoost expects numpy array without pandas Categorical columns
            if isinstance(m, CatBoostRegressor):
                input_data = X_fe.values
            else:
                # TabPFNRegressor (and other sklearn-like) expects DataFrame to match feature_names_in_
                input_data = X
            preds.append(m.predict(input_data))
        return np.tensordot(self.weights, preds, axes=[0, 0])


def load_ensemble_model(path: Path) -> WeightedEnsemble:
    """Load the ensemble pickle file containing a WeightedEnsemble."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_predictions(y_pred: np.ndarray, out_path: Path):
    """Save predictions DataFrame to CSV."""
    pd.DataFrame({'y_pred': y_pred}).to_csv(out_path, index=False)


def main():
    parser = argparse.ArgumentParser(
        description="Run prediction using a weighted ensemble pickle")
    parser.add_argument(
        "-d", "--dataset", required=True,
        help="Path to dataset root (contains fold subfolders)")
    parser.add_argument(
        "-m", "--model-file", required=True,
        help="Path to the final_model.pkl file")
    parser.add_argument(
        "-f", "--fold", type=int, default=None,
        help="Fold number to use (overrides any simulated fold)")
    parser.add_argument(
        "-o", "--output", required=True,
        help="CSV file to save predictions")
    args = parser.parse_args()

    # load raw test DataFrame
    dataset_dir = Path(args.dataset)
    X_test = get_test_data(str(dataset_dir), fold=args.fold)

    # load ensemble and predict
    ensemble = load_ensemble_model(Path(args.model_file))
    y_pred = ensemble.predict(X_test)

    # save results
    save_predictions(y_pred, Path(args.output))
    print(f"Saved predictions to: {args.output}")

if __name__ == '__main__':
    main()
