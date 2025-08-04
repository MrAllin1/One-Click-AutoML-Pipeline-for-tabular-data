#!/usr/bin/env python3
import argparse
import pickle
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from pytorch_tabnet.tab_model import TabNetRegressor

# Feature-engineering imported from training pipeline
from models.tree_based_methods.auto_ml_pipeline_project.auto_ml_pipeline.feature_engineering import engineer_features
# Use centralized data-layer methods
from data import get_test_data
from data.load import load_only_train_for_dataset

# ── LOGGER SETUP ───────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# ── MUST match the training-time definition exactly ─────────────────────────────
class WeightedEnsemble:
    """
    Wraps models supporting .predict(X) to return a weighted average of predictions.
    """
    def __init__(self, models, weights):
        self.models = models
        w = np.array(weights, dtype=float)
        self.weights = w / w.sum()

    def predict(self, X):
        preds = [m.predict(X) for m in self.models]
        return np.tensordot(self.weights, preds, axes=[0,0])

def main():
    p = argparse.ArgumentParser(description="Predict using the trained ensemble model")
    p.add_argument("--dataset",    required=True,
                   help="Name of dataset (folder under data/) to load")
    p.add_argument("--fold",       type=int, default=None,
                   help="Fold number for get_test_data (optional)")
    p.add_argument("--model-dir",  required=True,
                   help="Directory containing final_model.pkl and related artifacts")
    p.add_argument("--output",     default="ensemble_predictions.csv",
                   help="CSV file to save ensemble predictions")
    args = p.parse_args()

    # 1. Load ensemble
    model_dir = Path(args.model_dir)
    ensemble_path = model_dir / "final_model.pkl"
    if not ensemble_path.exists():
        logger.error("Ensemble model not found at %s", ensemble_path)
        sys.exit(1)

    # Now that WeightedEnsemble is defined in __main__, pickle.load will succeed
    with open(ensemble_path, "rb") as f:
        ensemble: WeightedEnsemble = pickle.load(f)
    models = ensemble.models
    weights = ensemble.weights
    logger.info("Loaded ensemble with %d models", len(models))

    # 2. Load full training data for feature engineering
    try:
        X_train_full, _ = load_only_train_for_dataset(args.dataset)
        logger.info("Loaded full training data: %d samples, %d features",
                    X_train_full.shape[0], X_train_full.shape[1])
    except Exception as e:
        logger.error("Failed to load training data: %s", e)
        sys.exit(1)

    # 3. Load test data via data layer
    try:
        X_test_raw = get_test_data(args.dataset, fold=args.fold)
        logger.info("Loaded test data: %d samples, %d features",
                    X_test_raw.shape[0], X_test_raw.shape[1])
    except Exception as e:
        logger.error("Failed to load test data: %s", e)
        sys.exit(1)

    # 4. Basic imputation and outlier capping
    X_test = X_test_raw.copy()
    # Impute missing
    for col in X_train_full.columns:
        if X_test[col].isna().any():
            if pd.api.types.is_numeric_dtype(X_train_full[col]):
                fill = X_train_full[col].median()
            else:
                fill = X_train_full[col].mode().iloc[0]
            X_test[col].fillna(fill, inplace=True)
    # Cap outliers
    for col in X_train_full.select_dtypes(include=[np.number]).columns:
        lo, hi = X_train_full[col].quantile(0.01), X_train_full[col].quantile(0.99)
        X_test[col] = X_test[col].clip(lo, hi)

    # 5. Feature engineering for tree-based model
    try:
        _, X_test_fe = engineer_features(X_train_full, X_test.copy())
        logger.info("Engineered tree-based features: %d cols", X_test_fe.shape[1])
    except Exception as e:
        logger.error("Feature engineering failed: %s", e)
        sys.exit(1)

    # 6. Prepare numpy array for TabNet/TabPFN
    X_test_np = X_test.astype(np.float32).values

    # 7. Generate per-model predictions
    preds = []
    for i, m in enumerate(models):
        if i == 0:
            # TabPFN: raw processed frame
            try:
                p_i = m.predict(X_test)
            except:
                p_i = m.predict(X_test_np)
        elif i == 1:
            # Tree-based: engineered features
            try:
                p_i = m.predict(X_test_fe)
            except:
                p_i = m.predict(X_test_fe.values)
        elif i == 2:
            # TabNet: numpy + inverse-scale
            raw = m.predict(X_test_np).ravel()
            # locate a scaler if present
            scaler = None
            s_files = list(model_dir.rglob("scaler_fold*.pkl"))
            if s_files:
                scaler = joblib.load(s_files[0])
            p_i = scaler.inverse_transform(raw.reshape(-1,1)).ravel() if scaler else raw
        else:
            # fallback
            try:
                p_i = m.predict(X_test)
            except:
                p_i = m.predict(X_test_np)
        preds.append(np.array(p_i).ravel())
        logger.info("Model %d predicted %d values", i, len(preds[-1]))

    # 8. Combine ensemble
    P = np.vstack(preds)
    y_ens = np.tensordot(weights, P, axes=[0,0])
    logger.info("Combined predictions into ensemble of length %d", y_ens.shape[0])

    # 9. Save
    out_df = pd.DataFrame({"prediction": y_ens})
    out_df.to_csv(args.output, index=False)
    logger.info("Saved predictions to %s", args.output)

if __name__ == "__main__":
    main()
