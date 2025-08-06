# evaluate_test.py
#!/usr/bin/env python3
"""
Evaluation script at project root: loads a trained model, loads test parquet via full paths,
applies feature engineering, aligns features, predicts on X_test, computes and prints R².
"""
import sys
import os
import argparse
import gzip
import pickle
from pathlib import Path
import pandas as pd
from sklearn.metrics import r2_score

# Ensure auto_ml_pipeline package is on PYTHONPATH
PROJECT_ROOT = os.path.dirname(__file__)
PACKAGE_PATH = os.path.join(PROJECT_ROOT, 'src', 'models', 'tree_based_methods', 'auto_ml_pipeline_project')
if PACKAGE_PATH not in sys.path:
    sys.path.insert(0, PACKAGE_PATH)

from auto_ml_pipeline.feature_engineering import engineer_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate model with R² on test parquet files"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained model .pkl file or directory of pkls.")
    parser.add_argument("--X_test", type=str, required=True,
                        help="Full path to X_test parquet.")
    parser.add_argument("--y_test", type=str, required=True,
                        help="Full path to y_test parquet.")
    return parser.parse_args()


def load_parquet(path):
    print(f"DEBUG: Loading parquet from {path}")
    df = pd.read_parquet(path)
    print(f"DEBUG: Loaded DataFrame with shape {df.shape} and columns: {list(df.columns)}")
    return df


def load_model(path):
    p = Path(path)
    if p.is_dir():
        print(f"DEBUG: Loading all models from directory {path}")
        models = [pickle.load(open(fp, 'rb')) for fp in sorted(p.glob('*.pkl'))]
        from sklearn.ensemble import VotingRegressor
        model = VotingRegressor([(f'm{i}', m) for i, m in enumerate(models)])
        print(f"DEBUG: VotingRegressor with {len(models)} models initialized")
        return model
    else:
        print(f"DEBUG: Loading single model from {path}")
        opener = gzip.open if p.suffix == '.gz' else open
        with opener(p, 'rb') as f:
            model = pickle.load(f)
            print(f"DEBUG: Loaded model type: {type(model)}")
            return model


def main():
    args = parse_args()

    # load test data
    X_test = load_parquet(args.X_test)
    y_test = load_parquet(args.y_test).squeeze()
    print(f"DEBUG: y_test series length={y_test.shape} | head=\n{y_test.head()}")

    # feature engineering
    X_test_fe, _ = engineer_features(X_test, X_test.copy())
    print(f"DEBUG: Feature-engineered test set shape: {X_test_fe.shape}")

    # load model
    model = load_model(args.model_path)

    # align test features to what model expects
    if hasattr(model, 'feature_names_'):
        expected = list(model.feature_names_)
        missing = [f for f in expected if f not in X_test_fe.columns]
        if missing:
            # Add missing columns (constant zero)
            print(f"WARNING: Test data missing features {missing}, filling with zeros")
            for f in missing:
                X_test_fe[f] = 0
        # Now reorder
        X_test_fe = X_test_fe[expected]
        print(f"DEBUG: Aligned test features order: {expected}")
    else:
        print("WARNING: Model has no feature_names_; skipping feature alignment.")

    # predict
    print("DEBUG: Generating predictions on test set...")
    preds = model.predict(X_test_fe)
    print(f"DEBUG: Sample predictions head=\n{preds[:5]}")

    # evaluate
    r2 = r2_score(y_test, preds)
    print(f"R² score on test set: {r2:.4f}")

if __name__ == '__main__':
    main()


# python evaluate_test.py \
#   --model_path   /work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template/models/tree_based_methods_output/bike_sharing_demand_fold11/ensemble.pkl \
#   --X_test       /work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template/data/bike_sharing_demand/11/X_test.parquet \
#   --y_test       /work/dlclarge2/latifajr-dl_lab_project/autoML/automl-exam-ss25-tabular-freiburg-template/data/bike_sharing_demand/11/y_test.parquet
