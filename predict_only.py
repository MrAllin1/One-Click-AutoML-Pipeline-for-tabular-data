# predict_only.py
#!/usr/bin/env python3
"""
Prediction script for CatBoost/LightGBM models with chunked inference and feature engineering.

Loads either a single X_test.parquet or multiple per-split under --test_root,
applies feature engineering, loads a trained model named tree_model.pkl (or ensemble of .pkl), performs chunked inference,
and saves a CSV of predictions.

Usage:
    python predict_only.py \
        --test_root /path/to/data/exam_dataset/1 \
        --model_root /path/to/model_directory \
        --output_path /path/to/predictions/predictions.csv \
        [--rows_per_chunk 2000]
"""
import os
import sys
import argparse
import gzip
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

# Ensure the auto_ml_pipeline package directory is on sys.path to import engineer_features
script_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust this path to where your package lives
pkg_path = os.path.join(script_dir, 'src', 'models', 'tree_based_methods', 'auto_ml_pipeline_project')
if pkg_path not in sys.path:
    sys.path.insert(0, pkg_path)

from auto_ml_pipeline.feature_engineering import engineer_features


def load_tests(root_path: str) -> pd.DataFrame:
    p = Path(root_path)
    single = p / 'X_test.parquet'
    if single.exists():
        df = pd.read_parquet(single)
        print(f"Loaded single X_test.parquet with shape {df.shape}")
        return df
    X_list = []
    for sub in sorted(p.iterdir()):
        if sub.is_dir():
            xp = sub / 'X_test.parquet'
            if xp.exists():
                X_list.append(pd.read_parquet(xp))
            else:
                print(f"Warning: missing X_test in {sub}")
    if not X_list:
        raise RuntimeError(f"No X_test.parquet found under {root_path}")
    df = pd.concat(X_list, ignore_index=True)
    print(f"Concatenated {len(X_list)} splits → shape {df.shape}")
    return df


def load_model(root: Path):
    if root.is_dir():
        model_file = root / 'tree_model.pkl'
        if model_file.exists():
            with open(model_file, 'rb') as f:
                print(f"Loading model from {model_file}")
                return pickle.load(f)
        models = [pickle.load(open(fp, 'rb')) for fp in sorted(root.glob('*.pkl'))]
        from sklearn.ensemble import VotingRegressor
        print(f"Loaded {len(models)} models into VotingRegressor ensemble")
        return VotingRegressor([(f'm{i}', m) for i, m in enumerate(models)])
    else:
        opener = gzip.open if root.suffix == '.gz' else open
        with opener(root, 'rb') as f:
            print(f"Loading model from {root}")
            return pickle.load(f)


def predict_in_chunks(model, X: pd.DataFrame, chunk_size: int) -> np.ndarray:
    preds = []
    for start in range(0, len(X), chunk_size):
        end = start + chunk_size
        preds.append(model.predict(X.iloc[start:end]))
    return np.concatenate(preds)


def main():
    parser = argparse.ArgumentParser(description='Predict and save CSV of model outputs')
    parser.add_argument('--test_root', required=True, help='Directory containing X_test.parquet or splits')
    parser.add_argument('--model_root', required=True, type=Path, help='Directory or file path for model; dir should contain tree_model.pkl')
    parser.add_argument('--output_path', required=True, type=Path, help='CSV file to write predictions')
    parser.add_argument('--rows_per_chunk', type=int, default=2000, help='Chunk size for inference (0 to disable)')
    args = parser.parse_args()

    # ensure output directory exists
    out_dir = args.output_path.parent
    os.makedirs(out_dir, exist_ok=True)
    print(f"Output directory ensured at {out_dir}")

    # load test data
    X_test = load_tests(args.test_root)

    # feature engineering
    print("Applying feature engineering...")
    X_test_fe, _ = engineer_features(X_test, X_test.copy())
    print(f"Engineered to {X_test_fe.shape[1]} features")

    # load model
    model = load_model(args.model_root)

    # predict
    if args.rows_per_chunk > 0:
        print(f"Predicting in chunks of {args.rows_per_chunk} rows...")
        preds = predict_in_chunks(model, X_test_fe, args.rows_per_chunk)
    else:
        print("Predicting all at once...")
        preds = model.predict(X_test_fe)

    # save predictions
    pd.DataFrame({'prediction': preds}).to_csv(args.output_path, index=False)
    print(f"Saved predictions to {args.output_path}")

if __name__ == '__main__':
    main()

# python predict_only.py   \
#     --test_root data/exam_dataset/1   \
#     --model_root models/tree_based_methods_output/exam   \
#     --output_path models/tree_based_methods_output/exam/predictions/predictions.csv   \
#     --rows_per_chunk 2000

# took 1h and 3min to train