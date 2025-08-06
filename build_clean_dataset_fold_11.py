#!/usr/bin/env python3
"""
build_clean_superconductivity.py

Concatenate original superconductivity folds 1–10, de-duplicate,
and create a new fold "11" with train/test split (default ~10%).

Usage:
    python build_clean_superconductivity.py [--test_size 0.10] [--random_state 42]
"""

import sys
from pathlib import Path
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split

# Ensure our src folder is on the import path
proj_root = Path(__file__).resolve().parent
sys.path.insert(0, str(proj_root / "src"))

# We'll use load_fold directly to avoid fold-11
from data.load import load_fold, get_available_folds, DATA_ROOT

def main(test_size: float, random_state: int):
    dataset = "superconductivity"
    ds_dir = DATA_ROOT / dataset

    # 1) Determine original folds (numeric < 11)
    folds = [f for f in get_available_folds(dataset, DATA_ROOT) if f < 11]

    # 2) Load & concatenate all train and test from folds 1–10
    X_tr_parts, y_tr_parts = [], []
    X_te_parts, y_te_parts = [], []
    for fold in folds:
        X_tr, X_te, y_tr, y_te = load_fold(dataset, fold, DATA_ROOT)
        X_tr_parts.append(X_tr)
        y_tr_parts.append(y_tr.squeeze())
        X_te_parts.append(X_te)
        y_te_parts.append(y_te.squeeze())

    X_all = pd.concat(X_tr_parts + X_te_parts, ignore_index=True)
    y_all = pd.concat(y_tr_parts + y_te_parts, ignore_index=True)

    # 3) Drop duplicates
    df_all = X_all.copy()
    df_all["target"] = y_all.values
    df_all = df_all.drop_duplicates(ignore_index=True)
    y_all = df_all["target"]
    X_all = df_all.drop(columns=["target"])

    # 4) Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_all, y_all,
        test_size=test_size,
        random_state=random_state,
        shuffle=True
    )

    # 5) Write fold 11
    out_dir = ds_dir / "11"
    out_dir.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(out_dir / "X_train.parquet")
    y_train.to_frame(name="target").to_parquet(out_dir / "y_train.parquet")
    X_test.to_parquet(out_dir / "X_test.parquet")
    y_test.to_frame(name="target").to_parquet(out_dir / "y_test.parquet")

    print(f"Built clean fold '11' for '{dataset}' in {out_dir}")
    print(f"  Train rows: {len(X_train)}, Test rows: {len(X_test)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build clean superconductivity fold '11' from folds 1–10"
    )
    parser.add_argument(
        "--test_size", type=float, default=(1378/13776),
        help="Fraction of data to hold out as test (default ≃0.10)."
    )
    parser.add_argument(
        "--random_state", type=int, default=42,
        help="Random seed for reproducibility."
    )
    args = parser.parse_args()
    main(args.test_size, args.random_state)
