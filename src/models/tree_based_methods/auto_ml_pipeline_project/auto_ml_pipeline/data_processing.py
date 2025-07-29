# auto_ml_pipeline/data_processing.py

"""
Handles data loading, missing values, type detection, outlier capping,
and merging splits—but only for training data.
"""
import pandas as pd
import numpy as np
import os

from config import RANDOM_SEED


def load_data(path):
    """
    Load a single split: expects X_train.parquet & y_train.parquet.
    Returns X_train, y_train.
    """
    X_train = pd.read_parquet(os.path.join(path, 'X_train.parquet'))
    y_train = pd.read_parquet(os.path.join(path, 'y_train.parquet')).squeeze()

    # Missing value handling
    for col in X_train.columns:
        if X_train[col].isna().any():
            if pd.api.types.is_numeric_dtype(X_train[col]):
                med = X_train[col].median()
                X_train[col].fillna(med, inplace=True)
            else:
                mode = X_train[col].mode()[0]
                X_train[col].fillna(mode, inplace=True)

    # Outlier capping
    for col in X_train.select_dtypes(include=[np.number]):
        lower = X_train[col].quantile(0.01)
        upper = X_train[col].quantile(0.99)
        X_train[col] = X_train[col].clip(lower, upper)

    return X_train, y_train


def load_all_splits(root):
    """
    Load all splits under `root/<split>/X_train.parquet` & `y_train.parquet`.
    Returns concatenated X_all, y_all.
    """
    X_list, y_list = [], []
    for split in sorted(os.listdir(root)):
        split_dir = os.path.join(root, split)
        if os.path.isdir(split_dir):
            X_tr = pd.read_parquet(os.path.join(split_dir, 'X_train.parquet'))
            y_tr = pd.read_parquet(os.path.join(split_dir, 'y_train.parquet')).squeeze()
            X_list.append(X_tr)
            y_list.append(y_tr)
    X_all = pd.concat(X_list, axis=0).reset_index(drop=True)
    y_all = pd.concat(y_list, axis=0).reset_index(drop=True)
    return X_all, y_all

def load_all_tests(root):
    """
    Load all splits under `root/<split>/X_test.parquet`, `y_test.parquet`.
    Returns concatenated X_test_all, y_test_all.
    """
    import os
    import pandas as pd

    X_list, y_list = [], []
    for split in sorted(os.listdir(root)):
        split_dir = os.path.join(root, split)
        if os.path.isdir(split_dir):
            X_te = pd.read_parquet(os.path.join(split_dir, 'X_test.parquet'))
            y_te = pd.read_parquet(os.path.join(split_dir, 'y_test.parquet')).squeeze()
            X_list.append(X_te)
            y_list.append(y_te)
    X_test_all = pd.concat(X_list, axis=0).reset_index(drop=True)
    y_test_all = pd.concat(y_list, axis=0).reset_index(drop=True)
    return X_test_all, y_test_all
