# feature_engineering.py

"""
Creates additional features, encodes categoricals.
"""
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

def engineer_features(X_train, X_test):
    # Copy to avoid modifying original
    X_tr = X_train.copy()
    X_te = X_test.copy()

    # Identify categorical columns
    cat_cols = []
    for c in X_tr.columns:
        if X_tr[c].dtype == 'object' or X_tr[c].dtype.name == 'category':
            cat_cols.append(c)
        elif pd.api.types.is_integer_dtype(X_tr[c]) and X_tr[c].nunique() < 0.05 * len(X_tr):
            cat_cols.append(c)

    # Cast to category dtype
    for c in cat_cols:
        X_tr[c] = X_tr[c].astype('category')
        X_te[c] = X_te[c].astype('category')

    # Ordinal encode for LightGBM fallback
    oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_tr[cat_cols] = oe.fit_transform(X_tr[cat_cols])
    X_te[cat_cols] = oe.transform(X_te[cat_cols])

    # Datetime expansion
    for c in list(X_tr.columns):
        if np.issubdtype(X_tr[c].dtype, np.datetime64):
            for attr in ['year', 'month', 'day', 'hour', 'weekday']:
                X_tr[f"{c}_{attr}"] = getattr(X_tr[c].dt, attr)
                X_te[f"{c}_{attr}"] = getattr(X_te[c].dt, attr)
            X_tr.drop(columns=[c], inplace=True)
            X_te.drop(columns=[c], inplace=True)

    # Drop only truly constant features (same value everywhere)
    for c in list(X_tr.columns):
        if X_tr[c].nunique() <= 1:
            X_tr.drop(columns=[c], inplace=True)
            X_te.drop(columns=[c], inplace=True)

    return X_tr, X_te
