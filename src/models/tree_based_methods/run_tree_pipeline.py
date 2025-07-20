# run_tree_pipeline.py
# pip install lightgbm==4.3.0 catboost==1.2.5 optuna==3.6.0 pandas pyarrow joblib
import gc, os, sys, json, warnings, joblib
from pathlib import Path

import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Determine script and models directory relative to this file
SCRIPT_DIR = Path(__file__).resolve().parent             # .../src/tree_based_methods
MODELS_BASE = SCRIPT_DIR / "models"                       # .../src/tree_based_methods/models

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    CatBoostRegressor = Pool = None


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #
def load_split(root: Path, split: int):
    """Return X_train, y_train, X_test, y_test for a numeric <split> dir."""
    base = root / str(split)
    X_train = pd.read_parquet(base / "X_train.parquet")
    y_train = pd.read_parquet(base / "y_train.parquet").squeeze("columns")
    X_test = pd.read_parquet(base / "X_test.parquet")
    y_test = pd.read_parquet(base / "y_test.parquet").squeeze("columns")
    return X_train, y_train, X_test, y_test


def mark_categoricals(df: pd.DataFrame, frac_threshold: float = 0.05):
    """
    Cast columns to category if:
        - dtype=='object'
        - dtype is integer-like *and* unique/rows < threshold.
    Returns list of category column names.
    """
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
            cat_cols.append(col)
        elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() < frac_threshold * len(
            df
        ):
            cat_cols.append(col)

    for c in cat_cols:
        df[c] = df[c].astype("category")
    return cat_cols


# --------------------------------------------------------------------------- #
#  LightGBM with Optuna                                                       #
# --------------------------------------------------------------------------- #
def tune_lgbm(X, y, cat_cols, n_trials=60, seed=0):
    dtrain = lgb.Dataset(X, y, categorical_feature=cat_cols, free_raw_data=False)

    def objective(trial):
        param = {...}  # same as before
        cv = lgb.cv(...)
        ...

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial.params
    best.update({...})

    booster = lgb.train(
        best,
        dtrain,
        num_boost_round=study.best_trial.user_attrs["best_iter"],
    )
    return booster, study


# --------------------------------------------------------------------------- #
#  CatBoost with Optuna                                                       #
# --------------------------------------------------------------------------- #
def tune_cat(X, y, cat_cols, n_trials=60, seed=0):
    ...  # same as before
    return final_model, study


# --------------------------------------------------------------------------- #
#  Main                                                                       #
# --------------------------------------------------------------------------- #
def run_dataset(root: Path, n_trials=60):
    # create models folder for this dataset under SRC/models/<dataset>
    out_dir = MODELS_BASE / root.name
    out_dir.mkdir(parents=True, exist_ok=True)

    for split in sorted(p.name for p in root.iterdir() if p.is_dir()):
        print(f"\n─── Dataset {root.name} | split {split} ─────────────")
        Xtr, ytr, Xte, yte = load_split(root, int(split))

        cat_cols = mark_categoricals(Xtr, frac_threshold=0.05)
        algo = "catboost" if len(cat_cols)/len(Xtr.columns) >= 0.30 else "lightgbm"

        print(...)

        if algo == "lightgbm":
            model, study = tune_lgbm(Xtr, ytr, cat_cols, n_trials=n_trials)
            preds = model.predict(Xte[Xtr.columns], num_iteration=model.best_iteration)
        else:
            model, study = tune_cat(Xtr, ytr, cat_cols, n_trials=n_trials)
            preds = model.predict(Xte.values)

        r2 = r2_score(yte, preds)
        print(f"   ▸ R² on hidden test split = {r2:0.4f}")

        # save model artifact
        joblib.dump(
            {"model": model,
             "algo": algo,
             "categorical_cols": cat_cols,
             "columns": list(Xtr.columns),
             "best_params": study.best_trial.params},
            out_dir / f"{algo}_split{split}.pkl",
            compress=("xz", 9),
        )

        # free RAM
        del model, Xtr, ytr, Xte, yte, preds
        gc.collect()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage:  python run_tree_pipeline.py <data_folder>")
        sys.exit(1)

    data_path = Path(sys.argv[1]).expanduser()
    run_dataset(data_path, n_trials=60)


# run by 
# python3 python run_tree_pipeline.py data/bike_sharing_demand