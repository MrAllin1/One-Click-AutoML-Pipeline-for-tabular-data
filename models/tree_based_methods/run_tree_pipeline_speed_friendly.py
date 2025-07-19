# # fast prototyping
# python run_tree_pipeline.py data/wine_quality          # 25 trials

# # crank up to 40 trials for the final run
# python run_tree_pipeline.py data/wine_quality --n_trials 40


# run_tree_pipeline_speed_friendly.py
# ---------------------------------------------------------------------
# Quick install (CPU):
#   pip install numpy==1.26.4 lightgbm==4.3.0 catboost==1.2.8 \
#               optuna==3.6.0 pandas pyarrow joblib
# ---------------------------------------------------------------------
import gc, sys, warnings, joblib
from pathlib import Path

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore", category=UserWarning)
optuna.logging.set_verbosity(optuna.logging.WARNING)

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    CatBoostRegressor = Pool = None
# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def load_split(root: Path, split: int):
    base = root / str(split)
    X_train = pd.read_parquet(base / "X_train.parquet")
    y_train = pd.read_parquet(base / "y_train.parquet").squeeze("columns")
    X_test  = pd.read_parquet(base / "X_test.parquet")
    y_test  = pd.read_parquet(base / "y_test.parquet").squeeze("columns")
    return X_train, y_train, X_test, y_test


def mark_categoricals(df: pd.DataFrame, frac_threshold: float = 0.05):
    cat_cols = []
    for col in df.columns:
        if df[col].dtype == "object" or pd.api.types.is_categorical_dtype(df[col]):
            cat_cols.append(col)
        elif pd.api.types.is_integer_dtype(df[col]) and df[col].nunique() < frac_threshold * len(df):
            cat_cols.append(col)
    for c in cat_cols:
        df[c] = df[c].astype("category")
    return cat_cols

# ------------------------------------------------------------------ #
#  LightGBM + Optuna                                                 #
# ------------------------------------------------------------------ #
def tune_lgbm(X, y, cat_cols, n_trials=25, seed=0):
    dtrain = lgb.Dataset(X, y, categorical_feature=cat_cols, free_raw_data=False)

    def objective(trial):
        param = {
            "objective": "regression",
            "metric":     "rmse",
            "boosting_type": "gbdt",
            "verbosity":  -1,
            "seed":       seed,
            "num_threads": 2,                       # ≤ Colab vCPU
            # narrower search space
            "learning_rate": trial.suggest_float("lr", 0.01, 0.15, log=True),
            "num_leaves":    trial.suggest_int("num_leaves", 32, 256, step=8),
            "max_depth":     trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 60),
            "subsample":        trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 5.0),
            "feature_pre_filter": False,
        }

        cv = lgb.cv(
            param,
            dtrain,
            nfold=5,
            seed=seed,
            num_boost_round=10_000,
            callbacks=[
                lgb.early_stopping(100, verbose=False),   # patience ↓
                lgb.log_evaluation(period=0),
            ],
        )
        first_key = next(iter(cv))
        best_iter = len(cv[first_key])
        trial.set_user_attr("best_iter", best_iter)
        return cv[first_key][-1]          # minimise CV-RMSE

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial.params | {
        "objective": "regression",
        "metric": "rmse",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "seed": seed,
        "num_threads": 2,
    }

    booster = lgb.train(
        best,
        dtrain,
        num_boost_round=study.best_trial.user_attrs["best_iter"],
    )
    return booster, study

# ------------------------------------------------------------------ #
#  CatBoost + Optuna                                                 #
# ------------------------------------------------------------------ #
def tune_cat(X, y, cat_cols, n_trials=25, seed=0):
    X_np   = X.values
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    def objective(trial):
        params = {
            "loss_function": "RMSE",
            "eval_metric":   "RMSE",
            "random_seed":   seed,
            "iterations":    10_000,
            "task_type":     "CPU",
            "thread_count":  2,
            # search space
            "learning_rate": trial.suggest_float("lr", 0.01, 0.2, log=True),
            "depth":         trial.suggest_int("depth", 4, 10),
            "l2_leaf_reg":   trial.suggest_float("l2", 1e-3, 10.0, log=True),
            "bagging_temperature": trial.suggest_float("bag_temp", 0.0, 1.0),
            "border_count":  trial.suggest_int("border_cnt", 32, 255),
            "od_type": "Iter",
            "od_wait": 100,
            "verbose": 0,
        }

        folds = KFold(n_splits=5, shuffle=True, random_state=seed)
        rmses = []
        for tr_idx, val_idx in folds.split(X_np):
            train_pool = Pool(X_np[tr_idx], y.iloc[tr_idx], cat_features=cat_idx)
            val_pool   = Pool(X_np[val_idx], y.iloc[val_idx], cat_features=cat_idx)
            model = CatBoostRegressor(**params)
            model.fit(train_pool, eval_set=val_pool, use_best_model=True, verbose=False)
            rmses.append(model.best_score_["validation"]["RMSE"])
        return np.mean(rmses)

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_trial.params | {
        "loss_function": "RMSE",
        "eval_metric":   "RMSE",
        "random_seed":   seed,
        "iterations":    10_000,
        "task_type":     "CPU",
        "thread_count":  2,
        "od_type": "Iter",
        "od_wait": 100,
        "verbose": 0,
    }

    final_model = CatBoostRegressor(**best)
    final_model.fit(Pool(X_np, y, cat_features=cat_idx), use_best_model=True)
    return final_model, study

# ------------------------------------------------------------------ #
#  Main loop                                                         #
# ------------------------------------------------------------------ #
def run_dataset(root: Path, n_trials=25):
    for split in sorted((p.name for p in root.iterdir() if p.is_dir()),
                        key=lambda s: int(s)):
        print(f"\n─── Dataset {root.name} | split {split} ─────────────")
        Xtr, ytr, Xte, yte = load_split(root, int(split))

        cat_cols = mark_categoricals(Xtr)
        algo = "catboost" if len(cat_cols) / len(Xtr.columns) >= 0.30 else "lightgbm"
        print(f"{Xtr.shape[0]:>6,d} rows | {Xtr.shape[1]} features "
              f"({len(cat_cols)} categorical → **{algo}**)")

        if algo == "lightgbm":
            if lgb is None:
                raise RuntimeError("pip install lightgbm")
            model, study = tune_lgbm(Xtr, ytr, cat_cols, n_trials=n_trials)
            preds = model.predict(Xte[Xtr.columns], num_iteration=model.best_iteration)
        else:
            if CatBoostRegressor is None:
                raise RuntimeError("pip install catboost")
            model, study = tune_cat(Xtr, ytr, cat_cols, n_trials=n_trials)
            preds = model.predict(Xte.values)

        print(f"   ▸ R² on hidden test split = {r2_score(yte, preds):.4f}")

        out_dir = root / "models"
        out_dir.mkdir(exist_ok=True)
        joblib.dump(
            {"model": model, "algo": algo, "categorical_cols": cat_cols,
             "columns": list(Xtr.columns), "best_params": study.best_trial.params},
            out_dir / f"{algo}_split{split}.pkl", compress=("xz", 9),
        )
        del model, Xtr, ytr, Xte, yte, preds
        gc.collect()

# ------------------------------------------------------------------ #
#  CLI                                                               #
# ------------------------------------------------------------------ #
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", help="Path to dataset folder")
    parser.add_argument("--n_trials", type=int, default=25,
                        help="Optuna trials per split (default 25)")
    args = parser.parse_args()

    run_dataset(Path(args.dataset).expanduser(), n_trials=args.n_trials)
