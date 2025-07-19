#!/usr/bin/env python
# train_final_model.py  ────────────────────────────────────────────────
"""
One-click AutoML for any dataset folder that follows the Freiburg-template:

    python train_final_model.py data/brazilian_houses
    python train_final_model.py data/wine_quality      --tier S-balanced
"""

import gc, sys, joblib, numpy as np, pandas as pd, optuna, warnings
from optuna.samplers import TPESampler
from pathlib import Path
from sklearn.model_selection import KFold
warnings.filterwarnings("ignore", category=UserWarning)

# ─────────────────────────────────────── libraries ─
try:
    import lightgbm as lgb
except ImportError:
    lgb = None
try:
    from catboost import CatBoostRegressor, Pool
except ImportError:
    CatBoostRegressor = Pool = None

# ─────────────────────────────── configuration tiers ─
TIERS = {
    "S-quick":    dict(n_trials=10,  early_stop=50,  folds=3, row_frac=0.70),
    "S-medium":   dict(n_trials=15,  early_stop=75,  folds=3, row_frac=1.00),
    "S-balanced": dict(n_trials=20,  early_stop=100, folds=5, row_frac=1.00),
    "S-slow":     dict(n_trials=30,  early_stop=100, folds=5, row_frac=1.00),
}

# default tier
CFG = TIERS["S-medium"]

# ───────────────────────────────────── helpers ─────────
def mark_cats(df, thr=0.05):
    cats = []
    for col in df.columns:
        if (df[col].dtype == "object"
            or pd.api.types.is_categorical_dtype(df[col])):
            cats.append(col)
        elif (pd.api.types.is_integer_dtype(df[col])
              and df[col].nunique() < thr * len(df)):
            cats.append(col)
    for c in cats:
        df[c] = df[c].astype("category")
    return cats

# ─────────────────────────────── lightgbm tuner ───────
def tune_lgbm(X, y, cats, cfg, seed=0):
    if cfg["row_frac"] < 1.0:
        samp = np.random.RandomState(0).choice(len(X),
                 int(cfg["row_frac"] * len(X)), replace=False)
        X, y = X.iloc[samp], y.iloc[samp]

    dtrain = lgb.Dataset(X, y, categorical_feature=cats, free_raw_data=False)

    def objective(trial):
        p = dict(
            objective="regression", metric="rmse", boosting_type="gbdt",
            seed=seed, num_threads=4, verbosity=-1,
            learning_rate=trial.suggest_float("lr", 0.01, 0.15, log=True),
            num_leaves=trial.suggest_int("leaves", 32, 256, step=8),
            max_depth=trial.suggest_int("depth", 3, 10),
            min_child_samples=trial.suggest_int("mcs", 5, 60),
            subsample=trial.suggest_float("sub", 0.5, 1),
            colsample_bytree=trial.suggest_float("col", 0.5, 1),
            lambda_l1=trial.suggest_float("l1", 0, 5),
            lambda_l2=trial.suggest_float("l2", 0, 5),
        )
        cv = lgb.cv(
            p, dtrain, nfold=cfg["folds"], seed=seed, num_boost_round=10_000,
            callbacks=[lgb.early_stopping(cfg["early_stop"], False),
                       lgb.log_evaluation(0)])
        trial.set_user_attr("best_iter", len(cv[next(iter(cv))]))
        return cv[next(iter(cv))][-1]

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=True)

    best = study.best_trial.params | dict(
        objective="regression", metric="rmse", boosting_type="gbdt",
        seed=seed, num_threads=4)
    model = lgb.train(
        best, dtrain, num_boost_round=study.best_trial.user_attrs["best_iter"])
    return model, study.best_trial.params

# ─────────────────────────────── catboost tuner ───────
def tune_cat(X, y, cats, cfg, seed=0):
    if cfg["row_frac"] < 1.0:
        samp = np.random.RandomState(0).choice(len(X),
                 int(cfg["row_frac"] * len(X)), replace=False)
        X, y = X.iloc[samp], y.iloc[samp]

    Xnp   = X.values
    cat_i = [X.columns.get_loc(c) for c in cats]

    def objective(trial):
        p = dict(
            loss_function="RMSE", eval_metric="RMSE", random_seed=seed,
            iterations=10_000, task_type="CPU", thread_count=4,
            learning_rate=trial.suggest_float("lr", 0.01, 0.2, log=True),
            depth=trial.suggest_int("depth", 4, 10),
            l2_leaf_reg=trial.suggest_float("l2", 1e-3, 10, log=True),
            bagging_temperature=trial.suggest_float("bag", 0, 1),
            border_count=trial.suggest_int("border", 32, 255),
            od_type="Iter", od_wait=cfg["early_stop"], verbose=0,
        )
        rms = []
        kf = KFold(cfg["folds"], shuffle=True, random_state=seed)
        for tr, vl in kf.split(Xnp):
            m = CatBoostRegressor(**p)
            m.fit(Pool(Xnp[tr], y.iloc[tr], cat_i),
                  eval_set=Pool(Xnp[vl], y.iloc[vl], cat_i),
                  use_best_model=True, verbose=False)
            rms.append(m.best_score_["validation"]["RMSE"])
        return np.mean(rms)

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=42))
    study.optimize(objective, n_trials=cfg["n_trials"], show_progress_bar=True)

    best = study.best_trial.params | dict(
        loss_function="RMSE", eval_metric="RMSE", random_seed=seed,
        iterations=10_000, task_type="CPU", thread_count=4,
        od_type="Iter", od_wait=cfg["early_stop"], verbose=0)
    final = CatBoostRegressor(**best)
    final.fit(Pool(Xnp, y, cat_i), use_best_model=True)
    return final, study.best_trial.params

# ─────────────────────────────────── main pipeline ────
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data", help="dataset folder (e.g. data/brazilian_houses)")
    ap.add_argument("--tier", choices=TIERS.keys(), default="S-medium",
                    help="speed/accuracy tier (default: S-medium)")
    args = ap.parse_args()

    cfg = TIERS[args.tier]
    root = Path(args.data).expanduser()
    print(f"▶ {root.name}  |  tier={args.tier}")

    # gather all labelled rows
    Xs, ys = [], []
    for sp in sorted(
        (p for p in root.iterdir() if p.is_dir() and p.name.isdigit()),
        key=lambda p: int(p.name)):
        Xs += [pd.read_parquet(sp/"X_train.parquet"),
               pd.read_parquet(sp/"X_test.parquet")]
        ys += [pd.read_parquet(sp/"y_train.parquet").squeeze(),
               pd.read_parquet(sp/"y_test.parquet").squeeze()]
    X, y = pd.concat(Xs, ignore_index=True), pd.concat(ys, ignore_index=True)
    cats = mark_cats(X)
    algo = "catboost" if len(cats)/len(X.columns) >= 0.30 else "lightgbm"
    print(f"Rows={len(X):,}  |  features={len(X.columns)}  "
          f"cats={len(cats)}  →  {algo}")

    if algo == "lightgbm":
        if lgb is None: sys.exit("install lightgbm first!")
        model, best = tune_lgbm(X, y, cats, cfg)
    else:
        if CatBoostRegressor is None: sys.exit("install catboost first!")
        model, best = tune_cat(X, y, cats, cfg)

    out_dir = Path("models") / root.name
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(dict(model=model, algo=algo,
                     categorical_cols=cats, columns=list(X.columns),
                     best_params=best),
                out_dir/"final_model.pkl", compress=("xz", 9))
    print("✅ saved", out_dir/"final_model.pkl")

if __name__ == "__main__":
    main()
