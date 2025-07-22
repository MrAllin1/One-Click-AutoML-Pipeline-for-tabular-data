#-------------------------------------
# auto_ml_pipeline/hyperparameter_tuning.py
#-------------------------------------
"""
Implements HPO with Optuna for LightGBM and CatBoost, with logging silenced.
"""
import optuna
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
import numpy as np
import lightgbm as lgb  # noqa
# LightGBM logging will be controlled via model constructor verbose flags
from catboost import CatBoostRegressor

from config import MAX_TRIALS, CV_FOLDS, RANDOM_SEED, EARLY_STOPPING_ROUNDS

# Note: LightGBM will suppress warnings via verbose flags on models

def tune_lgbm(X, y):
    """Tune LightGBM via Optuna using sklearn API and callbacks for early stopping."""
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'num_leaves': trial.suggest_int('num_leaves', 8, 256),
            'max_depth': trial.suggest_int('max_depth', 3, 15),
            'min_child_samples': trial.suggest_int('min_child_samples', 10, 100),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
            'random_state': RANDOM_SEED,
            'verbosity': -1,  # silence per-model logging
        }
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for tr_idx, val_idx in cv.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = lgb.LGBMRegressor(**params, n_estimators=1000, verbose=-1)
            model.fit(
                X_tr, y_tr,
                eval_set=[(X_val, y_val)],
                eval_metric='l2',
                callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS)]
            )
            preds = model.predict(X_val)
            scores.append(r2_score(y_val, preds))
        return -np.mean(scores)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=MAX_TRIALS)
    best = study.best_params
    final_model = lgb.LGBMRegressor(**best, n_estimators=study.best_trial.user_attrs.get('best_iteration', 1000), random_state=RANDOM_SEED, verbosity=-1)
    final_model.fit(X, y)
    return final_model, best


def tune_catboost(X, y):
    """Tune CatBoost via Optuna using sklearn-like API."""
    def objective(trial):
        params = {
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'depth': trial.suggest_int('depth', 4, 12),
            'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1, 10),
            'border_count': trial.suggest_int('border_count', 32, 255),
            'random_seed': RANDOM_SEED,
            'verbose': False,
            'allow_writing_files': False,
        }
        cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
        scores = []
        for tr_idx, val_idx in cv.split(X):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            model = CatBoostRegressor(**params, iterations=1000)
            model.fit(
                X_tr, y_tr,
                eval_set=(X_val, y_val),
                early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                verbose=False
            )
            preds = model.predict(X_val)
            scores.append(r2_score(y_val, preds))
        return -np.mean(scores)

    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))
    study.optimize(objective, n_trials=MAX_TRIALS)
    best = study.best_params
    final_model = CatBoostRegressor(**best, iterations=1000, random_seed=RANDOM_SEED, verbose=False, allow_writing_files=False)
    final_model.fit(X, y, verbose=False)
    return final_model, best
