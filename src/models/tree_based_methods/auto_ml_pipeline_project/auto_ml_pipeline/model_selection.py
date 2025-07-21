"""
Selects algorithm (LightGBM or CatBoost), trains with HPO, ensembles.
"""
import os

from sklearn.metrics import r2_score
from config import ENSEMBLE
from .hyperparameter_tuning import tune_lgbm, tune_catboost


def select_and_train(X_tr, y_tr, X_te, y_te, output_dir):
    """
    Chooses model based on categorical fraction, runs HPO, optionally ensembles.
    Returns dict of trained models and test metrics.
    """
    results = {}
    models = {}

    # Profile dataset
    cat_frac = sum(X_tr.dtypes=='category') / X_tr.shape[1]

    # First model choice
    use_cat = cat_frac > 0.3 or X_tr.nunique().max() > 50
    if use_cat:
        name1 = 'catboost'
        model1, _ = tune_catboost(X_tr, y_tr)
    else:
        name1 = 'lightgbm'
        model1, _ = tune_lgbm(X_tr, y_tr)

    preds1 = model1.predict(X_te)
    r2_1 = r2_score(y_te, preds1)
    models[name1] = model1
    results[name1] = r2_1

    # Ensemble second model
    if ENSEMBLE:
        name2 = 'lightgbm' if name1=='catboost' else 'catboost'
        if name2=='catboost':
            model2, _ = tune_catboost(X_tr, y_tr)
        else:
            model2, _ = tune_lgbm(X_tr, y_tr)
        preds2 = model2.predict(X_te)
        r2_2 = r2_score(y_te, preds2)
        models[name2] = model2
        results[name2] = r2_2

        blend = 0.5*preds1 + 0.5*preds2
        results['ensemble'] = r2_score(y_te, blend)

    return models, results