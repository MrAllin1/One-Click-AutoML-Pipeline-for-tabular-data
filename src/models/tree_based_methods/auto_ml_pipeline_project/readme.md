#-------------------------------------
# # AutoML Pipeline for Tree Based Model
# 
# This project implements a one-click pipeline that:
# - Loads a tabular dataset (single split or folder of splits)
# - Preprocesses data (missing values, categoricals, outliers)
# - Engineers features (datetime, interactions)
# - Selects LightGBM or CatBoost based on data profile
# - Tunes hyperparameters via Optuna (maximizing R²)
# - Optionally ensembles models
# - Saves the final model and reports test R²
