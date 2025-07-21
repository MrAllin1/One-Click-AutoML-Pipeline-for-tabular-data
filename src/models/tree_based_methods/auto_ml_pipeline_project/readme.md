#-------------------------------------
# # AutoML Pipeline for Tabular Regression
# 
# This project implements a one-click pipeline that:
# - Loads a tabular dataset (single split or folder of splits)
# - Preprocesses data (missing values, categoricals, outliers)
# - Engineers features (datetime, interactions)
# - Selects LightGBM or CatBoost based on data profile
# - Tunes hyperparameters via Optuna (maximizing R²)
# - Optionally ensembles models
# - Saves the final model and reports test R²
# 
# ## Usage for split-based evaluation
# ```bash
# python main.py --data_path /.../bike_sharing_demand/1 --output_dir ./results/split_1
# ```
# ## Usage for final model on all splits
# ```bash
# python final_train.py --dataset_path /.../bike_sharing_demand --output_dir ./results/final_model
# ```