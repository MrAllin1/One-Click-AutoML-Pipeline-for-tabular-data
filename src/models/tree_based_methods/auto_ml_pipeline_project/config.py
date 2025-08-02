# config.py

"""
Global configuration parameters for the AutoML pipeline.
"""

MAX_TRIALS = 450           # Optuna HPO trials 60 is the default
CV_FOLDS = 15              # Cross-validation folds
RANDOM_SEED = 42          # Random seed for reproducibility
GPU_SAMPLE_THRESHOLD = 1000  # Minimum rows to enable GPU training
EARLY_STOPPING_ROUNDS = 200   # Early stopping patience
ENSEMBLE = True          # Whether to train and blend both models
tb_log_subdir = "tenserboard_logs"