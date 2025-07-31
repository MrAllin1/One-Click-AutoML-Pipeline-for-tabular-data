# config.py

"""
Global configuration parameters for the AutoML pipeline.
"""

MAX_TRIALS            =   5    # → only 5 HPO trials instead of 450
CV_FOLDS              =   3    # → 3-fold CV to cut training by ~5×
RANDOM_SEED           =  42    # → keep for reproducibility
GPU_SAMPLE_THRESHOLD  =   0    # → with 0 rows threshold, GPU is always used if available
EARLY_STOPPING_ROUNDS =  10    # → stop after 10 rounds of no improvement
ENSEMBLE              = True  # → train just one model, no blending
